# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``terminate(wait=True)``.

The helper tests talk to a small fake Kubernetes API server over real HTTP, so
they exercise the actual Kubernetes client request path (query parameters,
retries and per-request timeouts) rather than a mock of it.
"""

import asyncio
import json
import math
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from kubernetes import client
from urllib3.exceptions import NewConnectionError

from k8s_agent_sandbox import deletion_wait
from k8s_agent_sandbox.deletion_wait import PendingDeletion
from k8s_agent_sandbox.k8s_helper import K8sHelper
from k8s_agent_sandbox.models import (
    SandboxDirectConnectionConfig,
    SandboxLocalTunnelConnectionConfig,
    SandboxTracerConfig,
)
from k8s_agent_sandbox.sandbox import Sandbox

NS = "demo"
CLAIM_PATH = f"/apis/extensions.agents.x-k8s.io/v1beta1/namespaces/{NS}/sandboxclaims/"
SANDBOX_PATH = f"/apis/agents.x-k8s.io/v1beta1/namespaces/{NS}/sandboxes/"
POD_PATH = f"/api/v1/namespaces/{NS}/pods/"

GONE = None  # a scripted 404
STALL = "stall"  # never answer until the server is closed


def claim(name="claim-1", sandbox=None, annotation=None):
    obj: dict[str, Any] = {"metadata": {"name": name}}
    if sandbox:
        obj["status"] = {"sandbox": {"name": sandbox}}
    if annotation:
        obj["metadata"]["annotations"] = {"agents.x-k8s.io/sandbox-name": annotation}
    return obj


def sandbox(name="sb-1", pod_annotation=None):
    obj: dict[str, Any] = {"metadata": {"name": name}}
    if pod_annotation:
        obj["metadata"]["annotations"] = {"agents.x-k8s.io/pod-name": pod_annotation}
    return obj


def pod(name):
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": name}}


class FakeKubeAPI:
    """A scripted API server. Each path answers from a list of states, one per
    GET; the last state repeats. A state is a body (200), ``GONE`` (404), an
    int status code, or ``STALL``.
    """

    def __init__(self) -> None:
        self.script: dict[str, list[Any]] = {}
        self.delete_status = 200
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _serve(self) -> None:
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                with outer._lock:
                    outer.requests.append({
                        "method": self.command,
                        "path": parsed.path,
                        "query": parse_qs(parsed.query),
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": self.rfile.read(length) if length else b"",
                    })
                    if self.command == "DELETE":
                        state: Any = outer.delete_status
                    else:
                        states = outer.script.get(parsed.path, [GONE])
                        state = states.pop(0) if len(states) > 1 else states[0]
                if state == STALL:
                    outer._closed.wait(30)
                    return
                if state is GONE:
                    code, body = 404, {"kind": "Status", "code": 404, "reason": "NotFound"}
                elif isinstance(state, int):
                    code, body = state, {"kind": "Status", "code": state}
                else:
                    code, body = 200, state
                payload = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_DELETE = _serve

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._closed.set()
        self.server.shutdown()
        self.server.server_close()

    def count(self, method: str, path: str) -> int:
        with self._lock:
            return sum(1 for r in self.requests if r["method"] == method and r["path"] == path)

    def deletes(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.requests if r["method"] == "DELETE"]


class SyncHelperCase(unittest.TestCase):
    def setUp(self):
        self.api = FakeKubeAPI()
        self.addCleanup(self.api.close)
        poll = patch.object(deletion_wait, "POLL_INTERVAL_SECONDS", 0.01)
        poll.start()
        self.addCleanup(poll.stop)
        # A caller client with the library's retry behaviour switched on, to
        # prove the wait path does not inherit it.
        self.config = client.Configuration()
        self.config.host = self.api.url
        self.config.retries = 3
        self.config.client_go_retries = True
        self.caller_client = client.ApiClient(self.config)
        self.addCleanup(self.caller_client.close)
        self.helper = K8sHelper(api_client=self.caller_client)

    def wait(self, pending=None, timeout=2.0):
        pending = pending or PendingDeletion.for_handle("claim-1", "sb-1", "pod-1")
        self.helper.delete_sandbox_claim_and_wait("claim-1", NS, timeout, pending)
        return pending

    def script(self, path, *states):
        self.api.script[path] = list(states)


class TestSyncHelperDeletionWait(SyncHelperCase):
    def test_sends_foreground_delete_for_the_claim_only(self):
        self.script(CLAIM_PATH + "claim-1", claim(), GONE)
        self.wait()
        deletes = self.api.deletes()
        self.assertEqual([d["path"] for d in deletes], [CLAIM_PATH + "claim-1"])
        self.assertEqual(deletes[0]["query"]["propagationPolicy"], ["Foreground"])

    def test_claim_404_with_live_sandbox_and_pod_is_not_success(self):
        self.script(CLAIM_PATH + "claim-1", GONE)
        self.script(SANDBOX_PATH + "sb-1", sandbox("sb-1", pod_annotation="pod-x"))
        self.script(POD_PATH + "pod-x", pod("pod-x"))
        with self.assertRaises(TimeoutError) as caught:
            self.wait(timeout=0.3)
        message = str(caught.exception)
        self.assertIn("Sandbox 'sb-1'", message)
        self.assertIn("Pod 'pod-x'", message)
        self.assertEqual(len(self.api.deletes()), 1)

    def test_returns_only_after_claim_sandbox_and_pod_are_all_gone(self):
        self.script(CLAIM_PATH + "claim-1", claim(), GONE)
        self.script(SANDBOX_PATH + "sb-1", sandbox("sb-1"), sandbox("sb-1"), GONE)
        self.script(POD_PATH + "sb-1", pod("sb-1"), pod("sb-1"), pod("sb-1"), GONE)
        self.wait(PendingDeletion.for_handle("claim-1", "sb-1", None))
        # Pod was read until it 404ed: pre-delete pass plus three polls.
        self.assertEqual(self.api.count("GET", POD_PATH + "sb-1"), 4)
        last_three = [r["path"] for r in self.api.requests[-3:]]
        self.assertEqual(last_three, [CLAIM_PATH + "claim-1", SANDBOX_PATH + "sb-1", POD_PATH + "sb-1"])

    def test_claim_delete_404_is_success_once_everything_is_gone(self):
        self.api.delete_status = 404
        self.wait()

    def test_lost_sandbox_retains_known_pod_across_timeout_and_retry(self):
        pending = PendingDeletion.for_handle("claim-1", "sb-1", "pod-1")
        self.script(POD_PATH + "pod-1", pod("pod-1"))
        with self.assertRaises(TimeoutError) as caught:
            self.wait(pending, timeout=0.3)
        self.assertIn("Pod 'pod-1'", str(caught.exception))
        self.script(POD_PATH + "pod-1", pod("pod-1"), GONE)
        self.wait(pending)

    def test_pod_annotation_learned_before_delete_is_retained_for_retry(self):
        pending = PendingDeletion.for_handle("claim-1", "sb-1", None)
        self.script(CLAIM_PATH + "claim-1", claim())
        self.script(SANDBOX_PATH + "sb-1", sandbox("sb-1", pod_annotation="pod-x"))
        self.script(POD_PATH + "pod-x", pod("pod-x"))
        with self.assertRaises(TimeoutError):
            self.wait(pending, timeout=0.3)
        self.assertIn("pod-x", pending.pods)
        # Claim and Sandbox vanish; only the learned Pod remains to be seen.
        self.script(CLAIM_PATH + "claim-1", GONE)
        self.script(SANDBOX_PATH + "sb-1", GONE)
        self.script(POD_PATH + "pod-x", pod("pod-x"), GONE)
        before = self.api.count("GET", POD_PATH + "pod-x")
        self.wait(pending)
        self.assertGreaterEqual(self.api.count("GET", POD_PATH + "pod-x") - before, 2)

    def test_adopted_sandbox_names_from_claim_are_followed(self):
        self.script(CLAIM_PATH + "claim-1", claim(sandbox="adopted", annotation="adopted-2"), GONE)
        self.script(SANDBOX_PATH + "adopted", sandbox("adopted"))
        self.script(SANDBOX_PATH + "adopted-2", GONE)
        self.script(POD_PATH + "adopted", pod("adopted"))
        with self.assertRaises(TimeoutError) as caught:
            self.wait(timeout=0.3)
        self.assertIn("Sandbox 'adopted'", str(caught.exception))
        self.assertGreater(self.api.count("GET", SANDBOX_PATH + "adopted-2"), 0)

    def test_disappeared_names_are_rechecked_until_one_clean_pass(self):
        # pre-delete, pass 1, pass 2, pass 3: no pass before the last has all
        # three absent at once, and earlier absences must not be remembered.
        self.script(CLAIM_PATH + "claim-1", claim(), GONE)
        self.script(SANDBOX_PATH + "sb-1", sandbox("sb-1", pod_annotation="pod-x"), GONE, sandbox("sb-1"), GONE)
        self.script(POD_PATH + "pod-x", pod("pod-x"), pod("pod-x"), GONE, GONE)
        pending = self.wait(PendingDeletion.for_handle("claim-1", "sb-1", "pod-x"))
        self.assertEqual(self.api.count("GET", CLAIM_PATH + "claim-1"), 4)
        self.assertEqual(self.api.count("GET", SANDBOX_PATH + "sb-1"), 4)
        self.assertEqual(pending.pods, {"pod-x", "sb-1"})

    def test_non_404_errors_propagate(self):
        self.script(POD_PATH + "pod-1", 403)
        with self.assertRaises(client.ApiException) as caught:
            self.wait()
        self.assertEqual(caught.exception.status, 403)

    def test_non_404_read_before_delete_prevents_the_delete(self):
        self.script(CLAIM_PATH + "claim-1", 403)
        with self.assertRaises(client.ApiException):
            self.wait()
        self.assertEqual(self.api.deletes(), [])

    def test_non_404_delete_error_propagates_without_polling(self):
        self.api.delete_status = 500
        with self.assertRaises(client.ApiException) as caught:
            self.wait()
        self.assertEqual(caught.exception.status, 500)
        # one pre-delete pass (claim, sandbox, pod), then the DELETE, nothing after
        self.assertEqual(self.api.requests[-1]["method"], "DELETE")

    def test_requests_use_a_private_retry_free_client_and_leave_the_callers_alone(self):
        self.caller_client.default_headers["X-Test"] = "kept"
        self.caller_client.cookie = "session=abc"
        self.script(CLAIM_PATH + "claim-1", GONE)
        self.wait()
        for request in self.api.requests:
            self.assertEqual(request["headers"].get("x-test"), "kept")
            self.assertEqual(request["headers"].get("cookie"), "session=abc")
        self.assertEqual(self.config.retries, 3)
        self.assertTrue(self.config.client_go_retries)
        # The caller's client was not closed and still works.
        self.assertIsNone(self.helper.get_sandbox_claim("claim-1", NS))

    def test_stalled_server_cannot_multiply_the_deadline_through_retries(self):
        # Control: the caller's own client retries a stalled GET.
        self.script(CLAIM_PATH + "control", STALL)
        with self.assertRaises(Exception):
            self.helper.custom_objects_api.get_namespaced_custom_object(
                "extensions.agents.x-k8s.io", "v1beta1", NS, "sandboxclaims", "control",
                _request_timeout=0.2,
            )
        self.assertGreater(self.api.count("GET", CLAIM_PATH + "control"), 1)

        self.script(CLAIM_PATH + "claim-1", STALL)
        pending = PendingDeletion.for_handle("claim-1", "sb-1", "pod-1")
        started = time.monotonic()
        with self.assertRaises(TimeoutError) as caught:
            self.wait(pending, timeout=0.5)
        elapsed = time.monotonic() - started
        self.assertEqual(type(caught.exception), TimeoutError)
        self.assertLess(elapsed, 1.5)
        self.assertEqual(self.api.count("GET", CLAIM_PATH + "claim-1"), 1)
        self.assertIn("unverified", str(caught.exception))
        self.assertIn("SandboxClaim 'claim-1'", str(caught.exception))
        self.assertIn("had not been attempted", str(caught.exception))
        self.assertEqual(self.api.deletes(), [])
        # Names stay pending for a retry.
        self.assertEqual((pending.sandboxes, pending.pods), ({"sb-1"}, {"pod-1"}))

    def test_connection_refusal_is_not_reported_as_deadline_expiry(self):
        self.api.close()
        with self.assertRaises(NewConnectionError):
            self.wait()

    def test_deadline_before_delete_does_not_claim_the_request_was_sent(self):
        with patch("k8s_agent_sandbox.k8s_helper.time") as clock:
            clock.monotonic.side_effect = [0, 0, 0, 0, 2]
            with self.assertRaises(TimeoutError) as caught:
                self.wait(timeout=1)
        self.assertIn("had not been attempted", str(caught.exception))
        self.assertEqual(self.api.deletes(), [])

    def test_stalled_delete_reports_unverified_and_does_not_retry_the_delete(self):
        self.script(CLAIM_PATH + "claim-1", claim())
        self.api.delete_status = STALL  # type: ignore[assignment]
        with self.assertRaises(TimeoutError) as caught:
            self.wait(timeout=0.5)
        self.assertIn("unverified", str(caught.exception))
        self.assertIn("call was attempted", str(caught.exception))
        self.assertEqual(len(self.api.deletes()), 1)

    def test_timeout_after_the_pre_delete_pass_reports_what_that_pass_saw(self):
        self.script(CLAIM_PATH + "claim-1", claim(), STALL)
        with self.assertRaises(TimeoutError) as caught:
            self.wait(timeout=0.5)
        message = str(caught.exception)
        self.assertIn("Still present at the last completed pass: SandboxClaim 'claim-1'", message)
        self.assertIn("call was attempted", message)


def build_sandbox(helper, pod_name="pod-1"):
    with patch("k8s_agent_sandbox.sandbox.Filesystem"), \
            patch("k8s_agent_sandbox.sandbox.CommandExecutor"), \
            patch("k8s_agent_sandbox.sandbox.create_tracer_manager", return_value=(MagicMock(), MagicMock())), \
            patch("k8s_agent_sandbox.sandbox.SandboxConnector"):
        box = Sandbox(
            claim_name="claim-1", sandbox_id="sb-1", namespace=NS,
            connection_config=SandboxLocalTunnelConnectionConfig(),
            tracer_config=SandboxTracerConfig(), k8s_helper=helper,
        )
    box._pod_name = pod_name
    return box


class TestSyncSandboxTerminate(unittest.TestCase):
    def setUp(self):
        self.helper = MagicMock()
        self.box = build_sandbox(self.helper)
        self.hook = MagicMock()
        self.box._on_terminated = self.hook

    def test_default_path_is_unchanged(self):
        self.box.terminate()
        self.helper.delete_sandbox_claim.assert_called_once_with("claim-1", NS)
        self.helper.delete_sandbox_claim_and_wait.assert_not_called()
        self.assertIsNone(self.box.claim_name)

    def test_wait_uses_the_wait_helper_with_names_known_before_close(self):
        self.box.terminate(wait=True, timeout=7)
        self.helper.delete_sandbox_claim.assert_not_called()
        (name, namespace, timeout, pending), _ = self.helper.delete_sandbox_claim_and_wait.call_args
        self.assertEqual((name, namespace, timeout), ("claim-1", NS, 7.0))
        # close_connection() clears _pod_name; the pending record must not.
        self.assertIsNone(self.box._pod_name)
        self.assertEqual((pending.sandboxes, pending.pods), ({"sb-1"}, {"pod-1"}))
        self.assertIsNone(self.box.claim_name)
        self.hook.assert_called_once()

    def test_timeout_keeps_claim_registry_and_pending_names_for_retry(self):
        self.helper.delete_sandbox_claim_and_wait.side_effect = TimeoutError("still there")
        with self.assertRaises(TimeoutError):
            self.box.terminate(wait=True)
        self.assertEqual(self.box.claim_name, "claim-1")
        self.hook.assert_not_called()
        first = self.helper.delete_sandbox_claim_and_wait.call_args.args[3]

        self.helper.delete_sandbox_claim_and_wait.side_effect = None
        self.box.terminate(wait=True)
        self.assertIs(self.helper.delete_sandbox_claim_and_wait.call_args.args[3], first)
        self.assertIn("pod-1", first.pods)
        self.assertIsNone(self.box.claim_name)
        self.hook.assert_called_once()

    def test_plain_terminate_after_failed_wait_is_delete_only_and_clears_state(self):
        self.helper.delete_sandbox_claim_and_wait.side_effect = TimeoutError("still there")
        with self.assertRaises(TimeoutError):
            self.box.terminate(wait=True)
        self.box.terminate()
        self.helper.delete_sandbox_claim.assert_called_once_with("claim-1", NS)
        self.assertIsNone(self.box.claim_name)
        self.assertIsNone(self.box._pending_deletion)

    def test_invalid_timeout_has_no_side_effects(self):
        for bad in (0, -1, math.nan, math.inf, -math.inf, True, "5", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.box.terminate(wait=True, timeout=bad)  # type: ignore[arg-type]
        self.assertTrue(self.box.is_active)
        self.assertEqual(self.box._pod_name, "pod-1")
        self.assertEqual([c for c in self.helper.mock_calls if c[0] != "__bool__"], [])
        self.hook.assert_not_called()

    def test_timeout_is_not_validated_when_not_waiting(self):
        self.box.terminate(timeout=0)  # type: ignore[call-arg]
        self.helper.delete_sandbox_claim.assert_called_once()

    def test_wait_is_keyword_only(self):
        with self.assertRaises(TypeError):
            self.box.terminate(True)  # type: ignore[misc]


class TestAsyncHelperDeletionWait(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        pytest.importorskip("kubernetes_asyncio")
        from kubernetes_asyncio import client as aclient
        from k8s_agent_sandbox.async_k8s_helper import AsyncK8sHelper

        self.api = FakeKubeAPI()
        self.addCleanup(self.api.close)
        poll = patch.object(deletion_wait, "POLL_INTERVAL_SECONDS", 0.01)
        poll.start()
        self.addCleanup(poll.stop)
        config = aclient.Configuration()
        config.host = self.api.url
        self.api_client = aclient.ApiClient(config)
        self.addAsyncCleanup(self.api_client.close)
        self.helper = AsyncK8sHelper(api_client=self.api_client)

    async def wait(self, pending=None, timeout=2.0):
        pending = pending or PendingDeletion.for_handle("claim-1", "sb-1", "pod-1")
        await self.helper.delete_sandbox_claim_and_wait("claim-1", NS, timeout, pending)
        return pending

    def script(self, path, *states):
        self.api.script[path] = list(states)

    async def test_sends_foreground_delete(self):
        self.script(CLAIM_PATH + "claim-1", claim(), GONE)
        await self.wait()
        deletes = self.api.deletes()
        self.assertEqual([d["path"] for d in deletes], [CLAIM_PATH + "claim-1"])
        self.assertEqual(deletes[0]["query"]["propagationPolicy"], ["Foreground"])

    async def test_claim_404_with_live_sandbox_and_pod_is_not_success(self):
        self.script(SANDBOX_PATH + "sb-1", sandbox("sb-1", pod_annotation="pod-x"))
        self.script(POD_PATH + "pod-x", pod("pod-x"))
        with self.assertRaises(TimeoutError) as caught:
            await self.wait(timeout=0.3)
        self.assertIn("Sandbox 'sb-1'", str(caught.exception))
        self.assertIn("Pod 'pod-x'", str(caught.exception))

    async def test_returns_after_everything_is_gone_and_retains_names_on_timeout(self):
        pending = PendingDeletion.for_handle("claim-1", "sb-1", "pod-1")
        self.script(POD_PATH + "pod-1", pod("pod-1"))
        with self.assertRaises(TimeoutError):
            await self.wait(pending, timeout=0.3)
        self.script(POD_PATH + "pod-1", pod("pod-1"), GONE)
        await self.wait(pending)
        self.assertGreaterEqual(self.api.count("GET", POD_PATH + "pod-1"), 3)

    async def test_non_404_errors_propagate(self):
        from kubernetes_asyncio import client as aclient
        self.script(POD_PATH + "pod-1", 403)
        with self.assertRaises(aclient.ApiException) as caught:
            await self.wait()
        self.assertEqual(caught.exception.status, 403)

    async def test_stalled_server_times_out_without_closing_the_callers_session(self):
        self.script(CLAIM_PATH + "claim-1", STALL)
        started = time.monotonic()
        with self.assertRaises(TimeoutError) as caught:
            await self.wait(timeout=0.5)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIn("unverified", str(caught.exception))
        self.assertEqual(self.api.deletes(), [])
        # The existing client is still usable.
        self.script(SANDBOX_PATH + "later", sandbox("later"))
        self.assertIsNotNone(await self.helper.get_sandbox("later", NS))

    async def test_cancellation_propagates_and_keeps_the_client_usable(self):
        self.script(POD_PATH + "pod-1", pod("pod-1"))
        task = asyncio.create_task(self.wait(timeout=30))
        await asyncio.sleep(0.2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.script(SANDBOX_PATH + "later", sandbox("later"))
        self.assertIsNotNone(await self.helper.get_sandbox("later", NS))


class TestAsyncSandboxTerminate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pytest.importorskip("kubernetes_asyncio")
        pytest.importorskip("httpx")
        from k8s_agent_sandbox.async_sandbox import AsyncSandbox

        self.helper = MagicMock()
        self.helper.delete_sandbox_claim = AsyncMock()
        self.helper.delete_sandbox_claim_and_wait = AsyncMock()
        with patch("k8s_agent_sandbox.async_sandbox.AsyncFilesystem"), \
                patch("k8s_agent_sandbox.async_sandbox.AsyncCommandExecutor"), \
                patch("k8s_agent_sandbox.async_sandbox.create_tracer_manager",
                      return_value=(MagicMock(), MagicMock())), \
                patch("k8s_agent_sandbox.async_sandbox.AsyncSandboxConnector") as connector:
            connector.return_value.close = AsyncMock()
            self.box = AsyncSandbox(
                claim_name="claim-1", sandbox_id="sb-1", namespace=NS,
                connection_config=SandboxDirectConnectionConfig(api_url="http://router:8080"),
                tracer_config=SandboxTracerConfig(), k8s_helper=self.helper,
            )
        self.box._pod_name = "pod-1"
        self.hook = MagicMock()
        self.box._on_terminated = self.hook

    async def test_default_path_is_unchanged(self):
        await self.box.terminate()
        self.helper.delete_sandbox_claim.assert_awaited_once_with("claim-1", NS)
        self.helper.delete_sandbox_claim_and_wait.assert_not_called()
        self.assertIsNone(self.box.claim_name)

    async def test_wait_passes_names_and_timeout_and_clears_state_on_success(self):
        await self.box.terminate(wait=True, timeout=7)
        self.helper.delete_sandbox_claim.assert_not_called()
        name, namespace, timeout, pending = self.helper.delete_sandbox_claim_and_wait.await_args.args
        self.assertEqual((name, namespace, timeout), ("claim-1", NS, 7.0))
        self.assertEqual((pending.sandboxes, pending.pods), ({"sb-1"}, {"pod-1"}))
        self.assertIsNone(self.box.claim_name)
        self.hook.assert_called_once()

    async def test_timeout_retains_claim_registry_and_pending_names(self):
        self.helper.delete_sandbox_claim_and_wait.side_effect = TimeoutError("still there")
        with self.assertRaises(TimeoutError):
            await self.box.terminate(wait=True)
        self.assertEqual(self.box.claim_name, "claim-1")
        self.hook.assert_not_called()
        first = self.helper.delete_sandbox_claim_and_wait.await_args.args[3]
        self.helper.delete_sandbox_claim_and_wait.side_effect = None
        await self.box.terminate(wait=True)
        self.assertIs(self.helper.delete_sandbox_claim_and_wait.await_args.args[3], first)

    async def test_cancellation_keeps_claim_registry_and_pending_names(self):
        started = asyncio.Event()

        async def hang(*args):
            started.set()
            await asyncio.Event().wait()

        self.helper.delete_sandbox_claim_and_wait.side_effect = hang
        task = asyncio.create_task(self.box.terminate(wait=True))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.box.claim_name, "claim-1")
        self.hook.assert_not_called()
        self.assertIsNotNone(self.box._pending_deletion)

    async def test_plain_terminate_after_failed_wait_is_delete_only(self):
        self.helper.delete_sandbox_claim_and_wait.side_effect = TimeoutError("still there")
        with self.assertRaises(TimeoutError):
            await self.box.terminate(wait=True)
        await self.box.terminate()
        self.helper.delete_sandbox_claim.assert_awaited_once_with("claim-1", NS)
        self.assertIsNone(self.box._pending_deletion)

    async def test_invalid_timeout_has_no_side_effects(self):
        for bad in (0, -1, math.nan, math.inf, -math.inf, True, "5", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await self.box.terminate(wait=True, timeout=bad)  # type: ignore[arg-type]
        self.assertTrue(self.box.is_active)
        self.assertEqual([c for c in self.helper.mock_calls if c[0] != "__bool__"], [])
        self.hook.assert_not_called()
