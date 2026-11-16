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

"""State shared by the sync and async ``terminate(wait=True)`` implementations.

The wait covers the claim, the Sandbox names this handle knows about and those
Sandboxes' Pod names. It is a bounded guarantee about those *known* objects:
it says nothing about objects a controller creates later, and a Pod that is gone
from the API is not proof that its processes stopped on an unreachable node.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from .constants import POD_NAME_ANNOTATION, SANDBOX_NAME_ANNOTATION

# Seconds between polling passes.
POLL_INTERVAL_SECONDS = 0.5


def validate_wait_timeout(timeout: float) -> float:
    """Return ``timeout`` as a float, or raise ``ValueError`` unless it is finite and positive."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
            or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"timeout must be a finite number of seconds greater than 0, got {timeout!r}")
    return float(timeout)


class DeadlineExceeded(Exception):
    """The shared deletion-wait budget ran out, possibly inside a request."""


@dataclass
class PendingDeletion:
    """Names to keep checking until all of them are absent in a single pass.

    Names are only ever added, so a name that was absent once is checked again.
    A handle keeps this record after a timeout so a retry still checks them.
    """

    claim_name: str
    sandboxes: set[str] = field(default_factory=set)
    pods: set[str] = field(default_factory=set)

    @classmethod
    def for_handle(cls, claim_name: str, sandbox_name: str, pod_name: str | None) -> "PendingDeletion":
        # With no recorded Pod name the Pod is named after its Sandbox.
        return cls(claim_name, {sandbox_name}, {pod_name or sandbox_name})

    def learn_claim(self, claim: dict[str, Any]) -> None:
        """Add the Sandbox names a claim reports (a warm pool may adopt a differently named one)."""
        status_name = ((claim.get("status") or {}).get("sandbox") or {}).get("name")
        annotation_name = ((claim.get("metadata") or {}).get("annotations") or {}).get(SANDBOX_NAME_ANNOTATION)
        self.sandboxes.update(n for n in (status_name, annotation_name) if n)

    def learn_sandbox(self, name: str, sandbox: dict[str, Any]) -> None:
        """Add the Pod a live Sandbox reports: its annotation if set, else its own name."""
        annotations = (sandbox.get("metadata") or {}).get("annotations") or {}
        self.pods.add(annotations.get(POD_NAME_ANNOTATION) or name)

    def timeout_error(
        self, timeout: float, present: list[str], unverified: str | None, delete_sent: bool,
    ) -> TimeoutError:
        parts = [
            f"Timed out after {timeout:g}s waiting for SandboxClaim '{self.claim_name}' and the Sandboxes "
            f"and Pods known for it to be deleted."
        ]
        parts.append(
            "Still present at the last completed pass: " + ", ".join(present) + "." if present
            else "Nothing was observed present at the last completed pass."
        )
        if unverified:
            parts.append(f"Status unverified: the request for {unverified} did not finish.")
        if delete_sent:
            parts.append(
                "The delete call was attempted; any deletion in progress is left alone. "
                "call terminate(wait=True) again to keep waiting."
            )
        else:
            parts.append("The delete call had not been attempted; call terminate(wait=True) again to retry.")
        return TimeoutError(" ".join(parts))
