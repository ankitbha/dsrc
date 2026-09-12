"""Load and run the exported DSRC bundle: one argmax over the whole network.

Parallel to `policy.actor_runtime.ActorRuntime`, but for the whole-network
controller rather than the 39-field local actor. No numpy mirror: section 5
of `plans/plan_task145_dsrc_policy_runtime.md` measured `greedy_actions` at
0.0123 ms p50 in torch eager on this machine for an 11,676-parameter
network, well inside a 200 ms end-to-end budget, so a mirror is available
if an Orin measurement (step 12) ever warrants one but is not built ahead
of that measurement.

**The refusal has no grandfather clause.** `ActorRuntime` accepts a bundle
exported before `contract_fingerprint` existed, because refusing every
older bundle already in the field would have been a bigger, unrelated
disruption. No DSRC bundle exists yet, so there is nothing to grandfather:
a manifest with no `network_fingerprint` at all is refused exactly like one
that carries the wrong value.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from perception.segment_state import (
    DEFAULT_NETWORK_DEFINITION_PATH,
    SEGMENT_BASIS_MEASURED,
    SegmentStateResult,
)
from policy.dsrc_contract import SPEED_ACTION_FRACTIONS, network_fingerprint

torch.set_num_threads(1)

#: Every segment was measured and the network ran.
OUTCOME_OK = "ok"
#: `SegmentStateResult` itself is why there is no action -- propagate its
#: own outcome rather than inventing a second name for the same fact.


@dataclass
class DsrcActResult:
    """One raw inference: a state in, an action and its Q-values out."""

    actions: np.ndarray  # (num_segments,) int64
    q_values: np.ndarray  # (num_segments, num_actions) float32
    latency_ms: float


@dataclass
class DsrcDecision:
    """One decision: either every segment was measured and the network ran,
    or it did not, and this says why."""

    outcome: str
    actions: np.ndarray | None
    q_values: np.ndarray | None
    segment_basis: tuple[str, ...]
    matched_links: tuple[int, ...]
    #: None when the network did not run this decision.
    latency_ms: float | None


def _load_network_definition(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


class DsrcRuntime:
    """Loads a DSRC bundle (`<prefix>.ts` + `<prefix>.json`), refuses a
    bundle whose network_fingerprint does not match the device's own network
    definition, and runs the argmax.

    `network_definition_path` is read independently of the manifest and is
    the runtime's own source of truth for what network it believes it is
    running on; the manifest is what the BUNDLE claims, and the two must
    agree.
    """

    def __init__(
        self,
        bundle_prefix: str,
        network_definition_path: Path | str = DEFAULT_NETWORK_DEFINITION_PATH,
    ) -> None:
        prefix = Path(bundle_prefix)
        manifest_path = prefix.with_suffix(".json")
        module_path = prefix.with_suffix(".ts")
        if not module_path.exists():
            raise FileNotFoundError(
                f"DSRC bundle not found at {module_path} - create one with\n"
                "  python3 policy/export_dsrc_policy.py --checkpoint "
                "results/checkpoints/mainz_here_best.pt --out models/dsrc_policy"
            )
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        definition = _load_network_definition(network_definition_path)
        self.network_id: str = definition["network_id"]
        self.feature_names: tuple[str, ...] = tuple(definition["feature_names"])
        self.segment_ids: tuple[tuple[str, ...], ...] = tuple(
            tuple(segment["edge_ids"]) for segment in definition["segments"]
        )
        self.segment_speed_limits_kmh: tuple[float, ...] = tuple(
            float(segment["speed_limit_kmh"]) for segment in definition["segments"]
        )
        self.num_segments = len(self.segment_ids)
        self.num_features = len(self.feature_names)
        self.num_actions = len(SPEED_ACTION_FRACTIONS)

        device_fingerprint = network_fingerprint(
            network_id=self.network_id,
            feature_names=self.feature_names,
            segment_ids=self.segment_ids,
            segment_speed_limits_kmh=self.segment_speed_limits_kmh,
        )
        bundle_fingerprint = self.manifest.get("network_fingerprint")
        if bundle_fingerprint is None:
            raise RuntimeError(
                "DSRC bundle manifest carries no network_fingerprint at all -- "
                "unlike the 39-field actor's contract_fingerprint, this check has "
                "no grandfather clause (no DSRC bundle predates it); re-export "
                "the policy with policy/export_dsrc_policy.py."
            )
        if bundle_fingerprint != device_fingerprint:
            raise RuntimeError(
                f"DSRC bundle was exported for network_fingerprint "
                f"{bundle_fingerprint} but this device's network definition "
                f"({network_definition_path}) is {device_fingerprint} -- same "
                "shape, different segment ids, feature order or speed limits; "
                "re-export the policy against this network definition."
            )

        bundle_fractions = tuple(self.manifest.get("action_fractions", ()))
        if bundle_fractions != tuple(SPEED_ACTION_FRACTIONS):
            raise RuntimeError(
                f"DSRC bundle action_fractions {bundle_fractions} does not match "
                f"the vendored {tuple(SPEED_ACTION_FRACTIONS)} -- same action "
                "count, different meaning per index; re-export the policy."
            )

        manifest_dims = (
            int(self.manifest.get("num_segments", -1)),
            int(self.manifest.get("num_features", -1)),
            int(self.manifest.get("num_actions", -1)),
        )
        expected_dims = (self.num_segments, self.num_features, self.num_actions)
        if manifest_dims != expected_dims:
            raise RuntimeError(
                f"DSRC bundle manifest declares (num_segments, num_features, "
                f"num_actions)={manifest_dims}, device expects {expected_dims}; "
                "re-export the policy."
            )

        self.module = torch.jit.load(str(module_path), map_location="cpu")
        self.module.eval()
        self.is_trained = bool(self.manifest.get("trained", False))
        # Warm the kernels so the first live decision is not an outlier.
        self.act(np.zeros((self.num_segments, self.num_features), dtype=np.float32))

    def act(self, state: np.ndarray) -> DsrcActResult:
        """Unconditional inference on a raw `(num_segments, num_features)` state.

        No coverage gate -- this is the function the golden-action
        demonstration (`tests/test_dsrc_runtime.py`) and any replay of
        recorded simulator states call directly. `decide` below is the
        live-path entry point that enforces full coverage before calling
        this.
        """
        t0 = time.monotonic()
        # copy=True: a read-only source (e.g. a view into a base64-decoded
        # buffer, as tests/test_dsrc_runtime.py's random-state block hands
        # in) would otherwise reach torch as a non-writable tensor.
        tensor = torch.from_numpy(np.array(state, dtype=np.float32, copy=True))
        with torch.inference_mode():
            logits = self.module(tensor)
            # SRC applies a softmax before the argmax (`src.rl.src_q.
            # greedy_actions`'s own docstring: it cannot change which index
            # is largest). Kept so this is the same port, not merely the
            # same choice -- section 1.7 measured 0 differences over 20,000
            # random states between the two.
            actions = torch.argmax(F.softmax(logits, dim=1), dim=1)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return DsrcActResult(
            actions=actions.numpy().astype(np.int64),
            q_values=logits.numpy().astype(np.float32),
            latency_ms=latency_ms,
        )

    def decide(self, segment_state: SegmentStateResult) -> DsrcDecision:
        """The live-path entry point: an action only when every segment was measured.

        Mirrors `SegmentStateResult` back into the decision it caused so a
        caller has both without re-deriving one from the other.
        """
        complete = segment_state.state is not None and all(
            basis == SEGMENT_BASIS_MEASURED for basis in segment_state.segment_basis
        )
        if not complete:
            return DsrcDecision(
                outcome=segment_state.outcome,
                actions=None,
                q_values=None,
                segment_basis=segment_state.segment_basis,
                matched_links=segment_state.matched_links,
                latency_ms=None,
            )
        result = self.act(segment_state.state)
        return DsrcDecision(
            outcome=OUTCOME_OK,
            actions=result.actions,
            q_values=result.q_values,
            segment_basis=segment_state.segment_basis,
            matched_links=segment_state.matched_links,
            latency_ms=result.latency_ms,
        )
