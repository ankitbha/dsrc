"""Vendored DSRC network: the whole-network controller's shape and identity.

`SrcQNetwork` mirrors `src.rl.src_q.SrcQNetwork`. `src.rl.src_q` imports only
torch (verified: no `sumolib`, no simulation environment stack), so nothing
here forces a device import to avoid a dependency the way `sim_contract.py`
does for the 39-field actor. It is vendored anyway because the deploy
integrity hash (`run_demo.py`'s `_build_provenance`, via
`scripts/record_deployed_commit.py`) covers only `deployment/jetson/`. A
`DsrcRuntime` that imported `src.rl.src_q` would execute code the hash says
nothing about, so every file the device runs for this controller stays
inside the tree the hash covers.

Only `forward` is vendored, not `greedy_actions`: the demonstration
(`tests/test_dsrc_runtime.py`) compares this runtime's action against
`src.rl.src_q.greedy_actions`, and that function must not be the thing
producing both sides of the comparison.

`tests/test_dsrc_contract.py` asserts this class matches
`src.rl.src_q.SrcQNetwork` in state-dict keys and layer shapes when the sim
package is importable, and separately asserts the frozen golden actions
(`specs/dsrc_golden_actions.json`), which need no such import at all.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import torch
import torch.nn as nn

#: SRC's action set as fractions of a segment's own speed limit, so it carries
#: to a network with a different limit rather than being a literal 30/45/60.
#: `src.sumo.mainz.SPEED_ACTION_FRACTIONS`.
SPEED_ACTION_FRACTIONS: tuple[float, ...] = (0.5, 0.75, 1.0)

#: SRC's `FEEDBACK_STEP`. `src.sumo.mainz.DECISION_INTERVAL_S`.
DECISION_INTERVAL_S: float = 60.0

#: What the deployed observation carries, in encoder order.
#: `src.sumo.mainz.HERE_FEATURES`.
HERE_FEATURES: tuple[str, ...] = ("speed", "free_flow", "jam_factor", "lanes", "length_km")


def jam_factor(speed_kmh: float, free_flow_kmh: float) -> float:
    """HERE's 0-10 congestion index, as the simulator's own stated model.

    Vendored from `src.sumo.mainz.jam_factor` for the same reason the network
    is vendored above: so `perception.segment_state` needs no `src.sumo`
    import. HERE does not publish its own formula and folds in incident data
    this project does not have, so the policy was trained against this
    stated ratio, not against HERE's own `jamFactor` field -- using the
    field instead would feed the network a different quantity than the one
    it was trained on.
    """
    if free_flow_kmh <= 0.0:
        return 0.0
    return float(min(10.0, max(0.0, 10.0 * (1.0 - speed_kmh / free_flow_kmh))))


class SrcQNetwork(nn.Module):
    """State-dict-compatible twin of `src.rl.src_q.SrcQNetwork`.

    Flatten every segment's features, emit every segment's Q-values. The
    hidden width is twice the input, matching the trained checkpoints
    (`stack.0.weight` is `[2*S*F, S*F]`).
    """

    def __init__(self, num_segments: int, num_features: int, num_actions: int) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.num_features = num_features
        self.num_actions = num_actions
        self.input_length = num_segments * num_features
        self.output_length = num_segments * num_actions
        self.stack = nn.Sequential(
            nn.Linear(self.input_length, 2 * self.input_length),
            nn.ReLU(),
            nn.Linear(2 * self.input_length, self.output_length),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """State `(segments, features)` to Q-values `(segments, actions)`."""
        logits = self.stack(torch.flatten(state))
        return logits.view((self.num_segments, self.num_actions))


def _polyline_hash(segment_polylines: Sequence[Sequence[Sequence[float]]]) -> str:
    """One hash over every segment's ordered (lat, lon) polyline points.

    A separate sub-hash rather than folding the raw points into
    `network_fingerprint`'s own payload: segment 5 alone carries 490
    vertices, and re-serialising every point of every segment on every
    fingerprint computation (device boot included) is wasted work when a
    single upfront hash says exactly as much about whether two definitions
    agree.
    """
    payload = json.dumps(
        [
            [[round(float(lat), 6), round(float(lon), 6)] for lat, lon in segment]
            for segment in segment_polylines
        ],
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def network_fingerprint(
    *,
    network_id: str,
    feature_names: Sequence[str],
    segment_ids: Sequence[Sequence[str]],
    segment_speed_limits_kmh: Sequence[float],
    segment_lanes: Sequence[float],
    segment_length_km: Sequence[float],
    segment_polylines: Sequence[Sequence[Sequence[float]]],
) -> str:
    """A short hash over everything a DSRC bundle's network identity depends on.

    Mirrors `sim_contract.contract_fingerprint`'s form: the shapes torch
    checks are not enough to catch a reordered or edited network
    definition -- two 12-segment, 5-feature networks are shape-identical --
    so the fingerprint covers the field NAMES, the ordered segment/edge ids,
    the per-segment speed limits, lane counts, lengths and polylines,
    everything the manifest table in plan section 3 names (amended: the
    table originally named only speed limits alongside feature names and
    segment ids, and left lanes and length_km -- two of the five features
    `HERE_FEATURES` reads -- and the polylines out; see the dated note in
    plans/plan_task145_dsrc_policy_runtime.md section 3). `segment_ids`
    here is the ordered list of ordered edge-id lists (what
    `data/mainz/mainz_segments.json` holds), not a separate id per segment.

    `json.dumps(..., sort_keys=True)` only reorders dict keys, never list
    contents, so swapping two entries of an ordered list changes the
    payload -- which is the point: reordering two segments or two edges
    within one changes nothing about the tensor shapes and must still
    change this hash.
    """
    payload = json.dumps(
        {
            "network_id": network_id,
            "feature_names": list(feature_names),
            "segment_ids": [list(ids) for ids in segment_ids],
            "segment_speed_limits_kmh": [round(float(v), 6) for v in segment_speed_limits_kmh],
            "segment_lanes": [round(float(v), 6) for v in segment_lanes],
            "segment_length_km": [round(float(v), 6) for v in segment_length_km],
            "segment_polyline_hash": _polyline_hash(segment_polylines),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
