#!/usr/bin/env python3
"""Generate specs/dsrc_golden_actions.json: the DSRC demonstration's reference file.

Runs `MainzEnv` under a trained checkpoint over the held-out test seeds
(`scripts/train_mainz_src.py`'s own `TEST_SEEDS`, `range(16, 21)`) and
records every decision's state, `src.rl.src_q.greedy_actions` output and
raw Q-values -- the SIMULATOR's own reference function, deliberately not
vendored, so `deployment/jetson/policy/dsrc_runtime.DsrcRuntime` (which
uses the vendored `SrcQNetwork`) is compared against an independent
implementation and not against itself (risk 3 in plan_task145). Also draws
20,000 states from a fixed seed over the plausible feature box -- the same
population plan section 1.7 measured 0 action mismatches on -- since 180
recorded decisions from 5 episodes cover far less of the input space than
training or a real drive would.

`tests/test_dsrc_runtime.py` reads this file and never regenerates it, the
same discipline `specs/transport_golden_frames.json` and
`test_transport_golden.py` already establish for this repo. Changing a
recorded value here is changing what the demonstration claims and needs a
deliberate re-run of this script, not a quiet edit.

**Storage.** The ~180 recorded states are small and stored as one hex
string per float32 (`state_hex`, `q_values_hex`) -- human-legible, and
exactly what plan section 5.1 asks for. The 20,000 random states are NOT
stored that way: at roughly 30 hex characters per state times 20,000, the
file would run to tens of megabytes for a "small frozen generated file"
(the precedent this follows, `specs/transport_golden_frames.json`, is
32 KB). Storing them instead as base64 of the raw float32 (states,
Q-values) and int8 (actions) buffers is equally bit-exact -- no decimal
rounding either way -- and keeps the file an order of magnitude smaller.

Usage:
  python3 scripts/export_dsrc_golden.py
  python3 scripts/export_dsrc_golden.py --checkpoint <path> --out <path>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "deployment" / "jetson")):
    if path not in sys.path:
        sys.path.insert(0, path)

from policy.dsrc_contract import network_fingerprint  # noqa: E402
from src.rl.src_q import SrcQNetwork, greedy_actions  # noqa: E402
from src.sumo.mainz import HERE_FEATURES, MainzEnv  # noqa: E402

#: scripts/train_mainz_src.py's own TEST_SEEDS -- read once, at the end,
#: and nowhere else during training. The golden file replays exactly that
#: held-out set.
TEST_SEEDS = tuple(range(16, 21))
DURATION_S = 2500.0
WARMUP_S = 300.0

DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "mainz_here_best.pt"
DEFAULT_NETWORK_DEFINITION = REPO_ROOT / "specs" / "dsrc_network_mainz.json"
DEFAULT_OUT = REPO_ROOT / "specs" / "dsrc_golden_actions.json"

RANDOM_STATE_COUNT = 20_000
RANDOM_SEED = 12345
#: The plausible feature box for a synthetic (segments, features) row:
#: speed and free_flow in km/h (free_flow's floor is above zero -- a zero
#: speed limit is not in this network), jam_factor in HERE's own 0-10
#: range, lanes and length_km spanning what data/mainz/mainz_segments.json
#: actually contains (plan section 1.3: 1.64-5.00 lanes, 0.79-9.12 km).
FEATURE_LOW = np.array([0.0, 30.0, 0.0, 1.0, 0.1], dtype=np.float32)
FEATURE_HIGH = np.array([120.0, 120.0, 10.0, 6.0, 10.0], dtype=np.float32)


def _hex_f32(x: np.ndarray) -> list:
    """Every element as an 8-hex-character string of its raw float32 bits."""
    flat = [v.tobytes().hex() for v in x.astype(np.float32).ravel()]
    return np.array(flat, dtype=object).reshape(x.shape).tolist()


def _b64(x: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(x).tobytes()).decode("ascii")


def _load_model(checkpoint: Path, num_segments: int, num_features: int, num_actions: int):
    model = SrcQNetwork(num_segments, num_features, num_actions)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_recorded(model, seeds) -> list[dict]:
    cases = []
    for seed in seeds:
        env = MainzEnv(seed=seed, duration_s=DURATION_S, warmup_s=WARMUP_S, features=HERE_FEATURES)
        try:
            state = env.reset()
            done = False
            while not done:
                tensor = torch.tensor(state, dtype=torch.float)
                with torch.no_grad():
                    q_values = model(tensor)
                action = greedy_actions(model, tensor)
                cases.append({
                    "seed": seed,
                    "state_hex": _hex_f32(np.asarray(state, dtype=np.float32)),
                    "action": action.tolist(),
                    "q_values_hex": _hex_f32(q_values.numpy().astype(np.float32)),
                })
                state, _reward, done = env.step(action.tolist())
        finally:
            env.close()
    return cases


def collect_random(model, count: int, num_segments: int, num_features: int, seed: int):
    rng = np.random.default_rng(seed)
    states = rng.uniform(
        FEATURE_LOW, FEATURE_HIGH, size=(count, num_segments, num_features)
    ).astype(np.float32)
    actions = np.zeros((count, num_segments), dtype=np.int8)
    q_values = np.zeros((count, num_segments, 3), dtype=np.float32)
    for i in range(count):
        tensor = torch.tensor(states[i], dtype=torch.float)
        with torch.no_grad():
            q = model(tensor)
        action = greedy_actions(model, tensor)
        actions[i] = action.numpy().astype(np.int8)
        q_values[i] = q.numpy().astype(np.float32)
    return states, actions, q_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--network-definition", type=Path, default=DEFAULT_NETWORK_DEFINITION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-states", type=int, default=RANDOM_STATE_COUNT)
    args = parser.parse_args()

    definition = json.loads(args.network_definition.read_text())
    num_segments = len(definition["segments"])
    num_features = len(definition["feature_names"])
    num_actions = 3

    model = _load_model(args.checkpoint, num_segments, num_features, num_actions)

    segment_ids = [list(s["edge_ids"]) for s in definition["segments"]]
    speed_limits = [float(s["speed_limit_kmh"]) for s in definition["segments"]]
    fingerprint = network_fingerprint(
        network_id=definition["network_id"], feature_names=definition["feature_names"],
        segment_ids=segment_ids, segment_speed_limits_kmh=speed_limits,
    )
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    recorded = collect_recorded(model, TEST_SEEDS)
    random_states, random_actions, random_q = collect_random(
        model, args.random_states, num_segments, num_features, RANDOM_SEED
    )

    document = {
        "network_id": definition["network_id"],
        "network_fingerprint": fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": str(args.checkpoint.resolve().relative_to(REPO_ROOT)),
        "num_segments": num_segments,
        "num_features": num_features,
        "num_actions": num_actions,
        "test_seeds": list(TEST_SEEDS),
        "duration_s": DURATION_S,
        "warmup_s": WARMUP_S,
        "recorded": recorded,
        "random": {
            "seed": RANDOM_SEED,
            "count": int(args.random_states),
            "feature_low": FEATURE_LOW.tolist(),
            "feature_high": FEATURE_HIGH.tolist(),
            "dtype_states": "float32",
            "dtype_actions": "int8",
            "dtype_q_values": "float32",
            "states_b64": _b64(random_states),
            "actions_b64": _b64(random_actions),
            "q_values_b64": _b64(random_q),
        },
    }
    args.out.write_text(json.dumps(document))
    print(
        f"wrote {args.out}: {len(recorded)} recorded decisions across "
        f"{len(TEST_SEEDS)} seeds, {args.random_states} random states "
        f"({args.out.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
