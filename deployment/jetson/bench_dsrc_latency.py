#!/usr/bin/env python3
"""Latency of the two task-145 DSRC stages: segment_assemble and dsrc_infer.

plan_task145 step 12 asks this to be measured on the Orin, over a replay of
one recorded drive, against section 6's prior (TorchScript dispatch ~4 ms
with multi-ms jitter for a network 3.3x larger, against a 200 ms end-to-end
budget). **This script's own runs are smoke runs on this development
machine, not the Orin measurement itself** -- section 8 of the plan states
plainly that step 12 needs the device and nothing before it does, and
implementer-145 had no device access. What this proves is that the harness
runs and produces a number in the right units; the Orin's own number is a
separate, later step.

Without --here-log, builds one synthetic link near each Mainz segment's own
first polyline point (full coverage, so dsrc_infer actually runs the
network on every iteration) and repeats it --ticks times. With --here-log,
replays a `logio.here_logger.HereLogger`-shaped run directory instead --
the real recorded-drive replay step 12 asks for, whenever one is available.

  python3 bench_dsrc_latency.py --ticks 300
  python3 bench_dsrc_latency.py --here-log ~/dsrc_logs/run_20260910_120000 --ticks 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

import numpy as np  # noqa: E402

from perception.segment_state import DEFAULT_NETWORK_DEFINITION_PATH, SegmentStateBuilder  # noqa: E402
from policy import export_dsrc_policy  # noqa: E402
from policy.dsrc_runtime import DsrcRuntime  # noqa: E402
from sensors.here_feed import HereFeed  # noqa: E402

REPO_ROOT = JETSON_DIR.parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "mainz_here_best.pt"


def _synthetic_here_feed(definition: dict, t_mono: float) -> HereFeed:
    """One link per segment, anchored at that segment's own first polyline
    point -- full coverage against the synthetic network export produces
    (specs/dsrc_network_mainz.json's own placement, not a real drive)."""
    feed = HereFeed()
    bodies = []
    for segment in definition["segments"]:
        lat, lon = segment["edges"][0]["polyline"][0]
        bodies.append({
            "location": {
                "length": 200.0,
                "shape": {"links": [{"points": [
                    {"lat": lat, "lng": lon},
                    {"lat": lat, "lng": lon + 0.0005},
                ]}]},
            },
            "currentFlow": {
                "speed": 15.0, "freeFlow": 16.67, "jamFactor": 3.0, "confidence": 0.9,
            },
        })
    body = json.dumps({"results": bodies}).encode("utf-8")
    feed.offer(status=200, body=body, received_t_mono=t_mono)
    return feed


def _here_log_feed(log_dir: Path, t_mono: float) -> HereFeed:
    """Replay the last body of a HereLogger run directory into a fresh feed."""
    index_path = log_dir / "here_index.jsonl"
    lines = index_path.read_text().splitlines()
    if not lines:
        raise SystemExit(f"{index_path} is empty -- nothing to replay")
    last = json.loads(lines[-1])
    body_path = log_dir / last["body_path"]
    feed = HereFeed()
    feed.offer(status=200, body=body_path.read_bytes(), received_t_mono=t_mono)
    return feed


def pctl(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "n": len(arr), "mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)), "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--network-definition", type=Path, default=DEFAULT_NETWORK_DEFINITION_PATH)
    parser.add_argument("--here-log", type=Path, default=None,
                        help="a HereLogger run directory to replay; default is synthetic")
    parser.add_argument("--ticks", type=int, default=300)
    args = parser.parse_args()

    definition = json.loads(args.network_definition.read_text())
    model, info = export_dsrc_policy.build_from_checkpoint(str(args.checkpoint), definition)
    import tempfile

    out_prefix = str(Path(tempfile.mkdtemp()) / "dsrc_policy")
    export_dsrc_policy.export(model, info, definition, out_prefix, checkpoint_path=str(args.checkpoint))
    runtime = DsrcRuntime(out_prefix, network_definition_path=str(args.network_definition))
    builder = SegmentStateBuilder(args.network_definition)

    t_mono = time.monotonic()
    feed = (
        _here_log_feed(args.here_log, t_mono) if args.here_log
        else _synthetic_here_feed(definition, t_mono)
    )

    assemble_ms, infer_ms = [], []
    coverage_outcomes: dict[str, int] = {}
    for _ in range(args.ticks):
        now = time.monotonic()
        t0 = time.monotonic()
        links, reading = feed.snapshot_links(now)
        segment_state = builder.build(links, reading, now)
        t1 = time.monotonic()
        decision = runtime.decide(segment_state)
        assemble_ms.append((t1 - t0) * 1000.0)
        # decide() short-circuits without running the network whenever the
        # coverage gate refuses (decision.latency_ms is None then); only a
        # decision that actually ran the network measures an inference.
        if decision.latency_ms is not None:
            infer_ms.append(decision.latency_ms)
        coverage_outcomes[decision.outcome] = coverage_outcomes.get(decision.outcome, 0) + 1

    source = str(args.here_log) if args.here_log else "synthetic (no --here-log given)"
    print(f"source: {source}")
    print(f"coverage outcomes over {args.ticks} ticks: {coverage_outcomes}")
    print("segment_assemble_ms (THIS MACHINE, not the Orin):", pctl(assemble_ms))
    if infer_ms:
        print("dsrc_infer_ms       (THIS MACHINE, not the Orin):", pctl(infer_ms))
    else:
        print(
            "dsrc_infer_ms       (THIS MACHINE, not the Orin): no tick ran the "
            f"network -- coverage outcomes were {coverage_outcomes}"
        )


if __name__ == "__main__":
    main()
