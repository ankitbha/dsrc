#!/usr/bin/env python3
"""Offline replay of a recorded run (plan_deployment.md Task P7).

Re-runs the full perception/observation/policy pipeline over the raw
video and logged GPS of a previous run, then compares the fresh outputs
against what was produced live. Determinism caveat: tracking and
relative-speed smoothing depend on inter-frame timing only through the
recorded frame order, so replay is reproducible run-to-run; divergence
from the live log indicates nondeterminism or a code change since the
recording - both worth knowing.

  python3 replay_demo.py --log ~/dsrc_logs/run_20260611_211500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from run_demo import build_components, load_config  # noqa: E402
from sensors.camera_stream import Frame  # noqa: E402
from sensors.gps_reader import GpsFix  # noqa: E402
from sensors.time_sync import now_mono, now_wall  # noqa: E402


def load_tick_records(metadata_path: Path) -> list[dict]:
    records = []
    with open(metadata_path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("type") == "tick":
                records.append(record)
    return records


def fix_from_record(record: dict) -> GpsFix:
    gps = record.get("gps", {})
    valid = bool(gps.get("valid"))
    return GpsFix(
        valid=valid,
        lat=gps.get("lat") if gps.get("lat") is not None else float("nan"),
        lon=gps.get("lon") if gps.get("lon") is not None else float("nan"),
        speed_mps=gps.get("speed_mps") if gps.get("speed_mps") is not None else float("nan"),
        heading_deg=gps.get("heading_deg") if gps.get("heading_deg") is not None else float("nan"),
        fix_quality=1 if valid else 0,
        num_sats=int(gps.get("num_sats") or 0),
        hdop=gps.get("hdop") if gps.get("hdop") is not None else float("nan"),
        # freshness is relative to the replay clock
        t_mono=now_mono(),
        t_wall=now_wall(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="run directory (with video.avi + metadata.jsonl)")
    parser.add_argument("--config", help="defaults to the config recorded with the run")
    parser.add_argument("--max-ticks", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.log).expanduser()
    video_path = run_dir / "video.avi"
    metadata_path = run_dir / "metadata.jsonl"
    if not video_path.exists():
        print(f"[replay] {video_path} missing - record with logio.video: true")
        return 1
    if not metadata_path.exists():
        print(f"[replay] {metadata_path} missing")
        return 1

    config_path = args.config or str(run_dir / "run_config.yaml")
    config = load_config(config_path)
    records = load_tick_records(metadata_path)
    print(f"[replay] {len(records)} live ticks loaded from {metadata_path}")

    _, _, pipeline, actor = build_components(config, "file:/dev/null", use_gps=False)
    pipeline.detector.warmup()

    video = cv2.VideoCapture(str(video_path))
    speed_deltas: list[float] = []
    action_matches = 0
    compared = 0
    peer_informed_ticks = 0
    i = 0
    while True:
        ok, image = video.read()
        if not ok or i >= len(records):
            break
        live = records[i]
        frame = Frame(image=image, frame_id=i, t_mono=now_mono(), t_wall=now_wall())
        tick = pipeline.step(frame, fix_from_record(live), detections_override=None)
        live_adv = live.get("advisory", {})
        if "recommended_speed_mps" in live_adv:
            speed_deltas.append(
                abs(tick.advisory.recommended_speed_mps - live_adv["recommended_speed_mps"])
            )
        if live.get("action") == tick.policy.action:
            action_matches += 1
        if live.get("n_peers"):
            peer_informed_ticks += 1
        compared += 1
        i += 1
        if args.max_ticks and i >= args.max_ticks:
            break
    video.release()

    if compared == 0:
        print("[replay] nothing compared - empty video or metadata mismatch")
        return 1
    stats = pipeline.stats.snapshot()
    # V2V peers are NOT replayable: `Tick.to_record` logs `n_peers`, a count, and not
    # the peer states themselves, so `pipeline.step` is called here without them. On a
    # run recorded with peers the divergence that follows is this tool's own missing
    # argument, and reporting it as "agreement" says the opposite of what it means --
    # the label promises that divergence indicates nondeterminism or a code change.
    # Measured on an otherwise identical pipeline: 3.3% agreement, max encoded delta
    # 0.59, and `downstream_congestion_estimate` falling from `measured` to
    # `fallback_neutral`. So the number is withheld rather than qualified.
    # Guarded the way the record below guards the same list. An empty one is a run
    # whose ticks carried no advisory, not a run with a delta of nan.
    speed_delta_line = (
        f"mean {np.mean(speed_deltas):.3f} m/s, max {np.max(speed_deltas):.3f} m/s"
        if speed_deltas else "no advisory in any replayed tick"
    )
    comparable = peer_informed_ticks == 0
    agreement = f"{action_matches / compared * 100:.1f}%" if comparable else (
        f"not comparable -- {peer_informed_ticks} of {compared} ticks were recorded "
        f"with V2V peers, which the record does not carry, so replay runs without them"
    )
    print(
        f"[replay] {compared} ticks replayed\n"
        f"  action agreement (all 4 heads): {agreement}\n"
        f"  |recommended speed delta|: {speed_delta_line}\n"
        f"  replay detect p50 {stats['detect_ms']['p50']:.1f} ms, "
        f"e2e p50 {stats['e2e_ms']['p50']:.1f} ms (replay clock)"
    )
    out = {
        "replayed_ticks": compared,
        # None, not a number, when the comparison cannot be made. A fraction here
        # would be read as a measurement of determinism and it would be a measurement
        # of what replay does not reconstruct.
        "action_agreement": (action_matches / compared) if comparable else None,
        "action_matches": action_matches,
        "peer_informed_ticks": peer_informed_ticks,
        "speed_delta_mean_mps": float(np.mean(speed_deltas)) if speed_deltas else None,
        "speed_delta_max_mps": float(np.max(speed_deltas)) if speed_deltas else None,
        "stats": stats,
    }
    with open(run_dir / "replay_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[replay] wrote {run_dir / 'replay_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
