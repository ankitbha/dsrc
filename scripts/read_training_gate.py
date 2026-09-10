#!/usr/bin/env python3
"""Read task 103's pre-registered gate off a training run. Three criteria, no fourth.

The gate was fixed in `plans/task_list.md` task 103 BEFORE the run, and this reads it
without renegotiating it:

  1. summed entropy falls below 90% of its maximum;
  2. the score trends up by more than the variation between consecutive updates;
  3. the action distribution leaves uniform, which needs the actor and is measured by
     `scripts/measure_action_distribution.py`.

    .venv/bin/python scripts/read_training_gate.py --run-dir <checkpoint dir>

The threshold for criterion 1 comes from the action profile rather than being typed
in: `speed_headway` has two three-valued heads, so the summed entropy maximum is
2 ln 3 = 2.1972 and 90% of it is 1.9775. The 4.394 of tasks 96 and 101 was ln 81 and
is not comparable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loaders import load_named_config  # noqa: E402
from src.rl.actions import ActionSpec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--training", default="mappo_sumo")
    parser.add_argument("--entropy-fraction", type=float, default=0.90)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    training = load_named_config("training", args.training)
    spec = ActionSpec(str(training["action_profile"]))  # type: ignore[arg-type]
    maximum = sum(math.log(spec.head_size(head)) for head in spec.active_heads)
    threshold = args.entropy_fraction * maximum

    rows = list(csv.DictReader((Path(args.run_dir) / "training_metrics.csv").open()))
    if len(rows) < 3:
        print(f"only {len(rows)} updates; the gate needs a curve")
        return 1
    entropy = [float(r["entropy"]) for r in rows]
    score = [float(r["score"]) for r in rows]

    # Criterion 2 compares the trend against the step-to-step variation, so a score
    # that wanders as much between neighbouring updates as it moves end to end is
    # not a trend. The slope is the least-squares one over the update index.
    n = len(score)
    mean_x = (n - 1) / 2.0
    mean_y = statistics.fmean(score)
    slope = (sum((i - mean_x) * (y - mean_y) for i, y in enumerate(score))
             / sum((i - mean_x) ** 2 for i in range(n)))
    step_sd = statistics.pstdev([score[i + 1] - score[i] for i in range(n - 1)])
    total_move = slope * (n - 1)

    result = {
        "updates": n,
        "entropy_max": round(maximum, 4),
        "entropy_threshold": round(threshold, 4),
        "entropy_first": round(entropy[0], 4),
        "entropy_last": round(entropy[-1], 4),
        "entropy_min": round(min(entropy), 4),
        "entropy_range": round(max(entropy) - min(entropy), 4),
        "criterion_1_entropy_falls": bool(min(entropy) < threshold),
        "score_first": round(score[0], 4),
        "score_last": round(score[-1], 4),
        "score_slope_per_update": round(slope, 5),
        "score_move_over_run": round(total_move, 4),
        "score_step_sd": round(step_sd, 4),
        "criterion_2_score_trends_up": bool(total_move > step_sd),
        "criterion_3": "run scripts/measure_action_distribution.py on latest_actor.pt",
    }
    print(json.dumps(result, indent=1))
    verdict = "PASS" if result["criterion_1_entropy_falls"] and result["criterion_2_score_trends_up"] else "FAIL"
    print(f"\ncriterion 1 (entropy below {threshold:.4f}): "
          f"{'pass' if result['criterion_1_entropy_falls'] else 'fail'} "
          f"-- lowest was {min(entropy):.4f}")
    print(f"criterion 2 (score move {total_move:+.4f} over step sd {step_sd:.4f}): "
          f"{'pass' if result['criterion_2_score_trends_up'] else 'fail'}")
    print(f"\ncriteria 1 and 2: {verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
