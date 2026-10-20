"""Re-scoring committed approach records on the network (2026-10-20). No SUMO."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rescore_network as rn  # noqa: E402
import run_jam_experiments as rje  # noqa: E402
from src.sumo.jam_episode import SCENARIOS  # noqa: E402


def _trace(cap=7440.0):
    out, t = [], 0.0
    while t <= cap + 1e-9:
        speeds = [50.0] * 36
        if 1500.0 <= t < 2500.0:
            speeds[4] = 10.0      # a network jam
        if 1300.0 <= t < 3000.0:
            speeds[30] = 12.0     # a longer one on an approach road
        out.append({"t": t, "speed_kmh": speeds, "density": [0.05] * 36, "count": [1] * 36,
                    "held": 0, "pending": 0, "running": 1100, "on_approach": 300,
                    "queue_m": [0.0] * 8, "slow_near_insertion": [0] * 8,
                    "k_hat_mean": [None] * 36, "k_true_mean": [None] * 36})
        t += 10.0
    return out


def _old_record(tmp_path, trace):
    """A record as the runner wrote it before 2026-10-20: `jam` over all 36 and
    `jam_network` beside it, with no `network_min`."""
    boundary = rje.Context(boundary="approach").boundary_object()
    spec = rje.Spec("M", "plan", "-", "B9", 16, 7200.0)
    from src.analysis.jam_metrics import load_v_jam

    new = rje.approach_metrics(trace, SCENARIOS["B9"], spec, load_v_jam()[0], boundary)
    old = {old_name: new[name] for old_name, name in rn.RENAMED.items()}
    old["storage"] = {k: v for k, v in new["storage"].items() if k != "network_min"}
    path = tmp_path / "M" / spec.trace_name()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"key": spec.key, "trace": trace}))
    record = {"type": "episode", "key": spec.key, "block": "M", "arm": "plan", "variant": "-",
              "scenario": "B9", "seed": 16, "duration_s": 7200.0, "m": 1.0,
              "penetration": 1.0, "bias": 1.0, "pair_sigma_scale": 1.0, "params": {},
              "flow": 3500.0, "trace_file": "/gone/worktree/trace.json", **old}
    return record, path, boundary, load_v_jam()[0]


def test_the_old_numbers_come_back_under_their_new_names(tmp_path):
    record, path, boundary, v_jam = _old_record(tmp_path, _trace())
    out = rn.rescore(record, path, v_jam, boundary)
    assert out["jam"] == record["jam_network"] and out["jam_all"] == record["jam"]
    assert "jam_network" not in out and out["jam"]["columns"] == list(range(12))
    # The network jam (segment 4, 1,000 s) is what `jam` measures; the approach one is not.
    assert out["jam"]["jam_time_min"] == pytest.approx(1000.0 / 60.0)
    assert out["jam_all"]["jam_time_min"] > out["jam"]["jam_time_min"]
    assert out["jam"]["onset_segment"] == 4 and out["jam_all"]["onset_segment"] == 30
    # 800 vehicles on the network at every 10 s sample of [1,140, 7,440] s.
    assert out["storage"]["network_min"] == pytest.approx(631 * 10.0 * 800.0 / 60.0)
    assert out["trace_file"] == str(path) and out["flow"] == 3500.0
    assert out["rescored"]["from_trace_file"] == "/gone/worktree/trace.json"


def test_a_number_that_does_not_come_back_is_refused(tmp_path):
    record, path, boundary, v_jam = _old_record(tmp_path, _trace())
    record["jam_network"] = {**record["jam_network"], "jam_time_min": 1.0}
    with pytest.raises(SystemExit):
        rn.rescore(record, path, v_jam, boundary)


def test_a_trace_of_another_episode_is_refused(tmp_path):
    record, path, boundary, v_jam = _old_record(tmp_path, _trace())
    path.write_text(json.dumps({"key": "M/plan/-/B9/17", "trace": _trace()}))
    with pytest.raises(SystemExit):
        rn.rescore(record, path, v_jam, boundary)


def test_a_committed_record_re_scores_exactly():
    # One committed approach record and its trace, when the trace is on this machine.
    traces = REPO_ROOT / "outputs" / "jam_approach" / "run_5cea9d3" / "traces"
    line = next(line for line in (REPO_ROOT / "results/jam_approach/P.jsonl").read_text()
                .splitlines() if '"P/rule/m0_p0.5/B9/16"' in line)
    record = json.loads(line)
    path = traces / "P" / "rule_m0_p0.5_B9_16.json"
    if not path.exists():
        pytest.skip("the committed traces are not on this machine")
    from src.analysis.jam_metrics import load_v_jam

    out = rn.rescore(record, path, load_v_jam()[0],
                     rje.Context(boundary="approach").boundary_object())
    assert out["jam"]["jam_time_min"] == record["jam_network"]["jam_time_min"]
    assert out["storage"]["network_min"] == pytest.approx(
        record["storage"]["system_excess_min"] - record["storage"]["approach_excess_min"],
        rel=1e-9)
