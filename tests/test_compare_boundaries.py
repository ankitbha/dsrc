"""The boundary comparison (approach plan section 7.1 step 4, step S8). No SUMO."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_boundaries as cb  # noqa: E402


class TestPairRows:
    def test_rows_pair_by_seed(self):
        out = cb.pair_rows({16: 10.0, 17: 12.0}, {17: 13.0, 16: 7.0})
        assert out["seeds"] == [16, 17] and out["differences"] == [-3.0, 1.0]
        assert out["mean"] == -1.0
        assert out["seeds_only_gated"] == [] and out["seeds_only_approach"] == []

    def test_a_missing_seed_is_listed_not_dropped_silently(self):
        out = cb.pair_rows({16: 10.0, 17: 12.0, 18: 5.0}, {16: 7.0, 17: 13.0, 19: 1.0})
        assert out["seeds"] == [16, 17]
        assert out["seeds_only_gated"] == [18] and out["seeds_only_approach"] == [19]


def _trace(path, running, held, pending, cap):
    trace = [{"t": float(t), "running": running, "held": held, "pending": pending}
             for t in range(0, int(cap) + 10, 10)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": "x", "trace": trace}))


def _jam(minutes, prevented=False, held=0.0, pending=0.0):
    return {"jam_time_min": minutes, "prevented": prevented, "held_min": held,
            "pending_min": pending}


def _write(directory, block, header, episodes):
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "header", **header})]
    lines += [json.dumps({"type": "episode", "key": f"{block}/{e['arm']}/{e['variant']}/"
                          f"{e['scenario']}/{e['seed']}", "block": block, **e})
              for e in episodes]
    (directory / f"{block}.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture()
def dirs(tmp_path):
    gated, approach, traces = tmp_path / "gated", tmp_path / "approach", tmp_path / "traces"
    g_eps, a_eps = [], []
    for arm, variant in (("no_control", "-"), ("rule", "m0"), ("lookahead", "J2")):
        for seed in (16, 17, 18):
            jam = {"no_control": 100.0, "rule": 5.0, "lookahead": 20.0}[arm]
            g_eps.append({"arm": arm, "variant": variant, "scenario": "B9", "seed": seed,
                          "duration_s": 7200.0, "flow": 3500.0 + seed,
                          "jam": _jam(jam, held=100.0, pending=10.0)})
            _trace(traces / "M" / f"{arm}_{variant}_B9_{seed}.json", 1000, 3, 2, 7200.0)
            if seed == 18 and arm == "rule":
                continue  # the approach boundary lacks this seed
            a_eps.append({"arm": arm, "variant": variant, "scenario": "B9", "seed": seed,
                          "duration_s": 7200.0, "flow": 3400.0 + seed,
                          "jam": _jam(jam + 2.0, pending=1.0),
                          "jam_network": _jam(jam), "jam_approach": _jam(2.0),
                          "storage": {"approach_excess_min": 40.0,
                                      "system_excess_min": 2000.0 + seed},
                          "approach_short": False})
    _write(gated, "M", {}, g_eps)
    _write(approach, "M", {"boundary": "approach", "W": 240.0,
                           "approach": {"T_A": {"140": 190.0}},
                           "config": {"selection": {"best_arm": "rule",
                                                    "best_params": {"m_on": 0.0},
                                                    "J_th": 2.0}}}, a_eps)
    return gated, approach, traces


def test_the_comparison_pairs_every_quantity_and_lists_the_missing_seed(dirs):
    gated, approach, traces = dirs
    assert cb.main(["--gated", str(gated), "--approach", str(approach),
                    "--gated-traces", str(traces)]) == 0
    out = json.loads((approach / "boundary_comparison.json").read_text())
    row = out["blocks"]["M"]["rows"]["rule/m0/B9"]
    assert row["gated"]["jam_time_min_mean"] == 5.0
    assert row["approach"]["jam_network_time_min_mean"] == 5.0
    assert row["approach"]["jam_all_time_min_mean"] == 7.0
    assert row["approach"]["jam_approach_time_min_mean"] == 2.0
    # Gated storage is held + pending; approach storage is approach excess + pending.
    assert row["gated"]["held_plus_pending_min_mean"] == 110.0
    assert row["approach"]["approach_excess_plus_pending_min_mean"] == 41.0
    # Gated system excess from the trace: 1,005 vehicles on [900, 7,200] every 10 s.
    assert row["gated"]["system_excess_min_mean"] == pytest.approx(10.0 * 1005 * 631 / 60.0)
    diff = row["approach_minus_gated"]
    assert diff["network_jam_time_min"]["seeds"] == [16, 17]
    assert diff["network_jam_time_min"]["seeds_only_gated"] == [18]
    assert diff["flow"]["mean"] == -100.0
    assert diff["storage_min"]["mean"] == pytest.approx(41.0 - 110.0)
    assert "not a controlled comparison" in out["note"]


def test_the_three_arm_contrasts_on_the_approach_boundary(dirs):
    gated, approach, traces = dirs
    result = cb.compare(gated, approach, traces)
    contrast = result["contrasts"]
    assert contrast["two_layer_controller"] == "rule/m0"
    assert contrast["lookahead"] == "lookahead/J2"
    b9 = contrast["scenarios"]["B9"]
    vs_nc = b9["rule/m0 - no_control/-"]
    assert vs_nc["jam_network_time_min"]["mean"] == -95.0
    assert vs_nc["jam_all_time_min"]["seeds"] == [16, 17]
    assert vs_nc["jam_all_time_min"]["seeds_only_other"] == [18]
    assert b9["rule/m0 - lookahead/J2"]["system_excess_min"]["mean"] == 0.0


def test_a_gated_jsonl_on_the_approach_side_is_refused(dirs, tmp_path):
    gated, _, traces = dirs
    with pytest.raises(SystemExit):
        cb.compare(gated, gated, traces)
