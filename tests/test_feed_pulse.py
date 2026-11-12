"""`MainzEnv.inject_pulse`: the suffix route, the vehicle count and their control.

The first test reads the committed files only; the second runs two 600 s SUMO episodes
on seed 16. Run this file in its own process.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.sumo import mainz
from src.sumo.mainz import MainzEnv, controlled_index

DATA = Path(__file__).resolve().parents[1] / "data" / "mainz"


def test_the_suffix_route_for_site_7_runs_from_its_first_edge_on_r277_to_the_exit():
    env = MainzEnv(seed=16, step_length_s=0.5)
    route, edges = env.pulse_route(7)
    segments = json.loads((DATA / "mainz_segments.json").read_text())
    assert route == "r277"
    assert edges[0] == "505" and edges[0] in segments[7]
    assert edges[-1] == "exit_tail"


def _run(pulse: bool, penetration: float = 0.5) -> dict:
    env = MainzEnv(seed=16, step_length_s=0.5, duration_s=600.0, penetration=penetration)
    record = None
    first_edges = {}
    try:
        env.reset()
        if pulse:
            record = env.inject_pulse(site=7, n=20, start_s=350.0, duration_s=60.0, seed=16)
        for _ in range(5):
            env._command([1] * 12)

            def look(t):
                if record is None:
                    return
                for vehicle in mainz._sumo.vehicle.getIDList():
                    if vehicle.startswith("pulse_") and vehicle not in first_edges:
                        first_edges[vehicle] = mainz._sumo.vehicle.getRoadID(vehicle)

            env.advance_ticks(60.0, on_tick=look)
        metrics = env.metrics()
        controlled = {v: env.controlled(v) for v in (record["ids"] if record else [])}
        positions = {v: env.position(v) for v in (record["ids"] if record else [])}
        size = env.schedule_size
    finally:
        env.close()
    return {"metrics": metrics, "record": record, "controlled": controlled,
            "positions": positions, "first_edges": first_edges, "size": size}


def test_a_pulse_of_20_adds_exactly_20_vehicles_and_follows_the_floor_rule():
    base = _run(pulse=False)
    pulsed = _run(pulse=True)

    def accounted(run):
        m = run["metrics"]
        return m["arrived"] + m["running"] + m["waiting_to_enter"] + m["held_at_entry"]

    assert accounted(pulsed) - accounted(base) == 20
    record = pulsed["record"]
    assert len(record["ids"]) == 20 and record["route_edges"][0] == "505"
    assert record["departs"][0] == pytest.approx(351.5)
    assert record["departs"][-1] == pytest.approx(408.5)
    n_schedule = base["size"]
    for j, vehicle in enumerate(record["ids"]):
        assert vehicle == f"pulse_7_{j}"
        assert pulsed["positions"][vehicle] == n_schedule + j
        assert pulsed["controlled"][vehicle] == controlled_index(n_schedule + j, 0.5)
    assert sum(pulsed["controlled"].values()) == math.floor((n_schedule + 20) * 0.5) - \
        math.floor(n_schedule * 0.5)
    assert record["controlled"] == sum(pulsed["controlled"].values())
    # Every pulse vehicle seen entered on the site's first edge.
    assert pulsed["first_edges"] and set(pulsed["first_edges"].values()) == {"505"}


def test_pulsed_and_unpulsed_runs_are_identical_until_the_pulse_departs():
    """Amendment 1, A8 (R1-8): the pulse must not change the run before it leaves."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_mainz_checkpoints as emc

    from src.sumo.feed_episode import Mode, run_feed_episode

    model = emc.load_arm_model("here", Path(__file__).resolve().parents[1] / "results"
                               / "checkpoints", 12).model
    base = run_feed_episode(model, 16, "mainz", Mode.published(), duration_s=1500.0)
    pulsed = run_feed_episode(model, 16, "mainz", Mode.published(), duration_s=1500.0,
                              pulse={"site": 7, "n": 20, "start_s": 1200.0,
                                     "duration_s": 60.0})
    fields = ("speed_kmh", "density", "count", "held", "pending", "running", "arrived")
    before = [(a, b) for a, b in zip(base["trace"], pulsed["trace"]) if a["t"] <= 1200.0]
    assert len(before) == 120
    for a, b in before:
        assert a["t"] == b["t"]
        assert all(a[f] == b[f] for f in fields), a["t"]
    # And the pulse does arrive in the run afterwards.
    after = [(a, b) for a, b in zip(base["trace"], pulsed["trace"]) if a["t"] > 1200.0]
    assert any(a["count"] != b["count"] for a, b in after)
    assert pulsed["pulse"]["departs"][0] == 1201.5
