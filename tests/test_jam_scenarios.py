"""The jam program's three demand files (plan section 4.2, step S1).

They are built with the committed builder beside the shared network by the commands in
plan section 4.2. The unit tests read the committed files and need no SUMO; the sanity
test at the bottom starts SUMO on each, so run this file in its own process.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import train_mainz_src as tms  # noqa: E402

DATA = REPO_ROOT / "data" / "mainz"
#: sha256 of `mainz.net.xml` at the plan's base commit d699ac2; a `--demand-only` build
#: must not change it.
NET_SHA256 = "037de791004725df8c125d796fe9f83b6092087b3290cb33160de2454a601cb3"
#: Measured with the builder in the planning scratchpad (plan step S1).
DEPARTURES = {"mainz_bg": 7144, "mainz_b7500": 7468, "mainz_b9000": 7594}
BURST_RATE = {"mainz_b7500": 7500.0, "mainz_b9000": 9000.0}
BACKGROUND = 3572.0
ENTRIES = 8


def _departs(name: str) -> list[float]:
    return [row["depart"] for row in json.loads((DATA / f"{name}_schedule.json").read_text())]


def _rate(departs: list[float], start: float, end: float) -> float:
    return sum(1 for d in departs if start <= d < end) * 3600.0 / (end - start)


@pytest.mark.parametrize("name", sorted(DEPARTURES))
def test_departure_count(name):
    assert len(_departs(name)) == DEPARTURES[name]


@pytest.mark.parametrize("name", sorted(DEPARTURES))
def test_background_rate_before_900_s(name):
    assert abs(_rate(_departs(name), 0.0, 900.0) - BACKGROUND) <= 10.0


@pytest.mark.parametrize("name", sorted(BURST_RATE))
def test_burst_window_carries_its_rate(name):
    # Each entry's headway is constant inside an interval, so the count is the rate times
    # the window to within one vehicle per entry.
    count = sum(1 for d in _departs(name) if 900.0 <= d < 1200.0)
    assert abs(count - BURST_RATE[name] * 300.0 / 3600.0) <= ENTRIES


@pytest.mark.parametrize("name", sorted(BURST_RATE))
def test_background_resumes_after_the_burst(name):
    assert abs(_rate(_departs(name), 1200.0, 7200.0) - BACKGROUND) <= 10.0


def test_background_has_no_burst():
    departs = _departs("mainz_bg")
    assert abs(_rate(departs, 900.0, 1200.0) - BACKGROUND) <= ENTRIES * 12.0
    assert max(departs) < 7200.0


def test_the_network_is_unchanged():
    assert hashlib.sha256((DATA / "mainz.net.xml").read_bytes()).hexdigest() == NET_SHA256


@pytest.mark.parametrize("name", sorted(DEPARTURES))
def test_scenario_is_registered_beside_the_shared_network(name):
    paths = tms.SCENARIOS[name]
    assert paths["net"] == "data/mainz/mainz.net.xml"
    assert paths["segments"] == "data/mainz/mainz_segments.json"
    assert paths["routes"] == f"data/mainz/{name}.rou.xml"
    assert paths["gated_routes"] == f"data/mainz/{name}_routes.rou.xml"
    assert paths["schedule"] == f"data/mainz/{name}_schedule.json"
    for key in ("routes", "gated_routes", "schedule"):
        assert (REPO_ROOT / paths[key]).exists(), (name, key)


@pytest.mark.parametrize("name", sorted(DEPARTURES))
def test_a_600_s_no_control_episode_starts_and_inserts(name):
    from src.sumo.mainz import HERE_FEATURES

    env = tms.make_env(tms.SCENARIOS[name], seed=1, duration_s=600.0,
                       features=HERE_FEATURES, step_length=0.5, gate_entries=False)
    try:
        env.reset()
        for _ in range(5):
            env.release()
            env._advance(60.0)
        metrics = env.metrics()
    finally:
        env.close()
    assert metrics["time_s"] == pytest.approx(600.0)
    assert metrics["running"] > 0
    assert metrics["arrived"] > 0
