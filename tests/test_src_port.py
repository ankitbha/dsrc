"""The predecessor paper's formulation, ported: threshold reward, binding speed
bins, link-level observation, and demand that can be unbalanced.

The paper (arXiv:2506.11973) rewards `-alpha * 1[rho > rho*] + beta * v` summed over
segments, commands 30/45/60 km/h over 2-3 km super-segments once a minute, and
observes per-segment density, speed, gap, inflow and outflow. What this project built
instead was an eleven-term weighted reward, per-vehicle speed bins of 20/27/30 m/s
chosen once a second, and an observation dominated by the agent's own kinematics.

Each test here pins one of those differences, and each is paired with the control
that says the default behaviour is unchanged.
"""
from __future__ import annotations

import pytest

from src.config.loaders import load_named_config
from src.rl.rewards import CRITICAL_DENSITY_RATIO, build_threshold_reward
from src.sumo.env import SumoTopologyEnv


def _env(tmp_path, **overrides):
    config = {
        "topology": load_named_config("topology", "inverted_tree"),
        "demand": load_named_config("demand", "sumo_capacity_drop"),
        "human_model": load_named_config("human_model", "w99_calibrated"),
        "duration_steps": 60,
        "dt": 0.5,
        "work_dir": str(tmp_path),
    }
    config.update(overrides)
    return SumoTopologyEnv("inverted_tree", config)


class TestTheThresholdReward:

    def test_it_is_a_step_and_not_a_gradient(self):
        # The whole point of the shape. A segment at 0.29 and one at 0.31 differ by
        # the full penalty, while 0.31 and 0.99 do not differ at all. A continuous
        # congestion penalty pays for cleaning up congestion; this pays for not
        # crossing the line.
        metrics = {"a": {"mean_speed": 0.0}}
        below = build_threshold_reward(metrics, {"a": 0.29}, congestion_penalty=1.0, speed_weight=0.0)
        just_above = build_threshold_reward(metrics, {"a": 0.31}, congestion_penalty=1.0, speed_weight=0.0)
        far_above = build_threshold_reward(metrics, {"a": 0.99}, congestion_penalty=1.0, speed_weight=0.0)
        assert below == 0.0
        assert just_above == -1.0
        assert far_above == just_above

    def test_the_threshold_is_the_papers_and_is_exclusive(self):
        assert CRITICAL_DENSITY_RATIO == 0.3
        metrics = {"a": {"mean_speed": 0.0}}
        assert build_threshold_reward(metrics, {"a": 0.3}, speed_weight=0.0) == 0.0
        assert build_threshold_reward(metrics, {"a": 0.3001}, speed_weight=0.0) == -1.0

    def test_it_sums_over_segments_rather_than_averaging(self):
        # A network with two congested links is worse than one with a single
        # congested link, which an average cannot express.
        one = build_threshold_reward({"a": {}, "b": {}}, {"a": 0.9, "b": 0.1}, speed_weight=0.0)
        two = build_threshold_reward({"a": {}, "b": {}}, {"a": 0.9, "b": 0.9}, speed_weight=0.0)
        assert two == 2 * one == -2.0

    def test_speed_is_rewarded_linearly_and_survives_the_penalty(self):
        reward = build_threshold_reward(
            {"a": {"mean_speed": 20.0}}, {"a": 0.5},
            congestion_penalty=1.0, speed_weight=0.05)
        assert reward == pytest.approx(-1.0 + 1.0)

    def test_a_missing_ratio_is_treated_as_uncongested_not_as_congested(self):
        # A segment nothing was measured for must not read as a penalty, which
        # would make the reward worse the less it knows.
        assert build_threshold_reward({"a": {"mean_speed": 0.0}}, {}, speed_weight=0.0) == 0.0


class TestDensityRatios:

    def test_the_denominator_scales_with_the_lane_count(self, tmp_path):
        # `density` counts vehicles per kilometre over the whole segment, not per
        # lane, so a two-lane segment at 130 veh/km is at HALF of jam density. A
        # per-lane denominator would report every two-lane segment as twice as
        # congested as it is, and the critical threshold would fire at half the
        # real density.
        from src.sensing.local import JAM_DENSITY_VEH_PER_KM_PER_LANE

        env = _env(tmp_path, duration_steps=40, warmup_steps=200)
        try:
            env.reset(seed=7)
            for _ in range(40):
                env.step({})
            ratios = env.density_ratios()
            segments = env.get_segment_metrics()
            lanes = dict(env.view.lane_counts)
            assert ratios, "no segment was measured"
            for segment_id, ratio in ratios.items():
                expected = (float(segments[segment_id]["density"])
                            / (JAM_DENSITY_VEH_PER_KM_PER_LANE * lanes.get(segment_id, 1)))
                assert ratio == pytest.approx(expected)
            # And the run reached a state where the threshold could fire at all,
            # so the test above is not comparing two zeros.
            assert max(ratios.values()) > 0.0
        finally:
            env.close()


class TestTheSpeedBinsCanBeAbsolute:

    def test_absolute_bins_reach_the_simulator_and_the_default_does_not_change(self, tmp_path):
        # Paired on one seed: the same traffic, commanded `slow` every step, once
        # with the contract's own bins and once with the paper's 8.3 m/s.
        def run(bins):
            env = _env(tmp_path / f"b{bool(bins)}", duration_steps=120, warmup_steps=400,
                       **({"speed_bins_mps": bins} if bins else {}))
            try:
                env.reset(seed=7)
                speeds = []
                for _ in range(120):
                    env.step({agent: {"desired_speed_bin": "slow"} for agent in env.agent_ids})
                    speeds.extend(s.speed_mps for s in env.vehicle_snapshots() if s.role == "av")
                return sum(speeds) / max(len(speeds), 1)
            finally:
                env.close()

        contract = run(None)
        paper = run({"slow": 8.33, "nominal": 12.5, "fast": 16.67})
        assert paper < contract, (
            "the absolute bins did not bind harder than the contract's 20 m/s "
            f"(contract {contract:.2f} m/s, paper {paper:.2f} m/s)"
        )

    def test_a_bin_the_action_space_can_emit_must_have_a_value(self, tmp_path):
        # Silence here would command nothing for that bin, so a third of the action
        # space would be a no-op and look like a policy that had learned to do
        # nothing.
        env = _env(tmp_path, duration_steps=20, warmup_steps=200,
                   speed_bins_mps={"slow": 8.0})
        try:
            env.reset(seed=7)
            with pytest.raises(ValueError, match="every bin"):
                for _ in range(20):
                    env.step({agent: {"desired_speed_bin": "fast"} for agent in env.agent_ids})
        finally:
            env.close()


class TestBranchSplitReachesSumo:

    def _departures_per_entry(self, env):
        import xml.etree.ElementTree as ET

        route_file = env.work_dir / "demand.rou.xml"
        tree = ET.fromstring(route_file.read_text())
        routes = {r.get("id"): r.get("edges").split()[0] for r in tree.iter("route")}
        counts: dict[str, int] = {}
        for vehicle in tree.iter("vehicle"):
            entry = routes[vehicle.get("route")]
            counts[entry] = counts.get(entry, 0) + 1
        return counts

    def test_an_unbalanced_split_produces_unbalanced_demand(self, tmp_path):
        # The entry edge ids come from the built network rather than being typed
        # here: a name that matches no entry is ignored by `_entry_weights`, so a
        # guessed id would silently produce the uniform split and the test would
        # assert against the control it is meant to differ from.
        probe = _env(tmp_path / "probe", duration_steps=1, warmup_steps=0)
        probe.reset(seed=7)
        entries = sorted(probe.network.entry_edges())
        probe.close()
        assert len(entries) == 6, entries

        demand = dict(load_named_config("demand", "sumo_capacity_drop"))
        demand["branch_split"] = {entry: (3.0 if index == 0 else 1.0)
                                  for index, entry in enumerate(entries)}
        # A long SCHEDULE, not a long run: the route file covers warm-up plus
        # duration and is written at reset, so the counts are read without
        # stepping. At 20 steps the whole schedule was 10 s and the heavy branch
        # had 2 departures against 1, which cannot show a threefold ratio.
        env = _env(tmp_path / "split", demand=demand, duration_steps=2000, warmup_steps=0)
        try:
            env.reset(seed=7)
            counts = self._departures_per_entry(env)
            assert set(counts) == set(entries), counts
            assert counts[entries[0]] > 2.5 * counts[entries[1]], counts
        finally:
            env.close()

    def test_the_shipped_configs_are_still_uniform(self, tmp_path):
        # THE CONTROL. Every demand config declares `branch_split: {main: 1.0}`,
        # which names no entry edge. That must resolve to the uniform split every
        # recorded SUMO run was measured under, or this change silently reinterprets
        # every result in the project.
        env = _env(tmp_path / "uniform", duration_steps=2000, warmup_steps=0)
        try:
            env.reset(seed=7)
            counts = self._departures_per_entry(env)
            assert len(set(counts.values())) == 1, counts
        finally:
            env.close()


class TestDownstreamCongestionIsALinkReading:
    """The one observation field a metering controller needs, and it was wrong twice.

    It read the EGO segment's jam fraction despite its name, and it was forced to
    zero whenever the ego had no AV neighbour within sensing range. So the signal a
    controller uses to decide whether to hold back was about the link it had already
    entered, and it vanished exactly when the AV was isolated -- the low-penetration
    case the project exists to study.
    """

    def test_agents_on_one_segment_see_the_same_value(self, tmp_path):
        # The coherence property. Per-vehicle observations differ between
        # neighbours, so agents on one link choose different actions and their
        # effects average out: measured, the fleet-wide action mix sits at 0.307
        # with a standard deviation of 0.008. A link-level field is the same for
        # everyone on the link, which is what turns per-vehicle commands into a
        # segment-level speed limit.
        env = _env(tmp_path, duration_steps=200, warmup_steps=600, dt=0.5)
        try:
            env.reset(seed=7)
            for _ in range(200):
                env.step({})
            observations = env.get_local_observations()
            segment_of = {s.vehicle_id: s.segment_id for s in env.vehicle_snapshots()}
            groups: dict[str, list[dict]] = {}
            for agent_id, obs in observations.items():
                segment = segment_of.get(agent_id)
                if segment:
                    groups.setdefault(segment, []).append(obs)
            shared = [g for g in groups.values() if len(g) > 1]
            assert shared, "no segment held two agents, so nothing was compared"
            for group in shared:
                values = {round(float(o["downstream_congestion_estimate"]), 9) for o in group}
                assert len(values) == 1, values
            # The control: a per-vehicle field DOES differ inside the same groups,
            # so the assertion above is about the field and not about a degenerate
            # traffic state where every vehicle is identical.
            assert any(len({round(float(o["ego_speed"]), 6) for o in group}) > 1
                       for group in shared)
        finally:
            env.close()

    def test_it_reads_the_segment_ahead_and_not_the_ego_segment(self, tmp_path):
        env = _env(tmp_path, duration_steps=200, warmup_steps=600, dt=0.5)
        try:
            env.reset(seed=7)
            for _ in range(200):
                env.step({})
            observations = env.get_local_observations()
            segment_of = {s.vehicle_id: s.segment_id for s in env.vehicle_snapshots()}
            ratios = env.density_ratios()
            downstream = env.view.downstream_segments()
            checked = differing = 0
            for agent_id, obs in observations.items():
                segment = segment_of.get(agent_id)
                if not segment:
                    continue
                successors = downstream.get(segment, ())
                expected = max((ratios[n] for n in successors if n in ratios), default=0.0)
                assert float(obs["downstream_congestion_estimate"]) == pytest.approx(
                    min(1.0, expected))
                checked += 1
                if successors and abs(expected - ratios.get(segment, 0.0)) > 1e-9:
                    differing += 1
            assert checked > 0
            # Without this the assertion passes on the OLD implementation, which
            # returned the ego segment: it only discriminates where the link ahead
            # is in a different state from the link underneath.
            assert differing > 0, "no agent's downstream differed from its own link"
        finally:
            env.close()

    def test_it_survives_an_agent_with_no_av_neighbour(self, tmp_path):
        # The gate that used to zero it. A lone AV on a quiet leaf, with a congested
        # link ahead, must still see the congestion: that is the situation metering
        # exists for, and the one the gate blinded it in.
        from src.sensing.local import LocalObservationBuilder, SensingConfig, VehicleSnapshot
        from src.road.topology_factory import build_topology
        from src.safety import SafetyConstraints, SafetyState
        import numpy as np

        topology = build_topology("inverted_tree")
        downstream = topology.downstream_segments()
        origin = next(s for s in topology.segment_ids if downstream.get(s))
        ahead = downstream[origin][0]
        ego = VehicleSnapshot(
            vehicle_id="ego", role="av", segment_id=origin,
            lane_index=(origin, ahead, 0), lane_id=0, position=(0.0, 0.0),
            longitudinal_m=10.0, speed_mps=20.0, acceleration_mps2=0.0,
            free_flow_speed_mps=30.0,
        )
        builder = LocalObservationBuilder(SensingConfig())
        observation = builder.build_all(
            time_s=0.0, topology=topology, snapshots=[ego], current_av_ids=["ego"],
            safety_states={"ego": SafetyState()}, target_headways={"ego": 1.6},
            target_lanes={"ego": None},
            # A congested link ahead and nothing else on the road: no AV neighbour.
            segment_metrics={ahead: {"density": 200.0, "jam_fraction": 0.9}},
            constraints=SafetyConstraints(), rng=np.random.RandomState(7),
        )["ego"]
        assert observation["downstream_congestion_estimate"] > 0.0, (
            "an isolated AV was blind to a congested link ahead"
        )
        # The control: the fields that ARE built from sensed AV peers still fall to
        # zero, so the assertion above is about this field and not about the gate
        # having been removed wholesale.
        assert observation["merge_pressure"] == 0.0
