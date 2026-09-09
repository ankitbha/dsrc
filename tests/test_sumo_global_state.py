"""What the MAPPO critic and the reward actually receive from the SUMO env.

Found by an independent audit. `get_global_state` returned the metrics dict, and
`encode_physical_global_state` reads `time`, `segment_state` (10 segments x 11
fields) and `demand_state` (2 fields) -- none of which were present. The centralized
critic, which is the entire point of MAPPO over IPPO, was therefore fed
`[0, count/500, av_count/500, 0 x 112]`: two non-zero inputs out of 115, against 31
on highway_env. Both SUMO training runs are invalid as a result.

Three reward inputs were constants for the same reason. `new_collision_count` was
hardcoded 0, and `build_team_reward` prefers it over `collision_count` whenever the
key is present, so the -5.0 collision weight read a constant. `rolling_roadblock_score`
was hardcoded 0.0 while carrying -2.0, the largest remaining weight, and on
highway_env it is the term that penalises AVs holding every lane of a segment below
free flow. With `crashed_agent_ids` empty by construction, all three anti-degenerate
terms were absent at once.

And `fairness_jain` measured a different quantity under the same name: Jain's index
over instantaneous speeds rather than over completions per entry branch, which is
the branch-fairness objective `inverted_tree` exists to study.
"""
from __future__ import annotations

import pytest

from src.config.loaders import load_named_config
from src.rl.encoders import (
    PHYSICAL_DEMAND_FIELDS,
    SEGMENT_FIELDS,
    encode_physical_global_state,
    physical_global_state_dim,
)
from src.sumo.env import SumoTopologyEnv


def _run(tmp_path, steps=120, **overrides):
    config = {
        "topology": load_named_config("topology", "inverted_tree"),
        "demand": load_named_config("demand", "sumo_saturating"),
        "duration_steps": steps, "dt": 1.0, "warmup_steps": 300,
        "work_dir": str(tmp_path),
    }
    config.update(overrides)
    env = SumoTopologyEnv("inverted_tree", config)
    env.reset(seed=7)
    info = {}
    for _ in range(steps):
        _, _, _, _, info = env.step({})
    return env, info


class TestTheCriticReceivesTheNetwork:

    def test_most_of_the_critic_input_is_populated(self, tmp_path):
        env, _ = _run(tmp_path)
        try:
            encoded = encode_physical_global_state(env.get_global_state())
            nonzero = sum(1 for v in encoded if abs(float(v)) > 1e-9)
            assert len(encoded) == physical_global_state_dim()
            # highway_env supplies 31 of 115 at a comparable point. Two was the
            # defect; anything in the twenties means the network is reaching it.
            assert nonzero >= 20, (
                f"only {nonzero} of {len(encoded)} critic inputs are non-zero"
            )
        finally:
            env.close()

    def test_the_three_top_level_fields_are_present(self, tmp_path):
        env, _ = _run(tmp_path)
        try:
            state = env.get_global_state()
            assert state["time"] > 0.0
            assert state["active_vehicle_count"] > 0
            assert "active_av_count" in state
        finally:
            env.close()

    @pytest.mark.parametrize("field", sorted(SEGMENT_FIELDS))
    def test_every_segment_field_the_encoder_reads_is_supplied(self, tmp_path, field):
        env, _ = _run(tmp_path)
        try:
            segments = env.get_global_state()["segment_state"]
            assert segments, "no segments reported"
            for name, segment in segments.items():
                assert field in segment, f"{name} is missing {field}"
        finally:
            env.close()

    @pytest.mark.parametrize("field", sorted(PHYSICAL_DEMAND_FIELDS))
    def test_the_demand_fields_are_supplied(self, tmp_path, field):
        env, _ = _run(tmp_path)
        try:
            assert field in env.get_global_state()["demand_state"]
        finally:
            env.close()

    def test_the_reported_penetration_matches_the_config(self, tmp_path):
        env, _ = _run(tmp_path)
        try:
            demand = env.get_global_state()["demand_state"]
            assert demand["av_penetration"] == pytest.approx(0.2)
        finally:
            env.close()


class TestTheRewardInputsAreMeasuredNotConstant:

    def test_new_collisions_track_the_counter(self, tmp_path):
        # Not asserted to be zero: asserted to be the per-step change in the
        # counter, so a hardcoded constant fails whatever its value.
        env, info = _run(tmp_path, steps=60)
        try:
            assert info["metrics"]["new_collision_count"] == env.new_collisions_last_step
        finally:
            env.close()

    def test_the_roadblock_input_is_measured_and_varies(self, tmp_path):
        # `rolling_roadblock_score` reads 0.0 everywhere at a congesting demand, and
        # that is correct: `_rolling_roadblock_score` returns 1.0 only when AVs hold
        # every lane AND the segment is NOT jammed (jam_fraction <= 0.25,
        # queue_length == 0), because it penalises a rolling roadblock in
        # free-flowing traffic rather than AVs sitting in a genuine jam. Asserting a
        # non-zero score here would be asserting the wrong thing.
        #
        # So the guard is on its input. `all_lane_av_low_speed_occupancy` must be a
        # measured per-segment quantity, which is what a hardcoded 0.0 destroyed.
        env, _ = _run(tmp_path)
        try:
            segments = env.get_global_state()["segment_state"]
            occupancy = {name: s["all_lane_av_low_speed_occupancy"]
                         for name, s in segments.items()}
            assert occupancy, "no segments"
            assert len(set(occupancy.values())) > 1, (
                f"every segment reports the same occupancy: {occupancy}"
            )
            congested = [name for name, s in segments.items()
                         if s["mean_speed"] < 5.0 and s["av_count"] > 0]
            assert congested, "no congested segment with AVs to check"
            for name in congested:
                assert occupancy[name] > 0.0, (
                    f"{name} has AVs at {segments[name]['mean_speed']:.2f} m/s and "
                    f"reports occupancy {occupancy[name]}"
                )
        finally:
            env.close()

    def test_the_roadblock_score_follows_its_documented_rule(self, tmp_path):
        # Pins the rule rather than a value, so a change to the shared
        # implementation shows up here rather than silently altering the reward.
        from src.metrics.segment_metrics import _rolling_roadblock_score

        jammed = {"all_lane_av_low_speed_occupancy": 1.0, "jam_fraction": 0.8,
                  "queue_length": 40}
        rolling = {"all_lane_av_low_speed_occupancy": 1.0, "jam_fraction": 0.0,
                   "queue_length": 0}
        empty = {"all_lane_av_low_speed_occupancy": 0.0, "jam_fraction": 0.0,
                 "queue_length": 0}
        assert _rolling_roadblock_score(jammed) == 0.0
        assert _rolling_roadblock_score(rolling) == 1.0
        assert _rolling_roadblock_score(empty) == 0.0

    def test_fairness_is_over_branch_completions_not_speeds(self, tmp_path):
        env, info = _run(tmp_path, steps=300)
        try:
            branches = env.get_global_state()["branch_state"]["per_branch_completed"]
            assert branches, "no per-branch completions recorded"
            assert sum(branches.values()) > 0
            # Six entry branches feeding two nodes: with the demand split evenly the
            # index should be high but not exactly 1.0, and it must not equal the
            # Jain index over speeds, which is a different number entirely.
            speeds = [s.speed_mps for s in env.vehicle_snapshots()]
            over_speeds = (sum(speeds) ** 2 / (len(speeds) * sum(v * v for v in speeds))
                           if speeds and sum(v * v for v in speeds) > 0 else 1.0)
            assert info["metrics"]["fairness_jain"] != pytest.approx(over_speeds, abs=1e-6), (
                "fairness_jain still equals the Jain index over speeds"
            )
        finally:
            env.close()


class TestWarmupArrivalsCountTowardTheWindow:

    def test_throughput_is_not_depressed_at_the_start_of_an_episode(self, tmp_path):
        # The rolling window refilled from empty after a warm-up that had already
        # put the network in steady state, so throughput_recent read 5.4 over the
        # first 60 steps against 11.0 afterwards -- on the term the config gives the
        # largest positive weight, for half of every 120-step episode.
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 200, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            early, late = [], []
            for step in range(200):
                _, _, _, _, info = env.step({})
                value = info["metrics"]["throughput_recent"]
                (early if step < 60 else late).append(value)
            mean_early = sum(early) / len(early)
            mean_late = sum(late) / len(late)
            assert mean_early > 0.7 * mean_late, (
                f"throughput over the first 60 steps is {mean_early:.1f} against "
                f"{mean_late:.1f} afterwards, so the window still refills from empty"
            )
        finally:
            env.close()


class TestFairnessIsMeasuredOverEveryEntryBranch:
    """`_branch_completed` gained a key only on a branch's first completion, so
    Jain's index was computed over the branches that had completed a vehicle rather
    than over all six. One branch at 1 and five at 0 read 1.0000 against a true
    0.1667, and 95 of 120 steps of the default evaluation config reported exactly
    1.0: the term reported its best value in the state it exists to detect.
    """

    def _env(self, tmp_path, **overrides):
        config = {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 200, "dt": 1.0, "warmup_steps": 0,
            "work_dir": str(tmp_path),
        }
        config.update(overrides)
        return SumoTopologyEnv("inverted_tree", config)

    def test_every_entry_branch_is_in_the_denominator_from_reset(self, tmp_path):
        env = self._env(tmp_path)
        env.reset(seed=7)
        try:
            branches = env.get_global_state()["branch_state"]["per_branch_completed"]
            assert set(branches) == set(env.network.entry_edges())
            assert len(branches) == 6, f"six entry branches expected, got {branches}"
            assert set(branches.values()) == {0}
        finally:
            env.close()

    def test_a_single_branch_completing_alone_scores_one_sixth(self, tmp_path):
        # With six branches, one at any count n and five at zero, Jain's index is
        # n^2 / (6 n^2) = 1/6 whatever n is. This is the state the metric exists to
        # detect and the state the defect scored 1.0.
        env = self._env(tmp_path)
        env.reset(seed=7)
        try:
            for _ in range(200):
                _, _, _, _, info = env.step({})
                completed = env.get_global_state()["branch_state"]["per_branch_completed"]
                nonzero = [count for count in completed.values() if count > 0]
                if len(nonzero) == 1:
                    assert info["metrics"]["fairness_jain"] == pytest.approx(1.0 / 6.0), (
                        f"one branch serving alone scored "
                        f"{info['metrics']['fairness_jain']} over {completed}"
                    )
                    break
            else:
                pytest.fail("no step had exactly one branch with completions")
        finally:
            env.close()

    def test_a_partly_served_network_never_reports_maximal_fairness(self, tmp_path):
        # The general form of the two assertions above, over a whole episode: while
        # between one and five of the six branches have completed anything, the
        # index must be below 1.0. The defect reported exactly 1.0 on every such
        # step -- 95 of the 120 steps of this configuration.
        env = self._env(tmp_path, duration_steps=120)
        env.reset(seed=7)
        try:
            partly_served = 0
            for _ in range(120):
                _, _, _, _, info = env.step({})
                completed = env.get_global_state()["branch_state"]["per_branch_completed"]
                served = sum(1 for count in completed.values() if count > 0)
                if 0 < served < 6:
                    partly_served += 1
                    assert info["metrics"]["fairness_jain"] < 1.0, (
                        f"{served} of 6 branches served, fairness "
                        f"{info['metrics']['fairness_jain']} over {completed}"
                    )
            # Without this the assertion above would pass on an episode in which the
            # branches were never partly served, proving nothing.
            assert partly_served > 10, (
                f"only {partly_served} steps had between one and five branches served"
            )
        finally:
            env.close()


class TestArrivedTotalCountsTheEpisodeOnly:
    """`arrived_total` is what every recorded arrival figure is measured from and
    what the critic reads as `completed_vehicle_count`. Adding warm-up arrivals to
    it made it read 43 before the episode began and 153 against the same run's 110.
    """

    def test_no_arrivals_are_counted_before_the_episode_starts(self, tmp_path):
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 120, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            assert env.arrived_total == 0, (
                f"{env.arrived_total} arrivals counted during warm-up; the warm-up "
                "must fill the network without contributing to the episode's total"
            )
            # The control: the warm-up did run and vehicles did complete during it,
            # so the zero above is a scoping decision and not an empty network.
            assert len(env._arrivals) > 0
            assert all(time <= 0.0 for time in env._arrivals)
        finally:
            env.close()

    def test_the_total_equals_the_arrivals_observed_during_the_episode(self, tmp_path):
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 120, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            for _ in range(120):
                env.step({})
            # The two counters are computed by different SUMO calls -- a count from
            # `getArrivedNumber` and a per-vehicle attribution from
            # `getArrivedIDList` -- so their agreement is a real cross-check that
            # both cover the episode and only the episode.
            branch_total = sum(
                env.get_global_state()["branch_state"]["per_branch_completed"].values()
            )
            assert env.arrived_total > 0
            assert branch_total == env.arrived_total, (
                f"branch completions {branch_total} disagree with arrived_total "
                f"{env.arrived_total}; both must be scoped to the episode"
            )
        finally:
            env.close()


class TestSegmentFlowsReachEveryConsumer:
    """`inflow` and `outflow` were computed inside `get_segment_metrics`, which also
    advanced the segment map as a side effect. That getter runs two or three times
    per step and `get_global_state` runs last, so the critic and the logs compared
    the step against itself and read zero on both fields, on 2 of the 11 fields
    supplied per segment.
    """

    def _env(self, tmp_path, **overrides):
        config = {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 120, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path),
        }
        config.update(overrides)
        return SumoTopologyEnv("inverted_tree", config)

    def test_every_call_within_one_step_reports_the_same_flows(self, tmp_path):
        env = self._env(tmp_path)
        env.reset(seed=7)
        try:
            moved = 0
            for _ in range(40):
                env.step({})
                direct = env.get_segment_metrics()
                state_view = env.get_global_state()["segment_state"]
                for segment_id, metrics in direct.items():
                    moved += metrics["inflow"] + metrics["outflow"]
                    assert (metrics["inflow"], metrics["outflow"]) == (
                        state_view[segment_id]["inflow"],
                        state_view[segment_id]["outflow"],
                    ), f"{segment_id} reports different flows to different consumers"
            # Without this the assertion above is satisfied by two sets of zeros,
            # which is exactly the state the defect produced.
            assert moved > 20, f"only {moved} boundary crossings seen in 40 steps"
        finally:
            env.close()

    def test_the_critic_sees_flow_on_interior_segments(self, tmp_path):
        env = self._env(tmp_path)
        env.reset(seed=7)
        try:
            with_inflow, with_outflow = set(), set()
            for _ in range(120):
                env.step({})
                for segment_id, metrics in env.get_global_state()["segment_state"].items():
                    if metrics["inflow"]:
                        with_inflow.add(segment_id)
                    if metrics["outflow"]:
                        with_outflow.add(segment_id)
            assert len(with_inflow) >= 8, f"inflow only ever seen on {sorted(with_inflow)}"
            assert len(with_outflow) >= 8, f"outflow only ever seen on {sorted(with_outflow)}"
            # Interior segments carry no spawns and no exits, so they are the ones a
            # spawn-and-exit-only accounting leaves permanently at zero.
            assert {"tree_middle_b1", "tree_middle_b2", "tree_trunk_c"} <= with_inflow
        finally:
            env.close()

    def test_flow_accounts_for_every_change_in_segment_occupancy(self, tmp_path):
        # The invariant that makes the flows trustworthy rather than merely
        # non-zero: a segment's occupancy changes by exactly its inflow minus its
        # outflow, so no vehicle is counted twice and none is lost.
        env = self._env(tmp_path)
        env.reset(seed=7)
        try:
            previous_counts: dict[str, int] = {}
            checked = 0
            for step in range(60):
                env.step({})
                state = env.get_global_state()["segment_state"]
                if previous_counts:
                    for segment_id, metrics in state.items():
                        change = metrics["vehicle_count"] - previous_counts.get(segment_id, 0)
                        assert change == metrics["inflow"] - metrics["outflow"], (
                            f"step {step}, {segment_id}: occupancy changed by {change} "
                            f"against inflow {metrics['inflow']} and outflow "
                            f"{metrics['outflow']}"
                        )
                        checked += 1
                previous_counts = {s: m["vehicle_count"] for s, m in state.items()}
            assert checked > 400
        finally:
            env.close()


class TestThresholdsComeFromOnePlace:
    """The rolling throughput window was read from the top level of the config,
    where nothing writes it, while `HighwayTopologyEnv` reads it from
    `metrics.thresholds`. A config setting it there was honoured on one simulator
    and ignored on the other.
    """

    def _env(self, tmp_path, window):
        return SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 120, "dt": 1.0, "warmup_steps": 300,
            "metrics": {"thresholds": {"throughput_window_s": window}},
            "work_dir": str(tmp_path)})

    def test_the_declared_throughput_window_is_honoured(self, tmp_path):
        # `throughput_recent` counts arrivals inside the window, so a window ten
        # times as long must count more of them.
        def mean_recent(window):
            env = self._env(tmp_path, window)
            env.reset(seed=7)
            try:
                values = []
                for _ in range(120):
                    _, _, _, _, info = env.step({})
                    values.append(info["metrics"]["throughput_recent"])
                return sum(values) / len(values)
            finally:
                env.close()

        narrow = mean_recent(6.0)
        wide = mean_recent(60.0)
        assert wide > narrow * 5, (
            f"a 60 s window counted {wide:.2f} arrivals on average and a 6 s window "
            f"{narrow:.2f}; the declared window is not reaching the metric"
        )
