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

import statistics

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
        "human_model": load_named_config("human_model", "w99_calibrated"),
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
        # Scanned across every step, not read off the final one. Congestion moves
        # on this network: it used to sit permanently on tree_middle_b1, because an
        # unintended yield starved that approach (task 98), and a check on the last
        # step happened to work. With the merge corrected there may be no slow
        # segment at any particular instant.
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 200, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            distinct, checked = 0, 0
            for _ in range(200):
                # Commanded slow, so segments carrying AVs below 5 m/s exist by
                # construction. Waiting for traffic to produce them worked only on
                # the network whose capacity was set by a permanent yield, where one
                # approach crawled permanently; at this demand on the corrected road
                # no segment holds AVs that slow at any step.
                for agent_id in list(env.agent_ids):
                    env.command_speed(agent_id, 3.0)
                env.step({})
                segments = env.get_global_state()["segment_state"]
                occupancy = {name: seg["all_lane_av_low_speed_occupancy"]
                             for name, seg in segments.items()}
                if len(set(occupancy.values())) > 1:
                    distinct += 1
                for name, seg in segments.items():
                    # The metric requires an AV in EVERY lane, not merely a slow AV
                    # somewhere on the segment: `_all_lane_av_low_speed_occupancy`
                    # returns 0 unless every lane holds one. Asserting on any slow
                    # AV misstated the rule and passed only on the network whose
                    # capacity was set by a permanent yield, where the starved
                    # approach was packed enough for AVs to hold both its lanes.
                    lanes = seg["lane_av_counts"]
                    if (seg["mean_speed"] < 5.0 and lanes
                            and all(count > 0 for count in lanes.values())):
                        checked += 1
                        assert occupancy[name] > 0.0, (
                            f"{name} has an AV in every lane at "
                            f"{seg['mean_speed']:.2f} m/s and reports occupancy "
                            f"{occupancy[name]}"
                        )
            assert checked > 20, (
                f"only {checked} segment-steps had an AV in every lane below 5 m/s, "
                "so the assertion above was barely exercised"
            )
            assert distinct > 20, (
                f"segments reported the same occupancy on all but {distinct} steps"
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
            "human_model": load_named_config("human_model", "w99_calibrated"),
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
            "human_model": load_named_config("human_model", "w99_calibrated"),
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
            "human_model": load_named_config("human_model", "w99_calibrated"),
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
            "human_model": load_named_config("human_model", "w99_calibrated"),
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
            "human_model": load_named_config("human_model", "w99_calibrated"),
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
        #
        # On `inverted_tree_bottleneck`, and for 400 steps, because the interesting
        # case is rare elsewhere. A vehicle usually leaves one segment, spends a step
        # with no segment inside the junction, and appears on the next, which the
        # disappearance half of the accounting covers. Crossing directly from one
        # segment to another within a single step -- the case the transition half
        # covers -- happens once in 400 steps on `inverted_tree` and 46 times on the
        # bottleneck variant, so a 60-step run on the plain tree almost never
        # exercises it and a mutant deleting that half survived.
        env = SumoTopologyEnv("inverted_tree_bottleneck", {
            "topology": load_named_config("topology", "inverted_tree_bottleneck"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 400, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            previous_counts: dict[str, int] = {}
            checked = 0
            direct_transitions = 0
            for step in range(400):
                before = dict(env._segment_of)
                env.step({})
                state = env.get_global_state()["segment_state"]
                direct_transitions += sum(
                    1 for vehicle_id, segment_id in env._segment_of.items()
                    if before.get(vehicle_id) not in (None, segment_id)
                )
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
            assert checked > 3000
            # Without this the run could satisfy the invariant while never crossing
            # a segment boundary within a step, which is the half of the accounting
            # a surviving mutant deleted.
            assert direct_transitions > 10, (
                f"only {direct_transitions} direct segment-to-segment transitions, "
                "so the transition half of the accounting was barely exercised"
            )
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
            "human_model": load_named_config("human_model", "w99_calibrated"),
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


class TestTheAggregatesCoverEveryVehicle:
    """`mean_speed` and `active_vehicle_count` were computed from the snapshots,
    which skip vehicles inside a junction because they have no edge and so no
    segment. That is right for the per-segment metrics and wrong for a network-wide
    aggregate: junction-crossing vehicles are moving, so the reported mean speed sat
    below SUMO's own.
    """

    def test_the_reported_mean_speed_is_sumos_own(self, tmp_path):
        import statistics

        from src.sumo import env as env_module

        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 200, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            excluded = 0
            for _ in range(200):
                _, _, _, _, info = env.step({})
                ids = env_module._sumo.vehicle.getIDList()
                speeds = [env_module._sumo.vehicle.getSpeed(v) for v in ids]
                assert info["metrics"]["mean_speed"] == pytest.approx(
                    statistics.fmean(speeds) if speeds else 0.0, abs=1e-6)
                assert info["metrics"]["active_vehicle_count"] == len(ids)
                excluded += len(ids) - len(env.vehicle_snapshots())
            # Without this the assertions above would be satisfied by a run in which
            # no vehicle was ever inside a junction, where the two agree anyway.
            assert excluded > 100, (
                f"only {excluded} vehicle-steps were inside a junction, so the "
                "comparison never had anything to distinguish"
            )
        finally:
            env.close()


class TestTheAntiDegenerateTermsAreMeasured:
    """`rolling_roadblock_score` and `all_lane_av_low_speed_occupancy` were once
    hardcoded to 0.0. They were given real values, but mutants restoring the
    constants survived: the tests written at the time read the per-segment values
    rather than the aggregate the reward consumes, and compared two zeros in a run
    where no AV was ever holding a lane below free flow.

    `rolling_roadblock_score` carries -2.0, the largest negative weight after the
    collision term, and it is what stops a policy learning to sit across every lane
    of a segment. A constant 0.0 removes the only term that penalises exactly that.
    """

    def _run(self, tmp_path, commanded, penetration=0.7, steps=200):
        # 0.7, not 0.2. `all_lane_av_low_speed_occupancy` requires an AV in EVERY
        # lane of a segment, and with two lanes everywhere a fifth of the fleet
        # rarely manages it: the term fired on 0 segment-steps at 0.2.
        demand = dict(load_named_config("demand", "sumo_saturating"))
        demand["av_penetration"] = penetration
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": demand,
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": steps, "dt": 1.0,
            "warmup_steps": 300, "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            roadblock, occupancy = [], []
            for _ in range(steps):
                if commanded is not None:
                    for agent_id in list(env.agent_ids):
                        env.command_speed(agent_id, commanded)
                _, _, _, _, info = env.step({})
                roadblock.append(info["metrics"]["rolling_roadblock_score"])
                occupancy.append(info["metrics"]["all_lane_av_low_speed_occupancy"])
            return roadblock, occupancy
        finally:
            env.close()

    def test_holding_the_lanes_below_free_flow_raises_the_roadblock_score(self, tmp_path):
        idle, _ = self._run(tmp_path, commanded=None)
        blocking, _ = self._run(tmp_path, commanded=10.0)
        idle_mean = sum(idle) / len(idle)
        blocking_mean = sum(blocking) / len(blocking)
        # 0.10, and the arm is 10 m/s rather than 5. Measured on the corrected road
        # with the calibrated driving model: uncommanded scores 0.0000, 5 m/s scores
        # 0.0783, 10 m/s scores 0.2172 and 16 m/s scores 0.0472. The peak is at
        # 10 m/s, not at the slowest arm, because the term fires only where AVs are
        # slow on a CLEAR road: at 5 m/s they jam the segment they are on, and the
        # term's own second and third conditions then exclude it. On the previous
        # network, whose capacity was set by a permanent yield, 5 m/s scored 0.1350.
        assert blocking_mean > 0.10, (
            f"AVs held at 5 m/s scored {blocking_mean:.4f} on the term that exists "
            "to penalise exactly that"
        )
        # The control: without it a constant 0.15 would satisfy the assertion above.
        assert idle_mean < 0.05, f"uncommanded traffic already scores {idle_mean:.4f}"

    def test_holding_the_lanes_raises_the_occupancy_term(self, tmp_path):
        _, idle = self._run(tmp_path, commanded=None)
        _, blocking = self._run(tmp_path, commanded=10.0)
        # Measured with two lanes everywhere at penetration 0.7: commanded 0.53
        # against uncommanded 0.20, a factor of 2.6. The uncommanded figure is no
        # longer near zero because at this penetration AVs sometimes occupy both
        # lanes of a segment without being told to, which is the honest reading of
        # a metric that asks whether every lane holds a slow AV.
        assert sum(blocking) / len(blocking) > 0.40
        assert sum(idle) / len(idle) < 0.30

    def test_the_aggregate_is_the_mean_over_segments(self, tmp_path):
        # Pins the aggregation as well as the magnitude: the reward reads one number
        # per step and the per-segment values are what it is built from.
        demand = dict(load_named_config("demand", "sumo_saturating"))
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": demand,
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 60, "dt": 1.0,
            "warmup_steps": 300, "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            nonzero = 0
            for _ in range(60):
                for agent_id in list(env.agent_ids):
                    env.command_speed(agent_id, 5.0)
                _, _, _, _, info = env.step({})
                segments = env.get_segment_metrics()
                for field in ("rolling_roadblock_score", "all_lane_av_low_speed_occupancy"):
                    expected = sum(float(m[field]) for m in segments.values()) / len(segments)
                    assert info["metrics"][field] == pytest.approx(expected)
                    if expected > 0:
                        nonzero += 1
            assert nonzero > 20, f"only {nonzero} non-zero values, so the check is vacuous"
        finally:
            env.close()


class TestTheFirstObservedStepIsNotOneLargeInflow:
    """`_warm_up` seeds `_segment_of` from the vehicles already on the network. A
    mutant deleting that seeding survived: without it the first observed step
    reports every vehicle on the network as having just entered its segment, which
    is a 32-vehicle inflow spike on the critic's first input of the episode.
    """

    def test_the_first_step_reports_ordinary_flow(self, tmp_path):
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 20, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            occupied = len(env.vehicle_snapshots())
            assert occupied > 20, "the warm-up did not fill the network"
            env.step({})
            first = sum(m["inflow"] for m in
                        env.get_global_state()["segment_state"].values())
            later = []
            for _ in range(19):
                env.step({})
                later.append(sum(m["inflow"] for m in
                                 env.get_global_state()["segment_state"].values()))
            assert first <= max(later) + 2, (
                f"the first observed step reported {first} vehicles entering a "
                f"segment against at most {max(later)} on later steps, with "
                f"{occupied} already on the network when the episode began"
            )
        finally:
            env.close()


class TestThePenetrationIsVisibleToTheCritic:
    """`_network_census` classifies a vehicle as an AV by its type id. A mutant
    classifying every vehicle as an AV survived: `active_av_count` is one of the
    three top-level inputs the critic receives, and it would have read the total
    vehicle count at every penetration.
    """

    def _counts(self, tmp_path, penetration):
        demand = dict(load_named_config("demand", "sumo_saturating"))
        demand["av_penetration"] = penetration
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": demand,
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 100, "dt": 1.0,
            "warmup_steps": 300, "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            totals, avs = 0, 0
            for _ in range(100):
                env.step({})
                state = env.get_global_state()
                totals += state["active_vehicle_count"]
                avs += state["active_av_count"]
            return totals, avs
        finally:
            env.close()

    def test_no_vehicle_is_an_av_at_zero_penetration(self, tmp_path):
        totals, avs = self._counts(tmp_path, 0.0)
        assert totals > 1000
        assert avs == 0, f"{avs} AVs counted in a fleet declared to have none"

    def test_the_counted_share_follows_the_declared_penetration(self, tmp_path):
        totals, avs = self._counts(tmp_path, 0.2)
        share = avs / totals
        assert 0.1 < share < 0.35, (
            f"{share:.3f} of vehicle-steps were counted as AVs against a declared "
            "0.2; the classification is not reading the vehicle type"
        )


class TestFlowMagnitudeIsBoundaryCrossingsNotOccupancy:
    """A mutant counting every vehicle that stayed on its segment as both an inflow
    and an outflow survived the conservation invariant, because it adds one to each
    side and the difference is unchanged. The invariant alone therefore does not
    pin the magnitude, and the critic would have received a flow of roughly the
    vehicle count on every segment on every step.
    """

    def test_flow_is_far_smaller_than_occupancy(self, tmp_path):
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 120, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            inflow_total = occupancy_total = 0
            for _ in range(120):
                env.step({})
                state = env.get_global_state()["segment_state"]
                inflow_total += sum(m["inflow"] for m in state.values())
                occupancy_total += sum(m["vehicle_count"] for m in state.values())
            # Vehicles cross a segment boundary about once every few hundred metres
            # of travel, so boundary crossings per step are a small fraction of the
            # vehicles present. Measured at roughly 0.8 against 58.
            assert occupancy_total / 120 > 30, "the network was not occupied enough to compare"
            assert inflow_total < occupancy_total / 10, (
                f"{inflow_total} inflow events against {occupancy_total} "
                "vehicle-steps: the flow is tracking occupancy, not boundary crossings"
            )
        finally:
            env.close()


class TestMeteringIsNotScoredAsObstruction:
    """`_rolling_roadblock_score` forbids AVs holding every lane of a segment below
    free flow while that segment is neither jammed nor queued. Its three conditions
    only ever look at the segment being scored, so holding a clear segment slow
    BECAUSE the next one is congested -- speed metering, which the project calls a
    legitimate mechanism -- was scored identically to obstruction.

    Measured before the exemption, with every AV commanded to 10 m/s over 300
    steps: 767 segment-steps scored a roadblock, of which 295 had a jammed segment
    immediately downstream and 372 had a clear one.
    """

    def _holding(self, **overrides):
        record = {"all_lane_av_low_speed_occupancy": 1.0, "jam_fraction": 0.0,
                  "queue_length": 0}
        record.update(overrides)
        return record

    def test_a_clear_road_ahead_is_still_obstruction(self, tmp_path):
        from src.metrics.segment_metrics import _rolling_roadblock_score

        clear = self._holding(jam_fraction=0.0, queue_length=0)
        assert _rolling_roadblock_score(self._holding(), [clear]) == 1.0
        # No downstream segment at all -- the trunk -- is also unexcused, because
        # there is no traffic reason to be slow.
        assert _rolling_roadblock_score(self._holding(), []) == 1.0

    def test_a_jammed_road_ahead_excuses_the_hold(self, tmp_path):
        from src.metrics.segment_metrics import _rolling_roadblock_score

        assert _rolling_roadblock_score(
            self._holding(), [self._holding(jam_fraction=0.5)]) == 0.0
        assert _rolling_roadblock_score(
            self._holding(), [self._holding(queue_length=3)]) == 0.0
        # Just below the threshold the term still fires, so the boundary is the
        # same one the segment applies to itself.
        assert _rolling_roadblock_score(
            self._holding(), [self._holding(jam_fraction=0.25)]) == 1.0

    def test_any_congested_downstream_segment_excuses_it(self, tmp_path):
        from src.metrics.segment_metrics import _rolling_roadblock_score

        # The permissive reading, documented rather than measured: inverted_tree
        # has no diverge, so no topology here produces more than one.
        assert _rolling_roadblock_score(
            self._holding(),
            [self._holding(), self._holding(jam_fraction=0.9)]) == 0.0

    def test_the_first_three_conditions_still_hold(self, tmp_path):
        from src.metrics.segment_metrics import _rolling_roadblock_score

        # The control on the whole class: the exemption must not have turned the
        # term off. A segment that is itself jammed, or itself queued, or that has
        # no AV holding every lane, scores zero whatever is downstream.
        clear = [self._holding(jam_fraction=0.0, queue_length=0)]
        assert _rolling_roadblock_score(self._holding(jam_fraction=0.5), clear) == 0.0
        assert _rolling_roadblock_score(self._holding(queue_length=1), clear) == 0.0
        assert _rolling_roadblock_score(
            self._holding(all_lane_av_low_speed_occupancy=0.0), clear) == 0.0

    def test_metering_is_excused_in_a_live_run(self, tmp_path):
        # Penetration 0.7. The exemption can only fire where the roadblock
        # conditions hold, which needs an AV in EVERY lane, and with two lanes
        # everywhere a fifth of the fleet manages that on 80 segment-steps against
        # 186 at 0.7.
        demand = dict(load_named_config("demand", "sumo_saturating"))
        demand["av_penetration"] = 0.7
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": demand,
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 300, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            scored = excused = 0
            for _ in range(300):
                for agent_id in list(env.agent_ids):
                    env.command_speed(agent_id, 10.0)
                env.step({})
                for metrics in env.get_segment_metrics().values():
                    holding = (metrics["all_lane_av_low_speed_occupancy"] > 0
                               and metrics["jam_fraction"] <= 0.25
                               and metrics["queue_length"] == 0)
                    if metrics["rolling_roadblock_score"] > 0:
                        scored += 1
                    elif holding:
                        excused += 1
            assert excused > 100, (
                f"only {excused} segment-steps were excused as metering; the "
                "downstream mapping may not be reaching the metric"
            )
            # The other half must still be scored: the exemption licenses metering,
            # it does not license obstruction.
            assert scored > 100, f"only {scored} segment-steps still score a roadblock"
        finally:
            env.close()


class TestEveryWeightedRewardTermIsMeasured:
    """Five mutants that each replace a reward term with a constant survived two
    audits. Every one changes a number the project reports, and the reward is what
    a training run maximises, so a term that is silently constant is a term the
    policy is not being asked about.

    Contributions measured over 300 steps at the operating point: throughput_recent
    +1.2437, mean_speed +0.5825, queue_length_total -0.4497, fairness_jain +0.3767,
    speed_std -0.2192, jam_fraction -0.2083, hard_braking_count -0.1083.
    """

    def _metrics(self, tmp_path, steps=120, commanded=None):
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": steps, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            history = []
            for _ in range(steps):
                if commanded is not None:
                    for agent_id in list(env.agent_ids):
                        env.command_speed(agent_id, commanded)
                _, _, _, _, info = env.step({})
                history.append(dict(info["metrics"]))
            return history
        finally:
            env.close()

    def test_queue_length_total_is_the_sum_over_segments(self, tmp_path):
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 60, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            nonzero = 0
            for _ in range(60):
                _, _, _, _, info = env.step({})
                expected = sum(int(m["queue_length"])
                               for m in env.get_segment_metrics().values())
                assert info["metrics"]["queue_length_total"] == expected
                nonzero += expected > 0
            assert nonzero > 30, f"a queue formed on only {nonzero} of 60 steps"
        finally:
            env.close()

    def test_speed_std_is_the_spread_of_the_fleet(self, tmp_path):
        history = self._metrics(tmp_path)
        values = [m["speed_std"] for m in history]
        assert min(values) > 0.0, "speed_std reads zero on some step"
        assert statistics.mean(values) > 1.0, (
            f"mean speed_std is {statistics.mean(values):.3f}; a congesting network "
            "has a wide speed distribution"
        )

    def test_hard_braking_is_counted(self, tmp_path):
        # Braking is rare in ordinary traffic here, so the premise is made active by
        # commanding a low speed, which forces followers to decelerate.
        idle = self._metrics(tmp_path, commanded=None)
        braking = self._metrics(tmp_path, commanded=2.0)
        idle_total = sum(m["hard_braking_count"] for m in idle)
        braking_total = sum(m["hard_braking_count"] for m in braking)
        assert braking_total > idle_total, (
            f"commanding 2 m/s produced {braking_total} hard-braking events against "
            f"{idle_total} uncommanded"
        )
        assert braking_total > 20

    def test_jam_fraction_is_a_mean_and_not_a_maximum(self, tmp_path):
        # Aggregated as a max instead of a mean, jam_fraction reads 1.0 whenever any
        # single vehicle on a segment is below the queue speed, and the mean team
        # reward moves from +1.21 to -0.27 per step -- larger than the leading term.
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 60, "dt": 1.0, "warmup_steps": 300,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            strictly_between = 0
            for _ in range(60):
                env.step({})
                for segment_id, metrics in env.get_segment_metrics().items():
                    fraction = metrics["jam_fraction"]
                    assert 0.0 <= fraction <= 1.0
                    if 0.0 < fraction < 1.0:
                        strictly_between += 1
            # A maximum can only ever be 0.0 or 1.0, so a value strictly between the
            # two is what distinguishes the mean from it.
            assert strictly_between > 50, (
                f"only {strictly_between} segment-steps had a jam fraction strictly "
                "between 0 and 1, so this cannot tell a mean from a maximum"
            )
        finally:
            env.close()

    def test_the_collision_metric_carries_the_counter_not_a_constant(self, tmp_path, monkeypatch):
        # The existing test asserts the metric equals the attribute, and both are 0
        # in every real run: it compares two zeros. This drives collisions from
        # SUMO's side and reads the METRIC, which is what the reward consumes.
        import src.sumo.env as env_module

        reported = [(), ("v1", "v2"), ("v1", "v2"), ("v1", "v2", "v3"), (), ("v1",)]
        calls = iter(reported)
        monkeypatch.setattr(env_module._sumo.simulation, "getCollidingVehiclesIDList",
                            lambda: next(calls, ()))
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 6, "dt": 1.0, "warmup_steps": 0,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            seen = []
            for _ in range(6):
                _, _, _, _, info = env.step({})
                seen.append(info["metrics"]["new_collision_count"])
            assert seen == [0, 2, 0, 1, 0, 1], seen
            assert any(seen), "the metric never left zero, so this proves nothing"
        finally:
            env.close()
