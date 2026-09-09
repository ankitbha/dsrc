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
