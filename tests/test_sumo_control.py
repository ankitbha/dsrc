"""Observations out of the SUMO env, and AV commands into it.

The sensing model, the safety layer and the encoders are unchanged; this is the
wiring that feeds them SUMO's state and applies what they decide. The measurements
that matter are that an AV commanded slower actually slows, and that the metrics the
reward reads are populated from SUMO rather than defaulted.
"""
from __future__ import annotations

import pytest

from src.config.loaders import load_named_config
from src.sumo.env import SumoTopologyEnv


def _env(tmp_path, **overrides):
    config = {
        "topology": load_named_config("topology", "inverted_tree"),
        "demand": load_named_config("demand", "saturating"),
        "duration_steps": 200,
        "dt": 1.0,
        "work_dir": str(tmp_path),
    }
    config.update(overrides)
    return SumoTopologyEnv("inverted_tree", config)


class TestObservations:

    def test_each_av_gets_an_observation(self, tmp_path):
        env = _env(tmp_path)
        try:
            env.reset(seed=7)
            observations = {}
            for _ in range(120):
                observations, _, _, _, _ = env.step({})
                if observations:
                    break
            assert observations, "no AV observation after 120 steps"
            assert set(observations) == set(env.agent_ids)
            for observation in observations.values():
                assert observation["ego_speed"] >= 0.0
                assert observation["is_active"] is True
                assert "leader_gap" in observation
        finally:
            env.close()

    def test_a_leader_shortens_the_reported_gap(self, tmp_path):
        # The control that the gap is measured rather than defaulted: with traffic
        # present at a congesting demand, at least one AV must see a finite leader.
        env = _env(tmp_path)
        try:
            env.reset(seed=7)
            finite = 0
            for _ in range(200):
                observations, _, _, _, _ = env.step({})
                finite += sum(
                    1 for o in observations.values()
                    if o["leader_gap"] != float("inf")
                )
            assert finite > 0, "no AV ever reported a finite leader gap"
        finally:
            env.close()


class TestActuation:

    def test_an_av_commanded_slower_actually_slows(self, tmp_path):
        # Commanded in m/s rather than through a speed bin. At a congesting demand
        # an AV is already slow, so `slow` -- which decodes to the allowed speed
        # minus 10 -- would often be ABOVE its current speed and the test would
        # pass or fail on the traffic state rather than on the actuation.
        env = _env(tmp_path, demand=load_named_config("demand", "low"))
        try:
            env.reset(seed=7)
            for _ in range(150):
                env.step({})
                if env.agent_ids and env.av_speed(env.agent_ids[0]) > 8.0:
                    break
            if not env.agent_ids:
                pytest.skip("no AV appeared at low demand within 150 steps")
            target = env.agent_ids[0]
            before = env.av_speed(target)
            for _ in range(12):
                if target not in env.agent_ids:
                    pytest.skip("the AV left the network before the command landed")
                env.command_speed(target, 2.0)
                env.step({})
            after = env.av_speed(target) if target in env.agent_ids else 0.0
            assert after < before - 1.0, (
                f"commanded 2 m/s and speed went {before:.2f} -> {after:.2f} m/s"
            )
        finally:
            env.close()

    def test_an_uncommanded_av_is_left_to_sumo(self, tmp_path):
        # The control for the test above. If the env clamped speed regardless of
        # the command, the previous test would pass for the wrong reason.
        env = _env(tmp_path, demand=load_named_config("demand", "low"))
        try:
            env.reset(seed=7)
            for _ in range(120):
                env.step({})
                if env.agent_ids:
                    break
            target = env.agent_ids[0]
            speeds = []
            for _ in range(15):
                env.step({})
                if target in env.agent_ids:
                    speeds.append(env.av_speed(target))
            assert speeds, "the AV left the network"
            assert max(speeds) > 1.0, (
                "an uncommanded AV never exceeded 1 m/s, so something is holding it"
            )
        finally:
            env.close()


class TestMetrics:

    def test_the_reward_metrics_are_populated_from_sumo(self, tmp_path):
        env = _env(tmp_path, duration_steps=300)
        try:
            env.reset(seed=7)
            info = {}
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step({})
            metrics = info["metrics"]
            for field in ("mean_speed", "throughput_recent", "jam_fraction",
                          "collision_count", "queue_length_total"):
                assert field in metrics, f"{field} missing from metrics"
            assert metrics["mean_speed"] > 0.0
            assert metrics["collision_count"] == 0
        finally:
            env.close()

    def test_throughput_counts_arrivals(self, tmp_path):
        env = _env(tmp_path, duration_steps=300)
        try:
            env.reset(seed=7)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step({})
            # Over 300 s at a congesting demand vehicles must be completing, or the
            # throughput term in the reward is measuring nothing.
            assert env.arrived_total > 0, "no vehicle completed its route in 300 steps"
            assert info["metrics"]["throughput_recent"] >= 0
        finally:
            env.close()


class TestTheTwoVehicleTypesDifferOnlyInIdentity:
    """Penetration must not change the fleet, only who is controllable.

    An earlier version gave the human type a speedFactor spread and the AV type
    none, so it defaulted to 1.0. Raising penetration then made the fleet more
    homogeneous, which reduces speed variance and congestion: with no controller
    acting at all, mean speed went 6.15 m/s at 10% penetration to 11.29 at 40% and
    arrivals 169 to 224. Reported as a control effect, that would have been wrong.
    """

    def test_both_types_carry_the_same_speed_factor(self, tmp_path):
        env = _env(tmp_path)
        try:
            env.reset(seed=7)
            routes = (env.work_dir / "demand.rou.xml").read_text()
            # The DISTRIBUTION members, which is what the flows draw from. An
            # earlier version of this test excluded every line containing
            # `probability` -- that is, exactly those two -- and checked the two
            # top-level vTypes that no flow references. The test written to stop
            # the speedFactor confound returning could not fail.
            types = [line for line in routes.splitlines()
                     if "<vType " in line and "probability" in line]
            assert len(types) == 2, f"expected two distribution members, got {types}"
            factors = [line.split('speedFactor="')[1].split('"')[0] for line in types]
            assert factors[0] == factors[1], (
                f"the two vTypes have different speed factors: {factors}"
            )
            maxima = [line.split('maxSpeed="')[1].split('"')[0] for line in types]
            assert maxima[0] == maxima[1], maxima
        finally:
            env.close()

    def test_raising_penetration_alone_does_not_change_flow(self, tmp_path):
        # The measurement the confound would have corrupted: with no actions
        # applied, penetration must not move mean speed much, because an
        # uncontrolled AV is just a vehicle.
        results = {}
        for pen in (0.10, 0.40):
            demand = dict(load_named_config("demand", "sumo_saturating"))
            demand["av_penetration"] = pen
            env = _env(tmp_path / f"p{int(pen*100)}", demand=demand, duration_steps=600)
            try:
                env.reset(seed=7)
                terminated = truncated = False
                info = {}
                while not (terminated or truncated):
                    _, _, terminated, truncated, info = env.step({})
                results[pen] = info["metrics"]["mean_speed"]
            finally:
                env.close()
        low, high = results[0.10], results[0.40]
        assert abs(high - low) < 0.5 * max(low, 1.0), (
            f"penetration alone moved mean speed {low:.2f} -> {high:.2f} m/s, so the "
            "two vehicle types still differ in something other than controllability"
        )


class TestTheDemandLastsTheWholeEpisode:
    """The flow must cover the warm-up as well as the episode.

    It previously ended at `duration_steps * dt`, while the simulation runs
    `warmup + duration` steps, so demand stopped before the episode did. With a
    300-step warm-up and a 600-step episode that left a third of the run draining,
    and the final step reported mean_speed 0.000 on an empty network.
    """

    def test_vehicles_are_still_present_at_the_final_step(self, tmp_path):
        env = _env(tmp_path, duration_steps=600, warmup_steps=300)
        try:
            env.reset(seed=7)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step({})
            assert info["metrics"]["active_vehicle_count"] > 0, (
                "the network was empty at the final step, so demand ran out early"
            )
            assert info["metrics"]["mean_speed"] > 0.0
        finally:
            env.close()

    def test_the_schedule_covers_warmup_plus_duration(self, tmp_path):
        env = _env(tmp_path, duration_steps=600, warmup_steps=300)
        try:
            env.reset(seed=7)
            routes = (env.work_dir / "demand.rou.xml").read_text()
            departs = [float(line.split('depart="')[1].split('"')[0])
                       for line in routes.splitlines() if "<vehicle " in line]
            assert departs, "no vehicles were scheduled"
            # 900 s of simulation: a 300-step warm-up and a 600-step episode at
            # dt 1.0. The last departure must be near the end of it, or the episode
            # finishes on a draining network.
            assert max(departs) > 850.0, (
                f"the last vehicle departs at {max(departs):.1f} s of a 900 s run"
            )
            assert min(departs) < 50.0
        finally:
            env.close()


class TestEveryActionHeadActuates:
    """Three of the four heads of the action contract reached the simulator
    nowhere: the headway bin changed an observation field and nothing else, and
    lane preference and merge mode were discarded in `_apply_actions`. A policy
    trained on the `full` profile was choosing three of its four outputs against no
    consequence, which is not a control experiment.
    """

    def _run(self, tmp_path, action, steps=400):
        import src.sumo.env as env_module

        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": steps, "dt": 0.1, "warmup_steps": 3000,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            taus, left_lane, total = [], 0, 0
            for _ in range(steps):
                env.step({agent_id: dict(action) for agent_id in list(env.agent_ids)})
                for agent_id in list(env.agent_ids):
                    taus.append(float(env_module._sumo.vehicle.getTau(agent_id)))
                    left_lane += int(env_module._sumo.vehicle.getLaneIndex(agent_id)) == 1
                    total += 1
            return {
                "tau": sum(taus) / max(len(taus), 1),
                "left_fraction": left_lane / max(total, 1),
                "requests": env.lane_change_requests,
            }
        finally:
            env.close()

    def _action(self, **overrides):
        action = {"desired_speed_bin": "fast", "desired_headway_bin": "normal",
                  "lane_preference": "keep", "merge_mode": "normal"}
        action.update(overrides)
        return action

    def test_the_headway_bin_becomes_the_car_following_tau(self, tmp_path):
        normal = self._run(tmp_path, self._action(desired_headway_bin="normal"))
        largest = self._run(tmp_path, self._action(desired_headway_bin="largest"))
        assert normal["tau"] == pytest.approx(1.6, abs=0.05)
        assert largest["tau"] == pytest.approx(3.0, abs=0.05)

    def test_create_gap_adds_the_declared_merge_bonus(self, tmp_path):
        without = self._run(tmp_path, self._action(desired_headway_bin="largest"))
        with_gap = self._run(tmp_path, self._action(desired_headway_bin="largest",
                                                    merge_mode="create_gap"))
        # 0.8 s is `SafetyConstraints.merge_gap_headway_bonus_s`, which exists for
        # exactly this, rather than a number invented here.
        assert with_gap["tau"] - without["tau"] == pytest.approx(0.8, abs=0.05)

    def test_lane_preference_moves_the_fleet_between_lanes(self, tmp_path):
        # 600 steps, which is 60 s at dt 0.1. AVs enter in the right lane and
        # migrate, so the occupancy shift accumulates: it is +0.095 after 40 s and
        # +0.19 after 60 s.
        keep = self._run(tmp_path, self._action(lane_preference="keep"), steps=600)
        left = self._run(tmp_path, self._action(lane_preference="prefer_left_if_safe"),
                         steps=600)
        right = self._run(tmp_path, self._action(lane_preference="prefer_right_if_safe"),
                          steps=600)
        assert keep["requests"] == 0, "keeping lane should request no change"
        assert left["requests"] > 100 and right["requests"] > 100
        assert left["left_fraction"] > keep["left_fraction"] + 0.1, (
            f"preferring left put {left['left_fraction']:.3f} of AV-steps in the left "
            f"lane against {keep['left_fraction']:.3f} when keeping"
        )
        assert right["left_fraction"] < keep["left_fraction"] - 0.1

    def test_hold_lane_suppresses_changes(self, tmp_path):
        import src.sumo.env as env_module

        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 60, "dt": 0.1, "warmup_steps": 3000,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            for _ in range(60):
                env.step({agent_id: self._action(merge_mode="hold_lane",
                                                 lane_preference="prefer_left_if_safe")
                          for agent_id in list(env.agent_ids)})
            assert env.lane_change_requests == 0, (
                f"hold_lane still requested {env.lane_change_requests} lane changes"
            )
            for agent_id in list(env.agent_ids):
                # 1536, not 0. Bits 8-9 of a lane-change mode are the
                # collision-avoidance bits, so 0 disables safety as well as
                # motivation; see TestActionsCannotCauseACollision.
                assert (env_module._sumo.vehicle.getLaneChangeMode(agent_id)
                        == SumoTopologyEnv.HOLD_LANE_MODE)
        finally:
            env.close()

    def test_an_unsafe_change_is_still_sumos_to_refuse(self, tmp_path):
        # The requests are asks. SUMO's lane-change model may refuse any of them,
        # and the mode is the library default rather than one that forces a change,
        # because a controller that could force an unsafe change would defeat the
        # reason this simulator was chosen.
        assert SumoTopologyEnv.LANE_CHANGE_MODE == 1621
        left = self._run(tmp_path, self._action(lane_preference="prefer_left_if_safe"))
        assert left["left_fraction"] < 1.0, (
            "every AV-step ended in the left lane, so no request was ever refused"
        )


class TestActionsCannotCauseACollision:
    """SUMO's car-following being unable to produce a collision is the premise of
    this migration, and the actuation of the action heads has to preserve it.

    It did not. `hold_lane` set the lane-change mode to 0, meaning "no lane
    changes" -- but bits 8-9 of a SUMO lane-change mode are the collision-avoidance
    bits, so 0 also turns safety off. `changeLane` holds its choice for a duration,
    so a vehicle told to prefer a lane on one step and to hold its lane on the next
    carried on into the change with no safety checks.

    Measured over 3000 steps with actions varying per agent per step: lane
    preference and merge mode together produced 646 collisions, and all four heads
    703, while each head alone produced none. That is why this test varies the
    heads together and does not trust a single-head check.
    """

    CHOICES = {
        "desired_speed_bin": ("slow", "nominal", "fast"),
        "desired_headway_bin": ("normal", "larger", "largest"),
        "lane_preference": ("keep", "prefer_left_if_safe", "prefer_right_if_safe"),
        "merge_mode": ("normal", "create_gap", "hold_lane"),
    }

    def _run(self, tmp_path, heads, steps=1500):
        import random

        rng = random.Random(1)
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_burst"),
            "duration_steps": steps, "dt": 0.1, "warmup_steps": 3000,
            "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            acted = 0
            for _ in range(steps):
                actions = {}
                for agent_id in list(env.agent_ids):
                    action = {"desired_speed_bin": "fast", "desired_headway_bin": "normal",
                              "lane_preference": "keep", "merge_mode": "normal"}
                    for head in heads:
                        action[head] = rng.choice(self.CHOICES[head])
                    actions[agent_id] = action
                    acted += 1
                env.step(actions)
            return env.collision_count, acted, env.collision_checks
        finally:
            env.close()

    def test_no_combination_of_actions_collides(self, tmp_path):
        collisions, acted, checks = self._run(tmp_path, tuple(self.CHOICES))
        # The premise, and the control that the premise was actually tested: an
        # action was issued on most steps and the counter was read on every one.
        assert acted > 1000, f"only {acted} agent-actions were issued"
        # The counter is read on every step of the warm-up as well as the episode,
        # which is what makes a zero here a measured zero rather than an unread one.
        assert checks == 3000 + 1500
        assert collisions == 0, f"{collisions} collisions from the action heads alone"

    def test_the_pair_that_broke_it_is_covered(self, tmp_path):
        collisions, _, _ = self._run(
            tmp_path, ("lane_preference", "merge_mode"))
        assert collisions == 0, (
            f"{collisions} collisions from lane preference and merge mode together; "
            "neither head alone produces any, which is why they are varied together"
        )
