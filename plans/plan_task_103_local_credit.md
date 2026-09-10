# Task 103 — One configuration, everything enabled, one seed

## Short version

**The problem.** The policy has not learned in either SUMO training run. Entropy
finished at 99.1% of its maximum in the first (task 96) and the second was flat
across all 50 updates (task 101). The diagnosed cause is the credit-assignment
signal: with a team reward shared among about twelve agents, one agent's action
moves the reward by roughly 1% of the reward's own step-to-step noise.

**What this task runs.** ONE configuration with every change enabled at once, on
ONE seed, judged on whether the policy learns rather than on how large its effect
is. Five changes:

| change | where | why |
|---|---|---|
| per-agent local reward over the agent's own segment and the segments downstream of it, blended with the team reward | `src/sumo/env.py`, `src/rl/rewards.py`, `src/rl/trainers.py` | the diagnosed cause. The neighbourhood includes downstream because metering means slowing the own segment to relieve the next one, and an own-segment-only term would punish exactly that |
| one decision per simulated second, action held for the ten simulation steps in between | `src/rl/trainers.py`, `configs/training/mappo_sumo.yaml` | at dt 0.1 a single action changes the next observation almost not at all, so the per-action credit is diluted tenfold by the step size alone |
| reward re-weighted toward delay and smoothness | `configs/training/mappo_sumo.yaml` | `speed_std` is 3.6% of the current objective, and delay and stop count -- the axes the predecessor's gains were largest on -- were not measured at all |
| AV penetration 0.25, from 0.20 | `configs/demand/sumo_oversaturated.yaml` | matches the predecessor's experiments, and gives more agent-steps per episode |
| `deployed_fidelity: false` | `configs/training/mappo_sumo.yaml` | 13 of 39 observation slots are absent or held at a constant under it. Sim-to-real transfer is deferred work; this run is about whether a policy can be learned at all |

**Prerequisite.** The environment records neither travel time nor any stop or
smoothness measure, so four metrics are added first: `mean_travel_time_recent`,
`mean_delay_recent`, `stopped_fraction` and `mean_abs_jerk`.

**The pass/fail gate, fixed before the run.** One seed cannot detect a 5% effect --
the paired standard error across seeds is 5 to 10% -- so the gate is a within-run
signal and nothing else:

1. **Entropy falls** below 90% of its maximum, which is 1.978 of ln(9) = 2.197 for
   the 9-action `speed_headway` profile. The 4.38 figure from the earlier runs was
   against ln(81) and is not comparable.
2. **The score trends up** by more than the within-run variation across updates.
3. **The action distribution leaves uniform**: the modal action's share exceeds
   1/9 by a clear margin.

Passing all three earns a five-seed run reported on fresh evaluation seeds.
Failing means the next iteration is on the reward, not on more seeds.

## Scope boundary

- **No centralized arm, no oracle sweep, no heuristic sweep.** Ruled out by the
  user as time-consuming.
- **No ablations.** The five changes are deliberately confounded. Ablations come
  after something works.
- **One seed, then five.** Not five in one go.

## Selection discipline

The configuration is chosen on this run's training curve, on seed 7. If it earns a
five-seed run, that run reports on evaluation seeds disjoint from 7 and from the
training seeds, as task 99 already required.

## Open decisions

- **Whether the demand has headroom for a controller at all.** The metering oracle
  on the corrected two-lane road at `sumo_oversaturated` measured -0.8 and +0.8
  against no control, which is indistinguishable from doing nothing (task 102). The
  only positive number in that progression is +7.6 +/- 15.2 at congestion onset,
  which is one standard error. So a null here does not separate "the policy cannot
  learn" from "there is nothing at this operating point to learn". Recorded before
  the run rather than after it.
- **Whether the reward weights are right.** Fixed from a measurement of each
  metric's magnitude and per-step spread at this operating point with no control,
  taken before the run, so no term is a hundredth or a hundred times another by
  accident.

## Steps

1. Four metrics in `src/sumo/env.py`, each with a test that fails on the
   uninstrumented version.
2. Neighbourhood metrics per agent in `info`, so the reward can price them without
   the environment holding weights.
3. Measure every metric's mean and per-step standard deviation at
   `sumo_oversaturated`, two lanes, no control. Set the weights from it.
4. `build_local_reward` in `src/rl/rewards.py`; the blend in `collect_rollout`
   behind `local_reward_weight`, default 0.0 so no existing config changes.
5. Action repeat in `collect_rollout` behind `decision_interval_s`, default equal
   to `dt` so no existing config changes.
6. Config changes. Full suite green. Commit.
7. Train one seed. Read the three gate quantities off `training_metrics.csv`.
8. Record the result, pass or fail, in `plans/task_list.md`.

## Sign-off

- [ ] Every new metric has a test that fails without the instrumentation.
- [ ] `local_reward_weight` 0.0 and `decision_interval_s` unset reproduce the
      previous trainer behaviour exactly.
- [ ] The weights are set from a measurement, and the measurement is recorded.
- [ ] The gate is read off the training curve without being renegotiated.
