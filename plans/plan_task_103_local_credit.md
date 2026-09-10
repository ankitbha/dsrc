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

  **SETTLED, in part, by the measurement below.** At 2400 veh/h the road serves
  about 2000 veh/h for five minutes and about 1200 after it collapses, so the
  difference between collapsing and not collapsing is large. That is headroom in
  the environment; whether a decentralised policy can reach it is what the run
  asks.
- **Which demand.** SETTLED: `sumo_capacity_drop` at 2400 veh/h. Measured with no
  control at dt 0.1 over a 900 s episode after a 300 s warm-up, in 60 s buckets
  (speed in m/s, throughput in completions per 60 s):

  | minute | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | ... | 14 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 2400 speed | 6.6 | 7.2 | 7.0 | 6.8 | 5.9 | 3.6 | 2.8 | 2.3 | 2.2 | | 1.9 |
  | 2400 throughput | 16.3 | 19.8 | 28.0 | 32.1 | 32.7 | 30.5 | 11.6 | 8.3 | 9.0 | | 27.7 |
  | 3000 speed | 3.0 | 3.4 | 2.3 | 2.8 | 2.3 | 2.0 | 2.2 | 2.3 | 2.5 | | 1.3 |
  | 3000 throughput | 13.6 | 12.7 | 14.4 | 13.5 | 19.5 | 11.6 | 9.3 | 12.7 | 18.3 | | 13.5 |

  At 3000 the warm-up alone gridlocks the road. The episode is shortened to 600 s
  so it sits around the collapse rather than trailing five minutes of gridlock.
- **Whether the reward weights are right.** SETTLED from the same measurement,
  before the run. The contribution table is in
  `configs/training/mappo_sumo.yaml`; no term is more than five times another
  except throughput, which leads deliberately, and the total is +0.91 per step.

## A known bias, measured and left in place

An agent inside a junction has no segment and so no neighbourhood, so a sub-step in
which it is there contributes nothing to its local reward although the agent was
answerable for it. Measured over 11,489 agent-decisions at this operating point:
sub-step coverage 99.38%, and 1.02% of agent-decisions are covered for fewer than
all ten sub-steps. The resulting understatement is about 0.6% of the local reward
on the affected agents, which is far below its own variation, so the run was not
restarted for it.

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

- [x] Every new metric has a test that fails without the instrumentation. Three
      mutants run and killed: the neighbourhood reduced to the agent's own segment,
      the warm-up flow tail removed, and `_departure_time` cleared at the end of
      warm-up. Each was killed by the test written for it and by no other.
- [x] `local_reward_weight` 0.0 and `decision_interval_s` unset reproduce the
      previous trainer behaviour exactly. `tests/test_local_credit.py` asserts that
      two agents in different neighbourhoods receive identical rewards until the
      local weight is set, which is the control that makes the separation test
      meaningful.
- [x] The weights are set from a measurement, and the measurement is recorded. One
      uncontrolled episode at this operating point, taken before the run; the
      contribution table is in `configs/training/mappo_sumo.yaml`.
- [x] The gate is read off the training curve without being renegotiated.
      `scripts/read_training_gate.py` derives the entropy threshold from the action
      profile rather than taking it as an argument. Criteria 1 and 2 fail. Criterion
      3 as written had no numeric threshold, which is a defect in this plan and is
      recorded as task 109 rather than resolved after the fact.

## Outcome

**The gate fails, and the five changes were not what was wrong.** Measured while the
run was in flight: the advantage carries no action-attributable signal, at its own
shuffled noise floor, on every configuration tried -- four reward decompositions,
three hold lengths, five discount horizons, two critic input sets, four speed-bin
scalings, three operating points and three penetrations up to 100%. The instrument
reads 14 to 80 times the floor when an action-correlated advantage is supplied, so it
could have produced a positive reading in each. Tasks 105 to 109 carry the detail.

None of the five changes in this plan could have worked, and the plan could not have
known that: the instrument that shows it did not exist when the plan was written. The
lesson is task 105's -- bracket a statistic before comparing arms on it.
