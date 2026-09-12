# Detours

Work the project did that the paper does not rest on. None of it is deleted: it is the
record of how the current shape was arrived at, and several of its findings are the
reason for decisions that now look arbitrary.

**The single thing to know before reading any of it.** This material describes an earlier
formulation: a MAPPO policy over a 39-field local-sensing observation, trained on a ladder
of synthetic topologies, with `inverted_tree` as the road. **That is not the paper.** The
paper's controller is SRC's own, its observation is five fields a traffic API returns, and
its network is Mainz. An agent that reads the material below without this warning will
write the wrong paper.

The code it describes was deleted on 2026-09-12 and is in the git history.

## What is in here

| | from | why it is a detour |
|---|---|---|
| The MAPPO replication on `inverted_tree` | `task_list.md` section C | superseded by task 141; the flow-level half is the SRC port on Mainz |
| Findings 77 to 140 | `task_list.md` section K | that leg's measurements, including its nulls and the instrument defects behind them |
| The ordering correction and items 72 to 76 | `task_list.md` | sequenced that leg's training |
| The paused state of 2026-09-09 | `task_list.md` | where that leg stopped |
| `plan_simulations.md` | its own file | the original plan, whose premise is network-level control from local sensing |
| `project_plan.md` | its own file | the integrated plan, whose claim is "using only local sensing" and whose experiment matrix is the highway-env ladder |
| `result_simulation_leg_null.md` | its own file | the mappo leg's own closing record |
| `plan_task_67_merge_blind_collisions.md` | its own file | the 40% episode-truncation rate in highway-env, whose cause was `idmvehicle` following only within its own lane |
| `plan_task_84_sumo_simulator.md` | its own file | the migration of the topology ladder from highway-env to sumo |
| `plan_task_103_local_credit.md` | its own file | the credit-assignment investigation on the local-sensing formulation |
| `result_fleet_mix_sweep.md` | its own file | a penetration sweep on that formulation |
| `replication_state.json` | its own file | the mappo replication's run state |
| `result_task93_burst_evaluation.json` | its own file | the pre-registered burst evaluation's raw result |
| `result_task99_corrected_road.json` | its own file | the corrected-road fundamental diagram's raw result, on `inverted_tree` |

## What is worth reading even so

Four things in here are cited by the current work and are not superseded:

* **Task 98**, the unintended permanent yield that set `inverted_tree`'s capacity. It is
  the mirror of the Mainz port's own problem, where `netconvert` dropped the conflict
  areas instead of inventing one.
* **Tasks 105 to 108 and 124**, on what a gradient-norm instrument can and cannot resolve,
  and the error bars that were missing from it.
* **Task 92**, the retraction of a throughput gain that was a simulation step-size
  artefact.
* **Task 100**, that the calibrated driving model gives up the collision-free guarantee.

---

## C. Simulation: replicate the MAPPO throughput result on `inverted_tree`

> **SUPERSEDED 2026-09-11 by task 141. This section is history.** The flow-level half is
> now the SRC controller ported to SUMO on Mainz, in `plans/mainz_src_port.md`. Two
> reasons, both measured. `inverted_tree` has one link per super-segment against the
> paper's 2 to 3 km, so there is no aggregation in it for a super-segment observation to
> summarise. And under the fleet this section used there is no capacity drop to recover:
> served flow is flat within seed error across a 9x density range. The section is kept
> because its findings about the environment stand and are cited elsewhere.

**Replication, not discovery.** The task is to reproduce a published result in this
simulator with this project's settings, on one topology. No topology ladder, no
penetration sweep, no large-scale verification of the setting.

**`inverted_tree` is the chosen topology and the only one to be run.** Task 8
measured it congesting in 12 of 12 cells, which is the property a replication needs.
The other five are out of scope, `merge` and `straight_multilane` especially: they
congested in 0 of 12.

### Correction 2026-09-08: the task 8 result does not apply to MAPPO

This section read "BLOCKED IN SUBSTANCE by the task 8 result" on the strength of
"+0.12 to +0.14 m/s". **That number describes hand-written baselines.**
`src/analysis/simulator_health.py` calls `make_baseline(controller)` against
`REFERENCE_CONTROLLER = "no_av"`, and `src/baselines/registry.py` holds `no_av`,
`random_av`, `selfish_av`, `density_lookup`, `dynamic_speed_limit`,
`av_mediated_speed_harmonization`, `backpressure` and `cooperative_smoothing`. **None
of them is learned.**

**MAPPO is implemented and has never been trained.** `src/rl/trainers.py` defines
`MAPPOTrainer` with `critic_scope = "global"` and a critic over
`physical_global_state_dim() + local_obs_dim()` — a centralized critic with
decentralized actors, which is the algorithm the replication needs — distinct from
the `IPPOTrainer` and `SharedPPOTrainer` beside it. `scripts/train_policy.py` is the
entry point and supports `--resume-from` and `--resume-latest`. `outputs/` holds no
checkpoint.

So task 8 is evidence about controllers this project does not intend to ship, and it
is not evidence about the method the paper replicates. **What task 8 does still bind
are the conditions a training or evaluation run must meet**, and tasks 67 to 69 carry
those forward: the truncation rate, the concurrent AV count, and the run-to-run
term.

5. ~~Per-field variance audit to identify inert inputs before any ablation.~~
   **DONE** — `src/analysis/observation_audit.py` + `scripts/audit_observation_fields.py`,
   41 tests. Ran 162 conditions (6 topologies × 3 controllers × AV penetration
   {0.05, 0.10, 0.20} × 3 seeds, 120 steps), 10,569 samples.
   Plan: `scratchpad/plan_task_05_variance_audit.md`. Artifacts:
   `outputs/validation/observation_audit/`.

   **Findings that bind sections D–F:**

   - **Only 2 of 39 encoded fields are uninformative everywhere.** `is_active`
     is structurally constant at 1.0 — the observation map contains only active
     AVs, so the flag can never be false. `distance_to_next_merge` is hardcoded
     to `0.0` in `src/sensing/local.py`. Neither should be sensed, and neither
     may be ablated as if the result meant anything.
   - **8 fields are penetration-gated:** dead at 5% AV penetration, informative
     at 20% — `nearby_av_count`, `active_av_count_local`, `nearby_av_density`,
     `nearby_av_lane_distribution.{1,2}`, `downstream_congestion_estimate` and
     its `cooperation.*` twin, and `target_lane_rear_required_decel`. No field
     moves the other way. The local-aggregate cooperation block therefore only
     earns its sensing cost above roughly 10% penetration, which the sampling
     controller (task 29) should treat as a rate-allocation input rather than
     sensing unconditionally.
   - **Inertness is strongly topology-gated:** ring 18/39 constant,
     `straight_single_lane` 14/39, `merge` 8/39, `straight_multilane` 7/39,
     `inverted_tree` 4/39, `inverted_tree_bottleneck` 3/39. Any ablation must be
     read per topology, never pooled.
   - **Coverage caveat:** 60 of 162 conditions produced zero samples — 54 are
     `no_av` (correct by design) and 6 are non-ring topologies at 5% penetration
     where no AV spawns at all. The effective matrix is 102 conditions.
   - **`controlled_vehicles` is inert outside ring.** `HighwayTopologyEnv`
     clears `agent_ids` and defers to the demand spawner whenever continuous
     demand is active, i.e. on every topology except ring. AV population must be
     set through `demand.av_penetration`. This bit the first run and is recorded
     as Amendment 2 in the plan.
6. Sufficiency harness: evaluate a fixed policy under configurable observation
   degradation (field ablation, added lag, added noise, forced fallbacks).
   **OFF THE CRITICAL PATH 2026-09-08.** It exists to derive a sensing requirement
   specification, which the paper described above does not claim. Build it only if
   the replication lands with time to spare.
7. Baseline sweep with the current sensing defaults, to establish the reference
   the degraded conditions are measured against. **OFF THE CRITICAL PATH
   2026-09-08** — the only reference the paper needs is `no_av` against trained
   MAPPO on one topology, which task 69 measures directly.
8. ~~Exercise the topology ladder beyond ring so the study is not
   single-topology.~~ **DONE — superseded by a simulator health check**, which
   absorbed and extended it. `src/analysis/simulator_health.py` +
   `scripts/check_simulator_health.py`, 213 tests. Ran 72 cells / 648 runs
   (6 topologies × 4 demands × 3 penetrations × 3 controllers × 3 seeds, 120
   steps) in 18m37s. Plan: `scratchpad/plan_task_08_simulator_health.md`.
   Artifacts: `outputs/validation/simulator_health/`.

   **Result: 0 of 72 cells pass all four criteria. There is no usable operating
   point, and the simulator cannot currently support a flow-level claim.**

   | criterion | fails in |
   |---|---|
   | `baselines_separate` | 68/72 |
   | `episodes_complete` | 44/72 |
   | `throughput_holds` | 31/72 |
   | `congestion_reachable` | 27/72 |

   - **The controllers have no measurable effect, and crashes are not the
     reason.** Cells that complete separate in 7% of cases, cells that crash in
     5% — the hypothesis that truncation was hiding the effect is refuted. In the
     four cells that both congest *and* complete, measured on 2–3 congested
     shared seeds, the best controller moves mean speed by **+0.12 to +0.14 m/s**
     against a 1.0 m/s threshold. That is the cleanest measurement the grid
     offers and it is near-zero.
   - **Congestion is reachable but topology-structured.** `inverted_tree`,
     `inverted_tree_bottleneck` and `ring` congest in 12/12 cells,
     `straight_single_lane` in 9/12, and **`merge` and `straight_multilane` in
     0/12** — including `merge`/high, which congests at a single penetration but
     fails the per-seed rule.
   - **Only 24 of 216 (cell, controller) pairs were ever measured with a
     congested shared seed.** Most separation failures are therefore not
     evidence that a controller cannot help; the controller was evaluated where
     there was nothing to control. Without `congested_shared_seeds` the report
     would have read as 68 controller failures.
   - **40% of runs crash** (260 of 648 never reach their configured duration).
   - **The penetration axis is substantially noise.** The mechanism is correct —
     pooled realised spawn fraction is 0.200 against a nominal 0.20 — but only
     60–78 vehicles spawn per run, so the standard error is ±0.05 and single-seed
     realised penetration swings between 0.08 and 0.43 at a nominal 0.20. Per-cell
     verdicts rest on single seeds, so nominal 0.05 and 0.10 cells can realise the
     same fraction. Concurrent AV count, which is what actually acts as an
     actuator, peaks at 5–6 out of ~40 active vehicles.
   - **243 of the 648 runs are duplicates.** `burst` is bit-identical to `medium`
     on all six topologies (162 runs), and ring disables demand so its four
     demand levels collapse to one (81 runs). Config defects, out of scope here.

   **Next diagnostic when this is picked up** (not run): raise demand *and*
   episode length *and* penetration together, so concurrent AV count reaches
   double digits. Raising nominal penetration alone would leave the realised
   concurrent count in single figures for the same small-sample reason. That run
   distinguishes "too few actuators" from "these controllers do not control this
   simulator" — two diagnoses leading to completely different work.
9. Sensing model calibrated from drive measurements. **SCOPED 2026-09-08 to the
   five parameters that exist.** `src/sensing/local.py` has exactly five, and each
   now has a named source:

   | parameter | default | source |
   |---|---|---|
   | `latency_s` | 0.0 | Measured on the drives: p50 96.7 ms, p95 172.9 ms, p99 276.4 ms over 22,929 ticks |
   | `queue_speed_mps` | 5.0 | Ego GPS speed distribution, 22,734 valid fixes, p50 11.14 m/s |
   | `range_m` | 150.0 | Offline `replay_demo.py` over stored video — **needs task 63 first** |
   | `position_noise_std` | 0.0 | Published characterisations of monocular detectors |
   | `speed_noise_std` | 0.0 | Published characterisations of monocular detectors |

   **The two noise terms come from the literature by decision, not by omission.**
   Calibrating them requires an independent measurement of other vehicles' true
   position and speed, and the vehicle carried no radar, no lidar and no second
   instrumented car. Reprocessing frames yields the estimator's frame-to-frame
   self-consistency, which is a different quantity from error against truth. The
   paper must attribute these two to published characterisations and not to this
   vehicle.

   **`range_m` needs no further driving.** `deployment/jetson/replay_demo.py`
   re-runs the whole perception, observation and policy pipeline over a run's raw
   video and logged GPS. It applies to the six runs holding `video_index.jsonl`;
   `run_20260908_142253` recorded no video and `run_20260908_144642` has video with
   no index. Any distance it produces is only as good as the mount geometry, so task
   63 must land first and the horizon must be re-established for the real mount.
10. Sufficiency study proper: derive the sensing requirement specification.
    **OFF THE CRITICAL PATH 2026-09-08**, with task 6.
11. Sampling policy evaluated in simulation for the flow-level benefit one
    vehicle cannot demonstrate. **OFF THE CRITICAL PATH 2026-09-08.** The
    flow-level benefit the paper reports is the replication in task 69; crediting
    the *sampling* policy at flow level is a second claim the paper does not make.
67. **Fix the truncation rate before any training run.** Task 8 measured 260 of 648
    runs never reaching their configured duration, a 40% rate. Training against that
    corrupts the return signal: a bad policy and a broken episode become
    indistinguishable. This precedes task 68 rather than running beside it.

    Two config defects task 8 found belong here, because both shrink the usable
    grid: `burst` is bit-identical to `medium` on all six topologies (162 of 648
    runs), and ring disables demand so its four demand levels collapse to one (81
    runs). Only the `inverted_tree` half of either matters now.
68. **Train MAPPO on `inverted_tree`.** `scripts/train_policy.py --training mappo
    --topology inverted_tree`. This is the single missing artifact — no part of the
    simulation claim can be evaluated until a checkpoint exists.

    **Blocked on tasks 67 and 9.** Not by convention: `src/envs/topology_env.py:109`
    constructs `LocalObservationBuilder(SensingConfig.from_config(self.config))` and
    line 281 builds every agent observation through it, so the `sensing:` block in
    the training YAML is the actor's input distribution. Training under the current
    defaults and calibrating afterwards produces a policy trained on an observation
    model the paper does not claim. `configs/training/shared_ppo_deploysense.yaml`
    already carries such a block and is the file task 9 fills.

    **Record the training config beside the checkpoint.** Task 69 has to evaluate
    under the same block, and a checkpoint whose sensing block is unknown cannot be
    evaluated at all.

    **Run a short pilot first**, to shake out the loop and to measure how long one
    update takes. That measurement sets the watcher's stall threshold, which is a
    guess until something measures it. A pilot's result is never reported as a
    result.

    Three traps, each already established elsewhere in this list:

    - **`--controlled-vehicles` is inert outside ring.** `HighwayTopologyEnv` clears
      `agent_ids` and defers to the demand spawner whenever continuous demand is
      active, so AV population must be set through `demand.av_penetration`. This bit
      the first run of task 5 and is recorded there as Amendment 2.
    - **The defaults are too small.** `--duration-steps` defaults to 120, and task 8
      measured concurrent AV count peaking at 5–6 out of ~40 active vehicles. Raise
      demand, episode length and penetration **together** until concurrent AV count
      reaches double digits; raising penetration alone leaves the realised count in
      single figures for the small-sample reason task 8 records.
    - **Resume, do not restart.** This is the project's first long run. Use
      `--resume-from` / `--resume-latest`, and prove resumption works by killing a
      run before depending on it.
69. **Evaluate trained MAPPO against `no_av` on `inverted_tree` for throughput.**
    The replication claim, and the only flow-level number the paper makes.

    **Gate: the evaluation must use the same `sensing:` block task 68 trained
    under.** If the two differ, the measured difference is a train/test mismatch
    rather than the controller, and nothing in the output would say so. Re-use the
    recorded config verbatim rather than rebuilding it.

    Seeds are the axis to spend on, now that topology and demand are fixed. Task 8
    measured single-seed realised penetration swinging between 0.08 and 0.43 at a
    nominal 0.20, and only 24 of 216 (cell, controller) pairs were ever measured
    with a congested shared seed. A throughput difference must clear that
    run-to-run term before it is reported.

140. **DEFECT, found by reading the algorithm rather than measuring its output:
     credit leaks across episode boundaries.** 2026-09-10. Ankit's assessment was that
     the session had spiralled into measurement while the problem lay in the algorithm
     and the implementation. Auditing the credit-assignment path found this in
     minutes; twenty measurements had not.

     **The chain, all of it from code:**

     1. `SumoTopologyEnv.step` sets `terminated = False` unconditionally (env.py:430),
        deliberately and with a comment saying so. Every episode ends by
        `truncated = step_count >= duration`.
     2. `collect_rollout` records `done=bool(terminated or agent_id not in
        next_observations)` (trainers.py:365). **`truncated` is never consulted.**
     3. So at an episode boundary every vehicle still in the network records
        `done=False`.
     4. `collect_rollout` resets the env and keeps filling the SAME buffer.
     5. Vehicle ids are `f"v{index}_{len(departures)}"`, regenerated identically on
        every reset: the route file is rewritten deterministically and the seed
        changes only the AV/human type draw, not the ids. **Ids repeat across
        episodes.**
     6. `compute_returns_and_advantages` groups by `agent_id`, so episode 1's `v0_5`
        and episode 2's `v0_5` -- different vehicles -- become one trajectory.
     7. With `done=False`, GAE bootstraps across the reset:
        `delta = r + gamma * V(episode-2 state) - V(episode-1 state)`, and `last_gae`
        propagates backward from episode 2 into episode 1.

     **Blast radius, by whether a run crossed a boundary:**

     | run | decisions / per episode | |
     |---|---|---|
     | `align_sumo`, the +0.00 +/- 0.29 null | 150 / 600 | clean |
     | `align_src`, the -0.47 +/- 0.51 null | 20 / 20 | clean |
     | `segown2`, the -0.12 +/- 0.51 arm | 20 / 20 | clean |
     | `segown_highpower` | 60 / 20 | **affected** |
     | speed-only arm | 40 / 20 | **affected** |
     | matched baseline | 40 / 20 | **affected** |
     | calibration v1 and v2 | 40 / 20 | **affected** |
     | TRAINING `mappo_sumo` | 1800 / 600 | **affected** |
     | TRAINING `mappo_src` | 40 / 20 | **affected** |

     **This does NOT explain the null, and saying so matters more than the defect.**
     The three cleanest measurements never crossed a boundary and still read zero. The
     bug contaminates both training runs and the later arms -- including the
     calibration every sensitivity bound was derived from -- but the central result
     does not rest on them.

     **The magnitude is bounded and modest.** At gamma 0.9 and lambda 0.95 the
     backward decay is 0.855 per decision, so contamination reaches about six
     decisions back from a boundary; with a mean of 7.1 decisions per agent and 20 per
     episode, only agents departing near the boundary are touched.

     **The fix is NOT the one line I first said.** Adding `truncated` to the done flag
     stops the leak but treats a time-limit truncation as a true termination, which
     biases value targets downward -- the standard time-limit-bootstrapping error in
     the opposite direction. Correct handling needs three things: a per-transition
     `truncated` flag distinct from `done`; a bootstrap value captured at each
     boundary from the state at the END of the finishing episode, which the buffer
     currently cannot express since it holds one bootstrap per agent for the whole
     rollout; and a GAE recursion that bootstraps on truncation while resetting
     `last_gae`. Making agent ids unique per episode is worth doing regardless, since
     the grouping key silently merging two vehicles is its own hazard.

     **Two design defects sit alongside it and are the likelier explanation of the
     null, neither of which is a bug:** the speed bins are 8.33/12.5/16.67 m/s against
     a mean AV speed near 2.5 m/s, so all three do the same thing on 84% of AV-steps
     and the action is mostly inert; and the reward is a network aggregate delivered
     identically to every agent, so the per-agent advantage has almost no per-agent
     variation. Both were recorded this session as swept knobs rather than read as
     faults.

139. **CLOSED. The ported run was stopped at update 13 of 20, its gate fails on all
     three criteria, and the trained checkpoint REDUCES throughput.** 2026-09-10.
     Ankit stopped the run and then stopped all remaining work.

     **The gate, pre-registered at 20 updates and read at 13:**

     | criterion | threshold | reading | |
     |---|---|---|---|
     | 1. summed entropy falls | below 1.9775 of 2.1972 | lowest 2.1459 | fail |
     | 2. score trends up | by more than step sd 0.2625 | moved **-0.4679** | fail |
     | 3. action distribution leaves uniform | joint modal share above 0.20 | **0.1460** | fail |

     Criterion 2 fails in the WRONG DIRECTION: the score declined by more than the
     update-to-update noise rather than staying flat. Criterion 3 is measured on the
     update-13 actor over 1,503 observations; both heads sit near uniform and a
     randomly initialised actor reads 0.1378, so 0.1460 is initialisation drift.

     **The early stop does not rescue criterion 2 and the margin is computable.** For
     it to pass at update 20, updates 14 to 20 would have had to hold a score of
     -6.03 -- better than every update except the first, against a whole-run range of
     -6.811 to -5.777.

     **THE CHECKPOINT WAS EVALUATED AGAINST NO CONTROL AND IS WORSE. One seed of five
     completed before the stop, so this is a single paired observation and not a
     five-seed result:**

     | seed | no control | trained | difference | commands issued | mean speed |
     |---|---|---|---|---|---|
     | 7 | 198 | 178 | **-20** | 525 | 4.67 -> 4.18 m/s |

     The no-control arm reproduced its value of 198 exactly, matching the fleet-mix
     sweep and the paired threshold sweep on the same seed, so the harness is sound.
     The 525 commands confirm the policy was live rather than inert -- the failure
     mode where a control arm silently does nothing and reads as a null. **The trained
     policy served 20 fewer vehicles and slowed the network.** With n=1 this is one
     observation against a seed-to-seed spread of about 16 arrivals, so it is
     consistent with harm and does not establish its size.

     **What else stopped unfinished:** the matched `mappo_src` baseline at 40
     decisions had one seed of three (z +0.02, corr +0.0157, against the speed-only
     arm's +0.0153 on the same seed -- indistinguishable, which is the head-to-head
     answer even at n=1); the segment re-run that would have reported its correlation
     directly, repairing task 138's walk-back, never started. **So task 138 stands:
     the road-attached agent's null rests on a z the calibration shows cannot separate
     a correlation of 0 from 0.1, and that was not repaired.**

138. **WALK-BACK: the segment arms' null is WEAK, and "every candidate is closed"
     overstated it.** 2026-09-10. This is the fourth statement of these thresholds and
     it is the one that stops converting.

     Task 137's calibration used i.i.d. synthetic advantages where the real advantage
     is temporally autocorrelated, which showed as z = -0.59 at c = 0 instead of 0.
     Rebuilt with the real advantage as the noise component, the c = 0 row reads
     +0.24 and the curve moves:

     | c | 0.00 | 0.02 | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 | 1.00 |
     |---|---|---|---|---|---|---|---|---|
     | i.i.d. noise (task 137) | -0.59 | -0.34 | +0.18 | +1.23 | +3.50 | +7.01 | +10.54 | +22.02 |
     | real advantage | +0.24 | -0.03 | -0.25 | +0.03 | +1.90 | +5.35 | +8.97 | +22.02 |

     z crosses 2 at **c = 0.204**, not 0.134. More important than the crossing: below
     c about 0.2 the readings are NON-MONOTONE -- +0.24, +0.09, -0.03, -0.25, +0.03 --
     every one of them inside the null's own scatter. **The gradient-norm z on an arm
     this size cannot separate a correlation of 0 from one of 0.1.**

     **What this costs.** The segment arms report only a z. Their bounds were quoted
     as 0.15, then 0.05, then 0.07; the honest figure is of order 0.1 to 0.2, which is
     not a bound worth much. So the road-attached agent reads null on an instrument
     too blunt to have seen a small effect, and task 134's "the last candidate is
     closed" and the explanation table's "ruled out" overstate what was measured. The
     vehicle arms are unaffected: they report `corr(slow, A)` directly at a
     two-standard-error bar of 0.035, and a measured correlation needs no conversion.

     **The fix is not a fifth conversion.** The segment arm is being re-run reporting
     the correlation directly from the same batch, which is one line and removes the
     conversion entirely. Until that lands, the road-attached agent should be
     described as reading null at an unknown but coarse sensitivity, not as closed.

     **The pattern, since it cost four attempts.** Each correction fixed the method
     and none touched the data: a formula for a different statistic, then an
     assumption of linearity, then a calibration whose null structure did not match.
     The lesson is recorded as
     [[feedback_calibrate_a_statistic_before_quoting_its_sensitivity]]: inject known
     effect sizes, match the null's structure, and prefer a statistic that is
     interpretable without conversion.

137. **MEASURED: the gradient-norm z is strongly SUB-LINEAR in the action-advantage
     correlation, so the ceiling extrapolation of task 136 was optimistic.**
     2026-09-10. `scratchpad/calibrate_instrument.py`.

     Task 136 converted z to a correlation by extrapolating linearly from the
     instrument's ceiling and flagged that as approximate. This measures the curve
     instead. One rollout of `mappo_src` seed 7, the resampled-action null computed
     once on it, then advantages synthesised at known correlations:
     `A(c) = c*u + sqrt(1-c^2)*w`, with u the standardised indicator of `slow` and w
     noise orthogonalised against u. The construction is verified, not assumed: every
     target c is reproduced to four decimals.

     | c | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 | 0.75 | 1.00 |
     |---|---|---|---|---|---|---|---|---|---|
     | z | -0.48 | -0.34 | +0.18 | +1.23 | +3.50 | +7.01 | +10.54 | +16.39 | +22.02 |
     | z/c | -48.0 | -17.1 | 3.7 | 12.3 | 17.5 | 20.0 | 21.1 | 21.9 | 22.0 |

     **z/c is not constant.** It rises from 3.7 at c = 0.05 to 22.0 at c = 1.0, so the
     statistic is far less responsive to small correlations than to large ones, which
     is what a norm does: a gradient norm is the norm of a noise vector plus an
     aligned component, and a small aligned component adds almost nothing in
     quadrature. Linear extrapolation from the ceiling predicts z = 2 at c = 0.091;
     measured, z crosses 2 at **c = 0.134**, a factor of 1.47.

     **Every threshold, corrected by that factor:**

     | arm | SE of mean z | linear | corrected |
     |---|---|---|---|
     | vehicle, shared, 1 s | 0.29 | 0.005 | **0.007** |
     | vehicle, shared, 60 s | 0.51 | 0.017 | **0.025** |
     | road, own reward, 171 | 0.52 | 0.048 | **0.071** |
     | road, own reward, 533 | 0.66 | 0.050 | **0.074** |

     **The conclusions do not change and the bounds widen.** Every arm still reads
     null; the road arms now bound the correlation at about 0.07 rather than 0.05.

     **A limitation of this calibration, stated because it is real.** The synthetic
     advantages are i.i.d. where the real advantage is temporally autocorrelated, and
     that shows: at c = 0 the curve reads z = -0.59 rather than 0, because pure i.i.d.
     noise produces a lower gradient norm than the null's real advantages do. The
     SHAPE conclusion is robust to that offset -- z/c varies sixfold across the range
     -- but the crossing point carries it, so 1.47 is an estimate of the correction
     and not an exact factor.

     **The tightest bound on the vehicle arms does not come from z at all.** Those
     arms report `corr(slow, A)` directly -- +0.0153 and -0.0052 on the speed-only
     arm at n about 3,300, where two standard errors is 0.035. A directly measured
     correlation needs no conversion and no calibration. The segment arms do not
     report one, which is why they depend on this curve.

     This is the third statement of these thresholds in one day. The first applied a
     correlation's sampling error to a statistic that is not a correlation; the second
     used the instrument's own scale but assumed linearity; this one measured the
     curve. Related: [[feedback_run_your_guard_against_a_control]].

136. **CORRECTION: the segment arms' sensitivity was quoted from the wrong formula,
     and the extra episodes bought none.** 2026-09-10.

     Tasks 129 and 134 put the arms' exclusion thresholds at 0.15 and 0.09, taken from
     2/sqrt(n), the sampling error of a correlation coefficient. **These arms do not
     measure a correlation.** They measure a gradient-norm z, and the sampling error
     of a statistic they do not compute says nothing about what they can see.

     **The instrument carries its own scale.** Its ceiling advantage is
     `(action == slow) * 2 - 1`, standardised -- an affine function of the indicator
     of `slow`, so its correlation with the action is exactly 1.0. The z it produces
     is therefore what a correlation of 1.0 looks like on that batch:

     | arm | per-seed z at correlation 1.0 | cross-seed SE | correlation at two SE |
     |---|---|---|---|
     | 171 segment-decisions | 20.8, 22.7, 20.4 | 0.52 | **0.048** |
     | 533 segment-decisions | 30.6, 44.8, 3.3 | 0.66 | **0.050** |

     **Two things change.** The arms are about three times MORE sensitive than the
     retracted figures said, close to the vehicle arms' 0.03, so the road-agent
     candidate is closed more firmly than task 134 claimed. And **the extra episodes
     bought no sensitivity at all**: per-seed sensitivity rose and the cross-seed
     standard error rose with it, 0.52 to 0.66. The third seed of the higher-power arm
     is badly conditioned, reading 3.3 where the others read 30.6 and 44.8 -- a
     tenfold spread in instrument sensitivity across seeds of one arm, which is worth
     knowing before any future arm is sized by episode count.

     **The extrapolation is linear from a correlation of 1.0 and is approximate.** A
     gradient norm is the norm of a noise vector plus an aligned component, which
     grows more slowly than linearly while the aligned component is small, so the true
     thresholds are somewhat worse than 0.05.

     **SUPERSEDED BY TASK 137, which measured the curve instead of assuming it.** z/c
     rises from 3.7 at c = 0.05 to 22.0 at c = 1.0 and z crosses 2 at c = 0.134 where
     linearity predicts 0.091, a factor of 1.47. The thresholds in the table above
     become 0.071 and 0.074. The direction was right and the size was not; the numbers
     to quote are task 137's.

     Related: [[feedback_reconstructed_quantities_are_inferences]] -- a threshold
     derived by applying a formula to a statistic that was never computed is the same
     failure as a quantity backed out of an aggregate.

135. **The objective tracks throughput in the MEAN and not per application.**
     2026-09-10. Computed on task 133's own 25 cells, so it costs nothing extra.

     Averaged over seeds, the objective and the arrival count agree: both fall when
     metering is applied, and both differences are resolved at three of four mix
     levels. But across the twenty individual (mix, seed) treatment applications, the
     correlation between the change in the objective and the change in arrivals is
     **-0.035**, with the speed term at -0.182 and the penalty term at +0.318.

     **So the objective detects THAT metering happened and does not measure HOW MUCH
     throughput it cost.** A learner improving this objective is not thereby improving
     throughput beyond the coarse direction, which matters because throughput is the
     quantity the project claims.

     **The power is weak and the claim is limited to match.** At n=20 the
     two-standard-error bar on a correlation is 0.485, so this excludes a tight
     relationship and does not exclude a moderate one. A pooled correlation over all
     25 cells reads +0.582, but that mixes the shared treatment trend with seed
     bistability and is the wrong statistic for the question: both quantities falling
     together on average produces a positive pooled correlation whether or not their
     fluctuations are related. See [[opt506_pooled_median_inverts_ranking]] for the
     same shape of error.

134. **RESULT: the road-attached agent is a null at three times the power too, so
     the last candidate explanation is closed at an exclusion threshold of 0.09.**
     2026-09-10.

     Three episodes per rollout rather than one, which takes the arm from 171
     segment-decisions to 532 or 533 and the exclusion threshold for a correlation
     from 0.15 to 0.089. Same resampled-action null, same per-seed initialisation.

     | seed | segment-decisions | measured | null | z | ceiling |
     |---|---|---|---|---|---|
     | 7 | 532 | 0.30147 | 0.24089 +/- 0.07005 | +0.86 | 9.9 |
     | 17 | 533 | 0.14642 | 0.19563 +/- 0.04237 | -1.16 | 10.7 |
     | 27 | 533 | 0.28457 | 0.77083 +/- 0.44645 | -1.09 | 2.9 |

     **Mean z -0.46 +/- 0.66**, against -0.12 +/- 0.51 at the lower power. Both are
     centred on zero, and the higher-power reading does not move toward the ceiling,
     which at 9.9 and 10.7 on two of three seeds is well above anything measured.

     **The shared-reward version of the same arm was re-measured alongside it and
     reads +0.25 +/- 0.76** (+1.77, -0.61, -0.42), where it previously read
     -1.32 +/- 0.16 under a fixed initialisation and the invalid permutation floor.
     That retires task 123's structural explanation for its negativity: the negativity
     was the instrument. The argument may still be correct and no longer has an
     observation supporting it.

     **What is left untested** is the band between this arm's 0.09 and the vehicle
     arms' 0.03, and the paper's own formulation rather than this approximation of it,
     whose action is drawn from one representative vehicle's observation rather than
     from five per-super-segment fields. Widening to those fields widens the deployed
     contract.

133. **PAIRED: the objective discriminates fleet behaviour at least as well as an
     arrival count, and its entire resolved response is the speed term.** 2026-09-10.

     Task 131's pass was unpaired and could not resolve its own arms. This one pairs:
     each of five seeds fixes the traffic and the metering draw sequence, so the same
     seed at two mixes differs only in the mix. Its p=0 cells reproduce the fleet-mix
     sweep's reference arrivals EXACTLY, seed for seed -- 198, 194, 156, 179, 196 --
     and its p=0 reward over seeds 7, 17 and 27 reproduces task 131's +1.531 exactly,
     so the two instruments agree before any comparison is drawn.

     Paired differences from p=0, two-standard-error bars, n=5:

     | p | d reward | d speed term | d penalty term | d arrivals |
     |---|---|---|---|---|
     | 0.25 | **-1.51** +/- 0.54 | **-1.24** +/- 0.72 | **-0.28** +/- 0.19 | **-25.4** +/- 8.7 |
     | 0.5 | **-1.29** +/- 0.73 | **-1.47** +/- 0.82 | +0.18 +/- 0.37 | **-36.8** +/- 23.0 |
     | 0.75 | **-1.70** +/- 0.97 | **-1.74** +/- 0.93 | +0.04 +/- 0.35 | **-32.0** +/- 24.7 |
     | 1 | **-1.86** +/- 0.71 | **-1.72** +/- 0.83 | -0.14 +/- 0.50 | -27.8 +/- 32.8 |

     Bold means the bar excludes zero.

     **CORRECTS what task 131 said about noise.** Unpaired, the reward's arm bar
     (1.25) spanned its between-arm range and arrivals' did not, which reads as the
     objective being the noisier measure. Paired, it is not: the ratio of the
     difference to its own bar is 2.79, 1.76, 1.75 and 2.63 for the reward against
     2.91, 1.60, 1.29 and 0.85 for arrivals. At p=1 the objective resolves the change
     and the arrival count does not. The reward is a sound discriminator of fleet
     behaviour on this road, so **no part of the null is attributable to the objective
     being unmovable or noisy.**

     **The decomposition stands and is sharper than task 131 could show.** The speed
     term is resolved and negative at every mix level and is essentially the whole of
     the reward difference. The penalty term resolves at one of four and changes sign
     across them: -0.28, +0.18, +0.04, -0.14. **The half of the objective that pays
     for keeping a link below critical density does not respond consistently to this
     behaviour, and the half that does is the half that restates mean segment speed.**

     **A consequence worth stating for the training run.** Within this cut the
     objective is maximised at p=0, by not metering. A policy that has learned nothing
     and a policy that has learned to leave the fleet alone produce the same
     behaviour, so this objective cannot demonstrate learning through improvement on
     this axis. The cut is one dimension -- a uniform random command at one speed --
     and a selective policy is not bounded by it; the metering oracle covers the
     selective case with perfect information and gains nothing.

132. **The two waiters holding the queued segment arms could never have fired.**
     2026-09-10.

     Each was `while pgrep -f 'measure_segment_own_reward|local_reward_align|...'`.
     `pgrep -f` matches the full command line of every process, and the waiter's OWN
     command line contains that pattern as literal text, so each waiter matched itself
     and one matched the other. Both jobs they were holding -- `measure_segment_as_agent`
     and the higher-power own-reward arm -- would have waited indefinitely.

     Replaced with one shell script that runs the two in sequence, with no matching.
     The third process-identification failure of the session, after `pgrep -f "a\|b"`
     (where `\|` is a literal in this shell) and a `grep -E "train_mappo|measure_threshold"`
     against scripts actually named `train_policy.py` and `threshold_reward_sweep.py`,
     which reported two healthy jobs as dead and led to duplicates being started
     against the same checkpoint directory. Recorded as a memory: identify a job by
     PID or by a sentinel file it writes, never by a pattern.

131. **RESULT: the threshold objective is NOT flat, so the null stays in the mechanism
     rather than moving to the reward.** 2026-09-10.

     The open question was whether the objective can be moved by behaviour at all. If
     it were flat, no learner could exploit it however well credit were assigned, and
     the entire gradient investigation would have been measuring the attribution of a
     quantity with nothing to attribute. It is not flat.

     Each AV independently issues the config's own `slow` command (8.33 m/s) with
     probability p at each decision and is otherwise released to SUMO's car-following.
     Three seeds, 600 s observed after 300 s of fill.

     | p | reward/step | penalty term | speed term | arrivals |
     |---|---|---|---|---|
     | 0 | **+1.531** +/- 1.082 | -1.50 | 3.031 | 182.7 |
     | 0.25 | +0.290 +/- 0.567 | -1.86 | 2.150 | 156.0 |
     | 0.5 | +0.744 +/- 0.497 | -1.29 | 2.034 | 153.3 |
     | 0.75 | +0.315 +/- 0.079 | -1.61 | 1.925 | 141.7 |
     | 1 | **-0.136** +/- 0.707 | -2.01 | 1.874 | 130.7 |

     The objective falls by 1.667 across the range and is MAXIMISED at p=0, by not
     metering at all. Arrivals fall by 52.0 over the same range. Reward and outcome
     agree in direction, so the reward is a coherent objective on this road and the
     null is not about it.

     **The decomposition says something the total hides.** `congestion_penalty` is
     1.0, so the penalty term is exactly minus the count of segments over rho* and the
     speed term is the remainder. The speed term declines strictly monotonically and
     tracks arrivals. The penalty term does not order at all: -1.50, -1.86, -1.29,
     -1.61, -2.01. **The half of the objective designed to pay for keeping a link
     below critical density -- the anticipatory behaviour the whole mechanism is
     supposed to produce -- does not respond coherently to this behaviour cut, and
     the half that does is the half that restates mean segment speed.**

     **What this pass cannot settle.** The comparison is unpaired, and the road is
     bistable: at p=0 the reward's seed-to-seed standard deviation is 1.082, so its
     two-standard-error bar over three seeds is 1.25 against a between-arm range of
     1.667. Per-seed values at p=0 are +3.021, +1.088 and +0.485, tracking that seed's
     arrivals (198, 194, 156). A paired pass over five seeds is running; its p=0 cells
     reproduce the fleet-mix sweep's reference arrivals exactly, seed for seed.

     **It is also one cut through a much larger action space** -- a uniform random
     command at one speed value -- so it bounds what an unselective fleet can do to
     the objective, not what a selective policy could. The metering oracle covers the
     selective case with perfect information and gains nothing.

130. **CORRECTION: the threshold penalty fires on 1.5 of 9 segments, not 7 of 9.**
     2026-09-10, caught by reconciling the fleet-mix sweep's reward against the
     training score.

     I inferred "roughly 7 of 9 segments above critical" from the training score by
     assuming every segment runs at the NETWORK mean speed. That is vehicle-weighted:
     a nearly-empty free-flowing leaf has a segment mean speed of 25 m/s while
     contributing almost nothing to the network mean, so the sum of per-segment means
     is far above nine times it. The reconstruction missed the reported reward by 1.9
     out of 1.5, which is how the error surfaced.

     **Measured directly** by the sweep at p = 0 over a 600 s episode: **1.50 of 9
     segments over rho\*** with a reward of +1.531 per step.

     **What it changes.** The claim that the threshold reward "has real range where the
     eleven-term reward sat near zero" survives -- the range is roughly -1.5 to +4 per
     step rather than -7 to +1.3 -- and the claim that the penalty fires constantly
     does not. It fires on one or two links at a time, which is the bottleneck, and
     that is arguably the intended behaviour rather than a defect.

     **The lesson is the one this session keeps producing**: a quantity reconstructed
     from an aggregate is an inference, and it needs the same scepticism as a
     measurement. The sweep measured it in one line.

129. **RESULT: the road-attached agent is a null too, so every candidate
     explanation is now measured and none survives.** 2026-09-10.

     The segment-as-agent arm paid its OWN per-segment reward -- an agent attached to
     the road, persisting for the whole episode, with a reward that differs from its
     neighbours' -- reads **mean z -0.12 +/- 0.51** over three seeds (-0.44, -0.81,
     +0.89). Indistinguishable from zero, and the CLOSEST TO ZERO of any arm measured
     today, which is itself a small check on the corrected null: the arm with the most
     independent structure is the one that centres.

     **So the last explanation standing after task 128's eliminations does not
     survive either** -- but at a much coarser sensitivity than the arms before it,
     and the first version of this entry understated that by quoting ONE standard
     error where the rest of the project quotes two. At 171 segment-decisions the
     standard error of a correlation is 0.077, so the arm excludes correlations above
     **0.15**, against the vehicle arms' 0.03 at n about 5,500. It rules out a large
     effect and not a small one. The higher-power version, three episodes per rollout,
     takes n to about 513 and the exclusion threshold to **0.09** -- still three times
     coarser than the vehicle arms. It is queued.

     **The full set, all measured against the same tested null:**

     | arm | agent | reward | mean z |
     |---|---|---|---|
     | `mappo_sumo`, 1 s | vehicle | shared | +0.00 +/- 0.29 |
     | `mappo_src`, 60 s | vehicle | shared | -0.47 +/- 0.51 |
     | paired team-only | vehicle | shared | -0.44 +/- 0.56 |
     | local blend 0.5 | vehicle | **unshared** | -0.81 +/- 0.14 |
     | segment, network reward | **road** | shared | (void: fixed init) |
     | segment, own term | **road** | **unshared** | **-0.12 +/- 0.51** |

     **What this means, stated carefully.** On this network, at this operating point,
     no configuration tried produces an advantage that carries information about the
     action -- across two agent attachments, two reward scopes, two decision rates,
     four bin scalings, three operating points, three penetrations and two reward
     shapes. That is a strong statement about THIS road and a weak one about
     decentralised traffic RL in general, because a single network with fixed routes
     to one exit and a junction-limited capacity is one instance.

     **What has NOT been measured** is whether the objective can be moved by behaviour
     at all. That sweep is running. If the threshold reward turns out flat across the
     whole fleet-mix range, the null is about the reward rather than the learning, and
     the entire gradient investigation was measuring the attribution of a quantity
     that had nothing to attribute.

128. **RESULT: an unshared per-agent reward produces no action alignment either,
     and the z statistic has a residual bias I cannot explain.** Measured 2026-09-10,
     three seeds, network initialisation varying with the seed, work directories keyed
     by process id.

     | arm | per seed z | mean z | corr(chose slow, advantage) |
     |---|---|---|---|
     | team only, shared reward | -1.12, +0.67, -0.87 | -0.44 +/- 0.56 | **-0.0043** |
     | local 0.5, UNSHARED reward | -1.08, -0.61, -0.73 | -0.81 +/- 0.14 | **+0.0016** |

     **The correlation is the statistic to read, and it is zero.** At n about 5,500
     its standard error is about 0.0135, so both values are well inside noise. Giving
     each agent a reward that genuinely differs from its neighbours' produces no
     more action-advantage alignment than a shared one.

     **This rules out the explanation named in the result document** as the structure
     underneath every null here -- that the nulls came from paying every agent the
     same reward. They do not. At the vehicle level, with sensitivity to a
     correlation of about 0.016, an unshared reward is silent about the action too.

     **AND THE z STATISTIC HAS A RESIDUAL BIAS.** It reads consistently negative
     across every arm measured today: +0.00, -0.47, -1.32, -0.44, -0.81. Under a null
     that resamples the action from the same policy at the same state, the resampled
     action and the action actually taken are exchangeable -- both are draws from
     `pi(.|s)` with the same parameters, since no update intervenes between collection
     and measurement -- so z should centre on zero. It does not, and I cannot say why.

     **What that does to the readings.** The correlations are unaffected: they are a
     direct point-biserial statistic with an analytic standard error and no null
     construction at all. Every conclusion in tasks 105 to 128 that rests on "z near
     zero" should be re-read as resting on the correlation instead, and the z values
     should be treated as an instrument with a known offset until the offset is
     explained. The two agree on the substance -- no alignment -- which is why the
     conclusions stand.

127. **Two copies of one measurement ran at once, sharing their work directories.**
     2026-09-10, and it nearly produced numbers from two interleaved simulations with
     nothing looking wrong.

     **What happened.** A chained waiter was still holding the local-reward arm; I
     killed what I believed was that waiter, matching by pattern rather than by
     identity, and killed a different one. The surviving waiter then fired the arm at
     the same moment I started it by hand. Parent pids settled it: the second copy's
     PPID was the waiter's.

     **Why it matters.** Both wrote `/tmp/dsrc_localalign_<weight>_<seed>`, so each
     was rewriting the other's `net.net.xml` between write and read. That surfaces as
     an XML parse error from netconvert output that was never finished -- and when it
     does NOT error, the run simply mixes two simulations and reports a number.

     **What was done.** The 32 s overlap fell on the team-only seed-7 arm, so most
     rows were probably unaffected; "probably" is not a basis for a results table, so
     the whole arm was discarded and restarted. Work directories are now keyed by
     process id, which makes the collision impossible rather than unlikely.

     **The readings from the discarded run**, recorded so they are not quietly reused:
     local 0.5 at seed 7 z = -1.08 with correlation +0.0017, at seed 17 z = -0.61 with
     correlation -0.0105.

     **The rule this session keeps relearning.** Kill by pid, not by pattern; wait on
     a pid, not on a name; and check what a "seed" or a "process" actually is before
     trusting that two of them are two.

126. **The segment arms are UNDERPOWERED for the question they were built to
     answer.** Computed 2026-09-10 from each arm's own ceiling and null, before the
     segment readings were complete.

     The instrument's dynamic range is the synthetic ceiling over the null, and the
     detectable correlation follows from it and the null's spread:

     | arm | transitions | ceiling | z per unit rho | rho detectable at z = 2 |
     |---|---|---|---|---|
     | vehicle / shared, 1 s | 5,760 | 30.0 | 128.6 | **0.016** |
     | vehicle / shared, 60 s | 1,627 | 17.2 | 60.7 | **0.033** |
     | segment / own reward | 171 | 6.5 | 20.8 | **0.096** |

     **So the segment arm resolves about 0.10, against 0.033 for the vehicle arm at
     the same lever.**

     **CORRECTED, later the same evening: the sentence that followed was wrong.** It
     read "a null there tells us nothing the earlier measurements did not; it can only
     contribute if it reads POSITIVE". That conflates two different configurations.
     The vehicle arms bound the VEHICLE formulation; they say nothing whatever about
     an agent attached to a segment, which has a different action, a different
     trajectory length and a different reward. A null on the segment arm excludes
     rho above about 0.10 **for the segment formulation**, and that is a real
     exclusion -- the more so because if a road-attached agent had signal one would
     expect it to be large, that being the entire point of the bigger lever.

     What remains true is the power comparison: the segment arm is six times coarser
     than the vehicle arm, so its null is correspondingly weaker, and three episodes
     per rollout rather than one would take n from about 171 to about 510 and the
     detectable rho to roughly 0.05.

     **The cause is structural.** One action per segment per decision gives 9
     segments times 20 decisions, about 171 transitions, against 5,760 at the vehicle
     level. The formulation that fixes the credit assignment is the same one that
     starves the sample size, which is the same trade as task 115's -- a bigger lever
     costs data -- appearing at the level of the batch rather than the trajectory.

     **What would make it stronger**: three episodes per rollout instead of one,
     taking n to about 510 and the detectable rho to roughly 0.05, at three times the
     runtime. The earlier version of this paragraph made that conditional on the
     current seeds reading positive, on the mistaken ground that a null would be
     uninformative. It would not be: it is the only measurement bearing on the one
     explanation that survives task 128's eliminations, and tightening it from 0.10 to
     0.05 is the difference between excluding a large effect and excluding a moderate
     one.

125. **PREDICTION, recorded at update 3 of 20 so it can be wrong.** The ported
     run's entropy is falling monotonically for the first time in this project:
     2.1898, 2.1837, 2.1775, a slope of **-0.00611 per update**.

     Criterion 1 of task 113 needs summed entropy below **1.9775** by update 20,
     which from the starting 2.1898 requires **-0.01117 per update**. The observed
     rate is **55% of that**, and at the observed rate update 20 lands at **2.0736**.

     **So the prediction is that criterion 1 fails, at about 2.07, unless the decline
     accelerates.** Recorded now rather than after the fact, because "the entropy was
     falling" is the kind of observation that reads as encouraging in hindsight
     whatever the endpoint turns out to be.

     **What would make this interesting even so.** The previous run moved entropy
     0.0147 in TWENTY-FIVE updates; this one has moved 0.0123 in three. That is an
     eightfold difference in rate, and it is the first quantity in this investigation
     that has moved in the direction the mechanism requires. A policy that is
     genuinely becoming less uniform, but too slowly for a 20-update budget, is a
     different situation from one that never moves -- and it would argue for more
     updates rather than for another change of formulation.

     **RETIRED AT UPDATE 5: the decline was not a trend.** Entropy went 2.1898,
     2.1837, 2.1775, 2.1743, then **rose** to 2.1762. The steps are -0.0061, -0.0061,
     -0.0032, +0.0019 -- deceleration and then reversal. The slope over five updates
     is -0.00338, projecting to **2.13** at update 20, further from the 1.9775 gate
     than the 2.07 predicted at update 3.

     The score flattened at the same time: -6.567, -6.446, -6.463, -6.462, three
     consecutive updates inside 0.02.

     **This is what the gradient measurements predict.** A policy whose advantage
     carries no information about its actions drifts rather than descends, and four
     steps of -0.006, -0.006, -0.003, +0.002 is drift. The paragraph above was written
     from three points and should not have been given the weight of a trend; it is
     kept rather than deleted because the reasoning it contains -- that a slow genuine
     decline would argue for more updates -- is still the right test, and it simply
     did not apply.

     **The caveat that comes with it.** Three points are not a trend, entropy can fall
     because a policy is collapsing onto one action for reasons unrelated to the
     objective, and the joint modal share is what distinguishes those. That is
     criterion 3, and it is measured at the end by
     `scripts/measure_action_distribution.py`.

124. **Every error bar on a z-score in this investigation was built from ONE
     network draw.** Found 2026-09-10 from a standard error that was too good.

     The segment arm reported mean z **-1.32 +/- 0.16** over three seeds. A tight
     spread around a consistent value reads as a robust finding; it was one draw
     repeated. Every arm called `seed_everything(0)` before building the trainer, so
     **the actor and critic weights were bit-identical across all three seeds** --
     confirmed by comparing parameters directly, not inferred. The seeds varied the
     TRAFFIC only.

     **Why that produces exactly this signature.** With a fixed initialisation the
     relationship between the critic's value function and the actor's action
     distribution is the same function in every run. Where that relationship happens
     to be systematic -- which it can be, since both are functions of the same
     observation through independently initialised networks -- the sign repeats and
     the spread collapses. The arm was not measuring three samples of a population; it
     was measuring one network on three traffic realisations.

     **What this does to the readings.**

     | quantity | status |
     |---|---|
     | point estimates (+0.00, -0.47, -1.32) | stand: that is what those configurations read at that initialisation |
     | every `+/-` quoted on a z | understates the uncertainty; it reflects traffic variation alone |
     | the conclusions | probably unchanged, since a null at a narrow bar is still a null at a wider one -- but "probably" is not "measured" |

     **Fixed**: `seed_everything(seed)`, so the initialisation varies with the seed in
     `measure_action_alignment.py`, `measure_segment_as_agent.py` and
     `measure_segment_own_reward.py`. Arms are being re-run.

     **This is the fourth instrument fault this session**, after a set of comparisons
     that were floor-to-floor (task 105), a process check whose pattern could not
     match (task 116), and a null that was not exchangeable with the data (task 119).
     Each was caught, each is recorded, and the underlying conclusion has survived all
     four. The failure rate is in the measurements rather than in the system under
     test, which is worth weighing when deciding how much of this to rely on.

123. **The shared-reward segment arm is near-uninformative BY CONSTRUCTION, and
     its negative readings are an artefact of that.** Noticed 2026-09-10 from the
     readings being consistently negative under a null that is now valid.

     Under the valid resampled-action null the arm reads z = -1.03 and -1.57 (it read
     -1.18 and -1.75 under the invalid permutation floor, so the repair moved it
     toward zero by about 0.16 and did not explain it).

     **Why it cannot say much.** In that arm every segment is paid the SAME
     network-wide reward at every decision. So every segment's reward sequence is
     identical, and the whole cross-segment variation in the advantage comes from the
     critic's value estimates -- which at measurement time are randomly initialised.
     The GAE residual is `r + gamma V(s') - V(s)` with `r` common, so the advantage is
     essentially a function of `V` alone.

     **And that makes a spurious correlation available.** The advantage and the
     action are then both functions of the same observation, through two independently
     initialised networks. Nothing forces their relationship to be zero, and a
     systematic one of either sign is exactly what a consistently negative z looks
     like.

     **So the arm mostly measures the untrained critic**, not whether a road-level
     agent has a learnable signal. It is kept because it is the paper's own reward
     shape -- the paper sums per-segment terms and its agent is centralized, so the
     sum is the right credit THERE -- and its reading should be quoted with this
     attached rather than as evidence about segment-level control.

     **The own-reward arm is the one that tests the question.** Each segment paid its
     own term makes the reward sequences differ across segments, so the advantage
     carries something other than critic noise. Verified a genuine decomposition
     before it ran: a clear segment reads +1.000 and a jammed one -0.850, summing to
     the network's +0.150.

122. **RESULT: the macroscopic lever does not restore the learning signal.**
     Measured 2026-09-10 with `scripts/measure_action_alignment.py` against the valid
     resampled-action null, three seeds.

     | config | lever | mean z | ceiling |
     |---|---|---|---|
     | `mappo_sumo` | 1 s | +0.00 +/- 0.29 | 28 to 33 |
     | `mappo_src` | 60 s | **-0.47 +/- 0.51** | 17 to 18 |

     Per seed, `mappo_src`: -1.33, +0.44, -0.53.

     **The sensitivity, so the null is bounded rather than merely stated.** With a
     ceiling of about 17, a null of about 0.11 and a null standard deviation of about
     0.029, the expected z for a true action-advantage correlation `rho` is roughly
     60 x rho. The arm therefore resolves rho near 0.03, and the reading bounds rho
     within about +/- 0.017 of zero. This is not a null from an insensitive
     instrument.

     **What it means.** Holding one action for a simulated minute instead of a second
     -- with the paper's threshold reward, its speed bins, its discount horizon, and a
     link-level congestion signal the actor can see -- produces no more
     action-attributable signal than the one-second version. Ankit's spatio-temporal
     argument is right about the mechanism and does not, on this road, produce a
     learnable gradient at the vehicle level.

     **What it does NOT mean.** The lever change is confounded with a cost measured
     in task 115: at 60 s each agent makes 7.1 decisions in its life rather than
     90.5, a thirteenfold loss of temporal structure. The two pull opposite ways and
     this measurement cannot separate them. Separating them needs an agent that
     persists -- which is the segment arms, still running.

121. **PRE-COMMITTED: what decides whether the paused training run resumes.**
     Written 2026-09-10 while the measurement was still running, so the rule is not
     chosen to fit the number.

     The `mappo_src` run is paused at update 2 of 20 with a complete checkpoint. The
     question is whether finishing it is worth about eighty minutes.

     **The rule.** Resume if `mappo_src` under the valid null reads a mean z above
     two standard errors over its three seeds -- the same two-standard-error bar this
     project uses everywhere else. Otherwise do not resume, and report the run as it
     stands.

     **Why the gradient reading decides a training question.** The z measures whether
     the advantage carries information about the action at the START of training. If
     it does not, PPO has nothing to ascend on its first updates, and the run is
     predictable from the first two: score -5.777 then -6.567, entropy 2.1898 then
     2.1837 against a 2.1972 maximum. Spending eighty minutes to watch a flat curve
     is not worth it when the reason is already measured.

     **What a resume would NOT be.** Evidence that the port works. It would only be
     the pre-registered gate of task 113 being read on a run whose gradient is known
     to carry signal, which is the condition under which that gate means anything.

     **What a non-resume is.** A null on the ported formulation at the vehicle level,
     with the cause measured rather than inferred, and the two segment arms as the
     remaining open question.

     **DEPARTED FROM, the same evening, and the reason matters.** The run was resumed
     from update 3 before the third seed reported. That contradicts the rule above,
     so the grounds are stated rather than left implicit: the rule's rationale was
     COST -- "spending eighty minutes to watch a flat curve is not worth it" -- and
     the cost changed. Pausing was forced by three heavy jobs contending for ten
     cores; with one measurement left running the machine is idle, and Ankit asked
     whether the run was still going, which is a signal that the pre-registered gate
     of task 113 is wanted whatever the gradient says.

     It is NOT a departure because the numbers were disliked. The two seeds available
     when the decision was taken read -1.33 and +0.44, and the rule would have
     refused the resume on that basis; the resume happened anyway, on cost grounds,
     and the gate will be read as task 113 specifies.

120. **The instrument was repaired and the vehicle-level result reproduced.**
     Measured 2026-09-10 with `scripts/measure_action_alignment.py`, which replaces
     the permutation floor of tasks 105 to 119.

     **The valid null.** For each transition, resample an action from the policy at
     the SAME observation and keep the advantage:

         a'_i ~ pi(.|s_i),    g_null = mean_i A_i * grad log pi(a'_i | s_i)

     States, advantages and their temporal structure all survive; only the pairing
     between an advantage and the action that earned it is broken. Under the
     score-function identity its expectation is zero.

     **It is tested, which the old floor never was** (`tests/test_action_alignment.py`):
     an advantage aligned with the action reads z above 5; one drawn independently
     reads within 3; **a temporally smooth but uninformative advantage -- a random
     walk with lag-1 autocorrelation above 0.9, asserted in the fixture -- also reads
     within 3**, which is the case that broke the permutation floor; and a partial
     signal reads monotonically between them.

     **THE REPRODUCTION CHECK PASSED.** `mappo_sumo`, three seeds, the same rollouts
     the old instrument measured:

     | null | mean z | ceiling over null |
     |---|---|---|
     | permutation floor (tasks 105 to 108) | -0.10 +/- 0.42 | 25 to 39 |
     | resampled action (valid) | **+0.00 +/- 0.29** | 31 to 33 |

     Same conclusion, better precision. **So the seven-dimension table stands**: at
     the vehicle level with a one-second lever, the advantage carries no information
     about the action, and that is now established against a null that is valid
     rather than one that merely looked reasonable.

     **What is void.** The segment-as-agent arms measured under the old floor,
     z = -1.18 and -1.75, are artefacts of the biased null and are discarded rather
     than reported with a caveat. Two more arms were mid-flight under the same floor
     and were stopped rather than completed, because finishing them would have put
     numbers on the record that read as evidence.

119. **The permutation floor is not a valid null when advantages are temporally
     correlated, and the segment arms are where that bites.** Noticed 2026-09-10 from
     the pattern of the readings rather than from theory.

     Segment-as-agent with the shared network reward read z = -1.18 and -1.75 on its
     first two seeds: the measured gradient norm SYSTEMATICALLY BELOW its own
     shuffled floor. A measured value below its null is the signature of a null that
     is not exchangeable with the data.

     **The mechanism.** Permuting advantages across the batch destroys two things at
     once: the association between action and advantage, which is what the test is
     meant to isolate, AND the temporal correlation within each agent's trajectory. A
     segment agent now holds about 19 consecutive decisions whose advantages are
     smooth; shuffling makes that sequence rough, and a rough sequence cancels less
     when summed against `grad log pi`, so the floor comes out HIGH. The z is biased
     negative by the design of the null, not by the data.

     **Why it did not show at the vehicle level.** There the readings sat at zero
     rather than below it, so the bias was either small or masked. It is not safe to
     assume it was absent: the same mechanism applies wherever `group_by_agent` gives
     an agent a long trajectory, and at the 1 s lever an agent holds about 90.

     **What a valid null would be.** Permute advantages WITHIN each agent's
     trajectory, preserving its temporal profile and destroying only the alignment
     with the actions taken; or permute the ACTIONS while holding advantages fixed.
     Either isolates the quantity the test claims to measure. The current floor does
     not.

     **What this does to the readings so far.** The vehicle-level nulls of tasks 105
     to 108 are not overturned -- a z near zero is consistent with no signal under
     either null -- but their precision is now in question, and the segment arms'
     negative z values should NOT be read as "worse than random". They should be read
     as "this instrument cannot presently distinguish them from random".

118. **A limitation of the segment-as-agent test, stated before its arms
     reported.** 2026-09-10.

     The test samples one action per segment per decision and applies it to every AV
     there, which makes the effective agent the link. But the action is sampled from
     **one representative vehicle's observation**, and that observation is dominated
     by that vehicle's own kinematics: of the 39 inputs, only
     `downstream_congestion_estimate` is genuinely link-level, and the rest -- ego
     speed, leader gap, lane gaps, headway -- describe an arbitrary vehicle rather
     than the segment.

     **So the segment policy keys mostly on a quantity that is close to random with
     respect to the state it is meant to control.** A null on these arms therefore
     supports "an action chosen from one vehicle's kinematics does not help" more
     strongly than it supports "segment-level control does not help". The two are
     not the same claim and the test cannot separate them.

     **What a clean version needs** is link-level observation features -- the paper's
     five per super-segment, density, mean speed, mean gap, inflow and outflow -- so
     the policy is a function of the road rather than of whoever happens to be on it.
     That was built and then backed out (task 110) because six new slots pulled in
     the parity ledger and the Jetson observation builder; the smaller correction to
     `downstream_congestion_estimate` was taken instead. If the segment formulation
     is worth pursuing, those five fields are the prerequisite and the contract work
     is the cost.

     Recorded before the arms reported, so it is a limitation of the design and not
     an explanation reached for afterwards.

117. **What the gradient instrument can resolve at each level, worked out before
     the readings.** 2026-09-10. Every arm reports a synthetic action-correlated
     ceiling beside its permutation floor, and the RATIO of those two is the
     instrument's dynamic range on that batch. It varies enormously with the number
     of transitions, so the same z means different things at different levels.

     | level | transitions | ceiling over floor |
     |---|---|---|
     | vehicle-agent, 1 s lever | about 5,200 | 25 to 39 |
     | vehicle-agent, 60 s lever | about 1,600 | (measured per arm) |
     | segment-agent, 60 s lever | about 171 | **5.8** |

     Fewer samples means less cancellation in the shuffled gradient, so the floor
     sits higher relative to a perfect signal and the range compresses.

     **The sensitivity that follows.** Writing the measured ratio as
     `1 + rho * (ceiling_ratio - 1)` for a true action-advantage correlation `rho`,
     the segment-agent arm at a ceiling ratio of 5.8, a floor of 0.337 and a floor
     standard deviation of 0.092 needs:

     | rho | expected ratio | expected z |
     |---|---|---|
     | 0.20 | 1.96 | **+3.5** |
     | 0.10 | 1.48 | +1.8 |
     | 0.05 | 1.24 | +0.9 |

     So this arm resolves **rho above about 0.1** and is marginal below that. Weaker
     than the vehicle-level test, which resolves about 0.05, and not toothless: a
     null here excludes the size of effect that would matter, and does not exclude a
     small one.

     Recorded before the arm completed, so the sensitivity is not a caveat chosen
     after seeing whether the result was liked.

116. **Two harness defects from running several heavy jobs at once, recorded
     because neither announced itself.** 2026-09-10.

     **A measurement process died after two of three seeds with no error.** The
     gradient test writing `lever.log` completed `mappo_src` seeds 7 and 17 and then
     ended: no traceback, no error text, the log simply stops mid-table. Its command
     line does not match the `pkill` pattern used to pause the training run a moment
     earlier, so that is not the explanation, and there is nothing else to point at.
     Three SUMO simulations plus torch were resident at about 450 MB each, so memory
     pressure is the plausible cause and it is NOT established. The missing seed is
     re-run rather than a two-seed mean being reported as if three had been planned.

     **`pgrep -f <script name>` also matches the shell that LAUNCHED it.** Tested
     rather than asserted: with one measurement running, `pgrep -f segment_agent.py`
     returned two pids -- the python process and the `bash -c` wrapper that started
     it. So

         while pgrep -f lever_signal.py > /dev/null; do sleep 20; done

     inside such a wrapper can wait on itself and never terminate. Chains now wait on
     a pid, `while kill -0 $PID`, and it is worth knowing that the pid `pgrep`
     returns first is the WRAPPER's, not the python process's -- which happens to be
     equivalent, because the wrapper exits with its child.

     **AND MY OWN PROCESS CHECK WAS BROKEN AND REPORTED ABSENCE.** This is the one
     that matters. `pgrep -f "segment_agent\|lever_seed27"` in zsh searches for a
     LITERAL backslash-pipe: `pgrep` takes an extended regular expression, where the
     alternation is a bare `|`, so the escaped form matches a string that cannot
     exist. It returned zero, I read that as two measurement processes having died,
     and wrote it up. Nothing had died -- `ps aux | grep -E` a moment later showed
     the job running with three minutes of CPU.

     **What this costs if unnoticed.** A queued measurement that never starts reads
     as "not run yet" indefinitely; a partial result table looks complete unless the
     seed count is checked against the plan; and a broken liveness check reports a
     healthy job as dead, which is what nearly caused a running measurement to be
     relaunched on top of itself. All three are failures that look like success, and
     the third is the same shape as the noise floor that was never measured.

115. **The deepest difference from the paper is what the agent is ATTACHED to, and
     the macroscopic lever makes it worse before it makes it better.** Recorded
     2026-09-10 while the ported run was still training.

     **In the paper the controller is the ROAD.** A super-segment persists for the
     whole episode, so its agent takes twenty consecutive decisions on the same
     state and accumulates a trajectory. **In this project the controller is a
     VEHICLE.** It enters, traverses, and leaves.

     MEASURED over one rollout per configuration, rather than estimated. The first
     version of this entry said "4 or 5" from an arithmetic estimate off the mean
     travel time; the measurement says 7:

     | config | decisions | distinct agents | decisions per agent, mean / median / max | agents with ONE decision |
     |---|---|---|---|---|
     | `mappo_sumo`, 1 s lever | 150 | 57 | **90.5 / 99 / 150** | 0.0% |
     | `mappo_src`, 60 s lever | 20 | 227 | **7.1 / 7 / 19** | 6.2% |

     A thirteenfold reduction in temporal structure per agent, and four times as many
     distinct agents passing through -- 227 against 57 -- each contributing a short
     fragment instead of a trajectory. The agent count rises because the episode is
     twice as long and more congested, so more vehicles enter and leave.

     That is the hidden cost of the macroscopic lever here. It buys each decision a
     much larger effect and it leaves GAE almost no temporal structure per agent:
     with four transitions, an advantage is barely more than the reward minus the
     value. Both effects are real and they pull in opposite directions, which is a
     reason the ported configuration might show nothing even with every other gap
     closed.

     **The first two seeds of the gradient test are consistent with that**: `mappo_src`
     reads z = -1.82 and +0.21 against `mappo_sumo`'s completed -0.10 +/- 0.42, and
     the training score has gone from -5.777 to -6.567 over two updates. Neither is
     conclusive and the third seed is outstanding.

     **THE CANDIDATE THAT FOLLOWS, and it keeps decentralized execution.** Give the
     actor LINK STATE ONLY -- drop the per-vehicle fields from its input. Then every
     AV on a link computes the same action, the fleet implements a variable speed
     limit on that link, and the effective agent is the LINK, realised by whichever
     vehicles happen to be on it. The road-attached property is recovered without any
     vehicle reading a global state, which is the project's actual claim.

     It is the coherence argument taken to its conclusion: co-located agents already
     see the same `downstream_congestion_estimate`, and what stops them acting
     together is that their own kinematics differ and dominate the input.

     **It is testable before it is trained.** The gradient instrument
     (`scripts/measure_gradient_signal.py`, with the floor distribution of task 110)
     reads a z-score from three short rollouts, so masking the per-vehicle inputs and
     re-measuring costs minutes rather than an hour of training.

114. **The network is bistable at this operating point, and the branch imbalance
     is emergent rather than structural.** Measured 2026-09-10 by
     `scripts/measure_branch_asymmetry.py`, five seeds, 2400 veh/h, no control, mean
     density ratio over a 600 s episode after a 300 s warm-up.

     | seed | b1 | b2 | a1-a3 | a4-a6 | trunk | served |
     |---|---|---|---|---|---|---|
     | 7 | 0.237 | 0.325 | 0.222 | 0.026 | 0.062 | 198 |
     | 17 | 0.255 | 0.209 | 0.192 | 0.184 | 0.062 | 194 |
     | 27 | 0.189 | 0.297 | 0.216 | 0.212 | 0.047 | 156 |
     | 37 | **0.532** | **0.663** | 0.057 | 0.046 | 0.054 | 179 |
     | 47 | **0.427** | **0.572** | 0.032 | 0.081 | 0.060 | 196 |

     **Two regimes, from the same demand on the same network.** Seeds 37 and 47 put
     the congestion in the middles with the leaves nearly empty; seeds 7, 17 and 27
     put it on the leaves with the middles moderate. Which branch leads varies -- b2
     is denser on four of five seeds and b1 on one -- so the imbalance is not built
     into the road.

     **The road is symmetric, checked rather than assumed.** In the generated
     network, `b1` and `b2` are both `type="zipper"`, each takes six incoming lanes,
     and both connect into the trunk with lanes 0 to 0 and 1 to 1, every connection
     `state="Z"`. There is no priority difference to find.

     **A reading trap this exposes.** A low density ratio can mean free flow OR an
     empty road, and at fixed flow the two are the same measurement: 400 veh/h at 25
     m/s is 4.4 veh/km, and at 3 m/s it is 37. So `a4-a6` at 0.026 on seed 7 are
     carrying the same demand as `a1-a3` at 0.222 -- they are moving, not idle. An
     earlier note in this file read that as "b1 congested and b2 free", which is
     wrong on both halves.

     **Why it matters for every comparison in this project.** Served trips range from
     156 to 198 across five seeds of identical demand, a spread of 27%. That is where
     the paired standard deviation of 17 to 32 arrivals comes from, and it is a
     property of the operating point rather than of any controller. Comparisons here
     must be paired by seed, and a five-seed comparison can only detect effects above
     roughly 8%.

113. **PRE-REGISTERED: the predecessor paper's formulation, ported.** Written
     2026-09-10 BEFORE the run finishes, and before any of its numbers are seen.
     Ankit's instruction was to read the paper and port its reward; reading it showed
     the gap is not the reward alone.

     **What the paper actually does** (arXiv:2506.11973), against what this project
     built:

     | | the paper | `mappo_sumo` |
     |---|---|---|
     | agent | ONE centralized agent over all segments | ~39 independent AVs |
     | action | max speed for a 2-3 km super-segment | one vehicle's speed bin |
     | speed values | 30/45/60 km/h = 8.3/12.5/16.7 m/s | 20/27/30 m/s |
     | decision interval | 60 s | 1 s |
     | gamma | 0.9 per minute, ~10 min | 0.99 per second, 100 s |
     | observation | per-super-segment density, speed, gap, inflow, outflow | per-vehicle kinematics |
     | reward | `-alpha * 1[rho > rho*] + beta * v`, two terms | eleven weighted terms |
     | network | real highway, Mainz, ramp inflows | symmetric six-branch merge tree |
     | AVs | the compliance mechanism, 25/50/75/100% ablation | independent decision-makers |

     **Its action sets a speed limit over kilometres of road for a whole minute; ours
     set one vehicle's speed for one second.** That is the credit-assignment problem
     and the paper never had it. It also means tasks 105 to 108 do NOT predict this
     run: every one of those measurements was taken with a one-second lever, and what
     changes here is the lever rather than any knob they varied.

     **A CORRECTION to the claim that the new bins bind, made before the result.**
     Measured over 286,352 AV-steps at this operating point, 600 s after the warm-up:
     mean AV speed 4.67 m/s and MEDIAN 0.11 m/s -- over half of AV-steps are on a
     vehicle that is essentially stopped, where no command binds at any value.

     | command | binds on |
     |---|---|
     | 8.33 m/s | 21.4% |
     | 12.50 m/s | 18.6% |
     | 16.67 m/s | 15.2% |
     | 20.00 m/s (the contract's `slow`) | 10.0% |

     So the new bins bind about twice as often in absolute terms, but the band where
     the three values DIFFER FROM EACH OTHER -- `slow` bites and `fast` does not -- is
     21.4 minus 15.2 = 6.2%, against roughly 5% for the old ones. The bin change is a
     real improvement and a smaller one than "8 to 17 m/s is the band congested
     traffic is in" implied. The decision interval and the reward shape are the
     larger changes, and if this run shows something it should not be attributed to
     the bins.

     **The observation carries the reward's own quantity, checked before the run.**
     The threshold penalises a link whose density ratio exceeds 0.3, and the actor
     sees exactly that ratio for the link ahead. Over 1,140 agent-observations at
     this operating point: the spread across agents at one instant is 0.282 on
     average and 0.327 at most, so the field differentiates rather than sitting at a
     constant; 11.2% of agents have a link ahead genuinely over critical; and the
     observed field agrees with the truth on 100% of them, because it is not one of
     the noised fields.

     This rules out one explanation in advance. If the run does not learn, it is not
     because the quantity the reward pays for is invisible to the actor.

     **What `mappo_src` changes**, all of it from the paper: the two-term threshold
     reward at rho* = 0.3 of jam density; absolute speed bins of 8.33/12.5/16.67 m/s;
     one decision a minute with gamma 0.9; and
     `downstream_congestion_estimate` corrected to read the density of the link
     AHEAD, ungated on having an AV peer, so that co-located agents see the same
     thing and can act coherently. Decentralized execution is unchanged: nothing
     reads a global state at run time.

     **THE GATE, the same three criteria as task 103 and with criterion 3 given the
     numeric threshold it lacked.** Read off seed 7's training curve:

     1. summed entropy below 1.978, which is 90% of ln 9 = 2.197;
     2. the score trends up by more than the standard deviation of the change
        between consecutive updates;
     3. **the joint modal action share exceeds 0.20**, against a uniform 1/9 = 0.111
        and against the 0.138 a randomly initialised network already reads. "A clear
        margin" was not a threshold and is the defect recorded as task 109.

     **Then, and only if the gate passes**, the learned arm against `no_av` on
     evaluation seeds disjoint from training: completed trips as the primary metric,
     paired, two standard errors; mean delay, stopped fraction and jerk beside it.

     **The evaluation seeds, pinned now: 57, 67, 77, 87, 97, 107, 117, 127, 137,
     147.** The same ten task 99 used, and disjoint from everything training touches
     -- `collect_rollout` draws `seed + update` for each update and `+ episode_index`
     within it, so a 20-update run from seed 7 consumes 8 through 28. Ten seeds, not
     five, because this operating point is bistable: served trips range 156 to 198
     across five seeds of identical demand (task 114), and five seeds can only
     resolve effects above about 8%.

     **What a null would mean here, stated now.** Two caveats are already on record
     and neither is created after the fact. The paper's rho* = 0.3 barely fires on
     this road -- measured over three seeds, only `tree_middle_b1` crosses it, with
     peak ratios of 0.03 to 0.25 on the leaves, 0.33 on b1, 0.29 on b2 and 0.12 on
     the trunk -- so the reward is effectively "keep the bottleneck below critical
     and go fast elsewhere". And the network is a merge tree with fixed routes to one
     exit, where throughput is set by gap acceptance at the junction, while the paper
     ran a real highway with ramp inflows. A null would therefore not separate "the
     formulation does not transfer" from "the threshold never fired" or from "this
     road has no capacity to recover".

     **The measurement that would settle the first of those** is a critical ratio
     taken from this network's own fundamental diagram instead of imported. A first
     attempt pooled segments and produced a non-monotone curve, because a trunk at
     1890 veh/h and a leaf at 350 do not belong on one axis; it has to be per
     segment. The per-segment half of that run is usable, and it argues that 0.3 is
     about right. Density ratio at which each segment's flow is highest, three
     seeds, 1200 s episodes, no control:

     | segment | ratio at peak flow | peak flow veh/h |
     |---|---|---|
     | `tree_leaf_a1` | 0.325 | 340 |
     | `tree_leaf_a2` | 0.275 | 294 |
     | `tree_leaf_a3` | 0.425 | 349 |
     | `tree_middle_b1` | 0.525 | 687 |
     | `tree_leaf_a4`, `a5`, `a6` | 0.025 | 352-360 |
     | `tree_middle_b2` | 0.075 | 1110 |
     | `tree_trunk_c` | 0.125 | 1890 |

     The last two groups never congest, so their "peak" is just the density they run
     at and says nothing about a critical point. **The segments that DO congest peak
     between 0.275 and 0.525**, so the paper's 0.3 sits at the bottom of the measured
     critical band rather than outside it. For an anticipatory controller that is the
     right side to err on: the penalty starts as the link approaches capacity rather
     than after it has passed it. It is imported rather than measured, and it lands
     in the right place, which is worth stating as two separate facts.

112. **The one arm with a positive reading, on ten seeds: 1.66 standard errors,
     which does not clear the bar and is not nothing.** Measured 2026-09-10 by
     `scripts/measure_full_penetration_signal.py`, settling the loose end task 110
     left open.

     `sumo_saturating` at 100% penetration, ten seeds, the measured gradient norm
     against a 60-permutation floor distribution per seed:

     | seed | 7 | 17 | 27 | 37 | 47 | 57 | 67 | 77 | 87 | 97 |
     |---|---|---|---|---|---|---|---|---|---|---|
     | z | -0.30 | +1.30 | +1.45 | -1.92 | +0.34 | **+2.36** | +1.34 | +1.32 | +0.16 | +0.25 |

     **Mean z +0.630 +/- 0.379, which is 1.66 standard errors.** The project's bar
     is two standard errors, so this is reported as no effect. Stated fully rather
     than only as a verdict, because the shape matters: **eight of ten seeds are
     positive**, one reaches +2.36, and one reaches -1.92.

     **What it does and does not change.** It is the single place in this whole
     investigation where "there is no signal" might be wrong, and it is the arm to
     run more seeds on if the question is reopened. It does NOT reopen the headroom
     question: a learning signal at 100% penetration would say a policy could be
     trained, not that a trained policy would gain anything, and the metering oracle
     with perfect information still serves -0.8 and +0.8 more vehicles than no
     control. The two are separate and both would have to move.

     **It is also not the deployment's operating point.** 100% penetration means
     every vehicle on the road is controlled, against the 25% the predecessor paper
     used and the far lower fraction any deployment would have.

111. **RESULT of the pre-registered run, and it is a null on all three criteria.**
     Read 2026-09-10 by `scripts/read_training_gate.py` on the completed 25 updates
     of seed 7.

     | criterion | bar | measured | verdict |
     |---|---|---|---|
     | 1: summed entropy falls | below 1.9775, which is 90% of 2 ln 3 | lowest 2.1754, first 2.1847, range 0.0147 | **fail** |
     | 2: the score trends up | move exceeds the step-to-step standard deviation | move -0.1091, step sd 0.2186 | **fail** |
     | 3: the action distribution leaves uniform | "a clear margin" -- no threshold, see task 109 | joint modal share 0.1502 against 0.1111 uniform and 0.1378 at initialisation | **fail** |

     Final head probabilities over 300 decisions: `desired_speed_bin` 0.393 / 0.263
     / 0.344 at entropy 1.0854, `desired_headway_bin` 0.313 / 0.305 / 0.382 at
     1.0934, both against a 1.0986 maximum. **Most of the departure from uniform is
     the network's initialisation, which already reads 0.1378.**

     The gate required all three and none is met. The five changes of task 103 are
     reported as a null, and tasks 105 to 110 say why none of them could have
     worked.

110. **A DEFECT IN MY OWN INSTRUMENT, found by Ankit: the floor had no error bar.**
     Every ratio in tasks 105 to 107 divided the measured gradient norm by the norm
     from ONE random permutation of the advantages. One permutation is a single draw
     from the floor's distribution, not the floor. Ankit's question was direct:
     several arms read 10% or more above 1.0, so why are they called floor readings?

     **Measured with 60 permutations per rollout, three seeds, reporting where the
     measured value sits in the floor's own distribution in standard deviations:**

     | arm | floor mean | floor sd | mean z |
     |---|---|---|---|
     | `sumo_capacity_drop`, penetration 0.25 | 0.054 | 0.016 | -0.10 |
     | `sumo_saturating`, penetration 0.25 | 0.088 | 0.025 | -0.01 |
     | `sumo_saturating`, penetration 1.00 | 0.042 | 0.011 | **+0.82** |
     | `sumo_capacity_drop`, gamma 0.999 | 0.052 | 0.013 | +0.33 |

     **The floor's standard deviation is 20 to 30% of its mean.** So the ratios of
     1.12, 1.27 and 1.40 that looked like improvements are inside one standard
     deviation of the floor's sampling noise, and the per-seed z values swing from
     -0.30 to +1.45 on the same arm. The conclusion of tasks 105 to 107 survives,
     but it was not properly supported until now: the correct statistic is the
     z-score against a permutation distribution, not a ratio against one draw.

     **The largest reading is penetration 1.00 at +0.82**, which is under one
     standard deviation on three seeds and is nothing on its own. It is the one arm
     worth more seeds if the question is ever reopened.

     **A NAMING ERROR OF MINE, also from Ankit's question.** "Oracle" was used for
     two unrelated things and they read as contradictory. The **metering oracle** is
     a CONTROLLER: perfect state, hand-written, it drives real vehicles and serves
     -0.8 and +0.8 more of them than no control. The "oracle" in the gradient probes
     drives nothing -- it is a synthetic advantage vector, +1 where the policy chose
     one value and -1 otherwise, injected into the gradient formula to check the
     statistic can move at all. One is about traffic, the other calibrates a
     measuring instrument. It is renamed **ceiling** throughout the scripts.

109. **A defect in my own pre-registration, recorded before the final numbers.**
     Task 103's criterion 3 reads "the modal action's share exceeds 1/9 by a clear
     margin". **"A clear margin" is not a threshold**, so unlike criteria 1 and 2 it
     cannot be evaluated mechanically and could be resolved either way once the
     number was in front of me. That is the failure a pre-registration exists to
     prevent, and I wrote it.

     It is recorded now, with the number already visible at update 23 of 25 and
     before the run finished, so the record shows what the criterion was worth
     rather than how it was applied. The reading at update 23, over 13,531
     observations:

     | head | mean probability | entropy | maximum |
     |---|---|---|---|
     | `desired_speed_bin` | 0.423 / 0.257 / 0.320 | 1.0778 | 1.0986 |
     | `desired_headway_bin` | 0.351 / 0.307 / 0.342 | 1.0970 | 1.0986 |

     Joint modal share 0.1486 against a uniform 0.1111, which is 34% above uniform
     in relative terms and 0.037 in absolute terms.

     **How it should have been written**: a numeric bar, for instance a modal share
     above 0.2, or a summed entropy below the criterion 1 threshold, which would
     have made criterion 3 redundant and revealed that at the time of writing.

     **It does not change this run's verdict**, because criteria 1 and 2 are
     numeric and both fail, and the gate required all three.

     **One observation worth keeping, with its own control.** The head that moved
     is `desired_speed_bin`, toward `slow`, and the head that did not is
     `desired_headway_bin`, which sits at 1.0970 of a 1.0986 maximum. That is the
     opposite of what task 105 would predict: the speed head is the one measured to
     be equivalent across its values on 97% of decisions, and the headway head sets
     `tau`, which affects car following at any speed.

     The control is the same measurement taken at update 1, when the actor had had
     one gradient step and was effectively at its initialisation:

     | update | speed head probabilities | speed head entropy | joint modal share |
     |---|---|---|---|
     | 1 | 0.362 / 0.326 / 0.312 | 1.0966 | 0.1378 |
     | 23 | 0.423 / 0.257 / 0.320 | 1.0778 | 0.1486 |

     **A randomly initialised network is already 0.1378 rather than 0.1111**, so
     most of the departure from uniform at update 23 is initialisation and not
     learning. What moved over 23 updates is the speed head's modal share, from
     0.362 to 0.423, and its entropy by 0.0188 of a 1.0986 range -- 1.7%. That is
     movement in the direction speed metering would need, at a rate consistent with
     the roughly 1% action-advantage correlation of task 107, and it is far too
     small to clear any of the three criteria.

108. **The control on tasks 105 to 107: a trained actor reads the same as a fresh
     one.** Measured 2026-09-10 by `scripts/measure_trained_actor_signal.py`.

     Every gradient reading in tasks 105 to 107 was taken at a randomly initialised
     actor. If the correlation between the advantage and the action rises as the
     policy trains, those readings describe a starting condition rather than the
     problem, and the conclusion drawn from them is wrong. The seed-7 run had
     reached 23 updates, so its checkpoint answers this directly. Same environment,
     same seeds, same reward, same floor and ceiling:

     | actor | measured over floor | per seed | oracle over floor | implied correlation |
     |---|---|---|---|---|
     | fresh initialisation | 0.848 | 0.608, 1.330, 0.607 | 26.2 | -0.0060 |
     | 23 updates of training | 0.812 | 0.465, 0.640, 1.332 | 34.0 | -0.0057 |

     **Indistinguishable.** The measurement is about the environment and the reward,
     not about where the policy happens to start, so tasks 105 to 107 stand.

     **A defect in this script, caught by the failure rather than by the result.**
     The first version read the checkpoint path from `sys.argv` AFTER replacing
     `sys.argv` for the config loader, so the path was None. It raised rather than
     loading nothing silently, which is the only reason it did not report a fresh
     actor twice under two labels and call them the same.

107. **Penetration does not produce a learning signal either, up to 100%.**
     Measured 2026-09-10 by `scripts/measure_penetration_signal.py`, closing the
     cheapest branch of task 106's option 2. Two demands, three penetrations, three
     seeds each, shipped bins, with the same floor and ceiling controls. The last
     column is the implied correlation between the advantage and the action,
     `(measured/floor - 1) / (oracle/floor - 1)`:

     | demand | penetration | AVs present | measured over floor | oracle over floor | implied correlation |
     |---|---|---|---|---|---|
     | `sumo_saturating` | 0.25 | 12.4 | 1.055 | 19.1 | +0.0030 |
     | `sumo_saturating` | 0.50 | 24.7 | 0.869 | 27.0 | -0.0050 |
     | `sumo_saturating` | 1.00 | 50.3 | 1.403 | 45.1 | +0.0091 |
     | `sumo_capacity_drop` | 0.25 | 42.4 | 0.848 | 26.2 | -0.0060 |
     | `sumo_capacity_drop` | 0.50 | 86.5 | 0.589 | 35.1 | -0.0121 |
     | `sumo_capacity_drop` | 1.00 | 168.2 | 0.960 | 80.2 | -0.0005 |

     **Every value is within 0.012 of zero and the sign is random.** At 100%
     penetration the policy commands the entire fleet -- 168 vehicles at 2400 veh/h,
     50 at 1200 -- and one agent's action still does not correlate with its own
     advantage. That is multi-agent credit assignment in its pure form: the other
     167 agents are exploring at the same time, and their contribution to the return
     swamps the one being credited.

     **The complete list of what has now been measured against the floor.** Every
     row uses the shuffled advantage as its floor and an action-correlated advantage
     as its ceiling, three seeds each:

     | varied | range | best measured over floor |
     |---|---|---|
     | reward decomposition | team, neighbourhood, own-vehicle, both | 0.850 |
     | action hold length | 1 s, 5 s, 20 s | 0.945 |
     | discount horizon | 10 to 1000 decisions | within 1.1x |
     | critic input | with and without privileged neighbourhood | within 1.05x |
     | speed bin scaling | four schemes, binding share 6% to 29% | 1.020 |
     | operating point | 900, 1200, 2400 veh/h | 1.269 |
     | AV penetration | 0.25, 0.50, 1.00 | 1.403 |

     Nothing clears the floor. The oracle reads 14 to 80 times it throughout, so the
     instrument was capable of a positive reading in every one of those arms.

     **WHAT IS AND IS NOT ESTABLISHED, stated precisely.** The metering oracle of
     task 102 is a specific heuristic given perfect state, so it is a lower bound on
     what the best controller could do and NOT an upper bound: "no controller can
     gain here" is not proven and cannot be proven this way. What is established is
     narrower and still decisive for the plan: **the mechanism this project posits --
     in-stream AV speed modulation -- neither gains when given perfect information
     nor presents a learnable gradient, on this road, at every operating point and
     penetration tried.**

     **The recommendation, now on evidence.** Report the simulation leg as a null,
     which is task 106's option 1. The reasoning is the order of the two problems:
     a counterfactual advantage estimator (option 3, COMA-style) is precisely
     targeted at the measurement above and would probably raise the correlation, but
     it fixes the learning of a mechanism that gains nothing when handed perfect
     information. Spending a research change on a better estimator is worth it only
     after an oracle shows headroom for some controller to reach, and the cheap way
     to look for that headroom is more oracles, not more training. The one positive
     figure on record remains +7.6 +/- 15.2 arrivals at congestion onset, which is
     one standard error.

106. **RETRACTION OF MY OWN RECOMMENDATION: rescaling the speed bins does not
     help, and neither does the operating point.** Measured 2026-09-10, minutes
     after task 105 recommended the rescale. The recommendation was an argument from
     a mechanism; this is the measurement, and it contradicts it.

     **The prediction that failed.** Task 105 established that the three speed values
     are equivalent on 97% of decisions and inferred that a rescaling which binds
     would produce a gradient. Four arms, shipped bins against three rescalings,
     three seeds each, with the same floor and ceiling controls:

     | speed bins | measured over floor | decisions the command binds on |
     |---|---|---|
     | shipped: 20 / 27 / 30 m/s | 0.848 | 6.0% |
     | prevailing segment speed as the context, no 12 m/s floor | **0.727** | 19.1% |
     | 0.60 / 0.85 / 1.00 of the vehicle's own speed | **1.020** | 29.2% |
     | 0.50 / 0.75 / 1.00 of the vehicle's own speed | **0.415** | 20.3% |

     **The binding share rose fivefold and the gradient stayed at its floor.** The
     recommended option 1 is the worst of the three rescalings. So inertness was a
     true description of the action and not the cause of the missing gradient.

     **The operating point does not do it either.** Shipped bins, demand varied:

     | demand | veh/h | AVs present | mean speed | measured over floor | oracle over floor |
     |---|---|---|---|---|---|
     | `sumo_burst` | 900 | 5.6 | 21.04 | 1.018 | 14.0 |
     | `sumo_saturating` | 1200 | 9.9 | 13.29 | 1.269 | 20.8 |
     | `sumo_capacity_drop` | 2400 | 42.4 | 4.70 | 0.848 | 26.2 |

     At 900 veh/h the traffic holds 21.0 m/s, so the shipped bins DO bind, and there
     are 5.6 agents rather than 42, so one agent is eight times as large a share of
     the fleet. The ratio is 1.018. The per-seed spread is 0.80 to 1.68 and nothing
     clears the floor.

     **What the ratio implies about the correlation.** With the advantage written as
     a correlation `rho` with the action plus independent noise, the measured ratio
     is about `1 + rho * (oracle_ratio - 1)`. At an oracle ratio of 14 to 39, a
     measured ratio of 1.02 to 1.27 puts **rho at or below about 0.01**, and the
     instrument would see rho of 0.05.

     **The convergence that matters.** Two instruments built for different purposes
     now agree. The perfect-information metering oracle measured -0.8 and +0.8
     arrivals against no control on this road (task 102), which is indistinguishable
     from doing nothing. The policy-gradient signal-to-noise puts the correlation
     between one AV's speed choice and its own advantage at about 1%. **A single
     AV's speed choice on this network at these penetrations changes almost
     nothing, and that is why no policy learns: there is very little to find.** The
     oracle says it from perfect information and no learning; the gradient says it
     from the learning signal and no oracle.

     **The revised decision for the user, replacing the four options in task 105.**
     The speed-bin rescale is off the table -- measured, not argued. What is left:

     1. **Report the simulation leg as a null**, with two independent lines of
        evidence rather than one, and the deployment carrying the feasibility claim
        as it already does.
     2. **Change what the AVs can do**, not how their reward is priced: raise
        penetration well above 25%, coordinate them as platoons rather than
        independently, or give the junction an explicit meter rather than relying on
        in-stream vehicles. Penetration is the cheapest of these and is being
        measured now with the same instrument, at 0.25, 0.5 and 1.0 on two demands.
     3. **Change the advantage estimator** to a counterfactual one -- a COMA-style
        baseline that marginalises the agent's own action -- which is the standard
        answer to exactly this measurement and is a research change rather than a
        configuration one.

     No recommendation yet: the penetration measurement is in flight and it bears
     directly on option 2. Recording the retraction now rather than after it, so the
     failed recommendation is on the record in its own right.

105. **THE ADVANTAGE CARRIES NO ACTION SIGNAL, and the speed bins are why.**
     Measured 2026-09-10 with `scripts/measure_gradient_signal.py`. This is the
     answer to why tasks 96 and 101 produced no learning, and it retracts four
     comparisons made earlier the same day.

     **The instrument, and the two controls it needed.** With advantages normalised
     to unit standard deviation, the norm of `d(policy_loss)/d(actor)` measures how
     much the advantage CORRELATES with the action: a term uncorrelated with the
     action cancels across the batch and the norm falls as 1/sqrt(N), while an
     aligned one adds. `policy_loss` itself says nothing -- at a probability ratio
     of 1 it is minus the mean normalised advantage, which is zero by construction
     however informative the advantage is, which is why task 96 could not settle
     this from the loss.

     The floor is the same advantages permuted across the batch: identical
     distribution, no correlation with the action. The ceiling is an advantage built
     to correlate with the action, +1 where the policy chose `slow` and -1
     otherwise. Three seeds, 17,296 decisions on the shipped configuration:

     | advantage | gradient norm | over the floor |
     |---|---|---|
     | as measured | 0.0499 | **0.784** |
     | shuffled (the floor) | 0.0637 | 1.000 |
     | action-correlated (the ceiling) | 1.5971 | **25.1** |

     **The advantage sits at its own noise floor and the instrument can read
     twenty-five times it.** PPO therefore has nothing to ascend, which is exactly
     what a policy frozen at 99% of maximum entropy looks like.

     **Every arm tried is at the floor.** Same instrument, same controls:

     | arm | measured over floor |
     |---|---|
     | team reward alone | 0.784 |
     | per-agent neighbourhood reward, blended 0.5 | 0.850 |
     | per-agent own-vehicle reward | 0.750 |
     | both together | 0.696 |
     | action held 1 s / 5 s / 20 s | 0.784 / 0.927 / 0.945 |
     | discount horizon 10 to 1000 decisions | within a factor of 1.1 |

     **FOUR RESULTS RETRACTED, all from the same session.** Before the floor existed
     I reported that the per-agent reward lowered the policy gradient (a paired ratio
     of 0.962), that the discount horizon does not matter, that privileged critic
     features do not help the gradient, and that a longer action hold does not help.
     Each was a comparison between two floors and none of them carried information.
     The paired construction was sound -- identical trajectories, verified by
     transition count and action sums -- and it was measuring a quantity that could
     not move.

     **What still stands from that work.** The critic regression, which is a
     different statistic: giving the centralized critic the agent's own
     neighbourhood raises out-of-sample R2 on the per-agent return from 0.808 to
     0.886, and leaves it at 0.854 against 0.858 under the team reward, which is the
     control that says the improvement is about the per-agent term. And the reward
     counterfactual of task 104, which is about the reward and not the advantage:
     one agent's action moves its own neighbourhood reward 17.8 times what its
     neighbours' actions move it. Both are real; neither reaches the gradient.

     **THE CAUSE. The action is inert on 97% of decisions.** `decode_speed_bin`
     returns `free_flow + offset` for offsets of -10, -3 and 0 m/s with a 12 m/s
     floor, so at a 30 m/s limit the three values are 20, 27 and 30 m/s.
     `setSpeed` is an upper bound that SUMO's car-following dominates, so on a
     vehicle already slower than the commanded value all three do the same thing. Of
     the 17,296 decisions above, **511 -- 3.0% -- were taken on a vehicle moving fast
     enough for the command to bind.** On that subset the ratio is 1.605, but at
     about 170 decisions per seed the per-seed values are 0.416, 3.811 and 0.588,
     which is too few to conclude. The 15.6% figure in task 104 is over AV-STEPS; 3%
     is over AV-DECISIONS, which is the population the gradient is built from.

     **The seed-7 run of task 103 is reported as a null in advance of finishing.**
     Nine of 25 updates at the time of writing: entropy 2.1847, 2.1783, 2.1811,
     2.1875, 2.1835, 2.1821, 2.1832, 2.1854, 2.1869 against a maximum of 2.1972 --
     flat to within 0.009, which is 0.4% of the range -- and the score random-walking
     between -0.254 and +0.119. It was left running rather than killed, so the
     pre-registered gate is read on the run as specified rather than on a run
     stopped when its numbers were disliked.

     **What this does NOT say.** It does not say a decentralised policy cannot help
     here. It says that with an action whose three values are indistinguishable on 97%
     of the decisions taken, no reward decomposition, discount horizon, critic input
     or hold length can produce a gradient, and none of the changes in task 103 could
     have worked. The environment question -- whether there is headroom at this
     operating point -- is still open and separate.

     **THE DECISION THIS NEEDS, and it is the user's.** Rescaling the speed bins
     changes what `desired_speed_bin` means on both sides of the deployed contract:
     the phone executes it and the Jetson logs it. Task 86 recorded that as the
     user's call and it now blocks the simulation leg. The options, with what each
     costs:

     1. **Keep the offsets, lower the floor, and make the context the local
        conditions rather than the lane limit.** `decode_speed_bin` already takes a
        `free_flow_speed_mps` argument; the SUMO env passes the lane limit. Passing
        the segment's prevailing speed and dropping the 12 m/s floor makes `slow`
        bind wherever the vehicle is moving. The head keeps its three names and its
        meaning becomes relative rather than absolute.
     2. **Rescale the offsets to a congested range**, for instance multiplicative
        0.6 / 0.85 / 1.0 of the vehicle's current achievable speed. Same effect,
        expressed on the action rather than on the context.
     3. **Leave the contract alone and change the operating point** so the traffic
        runs near 20 m/s, where the existing bins bind. That means a demand below
        the capacity collapse, which is the regime where the metering oracle
        measured +7.6 +/- 15.2 -- the one positive figure on record.
     4. **Leave everything and report the simulation leg as a null**, with this
        measurement as the reason.

     Recommendation: option 1. It is the smallest change that makes the head
     functional, it is what the argument name already says the parameter is, and it
     leaves the three action names -- which is what the deployed executor and the
     logs carry -- untouched.

104. **The credit-assignment diagnosis is now measured, and the speed head is
     inert on 84% of decisions.** Measured 2026-09-10 by
     `scripts/measure_credit_signal.py` while the task 103 run was in flight. Both
     halves were found by the same measurement.

     **The local reward carries the signal the team reward does not.** The
     measurement is a counterfactual on identical traffic: run the warm-up, hold one
     AV at 20 m/s for a 20 s window, repeat from the same seed holding it at 30 m/s,
     and record EVERY tracked agent's reward in both. Three seeds, three agents each,
     nine own and eighteen cross pairs:

     | quantity | value |
     |---|---|
     | agent i's local reward moved by agent i's own action | 22.82 |
     | agent i's local reward moved by another agent's action | 1.28 |
     | ratio, own over other | **17.8** |
     | team reward moved by one agent's action | 2.06 |
     | team reward over the same window, uncommanded | 193.03 |

     So one agent's action moves the team reward by **1.07%** of its magnitude, and
     moves its own neighbourhood reward by 17.8 times what its neighbours' actions
     move it. Task 96 diagnosed this from the policy loss; this measures it directly,
     and it is the justification for the per-agent term.

     **The control that had to come first.** Two uncommanded runs at the same seed
     differ by 0.000e+00 in the summed team reward. Without that, every difference
     above would be unattributable rather than small.

     **THE SPEED HEAD IS EQUIVALENT ACROSS ITS VALUES ON MOST DECISIONS.**
     `decode_speed_bin` returns `free_flow + offset` with offsets of -10, -3 and 0
     m/s and a floor of 12 m/s, and the SUMO env passes the lane limit as the
     context, so at a 30 m/s limit the three values are **20, 27 and 30 m/s**.
     `setSpeed` is an upper bound that SUMO's car-following then dominates, so a
     value above the speed the vehicle would take anyway changes nothing. Measured
     over 114,889 AV-steps at this operating point, mean AV speed 7.33 m/s:

     | value | m/s | share of AV-steps where it binds |
     |---|---|---|
     | `slow` | 20 | **15.6%** |
     | `nominal` | 27 | 0.9% |
     | `fast` | 30 | 0.07% |
     | (stopped, below 0.1 m/s) | | 25.6% |

     **On 84% of AV-steps all three values do the same thing, and on 99% `nominal`
     and `fast` do the same thing.** This is how the first version of the credit
     measurement was caught: it selected the three lowest-numbered agent ids, which
     after a 300 s warm-up are the oldest vehicles and so the deepest in the queue,
     and every commanded speed from 0.5 to 30 m/s produced a bit-identical
     trajectory. Exact zeros, not small numbers.

     **What this does and does not mean.** It is NOT that the head is unwired -- that
     was task 86 and it is fixed. The 15.6% of AV-steps where `slow` binds are the
     vehicles still moving fast as they approach the queue, which is exactly where
     speed metering has to act, so the mechanism the project studies IS expressible.
     What is lost is the other 84%, where the choice cannot matter, and those
     decisions put pure noise into the gradient. A vehicle stopped in a queue cannot
     help by any speed command, so part of that 84% is the operating point rather
     than the action space.

     **Consequence for task 103's gate, re-derived rather than moved.** Criterion 1
     asked for summed entropy below 1.978, which is 90% of ln(9). With the speed head
     equivalent across its values on 84% of decisions, an optimal policy is
     indifferent there, so the achievable mean summed entropy is about
     0.84 x ln(3) + a deterministic headway head, which is roughly 0.92. The gate
     threshold is therefore still reachable and is NOT changed. The confound is that
     a FAILURE of criterion 1 would be ambiguous between "the policy did not learn"
     and "most of its decisions had nothing to choose between".

     **The run was not restarted.** Rescaling the speed bins changes what the
     deployed actor's heads mean, on both sides of the contract, and task 86 already
     recorded that as the user's decision rather than mine. The run in flight is
     still informative under the reading above, and it costs no human time.

103. **PRE-REGISTERED: one configuration, everything enabled, one seed.** Written
     2026-09-10 BEFORE the run. The plan is `plans/plan_task_103_local_credit.md`.

     **What is being tested.** Whether a decentralised policy learns anything on
     this environment when every change that attacks the diagnosed cause is enabled
     at once. NOT how large its effect is: one seed cannot detect a 5% effect, since
     the paired standard error across seeds is 5 to 10%.

     **The five changes, deliberately confounded.** A per-agent reward measured over
     the agent's own segment and the segments downstream of it, blended half and
     half with the team reward; one decision per simulated second, with the action
     held for the ten simulation steps in between; the reward re-weighted to price
     delay, stopping and jerk; AV penetration 0.25; `deployed_fidelity` false.
     Ablations come after something works.

     **A sixth change that is not a treatment.** `sumo_burst` is replaced by
     `sumo_capacity_drop` at 2400 veh/h. Measured with no control at dt 0.1 over a
     900 s episode after a 300 s warm-up, in 60 s buckets: at 3000 veh/h the warm-up
     alone gridlocks the road, so the episode opens at 3.0 m/s with 70% of vehicles
     below 0.1 m/s and every minute of it is post-collapse; at 2400 the episode opens
     at 6.6 m/s and 33 completions per minute, collapses between minutes 5 and 6,
     and settles near 2.3 m/s. The episode is shortened to 600 s to sit around that
     collapse.

     **Latent demand does not engage on this road, and that is a finding.** Pending
     vehicles measured 0.0 at 2400 veh/h and 0.45 at 3000. Two lanes on every
     approach absorb the excess onto the carriageway rather than refusing entry, so
     the queue forms on the road. It reaches 181 only at 4500 veh/h, where the
     network is gridlocked from the first observed step. The saturated design's
     "served plus on-road plus latent" accounting therefore cannot separate the arms
     here; completed trips and delay have to.

     **THE GATE, fixed now.** Read off the training curve of seed 7, and nothing
     else:

     1. **Entropy falls** below 90% of its maximum. The maximum is ln(9) = 2.197 for
        the 9-action `speed_headway` profile; 90% is 1.978. The 4.38 figures from
        tasks 96 and 101 were against ln(81) = 4.394 and are NOT comparable.
     2. **The score trends up** by more than the variation between consecutive
        updates. The score is not comparable across configurations -- the local term
        changes its magnitude -- so only its trend within this run counts.
     3. **The action distribution leaves uniform**: the modal action's share exceeds
        1/9 by a clear margin.

     Passing all three earns a five-seed run, reported on evaluation seeds disjoint
     from the training seeds, exactly as task 99 required. Failing means the next
     iteration is on the reward, not on more seeds and not on more environment
     changes.

     **What a null would mean, stated now.** The metering oracle on this road at
     `sumo_oversaturated` measured -0.8 and +0.8 against no control, which is
     indistinguishable from doing nothing (task 102), and the only positive figure
     in that whole progression is +7.6 +/- 15.2 at congestion onset, which is one
     standard error. So a null here does not separate "a decentralised policy cannot
     learn this" from "there is nothing at this operating point to learn". It would
     be reported as the first, qualified by the second.

     **The metrics the five-seed run would report, also fixed now.** Primary:
     completed trips per episode, paired against `no_av` on the same traffic seeds,
     with the bar at two standard errors. Secondary: mean delay of completed trips,
     stopped fraction, mean absolute jerk, the trough of the 60 s rolling mean speed,
     summed `rolling_roadblock_score`, and collisions, which must be zero. Delay,
     stopping and jerk did not exist as measurements before this task; the first
     three are the axes the predecessor paper's gains were largest on.

     **One reading guard on the delay metric.** `mean_delay_recent` is a mean over
     COMPLETED trips, so a controller that stops a vehicle from completing improves
     it. That is why completed trips is the primary metric and why `latent_demand`
     and `throughput_recent` are reported beside it.

102. **Two lanes everywhere, and the oracle progression that made the case.**
     Decided by the user 2026-09-10; the measurements that led there are below,
     and they are a sequence of removed harms rather than a found benefit.

     The metering oracle -- perfect state, the project's own backpressure
     mechanism, no learning -- against no control, in the order the setup was
     corrected:

     | setup | metering vs no control |
     |---|---|
     | permanent yield at the merge, Krauss fleet | −16.6 +/- 7.8 |
     | zipper merge, calibrated W99, at congestion onset | **+7.6 +/- 15.2** |
     | zipper merge, W99, at the capacity peak (2100 veh/h) | −21 to −38 |
     | zipper merge, W99, past the peak (2400 veh/h) | −4 to −30 |
     | oversaturated, single-lane leaves | −10 to −32, latent unchanged |
     | **oversaturated, two lanes everywhere** | **−0.8 and +0.8** |

     **Each correction has removed harm and none has produced benefit.** The final
     row is the important one: with two lanes everywhere, metering is
     indistinguishable from doing nothing (−0.8 and +0.8 against a standard
     deviation of about 16), where on single-lane leaves it cost 10 to 32 vehicles.

     **Why the lane count mattered.** On a single-lane approach a slow vehicle
     cannot be overtaken, so an in-stream AV that slows is a rolling roadblock and
     not a meter. Ramp metering works because the meter sits beside the road; an AV
     is in it. The six entry leaves were one lane, so the mechanism the project
     exists to study was structurally unavailable on the segments where it would
     have to act. Both topologies now have two lanes on every approach; the
     bottleneck variant keeps its single-lane drop, which is intended rather than
     accidental.

     **The saturated design, and what it measures.** `sumo_oversaturated` holds
     3000 veh/h against a peak served flow near 1300, so the accounting closes:
     every scheduled vehicle is served, on the road, or latent. A controller must
     raise served and lower latent, and moving vehicles from the road into the queue
     is visible as one improving while the other worsens. Measured with no AVs at
     3000 veh/h on single-lane leaves: served settles near 1150 veh/h, the road
     holds about 480 vehicles, and latent grows about 0.44 vehicles per second once
     the leaves saturate. With two lanes the road absorbs far more, so at the same
     1200 s warm-up latent is 61 rather than 506 -- the two lane counts are NOT at
     the same point in their saturation trajectory, and comparisons across them are
     not like for like. Comparisons within a lane count are.

     **Three measurement defects found and fixed while doing this**, all in my own
     scripts rather than the simulator: the runs were sequential on one core of ten;
     a shared working directory had parallel runs overwriting each other's network
     file mid-read; and `ProcessPoolExecutor` hangs because libsumo keeps the
     simulation in module-level state and a forked child inherits a copy of it. The
     measurement now runs one independent subprocess per seed and takes control
     decisions at 1 Hz rather than 10 Hz, which is also the more realistic rate.

101. **RESULT on the corrected road: the environment is now right and the
     learning is not.** Run 2026-09-10 per the amended pre-registration (task 99):
     zipper merges, the calibrated Wiedemann-99 fleet, `sumo_burst`, 900 s episodes
     at dt 0.1, 50 updates of three episodes each, five policies, evaluated on ten
     traffic seeds disjoint from training. Rows in
     `plans/result_task99_corrected_road.json`.

     | arm | arrivals | trough m/s | recovery s | roadblock | collisions |
     |---|---|---|---|---|---|
     | `no_av` | **249.5 +/- 10.3** | 11.22 | 70.2 | 10.4 | 0 |
     | `density_lookup` | 248.0 +/- 14.1 | 12.15 | 82.0 | 22.6 | 0 |
     | `mappo` | 236.3 +/- 8.9 | 9.46 | 83.6 | 338.5 | 3.6 |

     Paired against `no_av` on the same traffic:

     | arm | metric | difference | verdict |
     |---|---|---|---|
     | `density_lookup` | arrivals | **−1.50 +/- 5.17** | **no effect** |
     | `density_lookup` | trough speed | +0.93 +/- 1.72 | no effect |
     | `density_lookup` | recovery | +4.03 +/- 87.39 | no effect |
     | `density_lookup` | roadblock | +12.16 +/- 4.70 | real |
     | `mappo` | arrivals | **−13.22 +/- 3.51** | **real** |
     | `mappo` | trough speed | −1.76 +/- 0.93 | no effect |
     | `mappo` | recovery | +13.38 +/- 44.80 | no effect |
     | `mappo` | roadblock | +328.01 +/- 25.62 | real |

     **The road fix moved the non-learning baseline from harmful to neutral.**
     `density_lookup` was −17.40 +/- 7.97 arrivals on the network with the permanent
     yield and is −1.50 +/- 5.17 now: indistinguishable from doing nothing. That is
     the clearest evidence that the earlier failures were the environment. A local
     density-and-queue heuristic can now act without paying for it.

     **MAPPO is still worse than doing nothing, and now more clearly so.** −13.22
     +/- 3.51 arrivals, a 5.3% reduction that clears the bar comfortably at ten
     seeds. It also causes 3.6 collisions against zero for both baselines, which is
     possible at all only because the calibrated model can collide (task 100).

     **Because the policy still did not learn.** Over 50 updates the score rose from
     1.636 to about 1.71 and plateaued, and entropy ended at 4.3822 against a
     maximum of 4.394 -- 99.7% of maximum, essentially uniform. Tripling the
     episodes per update improved the gradient enough to show a trend where the
     previous run had none, and not enough to move the policy. So this measures a
     near-uniform policy over the action space, and a near-uniform policy on this
     action space slows AVs at random, holds lanes (roadblock 338.5 against 10.4)
     and crashes occasionally.

     **What is now isolated.** The environment supports the phenomenon: the
     fundamental diagram has a 24% capacity drop, a perfect-information oracle
     scores positive at the operating point (+7.6 +/- 15.2), and the non-learning
     baseline is neutral rather than harmed. What remains is the learning problem,
     and it is quantified rather than guessed: a team reward shared among about
     twelve agents moves less than its own noise under one agent's action, and even
     at three episodes per update this is 1/17th of the per-update trajectory count
     Flow's benchmarks use, at 1/10th of their iteration count.

     **The next thing to try, and it is a decision.** Three options, in increasing
     order of departure from the current design: raise the trajectory count per
     update towards Flow's 50, which is a pure compute cost; give each agent a
     reward component it can move on its own, which changes the objective; or adopt
     Flow's formulation of one policy emitting all AVs' actions jointly, which
     abandons decentralised execution and so the project's premise. The first is the
     only one that does not change what is being claimed.

100. **The calibrated driving model gives up the collision-free guarantee, and
     that is the same trade in both directions.** Measured 2026-09-10 at 1200 veh/h
     with no AVs, three 600 s runs: SUMO's default Krauss produces **0** collisions
     and the calibrated Wiedemann-99 produces **2**.

     This is not a defect in either model. Krauss computes a collision-free safe
     speed exactly and recovers from a disturbance immediately, which is why task 84
     chose SUMO over highway_env in the first place -- "SUMO's car-following cannot
     produce a collision" was the premise of the whole migration. The same property
     is why it has no capacity drop and nothing for a controller to recover: a model
     that never over-brakes cannot produce the stop-and-go instability the
     controller exists to damp. W99's psycho-physical thresholds do over-brake, on
     purpose (CC2 and CC6 "help introduce stop-and-go dynamics"), and the price is
     that a collision becomes possible.

     **Three consequences, recorded rather than resolved.**

     - Every "zero collisions" claim in this project is now conditional on the
       driving model. The claims stand for Krauss; under W99 the rate is small but
       not zero. `TestNoCollisionsEver` and the action-head collision guard both
       exercise the Krauss path, so they still pass and now cover less than their
       names suggest.
     - `crash_penalty` in the reward is no longer inert on SUMO. It was recorded as
       structurally absent because `crashed_agent_ids()` returns nothing and SUMO
       could not collide; under W99 the collision counter does move, so the -5.0
       collision weight can now charge a policy.
     - The safety layer matters again. It was reasonable to leave
       `apply_safety_layer` unrun on SUMO while the simulator's own model was the
       guarantee. It is not reasonable under W99, and whether to run it is a
       decision that now has consequences.

     The honest framing for the paper is that the simulator offers a choice between
     a fleet that cannot crash and a fleet that can congest, and the phenomenon
     under study requires the second.

99. **AMENDED PRE-REGISTRATION for the re-run on the corrected road.** Written
    2026-09-10, before training, superseding task 93. Amended for two reasons that
    are both about the instrument and neither about a result: the road and the
    driving model changed (task 98 and the W99 calibration), and the variance
    changed with them.

    **What changed in the setup.** Merge nodes are zipper junctions. The fleet uses
    the predecessor paper's calibrated Wiedemann-99 parameters, so the fundamental
    diagram has a capacity drop: served flow peaks at 1298 veh/h and falls 24%,
    where the old road rose monotonically to 1110 with no drop at all.
    `sumo_saturating` is 1200 veh/h, the congestion onset. `sumo_burst` keeps a
    sustainable 900 base doubled for 150 s, re-profiled as speed 23.0 to 10.2 m/s
    with the queue peaking at 23 and recovering to 19.5.

    **The evaluation seeds, named before the run: 57, 67, 77, 87, 97, 107, 117,
    127, 137, 147.** All ten are disjoint from the five the policies trained on
    (7, 17, 27, 37, 47). The trainer seeds each episode as `seed + update`, so a
    shared base seed does not reproduce a training episode exactly, but evaluating
    on unseen traffic realisations removes the question.

    **Seeds: 10 for the evaluation, not 5.** The oracle's paired standard deviation
    at this operating point is 15.2 arrivals against 7.8 on the old road, because
    the congestion is now genuinely stochastic. At five seeds the two-standard-error
    bar admits only effects above 13.6 arrivals, or 5.4%; at ten it is 9.5, or 3.8%.
    Training stays at five seeds because it costs hours where evaluation costs
    minutes, and the evaluation is where the power is needed.

    **Training budget: three episodes per update, 50 updates.** The previous run's
    policy never left its initialisation, and the diagnosis was gradient noise
    rather than step count: a team reward shared among twelve agents moves less than
    its own noise under one agent's action, and one episode per update is 1/50th of
    the gradient quality Flow's benchmarks use. Three episodes per update at 50
    updates costs the same wall clock as the previous 100 single-episode updates.

    **Everything else is unchanged from task 93** and is what will be reported:
    completed trips as the primary metric; trough speed, recovery time, roadblock
    sum and collisions as secondaries; `no_av` and `density_lookup` as comparators;
    the final checkpoint rather than the best; a two-standard-error bar paired on
    seed; no commanded-speed arm as a comparator; and a null reported as a null.

    **Recorded before the run, so it cannot be claimed afterwards:** the oracle on
    the corrected road at this operating point scores +7.6 +/- 15.2, which is about
    one standard error. The honest prior is that any effect available here is small
    enough that ten seeds may still not resolve it.

98. **Why nothing works: the network has an unintended permanent yield, and it
    sets capacity.** Asked 2026-09-10 why MAPPO fails here when Flow reports gains
    on similar networks, and whether the sensing model is the cause. It is not the
    sensing model. `inverted_tree` as generated is not the network it was meant to
    be.

    **The evidence, in the order it was found.**

    A perfect-information oracle also fails. `scripts/measure_oracle_metering.py`
    slows AVs only while the segment downstream of them is congested -- the
    project's own declared backpressure metering -- reading true simulator state
    rather than the sensing model, with no learning involved. Five seeds on
    `sumo_burst`: `no_av` 259.8 +/- 7.2 arrivals, metering at 8 m/s
    −16.6 +/- 7.8, at 12 m/s −6.4 +/- 4.4, at 16 m/s −1.2 +/- 2.6. Nothing beats
    inaction and the gentlest intervention is merely the least harmful. Whatever is
    wrong is upstream of both sensing and learning.

    Neither topology has a capacity drop at dt 0.1. Served flow rises
    monotonically with demand -- 890, 908, 950, 1100, 1110 veh/h on `inverted_tree`
    at 900 to 2400 veh/h offered, and 802 to 1068 on the bottleneck variant. A
    capacity drop, throughput FALLING past a critical point, is the inefficiency
    every mixed-autonomy control result exploits; Flow's bottleneck benchmark has
    one, and its paper describes the opportunity as arranging vehicles "so that
    they merge optimally without the sharp decelerations that eventually give rise
    to the bottleneck".

    **Then the segment profile gave it away.** At 1800 veh/h with no AVs:

    | segment | lanes | mean speed | queue | vehicles |
    |---|---|---|---|---|
    | `tree_middle_b1` | 2 | **1.74** | **64.4** | 69.3 |
    | `tree_middle_b2` | 2 | 22.76 | 0.0 | 5.9 |
    | `tree_trunk_c` | 2 | 22.87 | 0.0 | 12.1 |

    `b1` is jammed solid between neighbours that are both free-flowing, and the
    trunk it feeds is nearly empty. That is not congestion physics.

    **The cause is in the generated network.** Every edge is written with
    `priority="-1"`, so `netconvert` broke the tie at junction `c` by geometry. The
    built connections read `state="M"` for `tree_middle_b2` into the trunk and
    `state="m"` for `tree_middle_b1`: b1 yields permanently, so its three leaves
    (a1, a2, a3) starve behind a yield that never clears while the trunk runs
    empty.

    **Confirmed by fixing it.** Rebuilding the merge nodes as SUMO `zipper`
    junctions, which alternate between approaches, no AVs, three seeds, arrivals
    per 600 s:

    | veh/h | as built | zipper merge |
    |---|---|---|
    | 900 | 148.3 | 148.0 |
    | 1500 | 158.3 | **252.3** |
    | 1800 | 183.3 | **252.0** |
    | 2400 | 185.0 | **268.0** |

    Capacity goes from about 1110 veh/h to about 1600, and the two middles
    symmetrise: b1 rises from 1.02 to 11.95 m/s while b2 falls from 22.87 to 11.26,
    with the starved leaves clearing (a5 from 4.70 to 23.63). At 900 veh/h nothing
    changes, because that is below capacity either way.

    **What this invalidates.** Every capacity figure in this project measured the
    yield rather than the road, and both demand configs were chosen against it. It
    also means **branch fairness -- the objective `inverted_tree` exists to study --
    was structurally unattainable**: b1's three branches could never be served
    equally with b2's, whatever a controller did. And it explains why every
    controller tried so far does harm: the constraint is right-of-way, and slowing
    vehicles that are already yield-limited only reduces what arrives.

    **On Flow specifically, four differences beyond this one**, checked against
    Vinitsky et al. 2018 rather than recalled: its benchmarks are the figure-eight,
    merge, grid and bottleneck -- there is no tree; they use a single centralized
    policy emitting all AVs' continuous accelerations, not decentralized agents
    sharing one reward, so the policy's action moves the reward substantially; they
    train 500 iterations of 50 rollouts each, against our 100 updates of one
    rollout; and their merge reward is dense and normalised in every vehicle's
    velocity, where ours is led by a 60 s rolling throughput count whose response
    to an action arrives long after the GAE horizon. Their own merge benchmark
    "started to gradually degrade after certain iterations, suggesting that the
    problem is difficult to solve with existing optimization methods".

    **DECISION FOR THE USER, because it changes the research object.** Fixing the
    merge makes `inverted_tree` the network it was described as, but it invalidates
    every capacity table and both demand levels, and the operating point, the burst
    scenario and the dt-convergence result all need re-measuring on the corrected
    road. The alternative -- keeping it -- means studying a network whose capacity
    is set by an arbitrary tie-break and whose fairness objective is unreachable.

97. **RESULT of the pre-registered run: MAPPO does not beat doing nothing, and is
    measurably worse.** Executed 2026-09-09 exactly as task 93 fixed it in advance:
    five seeds (7, 17, 27, 37, 47), `sumo_burst`, 900 s episodes at dt 0.1, the
    final checkpoint of each seed, comparators `no_av` and `density_lookup`,
    completed trips as the primary metric, a two-standard-error bar paired on seed.
    Produced by `scripts/evaluate_burst_scenario.py`.

    | arm | arrivals | trough m/s | recovery s | roadblock | collisions |
    |---|---|---|---|---|---|
    | `no_av` | **259.8 +/- 7.2** | 8.80 | 264.5 | 32.9 | 0 |
    | `density_lookup` | 242.4 +/- 19.2 | 8.10 | 342.3 | 173.7 | 0 |
    | `mappo` | 242.8 +/- 17.4 | 8.27 | 213.6 | 1365.3 | 0 |

    Paired against `no_av` on the same seeds:

    | arm | metric | difference | verdict |
    |---|---|---|---|
    | `density_lookup` | arrivals | −17.40 +/- 7.97 | **real** |
    | `density_lookup` | trough speed | −0.70 +/- 0.53 | no effect |
    | `density_lookup` | recovery | +17.17 +/- 75.35 | no effect |
    | `density_lookup` | roadblock | +140.80 +/- 36.45 | **real** |
    | `mappo` | arrivals | **−17.00 +/- 7.55** | **real** |
    | `mappo` | trough speed | −0.53 +/- 1.27 | no effect |
    | `mappo` | recovery | −91.40 +/- 87.66 | no effect |
    | `mappo` | roadblock | +1332.38 +/- 134.75 | **real** |

    **The primary metric says both controllers reduce throughput.** MAPPO completes
    17.0 +/- 7.6 fewer trips than an uncontrolled fleet, a 6.5% reduction, and the
    project's existing non-learning baseline is indistinguishable from it at
    −17.4 +/- 8.0. Zero collisions in all fifteen runs.

    **Nothing else clears the bar.** The trough is unchanged for both. MAPPO's
    recovery is 91 s faster on average but the spread is 88, so it does not clear
    two standard errors; on the pre-registration that is no effect, and it is
    recorded here rather than promoted.

    **The one term that moves decisively is the wrong one.** MAPPO's
    `rolling_roadblock_score` is 1365 against 33 for uncontrolled traffic, 41 times
    higher and far outside the noise. A policy that emits `slow` about a third of
    the time holds lanes below free flow constantly, which is exactly what that
    term exists to detect.

    **The qualification, recorded BEFORE this result was measured (task 96): the
    policy did not learn.** Entropy ended at 4.3548 against a maximum of 4.394, so
    it finished at 99.1% of maximum entropy, and the score and throughput were flat
    across all 100 updates. This measures a near-uniform policy over the action
    space more than it measures what MAPPO can do. The honest reading is therefore
    narrow: **with a team reward shared among about twelve agents, 100 updates of
    MAPPO produced no learning signal, and acting near-randomly over this action
    space costs 6.5% of throughput.**

    **It is consistent with task 92.** There is no throughput effect at a converged
    step size for a policy to find, so a policy that finds nothing is the expected
    outcome, and one that acts anyway does harm. What this run does NOT establish
    is that a policy with a working learning signal would fail; that question needs
    the credit-assignment problem addressed first, and it is the natural next task.

96. **The policy is not learning, and the cause is the credit-assignment signal
    rather than a defect.** Diagnosed 2026-09-09 at update 26 of the pre-registered
    run, before spending the remaining four hours on it.

    **The symptom.** Over 26 updates and five seeds: score flat at 1.60 to 1.73,
    throughput flat at 16.7 to 16.9 arrivals per 60 s, and entropy 4.3714 to 4.3674
    against a maximum of ln(81) = 4.394. The policy moved 0.004 in entropy and is
    effectively frozen. Zero collisions throughout.

    **A wrong diagnosis, and the control that caught it.** The objective is 99.96%
    value loss: `loss` about 116, of which `value_coef * value_loss` is 115.9,
    against a policy loss of -0.0004 and an entropy bonus of 0.044. Measured
    gradient norms on one minibatch: the critic's is 79.7 and the actor's 0.378,
    and `ppo_update` clips ONE norm over both networks, so `max_grad_norm` 0.5
    scaled every parameter by 0.0063. That looks decisive -- the actor apparently
    trained at 0.6% of its intended rate -- and it is wrong.

    **Adam makes a uniform gradient rescaling irrelevant.** Its update is
    `lr * m / sqrt(v)`, and scaling every gradient by a constant scales both
    moments, so the step is unchanged. Measured directly: 20 Adam steps on the same
    problem move a parameter by 0.383268 with unscaled gradients and 0.383267 with
    gradients scaled by 0.0063. The separate-clipping change was reverted rather
    than shipped, because it changes shared code that every experiment in this
    project uses and it has no measurable effect under Adam. The test written for
    it passed against the un-separated control, which is how the wrong diagnosis
    was caught before it was acted on.

    **What is left, and it is not a bug.** With advantages normalised, a policy
    loss of -0.0004 means the ratio barely leaves 1, which means the advantages
    carry almost no information about the actions. That is the expected shape of a
    shared team reward divided among about twelve agents: one agent's choice moves
    the team reward by far less than the noise in it over the GAE horizon of 20 s.
    The signal is weak because the problem is configured that way, not because a
    knob is set wrong.

    **The run continues unchanged.** This is what the pre-registration exists for:
    it fixed the metric and the bar before any of this was visible, so a null gets
    reported as a null. Task 92 already retracted the evidence that a throughput
    effect exists here at a converged step size, and a policy that cannot find a
    signal that is not there is the consistent outcome rather than a surprise.

95. **Round 3 leftovers, recorded rather than fixed, with the reason.**

    - **The critic's `time` input is far outside its normalisation range.**
      `FIELD_SCALES["time"]` is 120.0, so at the 900 s episodes `mappo_sumo` now
      runs it reaches 7.5 while every other input sits near [0, 1]. It is one of
      only three top-level inputs among the critic's 115. NOT changed: `FIELD_SCALES`
      is shared with `sim_contract`, this is a conditioning problem rather than a
      correctness one, and a finite-horizon value function should see the clock. If
      it is changed it should become the episode duration so the feature spans
      [0, 1], and that must not be done while a pre-registered run is in flight.
    - **`get_segment_metrics(snapshots)` ignores its argument when the cache is
      warm.** No external caller passes it -- `base_ctde_env.py`, `run_baseline.py`,
      `evaluate_policy.py` and `validate_topology_baselines.py` all call it with no
      argument -- so it is latent, but the signature invites a caller to be silently
      ignored. The fix is to hold the step's snapshots on the environment and drop
      the parameter. NOT done while a training run is in flight, because it changes
      the environment the run is training against and the evaluation must use the
      same one.
    - **`all_lane_av_low_speed_occupancy` reaches the reward at weight zero.** It
      has no entry in `DEFAULT_REWARD_WEIGHTS` and no config overrides one. Not a
      defect: the per-segment version is one of the eleven fields the critic reads,
      and `rolling_roadblock_score`, which carries -2.0, is built from it. Now
      documented at the point it is computed so nobody reads it as a penalised
      quantity.
    - **The comment round 3 reported as misattributing the -2.0 weight does not
      reproduce.** At HEAD that comment sits above `rolling_roadblock_score` and
      describes it correctly.

94. **The action heads could cause collisions, and the first training run was
    invalid because of it.** Found 2026-09-09 by the collision counter, on the
    smoke evaluation of a two-update checkpoint: 30 collisions on the learned arm
    against 0 for every baseline. SUMO's car-following being unable to produce a
    collision is the premise of the whole migration, so this was a defect in the
    actuation I had just added, not in SUMO.

    `hold_lane` set the lane-change mode to 0. Bits 8-9 of a SUMO lane-change mode
    are the collision-avoidance bits, so 0 does not mean "no lane changes", it
    means "no lane changes and no safety". Separately, `changeLane` holds its
    choice for a duration, so a vehicle told to prefer a lane on one step and to
    hold its lane on the next carried on into the change with the checks removed.

    Measured over 3000 steps, actions varying per agent per step:

    | heads varied | collisions |
    |---|---|
    | speed alone, headway alone, lane alone, merge alone | 0 each |
    | speed + lane | 0 |
    | speed + merge | 0 |
    | headway + merge | 0 |
    | **lane + merge** | **646** |
    | all four | 703 |

    **Every single head was clean and the pair was not**, which is why the guard
    now varies the heads together. A single-head test would have passed throughout.

    Two fixes, and each is sufficient on its own -- confirmed by reverting them
    separately, where either alone keeps the count at zero and only the shipped
    combination fails. Both are kept because both are right independently:
    `hold_lane` now uses mode 1536, which clears the vehicle's own motivations to
    change lane while leaving collision avoidance on, and it cancels any pending
    change by requesting the current lane.

    **The first five training runs were killed.** They had been training against an
    environment in which the policy's own actions could cause collisions, which
    contradicts the premise and makes the crash penalty fire on the environment's
    defect rather than the policy's behaviour.

93. **PRE-REGISTERED: what the MAPPO run will be judged on.** Written 2026-09-09
    BEFORE the run, because the previous headline was a number chosen after the
    fact from a sweep of constant commanded speeds, and Ankit's instruction was
    "no cheating and cherry picking". Everything below is fixed in advance. If the
    run does not clear these bars it is reported as a null.

    **What is run.** `configs/training/mappo_sumo.yaml` unchanged: MAPPO, `full`
    action profile (all four heads now actuate), `deployed_fidelity: true`, the
    cited sensing noise, dt 0.1, 900 s episodes on `sumo_burst`, one rollout per
    episode, 100 updates. Seeds fixed in advance: **7, 17, 27, 37, 47.** These are
    NOT the seeds the retracted arm tables were measured on (3, 7, 11, 19, 23),
    deliberately.

    **What it is compared against.** `no_av` at the same seeds and the same
    scenario, and `density_lookup`, the project's existing non-learning baseline.
    No commanded-speed arm is a comparator: a constant speed chosen from a sweep is
    an oracle, not a controller, and the sweep that produced one is retracted.

    **Primary metric, decided now: completed trips per episode.** One number,
    reported as a mean over the five seeds with its standard deviation and the
    paired difference against `no_av` on the same seeds.

    **Secondary metrics, also decided now**, because the scenario has a shape a
    single total hides:
    - the minimum of the 60 s rolling mean speed, which is the depth of the trough;
    - the time from the end of the burst until the queue returns below 10 vehicles,
      which is the recovery;
    - `rolling_roadblock_score`, summed, which is whether the policy bought its
      result with behaviour the contract forbids;
    - collisions, which must be zero.

    **The bar.** A gain is reported as real only if the paired difference against
    `no_av` exceeds two standard errors over the five seeds. Anything smaller is
    reported as no effect. No seed is dropped, no arm is selected, and the metric
    is not chosen after the numbers are seen.

    **What a null would mean, stated now so it cannot be reinterpreted later.**
    Task 92 retracted the evidence that a throughput effect exists here at a
    converged step size, so a null is the expected outcome rather than a
    disappointment, and it is still worth reporting: it would say that under the
    deployment's own sensing model, on this network, a policy with speed, headway,
    lane and merge control does not beat doing nothing. The alternative reading --
    that the reward does not ask for the right thing -- is guarded against by
    reporting arrivals directly rather than the reward.

92. **RETRACTION: the throughput gain was an artefact of the simulation step.**
    Measured 2026-09-09 after the user asked for `dt: 0.1` so the deployment's
    measured sensing latency could be represented. It retracts task 86 and moots
    task 91.

    `scripts/measure_step_size_convergence.py`, 600 s episode after 300 s of
    warm-up, five seeds, `sumo_saturating`, arrivals:

    | dt | steps | uncommanded | AVs at 10 m/s | gain |
    |---|---|---|---|---|
    | 1.0 | 600 | 132.0 +/- 6.4 | 155.2 +/- 10.5 | **+17.6%** |
    | 0.5 | 1200 | 134.4 +/- 5.9 | 157.8 +/- 16.0 | +17.4% |
    | 0.2 | 3000 | 157.6 +/- 12.3 | 160.4 +/- 7.2 | +1.8% |
    | 0.1 | 6000 | 168.4 +/- 11.3 | 161.2 +/- 8.6 | **−4.3%** |
    | 0.05 | 12000 | 173.0 +/- 8.4 | 161.6 +/- 8.6 | **−6.6%** |

    **The commanded arm barely moves: 155.2, 157.8, 160.4, 161.2, 161.6, a drift of
    6.4 arrivals across a twentyfold change in step size and well inside its own
    standard deviation. The uncommanded arm rises 31%, from 132.0 to 173.0.** The
    treatment is invariant to the numerical parameter and the control is not, so
    the difference between them was never a property of the traffic.

    **Mechanism.** SUMO's `--step-length` is the physics step, and the migration
    passed `dt` straight to it. A vehicle travelling 24 m/s advances 24 m per step
    at dt 1.0, so junction gap acceptance and car following were resolved at 24 m
    granularity and the uncommanded fleet lost throughput to the integration. A
    fleet held at 10 m/s advances 10 m per step and loses much less. Commanding a
    lower speed was buying back numerical resolution, not damping waves.

    **The project already knew this and the migration lost it.**
    `HighwayTopologyEnv` integrates at `physics_substeps: 10`, with a comment at
    `src/envs/topology_env.py:233-236` saying why: "decisions happen once per dt,
    but the physics must integrate at a finer grid (highway_env is built for ~10-15
    Hz): a single 1 s Euler step drives IDM vehicles through each other and to
    negative speeds". The SUMO env has no equivalent, so from the first commit of
    the migration the SUMO fleet integrated at 1 s where the highway_env fleet it
    was compared against integrated at 0.1 s. Every SUMO capacity figure, including
    the "junction-limited at about 900 veh/h" that the demand configs were chosen
    against, was measured on the coarse integration.

    **What is retracted.** Task 86's throughput gain, in all three of its recorded
    magnitudes: 52% at one seed, 28% at five, 17.6% after the fleet correction. At
    a converged step size holding AVs at 10 m/s does not raise throughput, it
    lowers it by 4 to 7%. Task 91's finding that 48.5% of the gain came from
    behaviour the contract forbids is moot, because there is no gain. The metering
    exemption committed in `2d313cc` is kept: it is a correct refinement of a
    metric that could not tell metering from obstruction, and it stands on its own
    reasoning, but the measurement that motivated it is withdrawn.

    **What this does NOT establish.** That no controller can raise throughput here.
    The retracted evidence came from an oracle -- every AV commanded to the same
    speed for a whole episode, which no policy can express. A learned policy acting
    on local conditions might still find something. What is gone is the evidence
    that an effect was there to be found, which is what justified the training run.

    **Open, and the user's call: whether the simulation leg still has a question.**
    The paper is a deployment story and the simulation exists to replicate that
    MAPPO works under the deployment-measured sensing model. That replication can
    still be run and reported -- including as a null -- but it should be commissioned
    knowing that the uniform-speed oracle now shows no effect to find at this
    operating point and topology.

91. **MOOT after task 92: there is no gain to be admissible or not.** The metering
    exemption is kept on its own reasoning. Original text follows.

    ~~The measured throughput gain is largely inadmissible under the project's own
    contract.~~ Measured 2026-09-09 while diagnosing why the reward ranks the
    commanded-speed arms differently from arrivals. The reward is not
    mis-specified; it is correctly refusing a strategy the project forbids.

    Reward decomposed by term, three seeds, 600 steps, per step:

    | arm | arrivals | reward | `rolling_roadblock_score` contribution |
    |---|---|---|---|
    | uncommanded | 132.7 | +0.919 | −0.011 |
    | 10 m/s | **162.3** | +1.181 | **−0.741** |
    | 15 m/s | 151.7 | +1.130 | −0.363 |
    | 20 m/s | 146.0 | **+1.243** | −0.015 |

    Every other term ranks 10 m/s first. `rolling_roadblock_score` at weight −2.0
    is the whole inversion: remove it and 10 m/s scores 1.922 against 20 m/s at
    1.258, which is the arrival order.

    **What the term measures.** `_rolling_roadblock_score` fires only when AVs
    occupy every lane of a segment at a mean speed more than 8 m/s below free flow,
    AND the segment's `jam_fraction` is at most 0.25, AND its queue length is zero.
    It is deliberately narrow: slow AVs while the road around them is clear. That
    is the README's prohibition on rolling roadblocks, made measurable.

    **Where it fires.** Over 300 steps at 10 m/s it fires on 767 segment-steps
    against 7 for both uncommanded traffic and 20 m/s. Not only on the single-lane
    leaves, where one AV trivially holds "every lane": 159 of them are on the
    two-lane `tree_middle_b2` and `tree_trunk_c`.

    **Metering or obstruction?** The guard checks jam and queue on its own segment
    only, so a segment held slow *because the next one is jammed* would be scored
    as obstruction although it is metering. Classifying the 767 firings by the
    state of the downstream segment at the same step:

    | downstream state | segment-steps | share |
    |---|---|---|
    | jammed (`jam_fraction` > 0.25) — metering | 295 | 38.5% |
    | queued but not jammed | 0 | 0.0% |
    | the trunk, which has no downstream segment | 100 | 13.0% |
    | **clear — obstruction by the project's definition** | **372** | **48.5%** |

    So it is not simply a mis-specified metric. Nearly half the firings are AVs
    holding a clear segment with a clear road ahead.

    **Consequence for the headline number.** The 17.6% gain at 10 m/s is achieved
    substantially through behaviour the project's contract forbids. The arms that
    do not trigger the penalty do not clearly beat doing nothing: at five seeds,
    20 m/s gives 138.8 +/- 10.6 against 132.0 +/- 6.4 uncommanded, which is within
    noise.

    **Decision for the user.** Three readings, and they lead to different papers.
    (a) The contract stands: the admissible gain is what a policy achieves without
    triggering the term, and on current evidence that is not distinguishable from
    zero — a null worth reporting, and the reason to report it is that the
    unconstrained gain is large. (b) The term is too strict on a segment whose
    downstream is jammed; exempting those 38.5% would license metering while still
    forbidding the 48.5%. That is a defensible refinement of the metric, not a
    weakening, but it must be made before the training run rather than after seeing
    the result. (c) The prohibition itself is reconsidered for single-lane
    approaches, where "hold every lane" cannot distinguish a roadblock from an
    ordinary slow vehicle.

    My recommendation is (b) plus reporting under both, because the exemption has a
    stated principle -- a jammed downstream segment is a traffic reason for being
    slow, which is exactly what the guard's other two conditions are testing for --
    and because the 48.5% that remains forbidden is the part that would make a
    reviewer uncomfortable.

90. **The two unmeasured sensing parameters now carry citations.** Ankit,
    2026-09-09: use prior papers to fill the values that could not be measured, or
    HERE data if possible.

    HERE does not apply. It is a segment-level traffic feed, so it can characterise
    aggregate speed on a link but not the per-vehicle position and speed error of a
    forward monocular camera, which is what these two parameters model.

    `configs/training/mappo_sumo.yaml` now cites Song, Lu, Zhang and Li,
    "End-to-end Learning for Inter-Vehicle Distance and Relative Velocity
    Estimation in ADAS with a Monocular Camera", ICRA 2020 (arXiv:2006.04082),
    on the TuSimple velocity benchmark. It measures this exact pair of quantities
    for this exact sensor and reports, for its full model, position MSE 10.23 m^2
    and range-averaged velocity MSE 0.86 m^2/s^2 — RMSE 3.20 m and 0.93 m/s.

    | parameter | was | now | source |
    |---|---|---|---|
    | `position_noise_std` | 1.5 | **3.2** | position MSE 10.23 m^2 |
    | `speed_noise_std` | 0.15 | **0.93** | velocity MSE 0.86 m^2/s^2 |

    The old values were carried over from `shared_ppo_deploysense` with no source
    and are optimistic against this citation by roughly 2x on position and 6x on
    speed. The cited figures belong to a state-of-the-art learned method, while
    this rig runs YOLOv8n with pinhole geometry and vehicle-width priors, so they
    are a floor on the noise rather than an estimate of it — which is the
    conservative direction for the claim: a policy that works under them would work
    under a better sensor.

    Song's velocity error is range-resolved — RMSE 0.39 near (< 20 m), 0.58 medium
    (20-45 m), 1.45 far (> 45 m) — and the range-averaged figure is taken because
    most vehicles observed at a 100 m range are beyond 45 m. Corroborating for
    speed alone, from a roadside camera validated against a GNSS and IMU reference:
    Bell et al., ISPRS Annals V-2-2020, average RMSE 0.625 m/s over four
    experiments.

    Verified against a noise-free run on the same seed: the injected error on
    `leader_gap` has a standard deviation of 3.00 m and on `leader_relative_speed`
    0.70 m/s over 376 paired observations. A test pins that the configured noise
    reaches the observations.

    **Only `mappo_sumo.yaml` is changed.** `mappo_deploysense.yaml` and
    `shared_ppo_deploysense.yaml` still carry 1.5 and 0.15, because their recorded
    results were produced under those values and changing them would silently
    supersede those results. Any future run on those configs should adopt the cited
    values first.

    **Still open: `latency_s`.** This one was measured — end-to-end latency over
    22,929 ticks on 2026-09-08 was p50 96.7 ms, mean 112.4 ms, p95 172.9 ms — and
    the obstacle is the simulation's time resolution, not a missing measurement.
    `SensingBuffer.frame_for_latency` selects whole recorded frames and the buffer
    records once per step, so at `dt: 1.0` any value in (0, 1.0] imposes a full
    second and overstates the measured delay tenfold. It is pinned at 0, which
    understates it by 97 ms. **Decision for the user:** running at `dt: 0.1` would
    make `latency_s: 0.1` represent the measured p50 almost exactly and would put
    the policy's decision rate at 10 Hz, closer to the deployment's 30 Hz tick than
    1 Hz is. The cost is that every capacity and arrival figure would need
    re-measuring a third time, and each episode becomes 6,000 steps rather than
    600.

89. **What the simulation study still needs.** Asked 2026-09-09: are the results
    for the paper in hand? Scope corrected by the user in the same exchange, and
    the correction matters: **the paper is a deployment story.** The simulation
    exists to replicate that MAPPO works under the sensing model measured on the
    deployment, not to run the nine-figure study in `plan_simulations.md` section
    8. Against that narrower target the gap is one training run, not a programme.

    **In hand.**

    - The effect is real and measurable in this simulator: AVs commanded to hold
      10 m/s raise arrivals 17.6% over 600 steps (155.2 +/- 10.5 against
      132.0 +/- 6.4, five seeds) and 20.9% over an hour, with zero collisions.
    - The action space can express a usable part of it: `slow` decodes to 13.94 m/s
      and gives 143.0 +/- 11.7, about half of what is available.
    - Every prerequisite for a valid run is now in place. The critic receives its
      115 inputs rather than 2; the fleet is the one the demand config declares;
      the episode is 600 steps, the horizon at which the effect exists; and
      `mappo_sumo.yaml` carries the deployment sensing block with
      `deployed_fidelity: true`.

    **Missing: the run itself.** `outputs/checkpoints/` is empty, and every SUMO
    training run to date is invalid on grounds found this session. One MAPPO run at
    `mappo_sumo` plus its evaluation against `no_av` is the deliverable.

    **Two things to settle before spending it.**

    1. **The reward does not rank the commanded-speed arms the way arrivals do.**
       Over 600 steps arrivals peak at 10 m/s and the reward at 15 m/s.
       `throughput_recent` is a 60 s window rather than the episode total, and
       `jam_fraction` at weight -2.0 grows as the network fills. A policy
       maximising this reward is not guaranteed to show the effect, so a null
       result would be uninterpretable.
    2. **The sensing model is only partly measured.** `range_m: 100.0` is
       deployment-derived and end-to-end latency was measured over 22,929 ticks,
       but `latency_s` is pinned at 0 because at a 1 s step any positive value
       means a full second, and `position_noise_std` and `speed_noise_std` are
       marked in the config as NOT measured, carried over from an earlier config
       and needing a published characterisation of monocular bounding-box ranging
       before the paper describes them as measured. A claim that MAPPO works
       "under the deployment's sensing model" rests on all three.

88. **Lower-severity items from the same audit.** Recorded, not fixed.

    - **The demand config's speed distribution is ignored on SUMO.** FIXED
      2026-09-09. `_write_routes` read only `speed_distribution.max_mps` and
      hardcoded the desired-speed spread as `normc(1,0.1,0.8,1.2)` on the lane
      limit, so a config declaring a 24.0 m/s mean produced a fleet desiring about
      30 m/s: every capacity measurement belonged to a fleet no config described.
      Every edge this builder writes carries the topology's single
      `speed_limit_mps`, so the configured distribution maps onto the factor
      exactly. Measured after the fix, per distinct vehicle at free flow: declaring
      24.0 gives 23.68, declaring 18.0 gives 17.88.

      **`spawn_min_gap_m` is deliberately not mapped, and the original wording of
      this item was wrong about it.** It claimed the minimum gap "falls from the
      configured 12 m to SUMO's vType default of 2.5 m, which raises jam density
      roughly fourfold". Those are two different quantities. On `highway_env`
      `spawn_min_gap_m` gates insertion — `_lane_has_spawn_gap` refuses a lane
      holding a vehicle within that distance — and changes no car-following
      behaviour. SUMO enforces insertion feasibility itself through the
      car-following model, which is the stronger criterion. Mapping the field to a
      vType `minGap` would change the standstill gap, and so jam density, from a
      config field that on the other simulator changes no physics at all.

      `branch_split` and `burst` remain unmapped: `branch_split` is `{main: 1.0}` in
      both SUMO demand configs and the builder already splits the rate evenly across
      the six entries, and `burst.enabled` is false in both.
    - **`mean_speed` and `active_vehicle_count` exclude vehicles inside junctions.**
      Measured: 700 of 48,590 vehicle-steps (1.44%) were on internal lanes, and the
      reported `mean_speed` was 6.157 m/s against SUMO's own 6.358 over all
      vehicles — 3.2% low, because junction-crossing vehicles are moving.
    - **`mappo_sumo.yaml` does not set `duration_steps`** — FIXED 2026-09-09, and it
      mattered more than "lower-severity" suggested. It defaulted to 120 while
      `rollout_steps` is 512. At 120 steps the throughput effect this configuration
      exists to learn is absent: AVs holding 10 m/s produce 19.7 arrivals against
      22.0 uncommanded, where at 600 steps the same comparison is 143.0 against
      111.8. The reward ranking inverts with it, placing 30 m/s first and 10 m/s
      last, so training at the default would have optimised against the effect under
      study. Now 600, with the measurement in the config and in
      `plans/plan_task_84_sumo_simulator.md`.
    - **Threshold sources differ between the simulators.** The SUMO env reads
      `queue_speed_mps` from `config["sensing"]` and `throughput_window_s` from the
      top level; `highway_env` reads both from `config["metrics"]["thresholds"]`, so
      a config setting them there is silently ignored on SUMO. FIXED 2026-09-09,
      and the item was half stale as the second audit round reported: by then
      `queue_speed_mps` already came from the shared
      `metric_thresholds_from_config`, but `throughput_window_s` was still read
      from the top level of the config, where nothing writes it. Every experiment
      config sets it under `metrics.thresholds`, so the window silently stayed at
      the 60 s default on SUMO whatever a config declared.

86. **RETRACTED by task 92 — the gain was a discretisation artefact.** The text
    below is kept as the record of how it was measured and corrected twice before
    being withdrawn; every arrival figure in it was taken at dt 1.0.

    ~~A 17.6% throughput gain exists, and the action space cannot reach it.~~
    Measured 2026-09-09 on SUMO, then re-measured twice after defects found in the
    measurement itself. This is the control effect the project has been trying to
    measure, and the reason no policy has found it.

    Every AV given the same commanded speed, 600 steps after a 300-step warm-up, at
    `sumo_saturating` (1050 veh/h, 20% penetration), five seeds:

    | command | arrivals | mean team reward |
    |---|---|---|
    | none | 132.0 +/- 6.4 | +0.823 |
    | 5 m/s | 123.4 +/- 19.9 | +0.520 |
    | 8 m/s | 148.8 +/- 9.0 | +0.915 |
    | **10 m/s** | **155.2 +/- 10.5** | +0.930 |
    | 12 m/s | 150.6 +/- 17.9 | +0.823 |
    | 15 m/s | 148.0 +/- 17.1 | **+1.034** |
    | 20 m/s | 138.8 +/- 10.6 | +0.987 |
    | 24 m/s | 137.0 +/- 4.7 | +0.956 |

    **Holding AVs at 10 m/s raises arrivals from 132.0 to 155.2, a 17.6% gain with a
    standard error of about 5.** Zero collisions throughout. The relationship is
    non-monotonic: 5 m/s is worse than doing nothing, so this is a genuine operating
    point rather than "slower is better".

    **The 52% figure this item first recorded is superseded twice over.** 167
    arrivals against 110 was one seed, taken with the fairness denominator defect
    present and on a fleet desiring 30 m/s where the demand config declares 24. The
    direction has survived every correction; the magnitude has fallen from 52% to
    17.6%.

    **The effect also does not exist at 120 steps**, which is what
    `mappo_sumo.yaml` was implicitly training at. No commanded arm beats an
    uncommanded fleet over 120 steps: the best is 28.8 +/- 2.9 against 27.8 +/- 3.1.
    The config now declares 600.

    That is the speed-harmonisation result the replication targets, and it is
    reproducible in one 600-step run.

    **The action space reaches about half of it — CORRECTED 2026-09-09.** This item
    first recorded that the action space could not express the effect at all, and
    that was true of the fleet then being simulated. `decode_speed_bin` returns the
    free-flow speed plus an offset (`slow` −10, `nominal` −3, `fast` 0), and
    free-flow is the vehicle's own desired speed. While the route writer ignored the
    demand config and gave every vehicle the 30 m/s lane limit, the bins decoded to
    20, 27 and 30 m/s, all inside the flat region where the effect is gone. With the
    fleet the config declares, they decode near 14, 20.5 and 23.5.

    Measured through the real action path, 600 steps, five seeds:

    | bin | decodes to | arrivals |
    |---|---|---|
    | uncommanded | — | 132.0 +/- 6.4 |
    | `slow` | 13.94 m/s | **143.0 +/- 11.7** |
    | `nominal` | 20.48 m/s | 133.6 +/- 4.3 |
    | `fast` | 23.53 m/s | 137.4 +/- 9.0 |

    `slow` recovers 11.0 of the 23.2 arrivals available between doing nothing
    (132.0) and the best commanded speed (155.2 at 10 m/s): about half the effect,
    at roughly two standard errors. The earlier measurement of arrivals identical at
    137 across all three bins and no action was taken on the undeclared fleet, where
    every bin landed above 20 m/s.

    So the actions were nearly equivalent rather than exactly equivalent, and the
    flat training curves and task 69's null still follow — but the remedy is smaller
    than this item first implied.

    **This supersedes task 85.** The entropy bonus really was 66% of the reward and
    `reward_scale: 1.0` really does fix that ratio, but changing it moved nothing --
    entropy went from −0.025 to +0.021 across 400 updates -- because the binding
    constraint is that the actions are equivalent. Task 85's arithmetic stands; its
    implied conclusion does not.

    **The decision this needs, and it is the user's. The case for it is now weaker
    than when it was first put.** The bins are part of the action contract shared
    with the deployed system through `sim_contract`, so rescaling them changes what
    the Jetson's actor emits as well. On the corrected measurements the question is
    whether to capture the remaining half of a 17.6% effect, not to make an
    unreachable effect reachable. Three routes:
    make the bins fractions of the free-flow speed rather than offsets from it
    (`slow` 0.33x gives 7.9 m/s on the declared fleet, which measured 148.8; a 0.42x
    fraction would give the 10 m/s that measured 155.2); decode them against the
    local traffic speed rather than the edge limit, so `slow` means slow *for these
    conditions*;
    or leave the contract and add absolute low-speed bins. The first is the smallest
    change and the third is the most explicit.

85. **The entropy bonus is 66% of the reward, so no policy has ever converged.**
    Found 2026-09-09 on SUMO, but it is not a SUMO defect: it applies to every
    training run this project has done.

    Measured at the metrics the 400-update SUMO run actually produced —
    `mean_speed` 7.33, `throughput_recent` 11.53, `jam_fraction` 0.128:

    | quantity | value |
    |---|---|
    | team reward | +1.2835 |
    | after `reward_scale` 0.05 | **+0.0642** — what the agent optimises |
    | entropy bonus at `entropy_coef` 0.01 and entropy 4.24 | **+0.0424** |
    | entropy bonus as a share of the reward | **66%** |
    | policy entropy against a 4.68 maximum | **91% of uniform** |

    PPO's default hyperparameters — learning rate 3e-4, value coefficient 0.5,
    entropy coefficient 0.01 — assume a reward of order 1. `reward_scale: 0.05`
    crushes this one to 0.064, so the entropy term is two thirds as large as the
    entire objective and the policy has almost no incentive to become
    deterministic. It held 91% of maximum entropy after 400 updates, and entropy
    moved −0.025 across them.

    **This is why every training run has looked flat.** The 100-update
    `highway_env` run showed entropy 3.426 → 2.830, which looked like convergence;
    at the SUMO operating point the reward is smaller still and the same
    coefficient dominates it. Neither run was learning much.

    **The fix is to stop scaling the reward down.** `reward_scale: 1.0` makes the
    reward 1.28 and the entropy bonus 3% of it, which is the ratio the default
    coefficients were chosen for. `reward_clip` at 10 already bounds the magnitude,
    so the scale-down was not protecting anything.

    Changing one thing at a time, so the effect is attributable.

81. **The reweighting worked: throughput improved 30% during training.** Done.

    100 updates, `mappo_deploysense` with `throughput_recent` at 0.10 and
    `jam_fraction` at −2.0, `inverted_tree` at the `saturating` demand, 3600-step
    episodes:

    | metric | first half | second half | change |
    |---|---|---|---|
    | `throughput_recent` | 10.779 | 14.062 | **+3.283 (+30%)** |
    | `entropy` | 3.426 | 2.830 | −0.596, converging |
    | `mean_speed` | 20.355 | 20.288 | −0.066, flat |
    | `jam_fraction` | 0.004 | 0.005 | +0.001 |
    | `collision_count` | 0.053 | 0.068 | +0.015 |

    Under the old weights the agent raised its own speed and left throughput
    untouched; under the new ones it does the reverse. That is the objective
    change working as intended, and it is the first time throughput has moved
    during training at all.

    **This is a training-metric result, not the replication.** The evaluation is
    still blocked: the reference is bistable at capacity and the arms do not
    complete reliably. See the retraction under task 80 and task 77.

82. **The `score` column no longer measures what is being optimised.** Open, small.

    `src/rl/trainers.py:134` computes `score = mean_speed − jam_fraction` and uses
    it for `best_score` and for choosing which checkpoint is "best". Since the
    reward became configurable and throughput-led, that expression is not the
    objective: a run whose throughput improves 30% while speed stays flat shows a
    `score` delta of −0.067, i.e. it looks slightly worse.

    So `actor.pt` is selected on a quantity the trainer is no longer maximising.
    `latest_actor.pt` is unaffected. The fix is to score with
    `build_team_reward(metrics, self.config.reward_weights)`, which is the thing
    actually being maximised.

83. **`config_resolved.yaml` is written only when training completes.** Open, small.

    Observed mid-run: the checkpoint directory held `actor.pt`, `critic.pt`,
    `latest_actor.pt`, `latest_critic.pt`, `trainer_state.pt` and
    `training_metrics.csv`, but no `config_resolved.yaml`; it appeared at
    completion. Task 69's evaluation harness refuses a checkpoint whose sensing
    block it cannot read, by design, so an interrupted or still-running training
    run cannot be evaluated even though its weights are on disk. Writing it
    alongside the first checkpoint would cost nothing.

80. **Controllers DO have a large measurable effect, once the simulator is fixed
    and measured at the right operating point.** PRELIMINARY — two seeds.

    At the `saturating` demand of 2000 veh/h, 600-step runs, after the
    collision-free bound, the node-geometry fix and the capacity measurement:

    | controller | AV penetration | jam | speed | throughput |
    |---|---|---|---|---|
    | `no_av` | 0.00 | 0.1083 | 10.70 | 11.5 |
    | `backpressure` | 0.10 | **0.0000** | 19.96 | **25.0** |
    | `backpressure` | 0.20 | 0.1000 | 17.69 | **31.5** |

    **A hand-written baseline at 10% penetration removes the congestion entirely
    and more than doubles throughput.** That is the effect task 8 concluded did not
    exist — it measured "+0.12 to +0.14 m/s against a 1.0 m/s threshold" and failed
    68 of 72 cells on `baselines_separate`.

    **Why task 8 could not see it.** Three reasons, each now measured:
    - Its demand levels bracket capacity without hitting it. `medium` (1800 veh/h)
      free-flows so there is nothing to relieve; `high` (2700) gridlocks so there is
      no throughput left to compare. The effect lives at 2000, which no config had.
    - Collisions dominated the dynamics. Crash queueing was most of the measured
      congestion — `merge` entirely, `inverted_tree` 61% — so a controller's effect
      was buried under crash-induced jams it could do nothing about.
    - Its reference controller was `no_av`, which cannot terminate early and so
      passed `episodes_complete` by construction, making the criterion uninformative.

    **What this changes.** Section C's premise, that the simulator cannot show a
    control effect, is refuted. The replication has something to replicate. It also
    means the MAPPO result must be compared against these baselines rather than
    only against `no_av`, because the bar is now high: a learned policy has to beat
    a hand-written one that already doubles throughput.

    **RETRACTED at five seeds. Do not quote the table above.** The five-seed check:

    | controller | pen | jam | speed | throughput | completed |
    |---|---|---|---|---|---|
    | `no_av` | 0.00 | 0.214 ± **0.334** | 11.41 ± **8.72** | 17.2 ± **16.2** | 5/5 |
    | `backpressure` | 0.10 | 0.000 | 19.24 | 21.0 | **1/5** |
    | `cooperative_smoothing` | 0.10 | — | — | — | **0/5** |
    | `backpressure` | 0.20 | 0.000 | 19.39 | 39.0 | **1/5** |
    | `cooperative_smoothing` | 0.20 | 0.000 | 20.08 | 23.0 | **1/5** |

    Two independent reasons the effect cannot be measured here:

    - **The reference is bistable at capacity.** `no_av` throughput is 17.2 with a
      standard deviation of 16.2, and jam 0.214 ± 0.334. Some seeds free-flow and
      some gridlock, which is what near-capacity traffic does. Two seeds happened
      to draw two congested ones, which is why the preliminary table looked clean.
    - **The treatment arms mostly crash.** 1 of 5 completed for `backpressure`,
      0 of 5 for `cooperative_smoothing` at 0.10. So each percentage above rests on
      one surviving run, and survivors are selected for not having crashed. This is
      the selection effect that already invalidated task 69's comparison.

    **And it contradicts something reported earlier in task 72.** `backpressure` at
    `high`/0.20 over 120 steps was 0 collisions and 6/6 completed. At the saturating
    demand over 600 steps it crashes 4 of 5. **The collision-free bound does not
    hold at longer durations**, which is consistent with the truncation arithmetic
    in task 72 and means task 77's residual is larger than two events.

    **So the claim that survives is narrow:** at this operating point a controller
    can drive jam to zero on the runs where it survives, and `no_av` cannot. Whether
    that is a throughput improvement is unmeasured, and cannot be measured until the
    runs complete reliably.


79. **The 120-step episode was hiding that `inverted_tree`/high is over-saturated.**
    Open, and it decides what task 74 can measure.

    Measured `no_av`, penetration 0.10, before the geometry fix:

    | topology | demand | steps | jam | speed | throughput | collisions |
    |---|---|---|---|---|---|---|
    | `merge` | high | 120 / 360 / 900 | 0.00 / 0.00 / 0.00 | 19.3 / 18.7 / 19.6 | 20 / 41 / 40 | 0 |
    | `inverted_tree` | high | 120 | 0.149 | 14.75 | 11.5 | 3 |
    | `inverted_tree` | high | 360 | 0.681 | **1.88** | **0.0** | 7 |
    | `inverted_tree` | high | 900 | **1.000** | **0.00** | **0.0** | 17 |

    **`inverted_tree` at high demand gridlocks.** Jam fraction reaches 1.0, mean
    speed 0.00 and throughput 0.0 — demand exceeds capacity, the network fills, and
    it never recovers. The 120-step episode measured only the transient before
    saturation, so every number this project has taken from that cell describes a
    filling network rather than a steady state.

    **After the geometry fix and the collision-free bound**, `low` and `medium` are
    clean at every duration tested — zero collisions and zero jam, speed 19.7 to
    22.7, throughput scaling with duration as a 60 s rolling window should:

    | demand | steps | jam | speed | throughput |
    |---|---|---|---|---|
    | low | 120 / 600 / 1800 | 0.000 | 22.4 / 22.3 / 22.7 | 6.5 / 13.5 / 14.5 |
    | medium | 120 / 600 | 0.000 | 20.4 / 19.7 | 9.0 / 26.5 |

    **The problem for task 74.** An hour-long run needs a demand that is congested
    AND moving. `low` and `medium` are free-flowing, so a controller has nothing to
    improve; `high` gridlocks, so there is no throughput to compare. Neither is
    usable, and the existing demand levels bracket the operating point without
    hitting it.

    **RESOLVED 2026-09-09 by measurement.** `configs/demand/saturating.yaml`, at
    2000 veh/h. Swept on 600-step runs, 2 seeds, after the collision-free bound and
    the geometry fix:

    | veh/h | jam | speed | throughput | |
    |---|---|---|---|---|
    | 1800 (`medium`) | 0.0000 | 19.66 | 26.5 | free flow, nothing to control |
    | **2000** | **0.1083** | **10.70** | **11.5** | **congested and moving** |
    | 2200 | 0.3867 | 8.66 | 15.5 | congested, heavier |
    | 2700 (`high`) | 1.0000 | 0.00 | 0.0 | gridlock |

    Capacity is about 1800 veh/h. `medium` sits at it, `high` is far past it. At
    2000 the network congests while still moving, and the throughput deficit
    against free flow — 11.5 against 26.5 — is the headroom a controller has to
    recover. Task 8's advice to raise demand was right in direction and would have
    overshot into gridlock.

78. **The simulator must present the sensing model the rig actually has.**
    **DECIDED BY THE USER 2026-09-09.** Supersedes the narrower task 71 ordering
    question, and blocks 73, 68 and 69.

    **The instruction:** the best possible representation in the simulator of the
    sensing model we deployed, using HERE to compute whatever HERE can compute.

    **The audit.** `src/analysis/observation_parity.py` already classifies all 39
    observation slots:

    | class | slots | meaning |
    |---|---|---|
    | `identical` | 8 | both sides compute the same thing the same way |
    | `approximated` | 18 | both compute it, by different mechanisms |
    | `substituted` | 7 | the sim has a real value; the rig substitutes a constant |
    | `structurally_absent` | 6 | the rig cannot produce it at all |

    **All six absent fields are rear-facing** — `follower_gap`,
    `follower_relative_speed`, `left_lane_rear_gap`, `right_lane_rear_gap`,
    `target_lane_rear_gap`, `target_lane_rear_required_decel` — because the live
    vehicle list is forward-camera derived and there is no rear sensor. The seven
    substituted are `ego_lane`, `time_since_last_lane_change`,
    `lane_changes_last_km`, `distance_to_downstream_bottleneck` and the three
    `nearby_av_lane_distribution` slots.

    **So the actor currently trains on 13 of 39 fields the deployed vehicle either
    cannot sense or replaces with a constant.** That is a third of its input, and it
    is the largest single reason a trained policy would not transfer.

    **What HERE can restore, and what it cannot.**
    - **Can:** road identity and geometry, hence route awareness for assigning
      camera detections to the ego's own road — the substance of task 71. And
      `distance_to_downstream_bottleneck`, from `jamFactor` on the segments ahead,
      which is currently substituted with a constant 0.4.
    - **Cannot:** any per-vehicle quantity. HERE's traffic flow API reports
      aggregate speed, free-flow speed and jam factor per road segment. It has no
      individual vehicles in it, so it cannot give a leader distance, a follower
      gap, or a lane distribution. Leader distance stays a camera measurement;
      HERE's contribution is knowing which road the camera is looking down.

    **Steps, in order:**
    1. A sensing-fidelity mode that presents the absent six and substituted seven
       exactly as the rig does, so the actor's input is what the vehicle can produce.
    2. HERE-derived `distance_to_downstream_bottleneck` from downstream `jamFactor`,
       replacing the substituted constant on both sides.
    3. Route-aware assignment of camera detections via map matching (task 71), which
       is what lets the live `leader_gap` mean what the sim's means.
    4. Only then retrain, because every step above changes the actor's input
       distribution.

    **Open:** whether the absent rear fields are dropped from the encoding entirely
    or held at the rig's constants. Dropping changes the vector width and every
    checkpoint; holding keeps the width and wastes six inputs. Recommendation: hold
    at the rig's constants, because the width is baked into `sim_contract` and the
    Jetson's actor runtime, and a width change is a far larger blast radius than six
    dead inputs.

77. **Two collisions survive the collision-free bound and the geometry fix.** Open.

    With the arcs joined, at `inverted_tree`/high/penetration 0.20 over 6 runs:
    `no_av` 4 collisions and 6/6 completed, `cooperative_smoothing` 2 and 5/6,
    `backpressure` **0 and 6/6**. So an AV arm is now fully collision-free and the
    human-only arm is not, which is the reverse of where this task started.

    **DIAGNOSED 2026-09-09, and it is density, not duration.** Per seed, at the
    `saturating` demand over 600 steps:

    | controller | seed | steps | completed | collisions | jam | throughput |
    |---|---|---|---|---|---|---|
    | `no_av` | 7 | 600 | yes | 0 | 0.217 | **0.0** |
    | `no_av` | 17 | 600 | yes | 0 | 0.000 | 23.0 |
    | `no_av` | 27 | 600 | yes | 0 | 0.000 | 34.0 |
    | `no_av` | 37 | 600 | yes | 0 | 0.064 | 29.0 |
    | `no_av` | 47 | 600 | yes | **9** | 0.790 | **0.0** |
    | `backpressure` | 7 | **225** | no | 2 | 0.000 | 29.0 |
    | `backpressure` | 17 | 600 | yes | 0 | 0.000 | 21.0 |
    | `backpressure` | 27 | **290** | no | 3 | 0.111 | 25.0 |
    | `backpressure` | 37 | **319** | no | 7 | 0.228 | 37.0 |
    | `backpressure` | 47 | **171** | no | 2 | 0.000 | 30.0 |

    **Three findings.**

    - **`no_av` is bimodal, not noisy.** Seeds 7 and 47 gridlock to throughput 0.0;
      seeds 17, 27 and 37 flow at 23 to 34. The 17.2 ± 16.2 reported under task 80
      was averaging two distinct regimes. Any reference at this operating point must
      report the modes or the count in each, never a mean.
    - **The bound fails for humans too**, not just AVs: seed 47 has 9 human-human
      collisions with no AVs present at all. So this is not an AV-control problem.
    - **AV runs die early, at steps 171 to 319**, and carry *higher* throughput
      (25 to 37) than the reference right up to the crash. They were working.

    **Why the bound fails.** It caps speed on the nearest vehicle ahead within a
    2.6 m lateral window. Density rises with run length until steady state -- 2000
    veh/h over 600 s spawns 333 vehicles against 67 over 120 s -- and at higher
    density the collisions measured earlier were side-by-side, at 1.9 to 3.7 m
    lateral, outside or at the edge of that window. A forward-looking window cannot
    see a conflict that is currently beside the vehicle and converging.

    **The fix is predictive rather than a wider window.** Widening to 4 m would make
    every adjacent-lane vehicle a longitudinal constraint and over-brake multi-lane
    sections. What is needed is closest-point-of-approach: for each pair, project
    both velocities, and if the predicted miss distance is under a vehicle width
    within the braking horizon, treat it as a conflict. That covers converging paths
    the current test misses and does not constrain parallel traffic that never
    meets.

## Ordering correction 2026-09-09: task 71 precedes training

Raised by the user, and it is right. Task 71 puts a route-aware leader gap on the
Jetson, and that lands in the **sensing model**, which `src/envs/topology_env.py`
uses to build every agent observation. It is therefore the same dependency that
already put task 9 ahead of task 68, and it changes three things:

- **`range_m`** stops being an optics estimate (100.0 m, from a 1.8 m vehicle
  spanning 14.4 px) and becomes measurable from the replay.
- **New error terms with no current analogue.** A map-matched gap carries
  map-matching error, polyline resolution, and a failure mode the sim does not
  model at all: matching the wrong road.
- **The parity class for `leader_gap`**, which is `identical` today only because
  every parity scene is single-arc.

So training before 71 produces a policy tuned to a precision the vehicle will not
have — the objection recorded in the task 67 round 3 audit, which was then only
half-acted on. **Revised order: 71, then 9, then 73, then 68, then 69.**

**The one thing this does not block.** A replication is a simulator claim, so the
throughput number does not depend on the Jetson. Training may proceed on the
current sensing block if the block is described as provisional; it may not if the
trained policy is the one to be deployed. That distinction is the user's.

75. **The arcs do not join. Vehicles are teleported sideways at every node.**
    Open, and it is the root cause of what task 72 could not reach.

    Measured distance between the end of each lane and the start of the lane
    `next_lane` sends a vehicle to:

    | transition | lateral jump |
    |---|---|
    | `a1/a2/a3_entry` → `('b1','c',0)` | **4.00 m** |
    | `('b1','c',0)` → `('c','exit',0)` | 2.00 m |
    | `('b1','c',1)` → `('c','exit',1)` | 2.00 m |
    | `('b2','c',1)` → `('c','exit',1)` | **10.00 m** |
    | `a4/a5/a6_entry` → `('b2','c',1)` | 0.04–0.09 m (these do join) |

    A vehicle crossing node `c` from `('b2','c',1)` is moved **ten metres**
    sideways, which is two lanes. No car-following bound can prevent a collision
    caused by a vehicle being placed into occupied space, which is why task 72's
    residual is exactly six side-by-side events at 1.9–3.7 m lateral separation,
    and why tightening MOBIL's `LANE_CHANGE_MAX_BRAKING_IMPOSED` from 2.0 to 0.05
    changed the count not at all.

    **It also explains two findings deferred from the task 67 audit.** The static
    successor map disagreed with the `lane_index` vehicles acquire on 9 of 9
    `('b2','c',1)` → `('c','exit',0)` transitions — because after a 10 m jump the
    geometrically nearest lane is not the steering target. And the one colliding
    pair the successor filter wrongly excluded collided 5–7 m past node `c`.

    So this is one defect with four symptoms, and fixing it is what makes
    collision-freedom achievable. The fix is that each lane's end must coincide
    with the start of its successor.

    Also noticed: `next_lane` on `('c','exit',k)` returns that same lane, giving a
    900 m self-jump. Harmless today because nothing walks past the exit, but it
    means the terminal arc has no proper successor.

76. **Most of this simulator's congestion was crash-induced queueing.** Open.

    Measured at high demand, penetration 0.10, `no_av`, with and without task 72's
    collision-free bound:

    | topology | collisions allowed | collision-free |
    |---|---|---|
    | `inverted_tree` | jam 0.2523 | **0.0990** |
    | `merge` | jam 0.0580 | **0.0000** |

    A crashed vehicle stops permanently and everything behind it backs up, so what
    the health check read as congestion was substantially a crash queue. On `merge`
    it was **all** of it, and its mean speed rose 14.13 → 20.11 m/s.

    **What this costs.** Task 8's `congestion_reachable` criterion was largely
    measuring crashes, so its topology ranking needs re-deriving. More importantly,
    a replication needs congestion for a controller to have anything to improve,
    and `inverted_tree` retains only 0.0990 — which is why it was the right
    topology to pick, but not obviously enough on its own.

    **Consequence for task 74:** reaching genuine congestion now has to come from
    demand and duration rather than from crashes. That is exactly task 8's own
    recommended diagnostic — raise demand, episode length and penetration together
    — arrived at from the opposite direction.

72. **Collisions must be impossible by construction, as in PTV Vissim.**
    **DECIDED BY THE USER 2026-09-09.** Blocks 73 and 74.

    **The instruction:** collisions are not something you generally see on a road,
    and a traffic simulator should make them structurally impossible rather than
    penalise them. PTV Vissim and SUMO both guarantee this in their car-following
    models; this simulator does not.

    **It is a prerequisite, not a preference, and the arithmetic says so.** At the
    current 120-step episode the AV arms complete 30 of 54, so per-step survival is
    0.995114 and the expected time to the first AV crash is **205 steps, about 3.4
    minutes**. Extrapolated:

    | episode | P(no AV crash) |
    |---|---|
    | 120 steps (current) | 0.556 |
    | 600 steps (10 min) | 0.053 |
    | 1200 steps (20 min) | 0.0028 |
    | **3600 steps (one hour)** | **2.2e-8** |

    An hour of simulated driving is unmeasurable until this lands. Every
    hour-long number would come from the vanishing fraction of runs that happened
    not to crash, which is the selection effect task 69 already had to work around
    at 7 of 18.

    **What it dissolves.** Task 67's entire premise — vehicles colliding because
    they cannot see each other — stops being a thing to mitigate. The `-5.0`
    `collision_count` reward weight becomes inert. `terminated` stops firing, so
    truncation ceases to be a criterion at all, and task 8's `episodes_complete`
    becomes trivially satisfied rather than structurally broken.

    **Three mechanisms are needed, in this order:**
    1. A safe-velocity cap for the in-lane leader — the rear-end case. A Krauss or
       Gipps bound, `v_safe = -b*tau + sqrt((b*tau)^2 + v_lead^2 + 2*b*gap)`, is
       collision-free by construction given both parties decelerate at `b`.
    2. The same cap against the merge-projected leader, reusing task 67's
       projection, which covered 27 of 51 measured collisions.
    3. Gap acceptance on lane changes — the lateral case, 6 of 51.

    **Design question, open:** enforce it once in the environment's substep loop,
    where `sub_dt` is known and it can cover humans and AVs together, or inside
    each vehicle model. The former gives one invariant and one place to test; the
    latter keeps each model self-contained. Recommendation: the substep loop,
    because a guarantee that lives in two models is a guarantee that can disagree
    with itself.

73. **Reweight the team reward so throughput is not 1.3% of the signal.**
    **DECIDED BY THE USER 2026-09-09.** Waits on 72.

    Measured over the 100-update run: `mean_speed` contributes 69.3% of the
    positive reward (weight 0.05 against a mean of 20.86) and `throughput_recent`
    contributes **1.3%** (weight 0.02 against a mean of 0.95) — a ratio of 55 to 1,
    because the weights do not normalise for scale. `throughput_recent` is a count
    of completions in a 60 s rolling window, so its magnitude also depends on the
    traffic state: about 0.95 during training, 5 to 15 at an evaluation's final
    step.

    Waits on 72 because the `-5.0` collision weight is currently a live term and
    becomes inert once collisions are impossible, which changes what the remaining
    weights have to balance against.

74. **Widen the evaluation to an hour of simulated driving.**
    **DECIDED BY THE USER 2026-09-09.** Waits on 72.

    The present evaluation is 18 conditions per arm at 120 steps, of which only 7
    had both arms complete. `throughput_recent` is an **integer** count, and it read
    5 to 15, so the resolution is one vehicle — 7% to 20% of the value. The paired
    result was +0.57 vehicles with four of seven conditions exactly equal, which is
    what that resolution produces rather than a measurement of the controller.

    An hour per run at `dt = 1.0` is 3600 steps, about 75 s of wall time per run at
    the measured 0.02 s per step. Seeds are the cheap axis now that topology and
    demand are fixed.

# Three-arm attribution, resolved 2026-09-09

The validator's round 3 finding 1 asked which of two changes the +17pp completion
gain belonged to, since `MergeAwareIDMVehicle` carries both a merge rule and a
low-speed stopping floor. Measured on the 81-run distinct grid, three arms:

| arm | completion | collisions (9 `no_av` conditions) |
|---|---|---|
| A plain IDM | 43/81 (53%) | 155 |
| B merge rule only | 57/81 (70%) | 126 |
| C merge rule + stopping floor | 57/81 (70%) | 138 |

**The merge rule accounts for the entire +17pp. The floor accounts for none of it,
and costs 12 collisions.** The floor was added to stop vehicles reversing through
zero speed, which it does — no vehicle reaches a negative speed with it — but the
extra stopping distance it imposes gives back 12 of the 29 collisions the merge rule
saves. From 5 m/s, unrestricted braking at 6 m/s² travels 2.08 m to rest while the
floor's geometric decay travels 3.26 m.

So the honest attribution is: the merge rule is the improvement, and the floor is a
correctness fix for the reversing defect that costs 12 collisions. It should stay,
because a vehicle driving backwards through traffic is a worse defect than 12
collisions, but it must not be credited with any of the completion gain.

# Task 77 resolved 2026-09-09: closest-point-of-approach helps, and does not solve it

`CollisionFreeMixin` now asks where a vehicle is GOING as well as where it is. For
every pair it projects both velocities, computes the time and distance of closest
approach, and imposes a stopping limit when the predicted miss is under 2.6 m
within a 4 s horizon. Parallel traffic in an adjacent lane is untouched, which is
what a wider lateral window could not have achieved.

Measured at the `saturating` demand over 600 steps, 5 seeds each:

| | before | with CPA |
|---|---|---|
| runs completing | 6/10 | **7/10** |
| total collisions | 23 | **14** |

**Collisions fall 39%. Completion improves by one run in ten. Four seeds get
worse** — `no_av` seeds 27 and 37 go from 0 collisions to 4 and 2, and
`backpressure` seed 17 stops completing. Braking for a predicted conflict creates
work for the vehicle behind, so the rule trades one collision mode for another.

**So collision-freedom is not achieved, and two mechanisms have now been tried.**
The safe-velocity bound took human-only collisions from 98 to 6 at short durations;
the geometry fix took the worst node discontinuity from 10 m to 0 m; CPA takes the
remaining total from 23 to 14. Each helped and none finished the job.

**The decision this now needs.** Constraining `highway_env`'s kinematics from
outside has produced diminishing returns across three attempts. PTV Vissim and SUMO
do not constrain a car-following model — their models cannot produce a collision in
the first place, because the safe speed is the model rather than a cap applied to
it. Continuing to add constraints is one option; replacing the vehicle model with
one that is collision-free by construction is another; running the replication on
SUMO, which is what the target papers use, is a third. That is a modelling decision
rather than a defect to fix, and it is the user's.

**Cost note.** CPA adds a second O(n²) pass per substep. A 600-step run at the
saturating demand went from about 12 s to about 40 s, so the 81-run grid is now
roughly 25 minutes rather than 8.

# PAUSED 2026-09-09

## The blocking unknown

**The collision-free bound is forward-looking only, and fails as density rises.**
`CollisionFreeMixin.perceived_safe_speed` in `src/vehicles/safe_following.py` caps
speed on the nearest vehicle ahead within a 2.6 m lateral window. At the
`saturating` demand the AV arms die at steps 171, 225, 290 and 319 of 600, and
`no_av` seed 47 has 9 human-human collisions with no AVs present, so it is not an
AV-control problem. The collisions are side-by-side at 1.9 to 3.7 m lateral —
outside or at the edge of that window — because a forward-looking test cannot see a
conflict that is currently beside the vehicle and converging.

**How to settle it.** Replace the lateral window with closest-point-of-approach:
for each pair, project both velocities, compute the time of closest approach and
the miss distance, and treat it as a conflict when the miss distance is under a
vehicle width inside the braking horizon. **Do not simply widen `SAFE_LATERAL_M`** —
4 m makes every adjacent-lane vehicle a longitudinal constraint and over-brakes
multi-lane sections, which `tests/test_collision_free.py::test_traffic_still_moves`
exists to catch.

**A second fact that blocks evaluation independently.** `no_av` at this demand is
**bimodal**, not noisy: seeds 7 and 47 gridlock to throughput 0.0 while 17, 27 and
37 flow at 23 to 34. A reference mean is meaningless there. Any comparison must
report the modes, or per-seed values, or use a demand below the bistable region.

## Exactly where I stopped

- `dsrc`, branch `main`, **clean tree, 0 commits ahead of `origin/main`**, HEAD
  `137b125`. Everything is pushed.
- Suites at HEAD: **simulator 337 passed**, **Jetson 2319 passed / 24 skipped**.
- **A trained checkpoint exists but will be lost.** The session scratchpad holds
  `train2/mappo_inverted_tree_full_seed7` with `actor.pt`, `critic.pt` and
  `config_resolved.yaml`, trained 100 updates under the throughput-led reward.
  The scratchpad is session-local. Retraining is about 40 minutes.
- Nothing is running. Two background jobs completed; a third measurement of the
  reference across more seeds was never started.

## What was learned that would otherwise be re-derived

- **Collisions were most of this simulator's congestion.** `merge` entirely,
  `inverted_tree` 61%. Task 8's `congestion_reachable` was largely measuring crashes.
- **The 120-step episode hid that `high` demand is over-saturated**: jam 0.149 at
  120 steps, 0.681 at 360, 1.000 at 900. Capacity is about 1800 veh/h.
- **`no_av` cannot fail `episodes_complete`** — it has no AVs and `terminated` tests
  AV crashes — so that criterion was uninformative for every task 8 cell.
- **The arcs did not join.** A vehicle crossing node `c` from `('b2','c',1)` was
  moved 10 m sideways. Fixed; worst jump is now 0.00 m on all six topologies.
- **The reward is a shared team reward**, not ego speed. Nine weighted network
  metrics. My earlier claim otherwise is corrected in the plan.
- **The 55-to-1 speed-to-throughput ratio was demand-specific.** At `medium`
  throughput averages 0.95; at `saturating` it averages 11.5 and the shipped weights
  already gave it 19%.
- Successor map after the geometry fix: `a1+a3 → ('b1','c',0)`, `a2 → ('b1','c',1)`,
  `a5 → ('b2','c',0)`, `a4+a6 → ('b2','c',1)`. Merge-conflict fixtures need a
  converging pair, so `a4`+`a6` or `a1`+`a3`.

## Task order to resume

1. **Task 77** — closest-point-of-approach in the bound. Then re-measure per seed;
   the target is every arm completing 600 steps.
2. **Re-measure the reference** with enough seeds to bound the bistability, and
   report modes rather than means.
3. **Task 74** — the hour-long evaluation, using `scripts/evaluate_replication.py`,
   which reads the sensing block from the checkpoint and refuses if it is absent.
4. **Task 82** — score the checkpoint on `build_team_reward`, not
   `mean_speed − jam_fraction`.
5. **Task 78 steps 2 and 3** — Jetson-side HERE work, which gates a deployable
   policy but not a simulator result.

## Open decisions not yet made

- **Whether the replication number waits for task 78 step 3.** A simulator claim
  does not depend on the Jetson; a deployable policy does. Recorded under task 78.
- **Whether the six rear fields are dropped from the encoding or held at the rig's
  constants.** Recommendation on record: hold, because the vector width is baked
  into `sim_contract` on both sides.
- **Whether to evaluate at `saturating` (bimodal) or find a demand below the
  bistable region.** Not yet investigated; 2200 was measured as congested and
  heavier, and may be more stable than 2000.


---

# `plan_simulations.md`

Moved here on 2026-09-12. The original plan, written before the deployment existed. Its
premise -- network-level congestion control realised through local sensing -- is the
formulation the project left.

# DSRC Plan

**Centralized training, decentralized execution for self-regulating autonomous vehicles that use local sensing to realize network-level congestion control through smooth speed/headway damping and conservative lane preferences.**

This is a natural extension of the current self regulating cars papers. Earlier sensys draft already motivates vehicle-level local sensing, density estimation, speed regulation, partial adoption, and learning/adaptation beyond the current lookup-table controller. 

Below is the project structure and a step-by-step execution plan.

---

# 1. Core hypothesis

The main hypothesis should be:

> A small fraction of autonomous vehicles, trained with centralized traffic-level feedback but deployed with only local noisy sensing, can physically realize network-level congestion-control policies through smooth desired-speed and desired-headway targets with conservative lane preferences.

This connects several ideas:

1. **Local density and speed advisories** are the interpretable non-learning version.
2. **Self-regulating AVs** are the infrastructure-free physical realization.
3. **CTDE RL** learns when and where AVs should harmonize speed, increase headway, hold lanes, or create merge gaps.

> AVs act as mobile actuators that implement traffic-control policies from inside the flow.

The v2 deployed cooperation model should use only aggregate traffic-state context available in each AV's public local observation, not infrastructure sensing, identity-level V2V messages, runner-aggregated fleet state, or joint lane-occupation plans. Each AV may observe nearby AV count, density, mean speed, queue estimates, downstream congestion estimates, segment target speed, and merge pressure as local observation fields. If no AV is nearby, the policy must fall back to individual local operation.

---

# Safety-Constrained Physical Flow Control

AVs should control flow through smooth longitudinal damping and cooperative gap creation, not through obstruction. Good performance should not be achievable through lane hogging, oscillatory lane changes, rolling roadblocks, or trapping human drivers.

Primary mechanisms:

```text
speed harmonization
adaptive headway control
cooperative merge gap creation
lane-change suppression near bottlenecks
backpressure-inspired speed metering
disturbance absorption / jam wave cancellation
```

The actor should output:

```text
desired_speed_bin
desired_headway_bin
lane_preference: keep | prefer_left_if_safe | prefer_right_if_safe
merge_mode: normal | create_gap | hold_lane
```

The safety/etiquette layer should enforce:

```text
minimum lane-change dwell time
maximum lane changes per km
safe front and rear gaps
rear braking limit for target-lane followers
bounded acceleration and deceleration
minimum contextual speed
no low-speed driving in uncongested conditions
no coordinated all-lane slowdown unless downstream congestion justifies it
```

Allowed communication:

```text
aggregate density
mean speed
queue estimate
downstream congestion estimate
segment-level target speed
merge pressure
```

Disallowed communication:

```text
joint lane occupation plans
AV-to-lane blocking assignments
coordinated roadblock formations
```

---

# 2. Four topology levels

## Level 1: Ring road

Purpose: show emergent stop-and-go wave damping.

```text
closed circular road
```

What it tests:

```text
wave damping
speed stabilization
partial AV penetration
local sensing sufficiency
```

Main actions:

```text
desired speed only
```

Primary metrics:

```text
speed variance
jam duration
wave amplitude
mean speed
recovery time after perturbation
```

This is the cleanest proof of concept.

---

## Level 2: Open straight highway

Purpose: introduce demand, inflow, outflow, throughput, and virtual detectors.

```text
inflow  --->  straight highway  --->  outflow
```

Use two versions:

```text
2A. single-lane straight highway
    isolates speed control

2B. multi-lane straight highway
    introduces conservative lane preferences and lane-change suppression
```

What it tests:

```text
throughput
travel time
lane utilization
demand sensitivity
burst response
```

Primary metrics:

```text
downstream throughput
mean travel time
speed variance
jam fraction
lane utilization
hard braking
```

This is the right “middle” topology because it adds traffic-demand realism without merge complexity.

---

## Level 3: Merge / bottleneck

Purpose: isolate cooperative gap creation, headway control, and conservative lane preference behavior.

Use a Y-merge.

```text
mainline ----\
              ---> downstream trunk
ramp --------/
```

What it tests:

```text
merge coordination
gap creation
conservative lane preference
bottleneck smoothing
queue reduction
```

Primary metrics:

```text
merge delay
queue length upstream of merge
trunk throughput
hard braking near merge
speed drop near bottleneck
lane distribution before merge
```

This is the necessary bridge between the straight road and the inverted tree.

---

## Level 4: Inverted tree

Purpose: final network-level stress test.

```text
A1 ----\
A2 ----- B1 ----\
A3 ----/         \
                  C ---- D ---- exit
A4 ----\         /
A5 ----- B2 ----/
A6 ----/
```

What it tests:

```text
multi-branch congestion propagation
network-level regulation
fairness across branches
spillback control
multi-merge coordination
```

Primary metrics:

```text
downstream trunk throughput
queue length per branch
mean travel time per origin branch
merge delay at each merge node
segment occupancy over time
speed variance per segment
spillback depth
branch fairness
```

Add a fairness metric. Otherwise a controller could improve trunk throughput by starving one branch.

Useful fairness metrics:

```text
std(branch travel times)
max branch queue length
min_branch_throughput / avg_branch_throughput
Jain fairness over branch throughputs
```

---

# 3. Updated project structure

The current repository is organized like this:

```text
dsrc/
  README.md

  configs/
    topology/
      ring.yaml
      straight_single_lane.yaml
      straight_multilane.yaml
      merge.yaml
      inverted_tree.yaml
      inverted_tree_bottleneck.yaml

    demand/
      low.yaml
      medium.yaml
      high.yaml
      burst.yaml

    human_models/
      cautious.yaml
      normal.yaml
      aggressive.yaml
      heterogeneous.yaml

    experiments/
      exp_ring_wave_damping.yaml
      ... planned final experiment configs ...

    training/
      shared_ppo.yaml
      ippo.yaml
      mappo.yaml

  src/
    envs/
      base_ctde_env.py
      topology_env.py
      wrappers.py

    road/
      topology_factory.py
      ring.py
      straight.py
      merge.py
      inverted_tree.py
      segment_graph.py

    demand/
      spawner.py
      demand_profiles.py
      route_sampler.py

    vehicles/
      behavior_profiles.py

    sensing/
      local.py

    safety/
      constraints.py
      etiquette.py
      safety_layer.py

    metrics/
      segment_metrics.py
      global_metrics.py
      safety_metrics.py
      fairness_metrics.py
      logger.py

    baselines/
      controllers.py
      registry.py

    rl/
      actions.py
      controller.py
      encoders.py
      models.py
      ppo.py
      rewards.py
      rollout_buffer.py
      trainers.py

  scripts/
    run_baseline.py
    train_policy.py
    evaluate_policy.py
    validate_project_interface.py
    validate_topology_baselines.py
    validate_training_eval.py
    run_experiment_matrix.py         # planned

  outputs/
    checkpoints/
    metrics/
    validation/
```

---

# 4. Baselines

You should use the following baseline ladder.

## B1. No AVs / human-only traffic

All vehicles use the default human driving model.

Purpose:

```text
lower-bound traffic performance
natural congestion formation
```

Use this for every topology and every demand level.

---

## B2. Random AVs

AVs receive local observations but choose random desired speed/lane commands.

Purpose:

```text
sanity check
shows improvement is not just due to AV presence
```

---

## B3. Selfish AVs / non-cooperative AVs

Each AV optimizes only its own progress.

Reward:

```text
+ ego speed
+ ego progress
- collision
- hard braking
- excessive lane changes
```

No global throughput reward.

Purpose:

```text
tests whether selfish autonomy worsens or fails to improve network flow
```

This is an important baseline because it contrasts “autonomous driving for myself” with “autonomous driving as traffic regulation.”

---

## B4. Density lookup controller

This is the direct continuation of your current paper.

Policy:

```text
local density estimate -> target speed bin
```

Example:

```text
low density      -> high target speed
medium density   -> moderate target speed
high density     -> reduced target speed
jam density      -> strong damping target speed
```

Purpose:

```text
simple interpretable self-regulation
non-learning baseline
```

This should be strong on ring and straight highway, weaker on merge/tree where headway and merge-gap behavior matter.

---

## B5. Dynamic speed limit: local AV speed advisory

This is an infrastructure-free approximation of a dynamic speed limit.

Each AV uses only its own local observation to choose a temporary speed advisory for itself.

```text
local density / queue estimate -> local AV speed advisory
```

Only the AV acts on the advisory. Human vehicles are affected only through ordinary car-following dynamics.

Purpose:

```text
interpretable local congestion metering
```

This baseline answers:

> How fast should an AV go given locally sensed congestion?

---

## B6. AV-mediated speed harmonization

This is the flow-smoothing version of local AV speed control.

Each AV smooths relative to nearby traffic speed, leader speed, and local flow using only its public local observation.

```text
local traffic speed estimate exists
only the observing AV realizes the target smoothly
human vehicles are influenced physically through car-following
humans may pass if safe
AVs do not change lanes solely to cover all lanes
AVs do not slow below local traffic-control targets by more than a small tolerance
```

Purpose:

```text
tests whether sparse AVs can damp local speed mismatch and waves
```

This is probably one of your strongest baselines/contributions. You can say:

> Sparse AVs can approximate speed-harmonization effects by acting as moving compliant vehicles, not roadblocks.

Compare:

```text
local density advisory with 5%, 10%, 20% AV penetration
local flow harmonization with 5%, 10%, 20% AV penetration
learned CTDE AV policy
```

---

## B7. Local backpressure-style control

Classic backpressure idea:

```text
pressure(edge) = upstream queue - downstream queue
```

At a merge, if one branch has high pressure and downstream capacity exists, that branch should be released more aggressively. If downstream congestion is high, upstream branches should be slowed.

For highways, implement pressure-inspired behavior not as a traffic light or infrastructure controller but as **local AV speed/gap regulation**:

```text
high upstream pressure + low downstream congestion:
    allow faster target speed

high downstream pressure:
    reduce upstream speed to avoid spillback

merge imbalance:
    use AVs to create gaps or meter branch inflow
```

Purpose:

```text
local network-pressure baseline
tests queue-aware regulation from each AV's observation
```

This is especially important for the inverted tree.

---

## B8. Cooperative adaptive cruise control / smoothing controller

Simple rule:

```text
AV slows down when local density is high or leader speed variance is high
AV maintains larger headway near bottlenecks
AV avoids unnecessary lane changes
AV uses only local observation and local aggregate cooperation fields
```

Purpose:

```text
strong hand-designed decentralized baseline
```

This gives reviewers a non-RL decentralized controller to compare against.


---

## B9. CTDE learned policy: speed + headway + conservative lane

Main method.

Actor:

```text
local noisy AV observation -> speed/headway bins + conservative lane preference + merge mode
```

Critic:

```text
global segment state during training
```

Execution:

```text
decentralized, local sensing only
```

This is your main proposed method.

---

# 5. Human driving models

Test robustness to multiple regular-vehicle models.

## H1. Cautious humans

```text
lower desired speed
larger headway
less aggressive lane changing
higher politeness
```

Expected behavior:

```text
fewer collisions
lower throughput
less unstable but slower
```

---

## H2. Normal humans

Default setting.

Use as the main result.

---

## H3. Aggressive humans

```text
higher desired speed
shorter headway
more frequent lane changes
lower politeness
higher acceleration/deceleration
```

Expected behavior:

```text
more stop-and-go waves
more hard braking
more merge conflicts
```

This is the most important stress test.

---

## H4. Heterogeneous humans

Mixture:

```text
30% cautious
50% normal
20% aggressive
```

This should be the main “realistic” setting.

---

# 6. Core environment API to standardize

Before adding more learning, standardize the environment interface.

Every topology should support:

```python
env.reset(config)
env.step(av_actions)
env.get_local_observations()
env.get_global_state()
env.get_segment_metrics()
env.get_episode_summary()
```

Every AV action should use the same format:

```python
action = {
    "desired_speed_bin": "slow" | "nominal" | "fast",
    "desired_headway_bin": "normal" | "larger" | "largest",
    "lane_preference": "keep" | "prefer_left_if_safe" | "prefer_right_if_safe",
    "merge_mode": "normal" | "create_gap" | "hold_lane",
}
```

Lane commands are conservative preferences only. Do not expose direct `left`/`right`, lane-coverage plans, or roadblock formations. The safety/etiquette layer decides whether any lateral command is legal, comfortable, and non-obstructive.

Every baseline should implement:

```python
class Controller:
    def act(self, local_obs, global_state=None):
        return av_actions
```

This makes all baselines interchangeable.

---

# 7. Step-by-step task plan

## Phase 1: Stabilize the environment

Focus only on environment correctness.

Tasks:

```text
1. Confirm that reset/step works without RL.
2. Confirm AV and RV creation works.
3. Confirm RVs follow default IDM/MOBIL behavior.
4. Confirm AVs accept desired speed and desired lane commands.
5. Confirm safety layer blocks unsafe lane changes.
6. Confirm vehicles are removed after exit.
7. Confirm no memory leak as vehicles spawn/despawn.
8. Confirm inactive vehicles are absent from AV/RV computation, controller inputs, rewards, and segment metrics.
```

Deliverable:

```text
one script that runs each topology with random actions for 1 episode
outputs a metrics CSV
renders or saves a basic visualization
```

Suggested script:

```bash
python scripts/run_baseline.py --topology ring --controller random_av
python scripts/run_baseline.py --topology straight_multilane --controller random_av
python scripts/run_baseline.py --topology merge --controller random_av
python scripts/run_baseline.py --topology inverted_tree --controller random_av
```

Do not start RL before this is stable.

---

## Phase 2: Build topology factory

Implement all four topologies with the same interface.

Tasks:

```text
1. ring.py
2. straight.py
3. merge.py
4. inverted_tree.py
5. topology_factory.py
6. segment_graph.py
```

Each topology should return:

```python
road_network
segment_ids
segment_lengths
entry_segments
exit_segments
merge_nodes
detector_locations
```

Deliverable:

```text
plots/topology_ring.png
plots/topology_straight.png
plots/topology_merge.png
plots/topology_inverted_tree.png
```

For each topology, verify:

```text
vehicles spawn correctly
vehicles follow routes correctly
segment IDs are correct
exit counting works
detectors count throughput
```

---

## Phase 3: Demand and traffic generation

Implement flow-based demand.

Tasks:

```text
1. Poisson vehicle spawning.
2. Demand levels: low, medium, high, burst.
3. Branch split ratios for merge/tree.
4. AV penetration rate.
5. Vehicle desired-speed distribution.
6. Human-driver type distribution.
```

Config example:

```yaml
demand:
  total_veh_per_hour: 2400
  av_penetration: 0.1
  branch_split:
    A1: 0.2
    A2: 0.2
    A3: 0.2
    A4: 0.2
    A5: 0.2
  burst:
    enabled: true
    start_s: 300
    end_s: 600
    multiplier: 1.8
```

Deliverable:

```text
demand sanity plots:
  vehicles spawned over time
  vehicles exited over time
  active vehicles over time
  per-branch arrivals
```

---

## Phase 4: Metrics and logging

Do this before baselines. Otherwise you will not know what works.

Implement per-step metrics:

```text
time
active vehicles
active AVs
completed vehicles
mean speed
speed std
jam fraction
hard braking count
collision count
lane changes
total queue length
throughput over recent window
```

Implement segment-level metrics:

```text
segment vehicle count
segment mean speed
segment density
segment queue length
segment jam fraction
segment AV count
segment inflow/outflow
```

Implement tree-specific metrics:

```text
queue per branch
travel time per branch
throughput per branch
branch fairness
merge delay per merge node
spillback depth
```

Deliverable:

```text
outputs/metrics/<experiment>/step_metrics.csv
outputs/metrics/<experiment>/segment_metrics.csv
outputs/metrics/<experiment>/episode_summary.json
```

This phase is crucial because your paper depends on network-level evidence.

---

## Phase 5: Local sensing model

Implement the observation model for AVs.

Each AV observation should include:

```text
is active
ego speed
ego acceleration
ego lane
current segment
distance to next merge
distance to downstream bottleneck
leader gap and relative speed
follower gap and relative speed
left-lane front/rear gaps
right-lane front/rear gaps
local density bins
local mean speed bins
segment-level local queue estimate
local active vehicle and AV counts
nearby AV count, density, mean speed, and lane distribution
optional nearby AV intent summary
```

Cooperation fields should be local aggregates only. They should not expose neighboring AV identities or direct V2V messages in v1. When `nearby_av_count` is zero, emit neutral aggregate values and require the controller to operate as an individual local policy.

Then add realism:

```text
distance-dependent detection probability
distance-dependent position noise
speed noise
latency buffer
field-of-view limit
occlusion optional
```

Deliverable:

```text
unit test comparing true local state vs noisy observed state
plots showing error vs distance
```

This directly connects to your current paper’s sensing-range/latency/noise story.

---

## Phase 6: Safety, etiquette, and physical-control layer

Safety has one DSRC controller path:

```text
all controllers: propose public v2 AV actions
HighwayTopologyEnv.step(): applies the common safety, etiquette, and physical-control layer
human drivers: continue to use highway-env IDM/MOBIL behavior where appropriate
```

The RL policy should not directly set unsafe acceleration. Learned controllers and baselines request speed/headway/lane-preference/merge-mode actions, and the common execution-time safety layer converts those requests into bounded physical behavior while reporting diagnostics and penalties.

Implement:

```text
desired speed bin -> target speed with acceleration limits
desired headway bin -> target headway
lane preference -> target lane only if safe and courteous
merge mode -> gap creation or lane-hold behavior
unsafe lane change -> blocked
short headway -> override speed downward
low TTC -> emergency safety behavior
low speed in uncongested conditions -> blocked
all-lane low-speed AV occupancy -> blocked unless downstream congestion justifies it
```

Safety checks:

```text
target lane exists
front gap sufficient
rear gap sufficient
time-to-collision safe
acceleration/deceleration bounded
speed limit respected
lane-change dwell respected
lane changes per km bounded
target-lane rear vehicle does not need excessive braking
```

Directional weighting:

```text
leader gap, leader relative speed, forward TTC, and downstream bottleneck distance are primary safety constraints
rear/follower checks remain required for lane changes
rear/follower pressure should not dominate longitudinal safety decisions
density/control objectives may use both upstream and downstream aggregates
```

Safety diagnostics should distinguish:

```text
safety_masked_action
etiquette_blocked_action
follower_disruption_blocked
external_safety_override
simulator_blocked_action
```

Deliverable:

```text
stress test with random AV commands
collision count should remain near zero or much lower than without safety layer
```

This is essential for a robotics venue framing.

---

## Phase 7: Non-learning baselines

Implement baselines in this order.

### 7.1 No AV

```bash
python scripts/run_baseline.py --controller no_av --topology ring
```

### 7.2 Random AV

```bash
python scripts/run_baseline.py --controller random_av --topology ring --av_penetration 0.1
```

### 7.3 Selfish AV

Local ego reward only.

```bash
python scripts/run_baseline.py --controller selfish_av --topology straight_multilane
```

### 7.4 Density lookup

Local density to target speed.

```bash
python scripts/run_baseline.py --controller density_lookup --topology ring
```

### 7.5 Dynamic speed limit

Local density and queue estimate to per-AV speed advisory. This is not an infrastructure speed-limit controller.

```bash
python scripts/run_baseline.py --controller dynamic_speed_limit --topology straight_multilane
```

### 7.6 AV-mediated speed harmonization

Local flow-matching and speed-mismatch damping from each AV's own observation.

```bash
python scripts/run_baseline.py --controller av_mediated_speed_harmonization --topology straight_multilane
```

### 7.7 Backpressure

Local pressure-inspired speed/headway metering near merge/tree bottlenecks.

```bash
python scripts/run_baseline.py --controller backpressure --topology inverted_tree
```

### 7.8 Cooperative smoothing

Hand-designed local smoothing controller that uses local aggregate fields without global state.

```bash
python scripts/run_baseline.py --controller cooperative_smoothing --topology inverted_tree
```

Deliverable:

```text
baseline comparison table before any RL
```

This lets you know whether the environment is producing meaningful effects.

---

## Phase 8: Topology-by-topology validation

Before RL, run the baseline ladder across every topology and demand regime with hard invariants plus directional sanity checks.

Deliverable:

```bash
python scripts/validate_topology_baselines.py --smoke
python scripts/validate_topology_baselines.py
```

Expected output:

```text
outputs/validation/task10/
  run_summary.csv
  directional_checks.csv
  validation_summary.json
  validation_summary.md
  runs/<controller>_<topology>_<demand>_seed<seed>/
```

Hard failures include broken reset/step, bad action keys, active-count mismatches, nonmonotonic completed counts, negative segment metrics, missing branch metrics, ring exits, missing high-demand spawns, and fairness values outside `[0, 1]`.

Directional warnings include sanity expectations such as selfish AVs having higher early speed than density-based controllers, density/harmonization improving saturated throughput or queues over random AVs in stressed straight roads, backpressure/cooperative smoothing helping merge or tree queues/fairness, and rolling-roadblock scores remaining near zero.

Directional warnings are not final paper claims. They are smoke signals for implementation sanity and should be interpreted across multiple seeds.

---

## Phase 9: RL training, simple first

Start with plain-PyTorch shared PPO and IPPO before full MAPPO. All algorithms use the same decentralized actor; only MAPPO uses `global_state` through a centralized critic during training.

Training order:

```text
1. Ring road, speed only.
2. Straight single-lane, speed only.
3. Straight multi-lane, speed + lane.
4. Merge, speed + lane.
5. Inverted tree, speed + lane.
```

Actor input:

```text
local noisy observation
local aggregate AV cooperation fields
```

Actor output:

```text
desired speed
desired headway
conservative lane preference
merge mode
```

Reward:

```text
global traffic reward shared across AVs
```

Initial reward:

```text
reward =
  + throughput
  + mean_speed
  - speed_variance
  - jam_fraction
  - hard_braking
  - collisions
  - excessive_lane_changes
```

For tree:

```text
reward =
  + trunk_throughput
  - total_queue_length
  - speed_variance
  - jam_fraction
  - merge_delay
  - fairness_penalty
  - hard_braking
  - collisions
```

Deliverable:

```text
one trained speed-only policy that beats no-AV and random on ring
```

Do not move to the tree until this works.

---

## Phase 9: MAPPO / CTDE

Once simple shared PPO works, move to CTDE.

Actor:

```text
π(a_i | local_obs_i)
```

Critic:

```text
V(global_state)
```

Global state:

```text
segment counts
segment densities
segment mean speeds
segment queues
AV counts per segment
merge queues
demand level
```

Training should use the same public v2 action interface as baselines. Unsafe or inappropriate learned actions are handled by the common execution-time safety layer, which reports masked, overridden, and blocked actions separately and supplies safety penalties for learning.

If no AVs are in the local neighborhood, the actor must fall back to individual operation using neutral aggregate cooperation fields.

For inverted tree, this can initially be a flat vector. Later, you can replace the critic with a graph neural critic.

Deliverable:

```text
MAPPO beats independent/shared PPO on merge and inverted tree
```

This is one of the main technical results.

---

## Phase 10: Main experiment matrix

After the method works, run the full evaluation.

Topologies:

```text
ring
straight_single_lane
straight_multilane
merge
inverted_tree
```

Controllers:

```text
no_av
random_av
selfish_av
density_lookup
dynamic_speed_limit
av_mediated_speed_harmonization
backpressure
cooperative_smoothing
CTDE_speed_only
CTDE_lane_only
CTDE_speed_plus_lane
```

Demand:

```text
low
medium
high
burst
```

AV penetration:

```text
0%
2.5%
5%
10%
20%
40%
```

Human model:

```text
cautious
normal
aggressive
heterogeneous
```

Sensing:

```text
perfect
realistic noise
high noise
latency 0.15 s
latency 0.5 s
limited range
```

Run at least:

```text
5 seeds for development
10+ seeds for final paper results
```

---

# 8. Main paper figures

Target these figures.

## Figure 1: System overview

Local AV sensing → speed/headway targets + conservative lane preference → physical damping → network and safety metrics.

## Figure 2: Four topologies

Ring, straight, merge, inverted tree.

## Figure 3: Speed heatmaps

Compare:

```text
no AV
selfish AV
density lookup
CTDE speed+headway+conservative lane
```

on ring or straight road.

## Figure 4: Throughput vs AV penetration

For straight, merge, and inverted tree.

## Figure 5: Queue length over time

Especially for inverted tree.

## Figure 6: Baseline comparison

Bar chart/table:

```text
travel time
throughput
jam fraction
fairness
```

## Figure 7: Human-driver robustness

Normal vs aggressive vs heterogeneous.

## Figure 8: Sensing robustness

Perfect sensing vs noisy sensing vs noisy+latency.

## Figure 9: Ablation

```text
speed only
lane only
speed + headway + conservative lane
local reward
global reward
CTDE critic
safety/etiquette diagnostics
rolling-roadblock score
```


---

# 9. Strongest story for the project

The final story should be:

> We ask whether a sparse fleet of autonomous vehicles can realize network-control effects from within the traffic stream without infrastructure sensing or obstruction. Using centralized training but decentralized execution, AVs learn local desired-speed and desired-headway targets with conservative lane preferences. A hard safety and etiquette layer prevents lane hogging, oscillatory lane changes, follower disruption, and rolling-roadblock behavior. Experiments across ring, straight highway, merge, and inverted-tree topologies show when local AV control can damp disturbances, create merge gaps, reduce spillback, and improve fairness under varying demand, human driving behavior, and sensing noise.


---

# `project_plan.md`

Moved here on 2026-09-12. The integrated plan. Its paper claim is regulation "using only
local sensing" and its experiment matrix is the highway-env topology ladder; both are
superseded. Its section 7, the final paper story, is the closest thing here to the current
argument and is worth reading beside
`paper_deploying_self_regulating_cars.md`.

# DSRC Integrated Project Plan

This is the top-level paper-facing roadmap. The subsystem details remain in:

- `plans/plan_simulations.md`
- `plans/plan_deployment.md`
- `plans/task_list.md`

## 1. Paper Claim

Sparse autonomous vehicles can regulate traffic from inside the flow using only local sensing, conservative public actions, and a common execution-time safety layer. The contribution is a deployability argument backed by a working system: how little must change in an ordinary vehicle for it to self-regulate, and how few such vehicles a road network needs.

Two consequences shape everything below. Nothing central sits in the real-time loop: the cloud trains and ships model updates offline, and communication is an observability hint, not a control dependency. And performance claims are separated by kind — traffic-control effects are measured in the original highway-env simulator through the project environment wrapper, while edge-feasibility claims are measured on the prototype hardware. Neither substitutes for the other.

## 2. System Stack

The project has two connected layers.

### Simulator and control foundation

The current repo provides:

```text
highway-env wrapper
topology ladder
demand generation
human-driver profiles
local AV sensing
public v2 action schema
common safety/etiquette/physical-control layer
metrics and logging
baseline controllers
model-free PPO/IPPO/MAPPO
```

This foundation establishes that sparse AVs can act as mobile actuators through smooth speed/headway targets and conservative lane preferences, and it is where every traffic-control number comes from.

### Advisory-only deployment prototype

The deployment plan demonstrates that local observation and policy inference run on vehicle-edge hardware:

```text
camera/GPS/optional OBD
perception and tracking
observation builder (sim-parity contract)
trained actor inference
safety/etiquette filter
dashboard and logs
no actuation
```

Sensing spans two devices, and the split is deliberate: the phone captures and forwards, while the Jetson owns every sensing setting as well as the policy. The phone holds none of the state that would justify a sensing decision.

The prototype is non-actuating. It is edge-feasibility evidence and the deployment argument's proof of existence, not an autonomous driving deployment.

## 3. Implementation Roadmap

### Phase 1: Preserve and validate the simulator foundation

Maintain the current environment, topology, demand, sensing, safety, metrics, baselines, and model-free RL stack. Keep running smoke validations before any behavioral change.

Primary scripts:

```text
scripts/run_baseline.py
scripts/train_policy.py
scripts/evaluate_policy.py
scripts/validate_project_interface.py
scripts/validate_topology_baselines.py
scripts/validate_training_eval.py
```

### Phase 2: Close the prototype hardware loop

The prototype is code-complete and validated on simulated drives. Remaining work is hardware, not software.

Deliverables:

```text
GPS device permissions resolved and rate confirmed
camera attached, selfcheck passing
mounted-camera calibration committed to config
recorded live run replayed for decision agreement
advisory hysteresis to damp leader-acquisition flicker
driver-facing readout showing the filtered cap, not the raw decode
in-vehicle advisory-only drive
```

### Phase 3: Extend prototype observation coverage

Roughly half the deployed observation is currently neutral fallback. Each item below converts placeholders into measured fields.

Deliverables:

```text
rear camera for follower and rear-gap fields
OBD-II speed as the primary ego-speed source
two-unit cooperative demo for nearby-AV and cooperation fields
map matching for distance-to-merge and downstream bottleneck
```

### Phase 4: Local-plus-aggregate simulation study

Replace global observation with the local-plus-aggregate contract and quantify what the aggregate buys, under one safety layer.

Deliverables:

```text
local-only, aggregate-assisted, and oracle controller comparison
message loss, delay, and staleness sweeps
sensing range and noise sweeps
compliance and penetration sweeps
bottleneck seeding vs uniform adoption at equal fleet size
```

### Phase 5: Build experiment launch infrastructure

Before final experiments, add launch and dry-run support for the complete matrix.

Deliverables:

```text
scripts/run_experiment_matrix.py
experiment configs under configs/experiments/
plot/table scripts
artifact manifest
```

The launcher should cover model-free baselines, learned-policy runs, robustness sweeps, and deployment metrics.

### Phase 6: Run final experiments and analysis

Only after all code paths and launchers are in place:

```text
run final training sweeps
run highway-env final evaluations
run robustness and partial-deployment sweeps
run deployment prototype measurements
generate figures and tables
```

## 4. Experiment Matrix

### Topologies

```text
ring
straight_single_lane
straight_multilane
merge
inverted_tree
inverted_tree_bottleneck
```

### Demand

```text
low
medium
high
burst
```

### Human models

```text
normal
heterogeneous
aggressive
```

### AV penetration

```text
5%
10%
20%
```

### Controller and method families

```text
no_av
random_av
selfish_av
density_lookup
dynamic_speed_limit
av_mediated_speed_harmonization
backpressure
cooperative_smoothing
SharedPPO
IPPO
MAPPO
```

### Observation regimes

```text
local-only
local plus aggregate
oracle/global (reference upper bound, not a deployable condition)
```

### Primary evaluation axes

```text
throughput
mean travel time
mean speed
speed variance
jam fraction
queue length
merge delay
spillback depth
branch fairness
hard braking
collisions
follower disruption
lane-change rate
rolling-roadblock score
sample efficiency
deployment latency/FPS
observation quality and field provenance
```

## 5. Safety And Evaluation Principles

- All controllers propose public v2 AV actions.
- The common DSRC safety layer in `HighwayTopologyEnv.step()` is the runtime enforcement path.
- The action to train on and measure is the one actually executed, not the raw proposal.
- Humans must remain passable when safe; performance must not come from obstruction.
- Branch fairness is required for merge/tree results.
- A network gain is never acceptable if bought by making human traffic less safe: report hard braking, collisions, and follower delay alongside every throughput result.
- The safety layer is specified, tested, and audited separately from the learned controller.

## 6. Deployment Link

The Jetson prototype is a first-class result, not an appendix. It establishes that the observation builder, actor inference, and safety filter fit inside a real-time budget on commodity hardware.

Deployment outputs:

```text
perception FPS
policy inference latency
end-to-end latency
ego-speed accuracy under dropout
observation quality and missingness
sim-to-prototype observation alignment
example dashboard
advisory-only safety statement
```

Two properties carry the deployment argument and should be reported explicitly: the observation contract is shared bit-for-bit with training, and every field is logged with its provenance, so a reader can see which values were measured and which were neutral fallbacks.

## 7. Final Paper Story

The paper should read as:

1. Highway congestion is a distributed stability problem, and the vehicle-side actuator has already been demonstrated; what remains is deployment.
2. Self-regulation needs no coordinator: each vehicle decides alone, and communication carries bounded aggregate state rather than commands.
3. The bill of changes that makes one ordinary car self-regulating, ending in a bounded advisory and a safety/etiquette filter.
4. A working edge prototype: sustained real-time operation and end-to-end latency well inside budget on commodity hardware, in advisory-only mode.
5. A minimal deployment model: because congestion is manufactured at bottlenecks, presence in a specific traffic stream beats market share, making a single fleet a sufficient launch vehicle.
6. The open networking questions this system raises: how little communication suffices, whether it can be private, what belongs in the cloud and how late it may be, robustness under partial deployment, and how safety should be verified.


---

# `result_simulation_leg_null.md`

Moved here on 2026-09-12. The MAPPO leg's own closing record. Written when that leg was stopped as a null, and still the best account of why it was.

# The simulation leg: what was measured, and why it is a null

One account of a result spread across task-list entries 92 to 134. It states what was
measured, what each measurement was controlled against, and exactly what is and is not
established. Sections below are in the order they were written; this summary is
current as of 2026-09-10 evening.

## Short version

**The mechanism this project posits -- in-stream AV speed modulation -- neither gains
when handed perfect information nor presents a learnable gradient, on this road.** Two
instruments built for different purposes agree:

- a perfect-information metering oracle serves **-0.8 and +0.8** more vehicles than
  no control, against a spread of about 16;
- the correlation between one AV's action and its own advantage is indistinguishable
  from zero: **+0.00 +/- 0.29** at a one-second lever and **-0.47 +/- 0.51** at a
  one-minute one, against an instrument that reads 17 to 33 times its null when a
  correlated advantage is supplied.

**The predecessor paper's own formulation was ported in full and is also a null at
the vehicle level.** Its two-term threshold reward, its 8.33/12.5/16.67 m/s speed
bins, one decision per simulated minute at gamma 0.9, and a corrected link-level
congestion signal every co-located AV sees identically. The paper's design does not
transfer to a vehicle-attached agent on this road.

**The objective is not the problem.** Paired over five seeds, the threshold reward
responds to fleet behaviour at every mix level and discriminates it at least as well
as counting arrivals does, so the null is not attributable to an unmovable or noisy
objective. Its entire resolved response is the speed term; the congestion-threshold
half, which is what pays for keeping a link below critical density, resolves at one
mix level of four and changes sign across them.

**Three things were NOT established when this section was first written. Two are now
measured and the third stands.**

1. **STANDS.** The oracle is one hand-written heuristic given perfect state, so it is
   a LOWER bound on the best controller and not an upper one. "No controller can gain
   on this network" is not proven and cannot be proven this way.
2. **RESOLVED.** Every arm that produced a clean null pays each agent the SAME reward, so those
   nulls cannot separate "the reward does not respond to individual actions" from
   "nothing responds to individual actions". Two arms have an unshared reward and BOTH
   HAVE NOW BEEN MEASURED: a per-agent local reward at the vehicle level reads a
   correlation of +0.0016 against the shared arm's -0.0042 at a standard error of
   0.0135, and a segment paid its own threshold term reads z -0.12 +/- 0.51. Neither
   separation appears, so the shared reward is not the reason.
3. **NULL, BUT AT A COARSE SENSITIVITY.** The paper's agent is a ROAD that persists for the episode; ours is a VEHICLE that
   makes 7.1 decisions and leaves. That difference is confounded with every lever
   measurement. The segment-attached agent that separates it reads null at both powers
   measured, -0.12 +/- 0.51 and -0.46 +/- 0.66. Scaled by the instrument's own
   only a z, which the calibration shows cannot separate a correlation of 0 from 0.1 on
   an arm this size.

**The instrument was rebuilt mid-investigation.** Its original permutation floor is
not a valid null when advantages are temporally correlated, and it made arms read
WORSE than random. The replacement resamples the action from the policy at the same
state; it has seven controls, and it reproduced the vehicle-level result at higher
precision, so the earlier work stands rather than falls. Readings taken under the old
floor were discarded rather than reported with a caveat.

## The environment is sound, and four defects had to be fixed to make it so

Each of these alone prevented any result. They are listed because the null means
nothing without them: a null on a broken environment is not a measurement.

| defect | what it did | fix |
|---|---|---|
| 1 s physics step | manufactured a 17.6% "throughput gain" that vanished at a converged step size | dt 0.1, and the result retracted (task 92) |
| unintended permanent yield at the merge | every edge had `priority="-1"`, so netconvert broke the tie by geometry and one approach was permanently minor | zipper junctions; capacity 1110 -> 1600 veh/h (task 98) |
| Krauss car-following | computes a collision-free safe speed exactly and recovers immediately, so there is no capacity drop and nothing to recover | the predecessor paper's calibrated Wiedemann-99, giving a 24% capacity drop (task 100) |
| single-lane approaches | a slow AV cannot be overtaken, so it is an obstruction and not a meter | two lanes on every approach (task 102) |

The oracle progression across those fixes is a sequence of removed harms and no
found benefit: -16.6 +/- 7.8, then +7.6 +/- 15.2 at congestion onset, then -21 to
-38 at the capacity peak, then -0.8 and +0.8 oversaturated with two lanes.

## The learning result, and the instrument that produced it

**The statistic.** With advantages normalised to unit standard deviation, the norm of
`d(policy_loss)/d(actor)` measures how much the advantage correlates with the action:
a term uncorrelated with the action cancels across the batch, an aligned one adds.
`policy_loss` itself says nothing, because at a probability ratio of 1 it is minus
the mean normalised advantage, which is zero by construction however informative the
advantage is.

**Its two controls, without which every reading is uninterpretable -- and the floor
had to be rebuilt.** The ceiling is a SYNTHETIC advantage built to correlate with the
action; it drives no vehicle and is not a controller, it exists to show the statistic
can move.

The floor was originally the same advantages permuted across the batch. **That is not
a valid null when advantages are temporally correlated**: it destroys the
action-advantage pairing AND each agent's temporal profile, so a smooth advantage
sequence becomes rough, cancels less against `grad log pi`, and the floor comes out
high. It was caught when two segment-level arms read a measured value systematically
BELOW their own null, which is that bias showing rather than data worse than random.

The valid null resamples an action from the policy at the SAME observation and keeps
the advantage, so states, advantages and their temporal structure all survive and only
the pairing breaks; under the score-function identity its expectation is zero. It is
tested (`tests/test_action_alignment.py`) against a signal it must see, noise it must
not, an autocorrelated-but-uninformative sequence, and a partial signal it must grade
-- none of which the permutation floor ever was.

**The result survived the repair.** On `mappo_sumo`, three seeds:

| null | mean z | ceiling |
|---|---|---|
| permutation floor | -0.10 +/- 0.42 | 25 to 39 |
| resampled action (valid) | **+0.00 +/- 0.29** | 31 to 33 |

Same conclusion, better precision. The table below therefore stands. Measured on the
shipped configuration, three seeds, 17,296 decisions:

| advantage | gradient norm | over the floor |
|---|---|---|
| as measured | 0.0499 | **0.784** |
| shuffled -- the floor | 0.0637 | 1.000 |
| action-correlated -- the ceiling | 1.5971 | **25.1** |

**Seven dimensions varied, none clearing the floor.** Three seeds each, floor and
ceiling on every arm:

| varied | range | best measured over floor |
|---|---|---|
| reward decomposition | team, neighbourhood, own-vehicle, both | 0.850 |
| action hold length | 1 s, 5 s, 20 s | 0.945 |
| discount horizon | 10 to 1000 decisions | within 1.1x |
| critic input | with and without privileged neighbourhood | within 1.05x |
| speed bin scaling | four schemes, binding share 6% to 29% | 1.020 |
| operating point | 900, 1200, 2400 veh/h | 1.269 |
| AV penetration | 0.25, 0.50, 1.00 | 1.403 |

At 100% penetration the policy commands the entire fleet -- 168 vehicles at 2400
veh/h -- and one agent's action still does not correlate with its own advantage.

**Those ratios are against ONE permutation, which is a single draw from the floor and
not the floor.** Estimated properly from 60 permutations per rollout, the floor's
standard deviation is 20 to 30% of its mean, so every ratio above is inside it. The
correct statistic is the z-score of the measured value in the permutation
distribution:

| arm | floor mean | floor sd | mean z over three seeds |
|---|---|---|---|
| `sumo_capacity_drop`, penetration 0.25 | 0.054 | 0.016 | -0.10 |
| `sumo_saturating`, penetration 0.25 | 0.088 | 0.025 | -0.01 |
| `sumo_saturating`, penetration 1.00 | 0.042 | 0.011 | +0.82 |
| `sumo_capacity_drop`, gamma 0.999 | 0.052 | 0.013 | +0.33 |

The largest is under one standard deviation. Penetration 1.00 was then run on ten
seeds: **mean z +0.630 +/- 0.379, which is 1.66 standard errors and does not clear
the project's two-standard-error bar.** Eight of the ten seeds are positive, one
reaches +2.36 and one -1.92. It is the single place in this investigation where "no
signal" might be wrong, and it is the arm to extend if the question is reopened.

It does not reopen the headroom question. A learning signal at 100% penetration would
say a policy could be trained, not that a trained policy would gain anything, and the
metering oracle still serves -0.8 and +0.8 more vehicles than no control. Both would
have to move. 100% penetration is also not a deployment operating point: every vehicle
on the road is controlled, against the 25% of the predecessor paper.

**The control on the instrument itself.** Every reading above was taken at a randomly
initialised actor, so a rising correlation with training would invalidate them. A
checkpoint 23 updates in reads 0.812 against a fresh actor's 0.848: indistinguishable.

## Two measurements that stand and do not reach the gradient

Both are real and neither changes the conclusion, which is worth stating so they are
not mistaken for support.

- **The reward counterfactual.** Holding one AV at 20 m/s for a 20 s window and
  repeating from the same seed at 30 m/s: the agent's own local reward moves 22.82,
  another agent's action moves it 1.28, a ratio of **17.8**. The team reward moves
  2.06 against a window total of 193.03, which is **1.07%**. So the per-agent reward
  is far better attributed than the team reward -- and neither survives into the
  advantage.
- **The critic regression.** Giving the centralized critic the agent's own
  neighbourhood raises out-of-sample R2 on the per-agent return from 0.808 to 0.886,
  and leaves it at 0.854 against 0.858 under the team reward, which is the control
  saying the improvement is about the per-agent term. The actor's gradient is
  unchanged.

## What was retracted, and by what

Recorded because the retractions are part of the result.

| claim | retracted by |
|---|---|
| AVs held at 10 m/s raise throughput 17.6% | the step-size sweep: the treatment was invariant to dt and only the control moved (task 92) |
| joint gradient clipping starved the actor | Adam is invariant to a uniform gradient rescale; 20 steps move a parameter 0.383268 unscaled and 0.383267 scaled by 0.0063 |
| the per-agent reward lowered the policy gradient; the horizon does not matter; privileged critic features do not help the gradient | all three were comparisons between two noise floors, taken before the floor was measured (task 105) |
| rescaling the speed bins would produce a gradient | three rescalings raise the binding share from 6% to 29% and leave the gradient at its floor; the recommended one is the worst (task 106) |
| the permutation floor is a valid null | two arms read systematically BELOW their own floor, which is the signature of a null not exchangeable with the data; replaced by resampling the action from the policy at the same state |
| every z-score error bar in the investigation | `seed_everything(0)` made the actor and critic weights bit-identical across seeds, so each bar came from ONE network draw; re-measured with the draw varying per seed |
| 7 of 9 segments run over critical density | inferred from a training score by assuming every segment ran at the vehicle-weighted network mean; measured directly at 1.5 of 9 (task 130) |
| one branch is congested and the other free | read off density ratios, but density is flow over speed, so a free-flowing link and an empty one look alike at fixed flow; both branches carried the same demand |
| the summed entropy declines monotonically | called from three updates; update 5 reversed it, and updates 7 and 10 reversed it again |
| the shared-reward segment arm reads consistently negative, and here is why | it reads +0.25 +/- 0.76 under the valid null with the initialisation redrawn per seed; the negativity was the instrument |
| the threshold objective is a noisier measure of behaviour than an arrival count | true unpaired and false paired: the ratio of each difference to its own bar is 2.79, 1.76, 1.75 and 2.63 for the reward against 2.91, 1.60, 1.29 and 0.85 for arrivals (task 133) |

## The pre-registered run

`configs/training/mappo_sumo.yaml`, seed 7, 25 updates of three episodes each, with
the gate fixed before the run. Criterion 1, summed entropy below 1.9775 of a 2.1972
maximum: the lowest reached is 2.1778. Criterion 2, the score trending up by more
than the variation between consecutive updates: it moves -0.1772 against a step
standard deviation of 0.2186. Criterion 3 as written carried no numeric threshold,
which is a defect in the pre-registration and is recorded as such rather than
resolved after the fact.

A randomly initialised actor already reads a joint modal share of 0.1378 against a
uniform 0.1111, and 23 updates take it to 0.1486, so most of the departure from
uniform is initialisation.

## SUPERSEDED IN SCOPE, 2026-09-10: this is a null about ONE formulation

Everything above was measured with a per-vehicle action taken once a second. Reading
the predecessor paper (arXiv:2506.11973) showed that is not the problem it solved.
Its agent sets a maximum speed for a 2-3 km super-segment once a MINUTE, from a state
of per-segment density, speed, gap, inflow and outflow, and rewards
`-alpha * 1[rho > rho*] + beta * v` -- two terms, not eleven. Its action is thousands
of times the lever, and it is a centralized controller with AVs as the compliance
mechanism, tested at 25 to 100% compliance.

**So the seven dimensions in the table above were all varied around a one-second
lever.** They do not bound what a macroscopic one does, in either direction. Ankit's
statement of it: each vehicle is changing decisions too quickly, nearby vehicles are
not matching, and too fine a control resolution devolves into noise.

**THE VEHICLE-LEVEL HALF OF THAT PORT IS NOW MEASURED, AND IT IS ALSO A NULL.**
`mappo_src` under the valid resampled-action null, three seeds: mean z
**-0.47 +/- 0.51** (per seed -1.33, +0.44, -0.53) against the one-second baseline's
+0.00 +/- 0.29. With a ceiling near 17 and a null standard deviation of 0.029 on a
null of 0.11, z is about 60 times the correlation AT THE CEILING. That slope does not
hold at small correlations -- z/c falls to 3.7 at c = 0.05, measured in task 137 --
so the bound from this arm is about 0.025 rather than the 0.017 an earlier version of
this paragraph gave. It remains a null from a sensitive instrument, not an insensitive
one, and the sensitivity is stated at the correlation it applies to.

So holding one action for a simulated minute -- with the paper's threshold reward, its
speed bins, its discount horizon, and a link-level congestion signal the actor can
see -- produces no more action-attributable signal than holding it for a second.

**One confound this cannot resolve.** The lever change carries a cost: at 60 s each
agent makes 7.1 decisions in its lifetime instead of 90.5, a thirteenfold loss of
temporal structure, because the agent is a vehicle that leaves rather than a road that
persists. A larger per-decision effect and a much shorter trajectory pull opposite
ways and this measurement cannot separate them. The segment-as-agent arms, where the
agent persists for the whole episode, are what can, and they have now reported: paid
its own per-segment reward, the road-attached agent reads -0.12 +/- 0.51 at 171
segment-decisions and -0.46 +/- 0.66 at 533. Neither separates from zero, so the
confound resolves in favour of neither side -- restoring the agent's temporal
structure does not recover a signal that the shorter trajectory was hiding.

`configs/training/mappo_src.yaml` ports the formulation. What it
changes and what was measured about each is task 113; the honest accounting of which
change is large is:

| change | size, measured |
|---|---|
| threshold reward | LARGE, but not as large as first claimed: the penalty fires on **1.50 of 9 segments** on average, measured directly by the fleet-mix sweep at p = 0 over a 600 s episode. The earlier figure of "about 7 of 9" was INFERRED from the training score using a bad approximation -- it assumed every segment runs at the network mean speed, but that is vehicle-weighted, so a nearly-empty free-flowing leaf contributes a segment mean of 25 m/s and almost nothing to the network mean |
| decision interval 60 s from 1 s | LARGE, and untested before this |
| `downstream_congestion_estimate` corrected to the link ahead, ungated | real: it was reading the ego link and vanishing whenever no AV was in range |
| speed bins 8.33/12.5/16.67 from 20/27/30 | SMALL: the band on which the three values differ is 6.2% against roughly 5%, because the median AV speed is 0.11 m/s and no command binds on a stopped vehicle |

## The structure underneath every null here: the reward is shared

Stated late, because it took the segment arms to make it obvious. **In every arm that
has produced a clean null, each agent is paid the SAME reward at each step.** The team
reward is a network aggregate; so is the threshold reward, summed over segments. With
a common reward the reward SEQUENCES are identical across agents, so the whole
cross-agent variation in the advantage comes from the critic's value estimates.

At the vehicle level that is not a defect, it is the configuration under test: a
shared reward that does not respond to an individual action is precisely the
credit-assignment problem. The measurement says so cleanly.

But it means a null on those arms cannot distinguish two things:

- the reward does not respond to individual actions;
- nothing responds to individual actions.

**Only an unshared reward separates them**, and exactly two arms have one: the
per-agent local reward of task 103, and the segment-as-agent arm paid its own term.

**THE FIRST HAS NOW BEEN MEASURED AND THE HYPOTHESIS IS WRONG.** Paired on identical
rollouts with the network draw varying, three seeds:

| arm | mean z | correlation between choosing `slow` and the advantage |
|---|---|---|
| team only, shared reward | -0.44 +/- 0.56 | **-0.0042** |
| local 0.5, UNSHARED reward | -0.81 +/- 0.14 | **+0.0016** |

At n about 5,500 the correlation's standard error is 0.0135, so both are noise and
they are indistinguishable from each other. **Giving each agent a reward that
genuinely differs from its neighbours' produces no more action-advantage alignment
than a shared one**, at sensitivity to a correlation of about 0.016.

So the shared reward is NOT the reason the advantage is silent. That was the best
remaining explanation and it is ruled out. What is left is the third item in the short
version -- the agent is a vehicle that makes 7.1 decisions and leaves, where the
paper's is a road that persists. The segment arm that tests it has since read null at
both powers run, and scaled by the instrument's own ceiling it would show a
correlation bound no tighter than about 0.1 (task 138).

**RETIRED, same day.** This section previously explained why the shared-reward
segment arm read consistently negative -- with a common reward its advantage is
essentially a function of the untrained critic, so the advantage and the action are
both functions of the same observation through independently initialised networks
(task 123). Re-measured against the resampled-action null with the network
initialisation redrawn per seed, that arm reads **+0.25 +/- 0.76** over three seeds
(+1.77, -0.61, -0.42), centred on zero. The -1.32 +/- 0.16 the explanation was built
for combined a fixed initialisation with the invalid permutation floor, so the
negativity was a property of the instrument and there is nothing left to explain. The
argument may still be correct; it no longer has an observation supporting it.

**THE SECOND HAS NOW BEEN MEASURED TOO, AND IT IS ALSO A NULL.** The segment-as-agent
arm paid each segment its own threshold term rather than the network sum -- verified a
genuine decomposition, in that a clear segment reads +1.000 and a jammed one -0.850,
summing to the network's +0.150 -- and was measured against the resampled-action null
with the network initialisation redrawn per seed:

| seed | segment-decisions | measured | null | z |
|---|---|---|---|---|
| 7 | 171 | 0.26741 | 0.30232 +/- 0.08000 | -0.44 |
| 17 | 171 | 0.24793 | 0.30957 +/- 0.07627 | -0.81 |
| 27 | 175 | 0.38380 | 0.31322 +/- 0.07969 | +0.89 |

Mean z **-0.12 +/- 0.51** over the three seeds, against an instrument ceiling of 6.2
to 6.6 on a supplied correlated advantage. So neither unshared-reward arm reads above
its null, and the reading is centred on the null rather than below it, which is what
the void permutation-floor arms did.

**CORRECTED, same day.** This paragraph put the arm's threshold at 0.15 from
2/sqrt(n), the sampling error of a correlation coefficient, which is the wrong formula
for an arm reporting a gradient-norm z. Scaled by the instrument's own ceiling the arm
reports only a z, and the calibration in task 138 shows that z cannot separate a
correlation of 0 from 0.1 on an arm this size. The
higher-power version runs three episodes per rollout instead of one, taking n to about
533 and buys no sensitivity, because the cross-seed standard
error rises with the per-seed sensitivity -- and still short of a clean test, which
needs the paper's five per-super-segment
observation fields and therefore a wider deployed contract. **It has now been
measured: -0.46 +/- 0.66 over three seeds (+0.86, -1.16, -1.09) at 532 to 533
segment-decisions**, and it buys no sensitivity, because the cross-seed standard error
rises along with the per-seed sensitivity. Tasks 129, 134 and 136.

## A property of the operating point that bounds every comparison here

The network is bistable. Five seeds of identical demand on an identical, structurally
symmetric road produce two regimes -- congestion in the middles with empty leaves, or
on the leaves with moderate middles -- and served trips range from 156 to 198, a 27%
spread. That is where the paired standard deviation of 17 to 32 arrivals comes from.
It belongs to the road, not to any controller, and it caps a five-seed comparison at
effects above roughly 8%. Task 114.

## The objective responds to behaviour; only its speed half does

The question prior to every gradient measurement is whether the objective can be moved
by behaviour at all. If it were flat, no learner could exploit it however well credit
were assigned, and the whole investigation would have been measuring the attribution
of a quantity with nothing to attribute. It is not flat.

Each AV independently issues the config's own `slow` command, 8.33 m/s, with
probability p at each decision and is otherwise released to SUMO's car-following.
Five seeds, each fixing both the traffic and the metering draw sequence, so the same
seed at two mixes differs only in the mix. Differences are from p=0 with
two-standard-error bars.

| p | d reward | d speed term | d penalty term | d arrivals |
|---|---|---|---|---|
| 0.25 | **-1.51** +/- 0.54 | **-1.24** +/- 0.72 | **-0.28** +/- 0.19 | **-25.4** +/- 8.7 |
| 0.5 | **-1.29** +/- 0.73 | **-1.47** +/- 0.82 | +0.18 +/- 0.37 | **-36.8** +/- 23.0 |
| 0.75 | **-1.70** +/- 0.97 | **-1.74** +/- 0.93 | +0.04 +/- 0.35 | **-32.0** +/- 24.7 |
| 1 | **-1.86** +/- 0.71 | **-1.72** +/- 0.83 | -0.14 +/- 0.50 | -27.8 +/- 32.8 |

Bold means the bar excludes zero. The measurement reproduces two independent
baselines before any comparison is drawn: its p=0 arrivals match the fleet-mix sweep's
reference seed for seed, 198, 194, 156, 179 and 196, and its p=0 reward over three
seeds matches the unpaired pass exactly.

**The objective resolves the behaviour change at every mix level, and at least as well
as counting arrivals does.** The ratio of each difference to its own bar is 2.79,
1.76, 1.75 and 2.63 for the reward against 2.91, 1.60, 1.29 and 0.85 for arrivals; at
p=1 the reward resolves the change and the arrival count does not. So no part of the
null is attributable to the objective being unmovable or noisy. It stays with the
mechanism and the credit assignment.

**Only one of the reward's two halves responds.** `congestion_penalty` is 1.0, so the
penalty term is exactly minus the count of segments over rho\* and the speed term is
the remainder. The speed term is resolved and negative at every level and is
essentially the whole difference. The penalty term resolves at one level of four and
changes sign across them. **The half of the objective designed to pay for keeping a
link below critical density -- the anticipatory behaviour the mechanism is supposed to
produce -- does not respond consistently to this behaviour, and the half that does is
the half that restates mean segment speed.**

**It tracks throughput in the mean and not per application.** Across the twenty
individual (mix, seed) treatment applications, the correlation between the change in
the objective and the change in arrivals is -0.035, with the speed term at -0.182 and
the penalty term at +0.318. The objective detects that metering happened; it does not
measure how much throughput it cost. At n=20 the two-standard-error bar on a
correlation is 0.485, so this excludes a tight relationship and not a moderate one.
The pooled correlation over all 25 cells is +0.582, but that mixes the shared
treatment trend with seed bistability and is the wrong statistic: two quantities that
both fall on average correlate positively whether or not their fluctuations are
related.

**Do not read this p against the fleet-mix sweep's p.** Both sweeps draw a Bernoulli
at each decision, at the same 1 s interval, on the same 6000-step episodes after the
same 3000-step fill -- their p=0 arms are identical, 198, 194, 156, 179 and 196 seed
for seed. What differs is the command. This one sets 8.33 m/s, a CEILING that SUMO's
car-following often already dominates: the network mean speed here is 1.9 to 2.6 m/s,
so the command frequently changes nothing. The fleet-mix sweep's `binding` scheme sets
0.6 times the vehicle's CURRENT speed, which always binds and compounds across
decisions. That is why the same nominal p costs 25 to 37 arrivals here and 56 to 147
there. The two measure different interventions and their p columns are not comparable.

**Two further limits.** The objective is maximised at p=0, by not metering at all, so within
this cut a policy that has learned nothing and a policy that has learned to leave the
fleet alone produce identical behaviour; improvement on this axis cannot demonstrate
learning. And the cut is one dimension, a uniform random command at a single speed
value, so it bounds what an unselective fleet can do to the objective rather than what
a selective policy could. The metering oracle covers the selective case with perfect
information and gains nothing.

## The explanations, and which survive

Every candidate for why the advantage is silent about the action, with what closed it.
The point of the table is that the eliminations are measurements rather than
arguments. Every row now has one. The last row's is still the weakest, but by less
than earlier text here claimed, and weaker than the row itself says: that arm reports
only a z, and a calibrated z cannot separate a correlation of 0 from 0.1 on an arm its
size (task 138). The other rows rest on directly measured correlations.

| explanation | status | what closed it |
|---|---|---|
| the reward does not respond to an individual action, because it is shared | **ruled out** | an unshared per-agent reward is equally silent: correlation +0.0016 against -0.0042 shared, standard error 0.0135 |
| the control resolution is too fine | **ruled out** | 60 s per decision reads -0.47 against 1 s at +0.00 |
| the speed bins do not bind | **ruled out** | 8.33 m/s binds on 21.4% of AV-steps against 10.0%, and changes nothing |
| the reward shape is wrong -- eleven weighted terms | **ruled out** | the paper's own two-term threshold reward gives the same result |
| the actor cannot see what the reward pays for | **ruled out** | it sees the exact density ratio of the link ahead, spread 0.282 across agents, agreeing with truth on 100% of observations |
| the critic cannot centre the advantage | **ruled out** | privileged neighbourhood features raise out-of-sample R2 from 0.808 to 0.886 and leave the gradient unchanged |
| penetration is too low | **ruled out** | 100% penetration, commanding the entire fleet, reads the same |
| the operating point has no headroom | **partly** | the metering oracle gains nothing, but it is one heuristic and a lower bound |
| **the agent is a VEHICLE that makes 7.1 decisions and leaves, where the paper's is a ROAD that persists** | **null, at a sensitivity too coarse to call it closed** | a segment paid its own reward reads -0.12 +/- 0.51 at 171 segment-decisions and -0.46 +/- 0.66 at 533, but these arms report only a z and the calibrated z cannot separate a correlation of 0 from 0.1 on an arm this size (task 138) |

**Every candidate now has a measurement against it, and the last one has two.** The
road-attached agent paid its own reward reads -0.12 +/- 0.51 at 171 segment-decisions
and -0.46 +/- 0.66 at 533. Both are centred on zero.

**How sensitive those arms are, corrected.** Earlier text here put their thresholds at
0.15 and 0.09, taken from 2/sqrt(n), the sampling error of a correlation coefficient.
That is the wrong formula: these arms report a gradient-norm z, not a correlation. The
instrument carries its own scale instead. Its ceiling advantage is an affine function
of the indicator of `slow`, so it has a correlation of exactly 1.0 with the action,
and the z it produces is what a correlation of 1.0 looks like on that batch: 20.8,
22.7 and 20.4 at the lower power and 30.6, 44.8 and 3.3 at the higher. Against the
observed cross-seed standard errors, a correlation of **0.048** and **0.050**
respectively would show at two standard errors.

So the arms are three times more sensitive than the retracted figures said, and **the
extra episodes bought no sensitivity**: per-seed sensitivity rose and the cross-seed
standard error rose with it, from 0.52 to 0.66. The third seed of the higher-power arm
is also badly conditioned, reading 3.3 where the others read 30.6 and 44.8, a
tenfold spread in instrument sensitivity across seeds of the same arm.

**THAT EXTRAPOLATION WAS LINEAR, THE CURVE IS NOT, AND THE CALIBRATED ANSWER IS THAT
THE z CANNOT DO THIS JOB.** Advantages were synthesised at known correlations, every
target reproduced to four decimals, with the noise component taken from the REAL
advantage so the curve passes through its own null:

| c | 0.00 | 0.02 | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| z | +0.24 | -0.03 | -0.25 | +0.03 | +1.90 | +5.35 | +8.97 | +22.02 |

z crosses 2 at c = 0.204, where linear extrapolation from the ceiling says 0.091. The
crossing is not the main point. Below c of about 0.2 the readings are non-monotone and
every one sits inside the null's own scatter, so **on an arm this size the gradient
norm cannot separate a correlation of 0 from one of 0.1.**

**This weakens the segment arms and not the vehicle arms.** The segment arms report
only a z, so their bound is of order 0.1 to 0.2 -- earlier text here said 0.15, then
0.05, then 0.07, each time correcting the conversion rather than the data. The vehicle
arms report `corr(slow, A)` directly, at two standard errors of 0.035, and a measured
correlation needs no conversion or calibration at all. The segment arm is being re-run
to report its correlation the same way. Tasks 137 and 138.

The conversion is load-bearing enough to be a tested instrument rather than a one-off:
`scripts/measure_alignment_calibration.py`, with `tests/test_alignment_calibration.py`
covering the construction it rests on -- that the synthesised advantage carries the
correlation it claims, that a noise vector already correlated with the action does not
leak into the target, and that an action column with no variance raises rather than
returning a zero correlation that would read as a measured null. Verified against a
control: removing the orthogonalisation fails seven of the eleven.

What remains untested is everything below about 0.1 on these arms, and a
clean test of the paper's own formulation rather than this approximation of it, which
needs its five per-super-segment observation fields and therefore widens the deployed
contract that the Jetson builder and the parity ledger both depend on.

## THE PORTED RUN, STOPPED AT UPDATE 13 OF 20, AND ITS GATE

Ankit stopped the run at update 13 on 2026-09-10. The gate was pre-registered at 20
updates, so criteria 1 and 2 are read on a SHORTER run than they were written for.
That cuts both ways and is recorded rather than argued: fewer updates give a trend less
chance to appear, and the trend that did appear points the wrong way.

| criterion | threshold | reading at update 13 | |
|---|---|---|---|
| 1. summed entropy falls | below 1.9775, of a 2.1972 maximum | lowest reached **2.1459** | fail |
| 2. score trends up | by more than the step-to-step variation | moved **-0.4679** against a step standard deviation of 0.2625 | fail |
| 3. action distribution leaves uniform | joint modal share above 0.20 | **0.1460** | fail |

**All three fail.** Criterion 3 is measured on the update-13 actor over 1,503
observations in the environment it trained in. Both heads sit near uniform: the speed
head at 0.3176 / 0.2994 / 0.3829 and the headway head at 0.2659 / 0.3529 / 0.3812,
summed entropy 2.1804 of a 2.1972 maximum. For scale, a randomly initialised actor
reads a joint modal share of 0.1378 and the earlier `mappo_sumo` run reached 0.1486
after 23 updates, so 0.1460 after 13 is initialisation drift rather than a learned
preference.

**The early stop does not rescue criterion 2, and the amount by which it does not is
computable.** For the criterion to pass at update 20, updates 14 to 20 would have had
to hold a score of -6.03. That is better than every update of the run except the
first, against a whole-run range of -6.811 to -5.777.

**Criterion 2 fails in the wrong direction, which is the part worth noting.** The
score did not stay flat; it declined, and by more than the update-to-update noise. The
curve: -5.777, -6.567, -6.446, -6.463, -6.462, -6.121, -6.353, -6.302, -6.304, -6.318,
-6.605, -6.731, -6.811.

**Entropy never approached its gate.** It fell 0.0439 over thirteen updates, from
2.1898 to 2.1459, against the 0.2197 fall the gate asks for, and it rose on four of
the twelve steps. An earlier claim here that the decline was monotone was retired
after update 5 reversed it, and updates 7 and 10 reversed it again.

## A defect in the credit assignment, found by reading rather than measuring

Recorded prominently because of how it was found. Twenty measurements did not surface
it; auditing the algorithm surfaced it in minutes, after Ankit said the problem was
likely in the algorithm and the implementation rather than in what was being measured.

**Credit leaks across episode boundaries.** `terminated` is always False on this
environment by construction, every episode ends by truncation, and the done flag
written into the rollout buffer consults `terminated` and agent presence but never
`truncated`. A vehicle still in the network at a boundary therefore records
`done=False`; the trainer resets the environment and keeps filling the same buffer;
vehicle ids are regenerated identically on every reset; and the advantage computation
groups by agent id. Two different vehicles sharing an id become one trajectory, and
GAE bootstraps backward across the reset.

**What it touches.** Both training runs, the higher-power segment arm, the speed-only
arm, the matched baseline, and both instrument calibrations all crossed a boundary.
The three measurements the central null rests on -- `align_sumo` at +0.00 +/- 0.29,
`align_src` at -0.47 +/- 0.51, and the own-reward segment arm at -0.12 +/- 0.51 -- ran
inside a single episode and did not.

**It does not explain the null, and that is the more important half.** The cleanest
measurements were unaffected and still read zero. What it does invalidate is the
sensitivity calibration every exclusion bound was derived from, on top of the
walk-back already recorded, and both training runs.

**Two design defects sit alongside it, and they are the likelier explanation. Neither
is a bug; both were logged this session as knobs that had been swept.**

- **The action is mostly inert.** The speed bins are 8.33, 12.5 and 16.67 m/s against
  a mean AV speed near 2.5 m/s, and all three command the same behaviour on 84% of
  AV-steps. An action that changes nothing five times in six cannot carry a gradient,
  and that is a property of the action space rather than a finding about decentralized
  control.
- **The reward gives each agent the same number.** It is a network aggregate delivered
  identically to every agent at every step, so cross-agent variation in the advantage
  comes almost entirely from the critic. That is algebra, not an empirical question,
  and it was tested seven ways instead of being read off.

## What was and was not built, stated exactly

Asked directly whether this implements super-segment control or only a coarse lever,
the answer is that the lever is coarse in time everywhere, coarse in space only in
arms that were never trained, and the STATE the controller acts on is a single
vehicle's kinematics throughout. Precisely:

| the paper | what was built | where |
|---|---|---|
| one decision per 60 s | one decision per 60 s | `mappo_src`, `decision_interval_s: 60.0` |
| one action per super-segment, applied to all vehicles on it | same, in the measurement arms only | `measure_segment_as_agent.py`, `segown_highpower.py` -- gradient measurements, never trained |
| | one action PER VEHICLE in the trained run | `mappo_src` is vehicle-attached; each AV samples from its own observation |
| super-segments of 2 to 3 km | links of 500 m (leaves), 600 m (middles and trunk), 300 m (bottleneck), nine of them | `configs/topology/inverted_tree.yaml` |
| state is per-super-segment density, speed, gap, inflow, outflow | state is one representative vehicle's 33-field local vector: `ego_speed`, `ego_headway_s`, `leader_gap`, `follower_gap`, lane gaps and so on | `LOCAL_OBS_FIELDS` in `src/rl/encoders.py` |

Of those 33 fields about five are link-level -- `downstream_congestion_estimate`,
`segment_target_speed`, `local_density_bin`, `local_mean_speed_bin`,
`local_queue_estimate`. The rest are per-vehicle kinematics.

**The paper's five fields already exist in this codebase.** `SEGMENT_FIELDS` carries
`density`, `mean_speed`, `queue_length`, `inflow` and `outflow` per segment, and the
CENTRALIZED CRITIC already consumes them. They are not routed to the actor. That is a
deliberate boundary rather than an oversight: the actor's input width is
`local_obs_dim()`, and the same width is pinned in the deployed contract that
`src/analysis/observation_parity.py` and the Jetson builder both check against. The
privileged-critic work was built specifically to add features without touching it.

**So the honest label for what was measured is "super-segment ACTION driven by a
single vehicle's OBSERVATION, on 500-to-600 m links".** That is not the paper's
controller, and no result here bounds the paper's controller. Widening the actor's
observation to the five per-segment fields is the one untested direction whose cost is
known: it breaks the deployed observation contract on both the Jetson and the phone.

## What this leaves

Four directions, of which the second has changed since it was first written here,
because one of its two halves has since been measured.

1. **Report the simulation leg as a null**, with the deployment carrying the
   feasibility claim as it already does. This is my recommendation.
2. **Change what the AVs can do** rather than how their reward is priced. Of the two
   candidates originally listed under this heading, the explicit junction meter is now
   measured and is not a direction: a perfect-information metering oracle serves 0.8
   fewer and 0.8 more vehicles than no control on the two demands, against a
   seed-to-seed spread of about 16 arrivals. Platoon coordination is untested and is
   what remains of this option.
3. **Test the paper's own formulation of the road-attached agent.** Raising the
   episode count is NOT the way: three episodes per rollout left the threshold where
   one episode did, because the cross-seed standard error rose along with
   the per-seed sensitivity (task 136). What is untested is the formulation itself.
   This arm draws its action from one representative vehicle's observation; the paper
   uses five per-super-segment fields, and adding them widens the deployed contract
   that the Jetson builder and the parity ledger depend on.
4. **Change the advantage estimator** to a counterfactual one. It is aimed exactly at
   the quantity that reads null and would probably raise the correlation.
5. **Move the congestion threshold, or establish that moving it does not help.** This
   is new, and it is the cheapest of the five: one sweep, not several days. The
   objective's anticipatory half does not respond to behaviour, and rho* = 0.3 of jam
   density fires on only 1.50 of 9 segments, so it is nearly inactive on this road.
   Either the threshold is misplaced and a placement exists that makes the term
   respond, or nothing the lever does moves any threshold. Those call for different
   work and the measurement separates them.

**On the order.** 5 comes first on cost alone: it is one sweep and it can retire an
option rather than open one. After that, an estimator that learns better is worth
building only after something shows headroom to reach, and oracles are the cheap way
to look for headroom, so 4 is last. For the same reason 3 comes before 4: it asks whether the mechanism is learnable
at the granularity the paper used, before spending on a better learner at the
granularity this project uses.

**CLOSED, 2026-09-10.** Ankit stopped the training run at update 13 and then stopped
all remaining work. Option 1 is what happened: the simulation leg is reported as a
null and the deployment carries the feasibility claim. Options 2 to 5 are recorded as
what a continuation would cost, not as work in progress. Nothing is running.


---

# `plan_task_67_merge_blind_collisions.md`

Moved here on 2026-09-12. The 40% episode-truncation rate in highway-env, whose cause was `IDMVehicle` following only within its own lane.

# Task 67 — AVs collide because the sensing model cannot see the vehicles that hit them

**Command:** `plan_dsrc_rec`. Every decision below was taken by recommendation
rather than put to the user. They are marked so nobody reads them as sign-off.

## Short version

Task 67 was filed as "fix the 40% episode-truncation rate". The rate is real and
worse on the one topology that matters, but truncation is a symptom. **Episodes end
because AVs crash, and AVs crash because `lane_gap_context` cannot report the
vehicle they are about to hit.** The safety layer brakes correctly for what it is
told; it is told the road ahead is clear.

Measured on `inverted_tree`, 108 runs mirroring the task 8 grid for one topology:

| controller | runs | completed | rate |
|---|---|---|---|
| `no_av` | 36 | 36 | **100%** |
| `cooperative_smoothing` | 36 | 7 | 19% |
| `backpressure` | 36 | 5 | 14% |
| **all** | **108** | **48** | **44%** |

`no_av` never fails because it has no AVs to crash. Completion falls to **0 of 6 at
`av_penetration` 0.20 on every demand level**: the failure rate rises with the number
of AVs, which is the signature of AVs being unable to see each other and the traffic
they merge into.

**51 terminating collisions, classified by the lane relation between the two
vehicles:**

| class | n | share |
|---|---|---|
| sibling merge arc — both arcs feed the same node | 27 | **53%** |
| same lane | 18 | 35% |
| adjacent lane, same arc | 6 | 12% |
| successor arc | 0 | 0% |

**Why the merge class is invisible.** `_lane_gaps` skips any neighbour whose
`lane_index` differs from the ego's (`src/sensing/local.py:362`). On `inverted_tree`
the graph is `a1_entry → b1`, `a2_entry → b1`, `a3_entry → b1`: several entry arcs
converge on one node. Two vehicles converging on that node are on different arcs, so
neither appears in the other's leader search, right up to the collision. The field
that should carry this, `distance_to_next_merge`, is **hardcoded to `0.0`** and was
recorded by task 5 as one of only two structurally uninformative fields in the whole
observation vector. The AV has no merge awareness on a merge topology.

**A measured run-up, `cooperative_smoothing`/medium/pen0.10/seed17:**

```
step  speed  sensed leader gap  true gap  commanded accel
  44   29.9                inf     145.0            +0.22
  47   27.2                inf      98.1            -0.27
  49   27.0              146.5      40.2            -0.04
  51   27.0              130.7      13.6            -0.01
```

The sensed gap is `inf` while the true gap closes from 176 m to 84 m, then reports
130.7 m when the truth is 13.6 m — it has locked onto a further vehicle on the ego's
own arc. Commanded deceleration decays to −0.01 m/s² as the gap collapses, because
the layer is braking for a leader 130 m away. The vehicle finally hit was on
`a4_entry → b2` while the ego was on `a5_entry → b2`: a sibling merge arc.

## Scope boundary

- **In scope:** the leader/follower/merge gap computation in `src/sensing/local.py`,
  and the `SafetyContext` fields fed from it in `src/envs/topology_env.py`.
- **Out of scope:** the safety layer's own braking law. It is not implicated: given a
  correct gap it brakes (−1.71 m/s² at 128 m in the trace above). If it turns out to
  brake too weakly once it can see, that is a separate task.
- **Out of scope:** highway-env's `IDMVehicle`. Human-on-AV rear-ends exist but are a
  third-party car-following model; see the open item below.
- **Out of scope:** every topology except `inverted_tree` for *measurement*. The fix
  is topology-agnostic because it lives in the sensing model, but only `inverted_tree`
  is re-measured.
- **Out of scope:** changing the termination rule. Ending an episode on an AV crash is
  the standard formulation and is what gives training its negative signal.

## Decisions, all taken by recommendation

| # | decision | taken | rejected, and why |
|---|---|---|---|
| D1 | Fix perception, not termination | Make the gap search see the vehicles that actually collide | Suppressing termination on collision hides the defect, and the safety filter is a headline claim of the paper |
| D2 | Merge conflicts get their own gap, not a fake leader | Add a merge-conflict gap and populate `distance_to_next_merge` | Reporting a merging vehicle as a "leader" would corrupt headway control, which is a different quantity |
| D3 | Route-aware search via the road graph | Walk `network.graph` successors/predecessors, accumulating lane lengths | Euclidean distance with a heading window is wrong on curves and links physically close but unconnected arcs |
| D4 | Adjacent-lane conflicts are in scope | Use `side_lanes` for the 12% class | Deferring leaves a known crash mode unaddressed for a small extra cost |
| D5 | Keep `range_m` as the visibility limit | Merge and successor searches stop at `range_m` | An unbounded search would make the AV clairvoyant and would not match the deployed system |
| D6 | Fix the sensing model only, then re-measure | Measure the residual before touching anything else | Changing two systems at once makes the delta unattributable |

## Open items for the user

1. **Human-on-AV rear-ends were raised as unrealistic, and they are 27% of the
   subset examined by collision partner.** They are not addressed by this plan.
   `IDMVehicle` has a bounded comfort deceleration, so when an AV brakes hard the
   follower can be unable to stop. The faithful comparison target (Flow, Vinitsky et
   al.) runs on SUMO, whose car-following is collision-free by construction, so a
   replication arguably should not have humans rear-ending anyone. **Recommendation
   if asked: after this fix, re-measure; if human-on-AV rear-ends remain material,
   make the human follower collision-free rather than changing the termination rule.**
   Flagged rather than decided because it changes the traffic model, not a bug.
2. **The 35% same-lane class may not be fully explained by this fix.** One sample had
   a correctly sensed gap of 7.8 m at 19.9 m/s onto a stopped vehicle — already
   unavoidable. Whether the leader became visible too late (a `range_m` or latency
   question, and therefore a task 9 question) or the layer braked too weakly is not
   yet established. Step 5 measures it.

## Steps

| # | step | done when |
|---|---|---|
| 1 | Characterise the same-lane class: log sensed gap, true gap and commanded accel for every same-lane collision | The 18 same-lane cases are split into "seen too late" and "seen, braked too weakly" with counts |
| 2 | `_route_gaps`: walk graph successors from the ego lane, accumulating `lane.length`, to find the true leader across arc boundaries; predecessors for the follower | Unit tests cover a two-arc chain, a three-arc chain, and the `range_m` cutoff |
| 3 | `_merge_conflict_gap`: find vehicles on sibling arcs sharing the ego's next node; return distance-to-merge-point for both, and the conflict gap | Unit tests cover two converging arcs, a non-merging pair, and a vehicle past the merge point |
| 4 | Populate `distance_to_next_merge` from step 3 instead of the hardcoded `0.0`; add the merge-conflict fields to `SafetyContext` | `observation_audit` no longer reports the field as structurally constant |
| 5 | Sanity tests: on `inverted_tree`, an AV converging with a sibling-arc vehicle reports a finite merge gap that decreases monotonically as they approach | Test fails against the current code |
| 6 | Re-run the 108-run grid and report completion before and after, by class | Completion rate and the class table are both reported; any class that did not improve is named |

## Risks

- **The observation vector changes.** `distance_to_next_merge` goes from constant
  `0.0` to a real value, and `leader_gap_m` changes on merge topologies. Nothing is
  trained yet, so no checkpoint is invalidated — but task 9's calibration and task 47's
  parity ledger both read these fields, and task 47 compares the sim sensing model
  against the live one. **The live model must be checked for the same defect**; if the
  Jetson's `observation_builder` has no merge concept either, the parity classification
  changes and task 47's ledger needs updating.
- **Cost per step.** The gap search is already O(n) per AV; a graph walk bounded by
  `range_m` adds a small constant. If the 108-run grid slows by more than ~25%, revisit.
- **A fix is new code.** The validator rounds exist for exactly this.

## Sign-off

- [ ] Steps 1–6 complete, tests passing
- [ ] Completion rate on `inverted_tree` reported before and after, by collision class
- [ ] Residual human-on-AV rear-end share reported, so open item 1 can be decided
- [ ] `validate_dsrc_3` clean

---

# PAUSED 2026-09-09

## The blocking unknown

**Nobody has measured whether making the safety layer yield at a merge reduces
collisions, and the whole of task 67 now rests on it.**

The sensing half is committed and **bought nothing**: the 108-run grid returns
48/108 completed both before and after, identical run for run. That is not a
disappointing result, it is an expected one — `LaneGapContext` gained
`merge_conflict_gap_m` and `distance_to_next_merge_m`, but **`SafetyContext` has no
such fields, so the safety layer never receives them**. The route-aware leader
search that *is* wired up addresses the successor-arc class, and the classification
measured **0 of 51** collisions in that class. So the committed change is correct and
almost inert on this topology.

**How to settle it.** Finish the wiring, then re-run the grid:

1. Add `distance_to_next_merge_m`, `merge_conflict_gap_m` and
   `merge_conflict_relative_speed_mps` to `SafetyContext` in
   `src/safety/safety_layer.py` (they exist already on `LaneGapContext`).
2. Make `_forward_ttc` and `_speed_and_acceleration` consider a merge hazard:
   priority by arrival, so when `merge_conflict_gap_m < distance_to_next_merge_m`
   the other vehicle gets the joining point and the ego must be able to stop short
   of it. Treat the joining point as a stationary obstacle in that case.
3. Pass the three fields through `_safety_context_for_vehicle` in
   `src/envs/topology_env.py` (around line 666, where `SafetyContext` is built from
   `gap_context`).
4. Re-run the grid and compare against 48/108.

The probe that answers it is already written:
`scratchpad/repro67.py` runs the 108-run grid in ~220 s and writes
`scratchpad/repro67.json`. **It lives in the session scratchpad and will be gone** —
it is 40 lines and re-derivable from the "Steps" table above; the grid is
`inverted_tree` × demands (low, medium, high, burst) × pen (0.05, 0.10, 0.20) ×
controllers (`no_av`, `cooperative_smoothing`, `backpressure`) × seeds (7, 17, 27),
120 steps, via `src.analysis.simulator_health.run_condition`.

## Exactly where I stopped

- Repo `dsrc`, branch `main`, **no worktree**. Not pushed: `main` is **ahead 1** of
  `origin/main` (`origin/main` = `61b65f0`).
- **Committed:** `6efc3bf` "Task 67: the gap search could not see across an arc
  boundary" — `src/sensing/local.py` (route-aware gaps, merge context) and 7 tests in
  `tests/test_local_sensing.py`. Full simulator suite green at that commit: **263
  passed**.
- **Uncommitted:** `tests/test_safety_layer.py` — 4 new tests in `TestMergeConflict`,
  **currently failing with `TypeError`** because `SafetyContext` does not accept the
  keyword arguments yet. This is intended TDD red, not a broken tree.
- **Not started:** steps 1–4 above; nothing in `topology_env.py` has been edited.
- `validate_dsrc_3` has **not** been run. Per `implement_dsrc` there is no push until
  validation passes.

## What was learned that would otherwise be re-derived

- **`longitudinal_m` restarts at every arc.** That is why the original code required
  an exact `lane_index` match: a raw subtraction across a boundary is meaningless.
  Any fix must accumulate lane lengths along `road_network.graph`.
- **`inverted_tree` geometry:** six 500 m entry arcs, `a1..a3 → b1` and `a4..a6 → b2`;
  `b1 → c` and `b2 → c` are 600 m and two-lane; `c → exit`. 12 lanes total.
- **Two fields are dead on this topology.** `distance_to_next_merge` is hardcoded to
  `0.0` in `src/sensing/local.py` (task 5 recorded this). `near_merge` in
  `SafetyContext` is set from `"merge" in segment_id`, and `inverted_tree` segments
  are named `tree_leaf_a1`, so it is always `False` there.
- **Collision classification, 51 terminating collisions:** sibling merge arc 27
  (53%), same lane 18 (35%), adjacent lane 6 (12%), successor arc 0.
- **Same-lane sub-split:** 10 of 18 were seen too late to stop, 8 had room. When a
  same-lane leader first became visible the mean gap was 74 m at 27.2 m/s, against a
  75 m stopping distance at 5 m/s² — the leader appears at almost exactly the
  distance where it is already too late.
- **The safety layer is not the defect.** Given a correct gap it brakes: −1.71 m/s² at
  128 m in the traced run-up. It decayed to −0.01 m/s² at a true 13.6 m gap because it
  was told the leader was 130.7 m away.
- **Baselines to reproduce before quoting a delta:** simulator suite 263 passed;
  Jetson suite 2308 passed / 24 skipped; grid completion 48/108.
- Use `.venv/bin/python`; system `python3` is 3.14 with no pytest.

## Task order to resume

1. Steps 1–4 under "The blocking unknown" — finish the wiring and measure.
2. If the merge yield works, decide the adjacent-lane class (6 of 51, 12%) — plan
   decision D4 says it is in scope via `side_lanes`, and it is not yet started.
3. `validate_dsrc_3`: one independent `opus` validator, 3 rounds, kept alive, every
   finding verified before acting. Then push.
4. `experiment_dsrc`: the full before/after grid and the class table for the plan's
   sign-off list.

## Open decisions not yet made

- **Open item 1 in the plan is still open and is the user's**: human-on-AV rear-ends
  were raised as unrealistic. They are not addressed. The recommendation on record is
  to re-measure after the merge fix and, if they remain material, make the human
  follower collision-free rather than change the termination rule.
- Whether the 8 same-lane collisions that "had room to stop" indicate the braking law
  is too weak. That would be outside this task's scope boundary and needs its own
  task.

---

# FINDING 2026-09-09: the road produces collisions for plain human traffic

**Three fixes were implemented and measured. None moved the completion rate.**

| change | grid completion |
|---|---|
| baseline | 48/108 (44%) |
| route-aware leader/follower across arc boundaries | 48/108 (44%) |
| + merge yield in the safety layer, priority by arrival | 48/108 (44%) |
| + entry lanes separated to a full lane width | 49/108 (45%) |

**Why none of them could have worked.** `no_av` completes 36 of 36, which the plan
read as evidence that the road is safe and the AVs are at fault. It is not evidence
of anything: `terminated = any(vehicle.crashed for vehicle in self._av_vehicles)`,
and a `no_av` run has no AVs, so it **cannot terminate early whatever happens on the
road**.

Counting collisions instead of terminations:

| controller | collisions per 120-step run | runs with at least one |
|---|---|---|
| `no_av` | median **17**, max 57 | 30 of 36 |
| `cooperative_smoothing` | median 6, max 52 | |
| `backpressure` | median 5, max 52 | |

**Plain IDM human traffic crashes more than the AV runs do.** The AVs are not driving
badly; they are driving in traffic where collisions are endemic, and only their own
collisions end an episode.

**The mechanism is that every node in this topology reduces lane count and nothing
sequences vehicles through it:**

| node | lanes in | lanes out |
|---|---|---|
| `b1` | 3 | 2 |
| `b2` | 3 | 2 |
| `c` | 4 | 2 |

`highway_env`'s `IDMVehicle` follows the vehicle ahead **in its own lane**. Two
vehicles arriving at `b2` on different incoming lanes have no mutual awareness in the
human model at all, so they interpenetrate. The same blindness I fixed for the AV's
sensing exists in the human car-following model, and that is where the collisions
come from.

**This also means task 8's `episodes_complete` criterion compared every AV controller
against a reference that passes by construction.** 44 of 72 cells failing that
criterion is not 44 cells of controller failure.

**What is now known to be NOT the cause:** the safety layer's braking law (it brakes
correctly when given a correct gap), the AV's leader perception across arcs (0 of 51
collisions were successor-arc), and the entry-lane geometry alone (separating them to
a full lane width bought 1 percentage point).

## The decision this needs, and it is the user's

Three routes, and they produce materially different work:

1. **Give the human model merge awareness at funnel nodes.** The faithful option: the
   replication target (Flow, Vinitsky et al.) runs on SUMO, whose car-following is
   collision-free by construction, so a faithful replication should not have humans
   driving through each other. Largest change, and it touches a third-party vehicle
   class. **Recommended if asked.**
2. **Change the topology so no node reduces lane count.** Cheapest, but the funnels
   are what create the congestion this topology exists to produce — task 8 measured
   `inverted_tree` congesting in 12 of 12 cells — so this likely removes the
   phenomenon under study.
3. **Accept the collisions and change the success criterion.** Stop treating
   `episodes_complete` as a gate, replace the `no_av` reference with one that can
   actually fail, and let MAPPO learn to sequence. Cheapest path to a trained policy,
   but the paper would be reporting throughput on a road where traffic collides.

Option 1 and option 3 are compatible: 3 unblocks training now, 1 makes the result
defensible later.

## Decision 2026-09-09: option 1, taken by the user

Give the human model merge awareness at funnel nodes. Rationale on record: the
replication target runs on SUMO, whose car-following is collision-free by
construction, so a faithful replication should not have human traffic driving
through itself.

**The mechanism, and a defect it exposes in this task's own earlier work.** The
standard way to make a merge safe is to project every vehicle approaching a shared
node onto that node's axis: a vehicle whose distance to the node is smaller arrives
first, so it is a leader at a gap of `d_ego - d_other`, and ordinary car-following
handles the rest.

`_merge_context` as committed picks the conflicting vehicle with the **smallest
distance to the node**, which is the vehicle furthest ahead, not the nearest one.
The crash data shows what that costs: at one impact it reported a conflict 13.4 m
from the node while the ego was 99.1 m from it — a projected gap of 85.7 m — and the
vehicle actually struck was 4.1 m away, alongside. The layer was handed the wrong
vehicle, which is why a correct yield rule fired on 59% of samples and changed
nothing.

Both the human model and the safety layer use the same projection.

---

# CORRECTION 2026-09-09: the first merge-aware result was an artifact

**What was reported and is wrong:** merge-aware humans cut collisions 212 to 120
(-43%) and lifted grid completion from 48/108 to 69/108.

**Why it was wrong.** `IDMVehicle.acceleration` uses a gap term of
`(desired_gap / d)^2`, which diverges as `d` approaches zero. Real vehicles never
reach a sub-metre gap because a collision is detected first, but a **projected**
leader can be placed a centimetre ahead, and the filter only excluded `gap <= 0`.
The result was unbounded braking: a `no_av` run reached a **mean_speed of
-2.6e11 m/s** at step 28, against 28.1 for plain IDM on the same seed. The
apparent collision reduction was vehicles being flung backwards, not merging.

That number feeds the trainer directly -- `score = mean_speed - jam_fraction` --
and `mean_speed` is also what task 8's `baselines_separate` criterion compares.

**The fix** clamps the merge acceleration to the vehicle's own `ACC_MAX`. Braking
is allowed to be hard; it is not allowed to be unbounded.

**The corrected measurement:**

| | baseline | with the blow-up | clamped, true |
|---|---|---|---|
| `cooperative_smoothing` | 7/36 | 16/36 | **11/36** |
| `backpressure` | 5/36 | 17/36 | **11/36** |
| overall completion | 48/108 (44%) | 69/108 (64%) | **58/108 (54%)** |
| `no_av` collisions, 12 conditions | 212 | 120 | **301** |

**The result is mixed and both halves are real.** AV episodes complete 10
percentage points more often, which is the criterion that gates training. Human
collisions in plain traffic rise 42%, because a vehicle yielding to a projected
conflict slows and the traffic behind it in its own lane does not always react in
time. That is the same rear-end mechanism raised as unrealistic at the start of
this task, now caused by the fix rather than merely present.

`mean_speed` across the grid is now 4.60 to 23.39 m/s, all physical.

## The decision this needs

1. **Keep it.** +10pp completion is real and training is what needs unblocking.
   The human collision rate is a known cost, recorded, and the paper does not
   claim collision-free traffic.
2. **Refine the yield test** so a vehicle only gives way when a collision is
   actually predicted, rather than whenever another vehicle reaches the node
   first. Should reduce the unnecessary slowdowns causing the extra rear-ends.
   **Recommended if asked** -- it addresses the regression rather than accepting it.
3. **Revert to plain IDM** and take option 3 from the previous decision instead:
   accept collisions, drop `episodes_complete` as a gate, let MAPPO learn.

**A separate defect worth its own task:** nothing anywhere validates that
`mean_speed` is physical. A value of -2.6e11 propagated into the training score,
the health-check criteria and the run records without a single guard noticing.

---

# TASK 68 OBSERVATION 2026-09-09: the reward is ego speed, not throughput

Noted while the first MAPPO run was in flight, before its result is known.

`src/envs/topology_env.py:578`:

```python
def _reward_for_vehicle(vehicle: ControlledVehicle) -> float:
    return 0.0 if vehicle.crashed else float(vehicle.speed)
```

That is the whole reward, less a `crash_penalty` of 2.0 subtracted in
`src/rl/trainers.py:198`. **There is no throughput term.** The agent maximises its
own speed; the paper's claim is a network-level throughput improvement.

**The two objectives conflict exactly where this topology is interesting.** At a
merge, yielding raises throughput and lowers the yielder's speed, so a
speed-maximising agent learns not to yield. The training diagnostics at update 35
of 100 are consistent with that reading, though 35 updates is too early to
conclude from:

| metric | first half | second half | change |
|---|---|---|---|
| `entropy` | 4.047 | 3.295 | −0.752, so the policy IS converging |
| `throughput_recent` | 0.836 | 0.852 | +0.016, noise |
| `mean_speed` | 20.891 | 20.666 | −0.225 |
| `collision_count` | 0.629 | 0.853 | **+0.225** |

**Why this is worth recording now.** If the finished run shows no throughput gain,
there are two very different explanations — the method does not transfer to this
simulator, or the agent was never asked for throughput — and they lead to
completely different work. Reward design is a modelling decision, so it is put
here rather than changed.

Vinitsky et al. and the Flow benchmarks generally use a system-level reward
(average speed over all vehicles, or a throughput term), not ego speed. A
replication whose reward is per-agent speed is not replicating the same
experiment.

**Do not read this as the result.** The run is unfinished; the measured outcome
goes in when it lands.

---

# TASK 69 RESULT 2026-09-09: the replication does not reproduce a throughput gain

Trained MAPPO, 100 updates, `inverted_tree`, `demand: medium`, seed 7, under the
`mappo_deploysense` sensing block. Evaluated on 18 conditions per arm — demands
(low, medium, high) × penetration (0.10, 0.20) × seeds (7, 17, 27) — with the
sensing block read from the checkpoint's own `config_resolved.yaml`.

## The numbers

| | `no_av` | MAPPO |
|---|---|---|
| episodes completed | 18/18 | **7/18** |
| mean throughput, completed runs only | 10.11 | 9.29 |

**Neither of those columns is a fair comparison and they must not be quoted as
one.** `no_av` completes 18 of 18 by construction — it contains no AVs, and
`terminated` tests AV crashes — so its completion rate carries no information. And
averaging throughput over 18 runs against 7 surviving runs is a selection effect:
the 7 are the conditions MAPPO did not crash out of.

**Paired on the 7 conditions where both arms completed:**

| condition | `no_av` | MAPPO | delta |
|---|---|---|---|
| low / 0.10 / 7 | 5.00 | 5.00 | 0.00 |
| low / 0.10 / 17 | 8.00 | 8.00 | 0.00 |
| low / 0.10 / 27 | 11.00 | 11.00 | 0.00 |
| low / 0.20 / 7 | 5.00 | 5.00 | 0.00 |
| low / 0.20 / 17 | 8.00 | 11.00 | +3.00 |
| low / 0.20 / 27 | 11.00 | 10.00 | −1.00 |
| medium / 0.10 / 27 | 13.00 | 15.00 | +2.00 |

Paired mean **+0.57 vehicles**, with **4 of 7 conditions exactly equal** and one
negative. Throughput here is an integer vehicle count, so the granularity is one
vehicle and the sample is seven. **This is not a throughput improvement; it is
noise on a sample too small to carry one.**

## What the policy did learn

`mean_speed` on the same 7 conditions: `no_av` 19.88 m/s, MAPPO **21.38 m/s**.
Over 100 updates, `entropy` fell 3.459 → 3.001, so the policy converged;
`throughput_recent` rose 0.895 → 1.002; `mean_speed` and `score` were flat.

**The agent learned exactly what it was paid for.** The reward is ego speed less a
crash penalty, recorded above before this run finished. MAPPO drives 1.5 m/s
faster than undisturbed traffic and crashes out of 11 of 18 evaluation runs. A
faster, less safe, no-more-productive policy is the correct optimum of that reward,
not a failure to learn.

## The decision, and it is the user's

1. **Change the reward to a system-level objective** — average speed over all
   vehicles, or an explicit throughput term — and retrain. This is what Vinitsky
   et al. and the Flow benchmarks optimise, so a replication arguably has to.
   Training costs about 25 minutes per run, so this is cheap to try.
   **Recommended if asked.**
2. **Keep the ego-speed reward and report the null.** Defensible only if the paper
   states plainly that the agent optimised individual speed, which is not the
   published experiment.
3. **Widen the evaluation before concluding** — more seeds, longer episodes, more
   demand levels. It would tighten the interval, but 4 of 7 paired conditions being
   exactly equal suggests the effect is absent rather than merely noisy at this
   sample size.

Options 1 and 3 compose: change the reward, then evaluate wider.

## What this result does NOT say

It does not show that MAPPO cannot improve throughput in this simulator. The
agent was never asked to. That distinction is why the reward was recorded before
the run rather than after.

---

# VALIDATOR ROUND 2 2026-09-09

All four numbers reproduced exactly, and the crashed-vehicle filter was checked
empirically rather than argued: yielding to wrecks gives **167** collisions and
14.04 m/s against **126** and 15.81 for skipping them, so it is better on both axes.

## Acted on, each verified here first

| finding | what it was | now |
|---|---|---|
| backwards driving | the clamp bounded acceleration, not speed. A projected gap does not shrink with the ego's speed, so ACC_MAX could be commanded at a standstill; vehicles reversed at **−7.5 m/s** and the mean over 40 vehicles read 28.1, hiding it | a stopping floor of `−speed / 1 s`, applied to the **composed** acceleration on **every** call. Flooring only the merge term left −0.87 m/s, because `act()` returned early when no conflict was in view |
| `min(candidates, key=abs)` untested | reverting it to the exact pre-fix bug left 52/52 passing | a ring test: a vehicle 75 m behind, 185 m ahead around the loop, must read as a follower |
| `_merge_context` filters untested | both could be deleted with 0 tests failing, and this is the half feeding the safety layer | three tests plus a live-vehicle control |
| docstrings and a dead field | `distance_to_next_merge_m` was plumbed through two dataclasses and read by nothing, while five tests passed it as input | removed from `SafetyContext`; `_forward_hazard`'s "stationary obstacle" wording corrected |

**Every one of those four is now mutation-caught**, verified in a throwaway mirror:
each mutation fails exactly one test, baseline and restored both 42 passed.

## A finding of my own, made while fixing theirs

`build_one` — the observation the **actor** trains on — never got the route-aware
gaps that `lane_gap_context` got, and `distance_to_next_merge` was still the literal
`0.0`. So the policy trained in task 68 could see neither a leader across an arc
boundary nor that a merge existed, on a topology whose every node is a merge.

The gaps are now fixed there too. **The merge distance is deliberately not**, and
task 47's parity ledger is what settled it: `deployment/jetson`'s observation
builder has no map matching and sets the field to `0.0` explicitly for sim parity.
Putting a real value in the sim observation would train the policy on information
the deployed vehicle cannot measure. The ledger failed the moment it was tried,
which is what that ledger is for.

**This is a decision the user may want to revisit:** giving the Jetson map matching
would let the policy use merge distance legitimately. Until then the sim must not.

## Residuals, measured and not fixed

- **The static successor map disagrees with the lane index vehicles acquire.**
  `_next_lane` returns `target_lane_index`, the steering target;
  `VehicleSnapshot.lane_index` is `get_closest_lane_index`, the geometrically
  nearest. 14 of 93 observed transitions disagree, systematically for
  `('b2','c',1) → ('c','exit',0)` (9 of 9), because the b2 arc ends at y = −6 while
  `c→exit` lane 0 starts at y = 0. Roughly 1% of ego observations lose their
  leader — the same mechanism as the entry-arc defect, one arc downstream.
- **The successor filter therefore excludes one pair that does collide**, 1 of 13
  classified collision pairs, 5–7 m past node `c`. The 48% false-conflict
  reduction stands; the filter is not sound.
  Both share one root cause and want one fix built on where vehicles physically
  travel rather than on the static map. That is a task, not a patch.
- **`predecessors[0]` explores a branching ancestor tree as a single chain.** Zero
  mismatches at `range_m` 150 on all six topologies; the trigger needs `range_m`
  above 600. Latent.
- **Half the wiring in `ae4f993` has never run.** `initial_human_vehicles` is 0 for
  every non-ring topology, so the reset-time spawn path never fires, and on `ring`
  every node has one incoming arc so no merge conflict can exist. Harmless, and it
  is why toggling the spawner alone is a sound experiment.

## Re-measured after these fixes

| | plain IDM | merge-aware |
|---|---|---|
| completion, 81 distinct runs | 43/81 (53%) | **57/81 (70%)** |
| collisions, 9 distinct `no_av` conditions | 155 | **138 (−11%)** |

The collision figure was −19% before the stopping floor. Part of that gain was
vehicles reversing out of trouble, and it is now gone.

---

# VALIDATOR ROUND 3, AND THE ATTRIBUTION SETTLED

Round 3 reproduced both headline numbers and found five things. The mutation table
closed every gap round 2 had reported, and the only surviving mutation
(`_stopping_floor` using `abs(speed)`) has no behavioural effect.

## The attribution, which round 3 showed was ambiguous

The stopping floor is not a merge fix: 73% of its interventions happen with no
merge conflict in view, and on `ring` — where the merge logic is provably inert,
every node having one incoming arc — it still changes behaviour. So the plain-IDM
control differed from the treatment in **two** ways, and the earlier statement
attributed the whole gain to "the merge rule".

Three arms, 243 runs, same seeds:

| arm | completion | collisions |
|---|---|---|
| A — plain IDM, neither change | 43/81 (53%) | 155 |
| B — merge rule only, floor disabled | **57/81 (70%)** | **126** |
| C — merge rule + stopping floor | 57/81 (70%) | 138 |

| change | completion | collisions |
|---|---|---|
| merge rule, A→B | **+17pp** | **−29** |
| stopping floor, B→C | **+0pp** | **+12** |

**The whole +17pp is the merge rule.** The floor buys no completion and costs 12
collisions, from the extra stopping distance: unrestricted −6 m/s² from 5 m/s
travels 2.08 m to rest, the floor's geometric decay 3.26 m. It is kept because a
vehicle reversing at −7.5 m/s is not a physical model, and 12 collisions is the
honest price of that correctness. Anyone quoting these numbers should quote C, and
say that B is the same result with an unphysical vehicle.

## Acted on

- **Noise asymmetry in the actor's input.** `_route_deltas` read the true
  `longitudinal_m` while the same-arc path read the noisy delta: at the training
  configs' 1.5 m, 3.308 m of spread same-arc against 0.000 m cross-arc. It now
  reuses the draw that neighbour already received.
- **The AV path was unfloored.** The env overwrites the AV command after
  `road.act()`. A `ControlledVehicle` at 0.3 m/s given −6.0 reached −1.5 m/s over
  three substeps. Measured headroom was +0.816 m/s over 16 runs. Both vehicle kinds
  now stop over the same 1 s horizon.
- **The parity ledger's `leader_gap` claim.** Scoped, not reclassified. See below.

## The inconsistency round 3 caught in this task's own reasoning

A real `distance_to_next_merge` was refused because the deployed vehicle has no map
matching. The route-aware leader gap uses **the same map** — a camera cannot know a
vehicle past a junction is on its route, nor accumulate lane length to it — and it
went into the actor's observation anyway.

Reverting is worse: an arc-blind actor beside a route-aware safety layer is the
larger defect. So the ledger description now scopes the identical claim to
single-arc scenes, which is all it was ever validated on: `_route_deltas` returns a
cross-arc neighbour **zero times across the entire parity suite**, because every
scene fixes one arc. Reclassifying to `approximated` cannot be done alone — the
suite requires a non-identical class to differ somewhere — so
`test_the_ledger_has_no_cross_arc_scene` fails the moment a cross-arc scene is
added, which is when reclassification becomes possible.

**Outstanding on task 47:** one cross-arc parity scene, then reclassify.

## Residuals, and one qualification worth acting on

Round 3 agreed the round 2 residuals are correctly scoped — the static successor
map versus the acquired `lane_index`, the single colliding pair the successor filter
excludes, `predecessors[0]`, the never-executed reset-time wiring — with one
qualification: **that assumption is now load-bearing in five places, not two.**
`_forward_lane_offsets`, the backward predecessor test, both successor filters, and
now the actor's observation. The surface grew this round rather than holding still,
and the parity inconsistency above is a direct consequence.

**So the next expansion of that assumption should be a deliberate decision, not a
side effect.** Recorded here so it is one.


---

# DECISION 2026-09-09: build route-aware leader gap on the Jetson

Taken by the user. **Recorded only; nothing implemented.** Filed as task 71 in
`plans/implementation_records.md`, with the scoping, the evidence that the data is already in
hand, and the two design questions that come first.

This resolves the inconsistency round 3 caught. The route-aware leader gap stays in
the actor's observation, and the deployed system will be given the means to produce
the same quantity, rather than the sim being cut back to what the vehicle can
currently see.

`distance_to_next_merge` is **not** covered by this decision and remains refused:
the flow API carries road geometry, not junction topology.

---

# CORRECTION 2026-09-09: the reward IS system-level. The earlier claim was wrong

**What was recorded above and in commit `1c8ee39` is wrong.** It said the reward is
ego speed with no throughput term. It is neither.

`_reward_for_vehicle` — `0.0 if crashed else vehicle.speed` — populates the
`rewards` dict that `env.step()` returns, and **the trainer ignores it**.
`src/rl/trainers.py:189` uses `build_team_reward(episode_metrics)`: one shared
system-level reward over nine network metrics, which is the MAPPO-style
formulation. I read the environment and assumed the trainer consumed it.

**What the reward actually is**, with each term's mean contribution over the
100-update run:

| metric | weight | mean value | contribution | share of positive |
|---|---|---|---|---|
| `mean_speed` | +0.05 | 20.860 | **+1.0430** | **69.3%** |
| `fairness_jain` | +0.50 | 0.885 | +0.4426 | 29.4% |
| `new_collision_count` | −5.00 | 0.044 | −0.2210 | |
| `rolling_roadblock_score` | −2.00 | 0.053 | −0.1059 | |
| `hard_braking_count` | −0.10 | 0.916 | −0.0916 | |
| `speed_std` | −0.02 | 3.484 | −0.0697 | |
| `throughput_recent` | +0.02 | 0.948 | **+0.0190** | **1.3%** |
| `queue_length_total` | −0.02 | 0.710 | −0.0142 | |
| `jam_fraction` | −1.00 | 0.013 | −0.0127 | |

Team reward +0.9894, times `reward_scale` 0.05, so +0.0495 per agent per step.

**The real finding, which the wrong one obscured.** Throughput is in the reward and
contributes **1.3%** of the positive signal. `mean_speed` contributes **69%**. The
ratio is **55 : 1**, because the weights do not normalise for scale: `mean_speed` is
about 21 in m/s while `throughput_recent` is about 0.95 in vehicles. The agent
optimises mean speed because that is where the reward is — which is exactly what
task 69 measured (+1.5 m/s of speed, no throughput gain).

**So the remedy changes.** It is not "add a system-level objective", which already
exists. It is to reweight so throughput is not 1.3% of the signal, or to normalise
the terms before weighting. That is a smaller change than a new reward, and it can
be done in `DEFAULT_REWARD_WEIGHTS` or per-config.

**How the error happened, since it is a repeat.** I read `_reward_for_vehicle` in
the environment, saw a plausible reward, and did not check whether the trainer
called it. The same class of mistake as reading `obs.feed_congestion` instead of
`obs_diagnostics.feed.downstream_congestion` on 2026-09-08: inspecting a producer
and assuming the consumer used it. Tracing the write path is the rule that would
have caught both.


---

# `plan_task_84_sumo_simulator.md`

Moved here on 2026-09-12. The migration of the topology ladder from highway-env to SUMO.

# Task 84 — Replace highway_env with SUMO

**Command:** `plan_dsrc_rec`. Decisions are taken by recommendation, not put to the
user, and marked as such.

## Short version

**Move the physics, road network and demand layers onto SUMO. Keep everything
above the sensing seam unchanged.**

Three attempts to make `highway_env` collision-free produced diminishing returns:
the safe-velocity bound took human-only collisions from 98 to 6 at 120 steps, the
node-geometry fix took the worst lateral discontinuity from 10 m to 0 m, and
closest-point-of-approach took the residual from 23 to 14 over ten 600-step runs
while making four seeds worse. Each helped; none finished.

**SUMO does not need constraining.** Measured before writing this plan, on the same
3-into-1 merge shape that broke `highway_env`, at 6000 veh/h into a two-lane exit —
three times past capacity — for 1800 steps with 134 vehicles on the network:
**zero collisions.** Its car-following model cannot produce one, because the safe
speed is the model rather than a cap applied over it.

It is also what the replication target uses. Vinitsky et al. and the Flow benchmarks
run on SUMO, so this removes a difference between our setup and theirs that was
never deliberate.

## The seam, and why the change is smaller than it looks

`VehicleSnapshot` is the boundary between the simulator and everything this project
built. Above it sit the sensing model, the safety layer, the etiquette filters, the
observation encoders, the policy and the trainers — none of which know what produces
the snapshots. Below it sits vehicle physics, the road network and demand.

All eleven `VehicleSnapshot` fields are available from TraCI:

| field | TraCI source |
|---|---|
| `vehicle_id` | the SUMO id |
| `role` | our own bookkeeping, unchanged |
| `segment_id` | edge id, mapped through the topology spec |
| `lane_index` | edge id plus lane index, translated |
| `lane_id` | `getLaneIndex` |
| `position` | `getPosition` |
| `longitudinal_m` | `getLanePosition` |
| `speed_mps` | `getSpeed` |
| `acceleration_mps2` | `getAcceleration` |
| `free_flow_speed_mps` | `getAllowedSpeed` |
| `crashed` | always `False`; SUMO does not produce collisions |

The environment's whole public surface is six methods — `reset`, `step`,
`get_segment_metrics`, `get_global_state`, `get_episode_summary`,
`get_local_observations` — so a SUMO-backed class implementing those is a drop-in.

## Scope boundary

- **In:** a new `src/envs/sumo_env.py`; SUMO network generation for `inverted_tree`
  and `inverted_tree_bottleneck`; demand as SUMO flows; AV actuation through TraCI;
  metric collection from TraCI.
- **Preserved unchanged:** `src/sensing/`, `src/safety/`, `src/metrics/` (its inputs
  are re-populated, not rewritten), `src/rl/`, `src/baselines/`, the observation
  encoders and `sim_contract`.
- **Becomes dead code:** `CollisionFreeMixin`, `MergeAwareIDMVehicle`,
  `SafeControlledVehicle`, and the merge-projection work in `src/sensing/local.py`
  that exists only because `highway_env` vehicles could not see across an arc.
- **Out:** the other four topologies. Only `inverted_tree` is needed, per the
  section C scope decision. `merge` and `straight_multilane` congested in 0 of 12
  cells and are not used.
- **Out:** the Jetson tree. Nothing under `deployment/` changes.

## Decisions, taken by recommendation

| # | decision | taken | rejected, and why |
|---|---|---|---|
| D1 | Where to cut | At `VehicleSnapshot` | Cutting higher would mean rewriting the sensing model, which is the project's own contribution and is already validated |
| D2 | TraCI or libsumo | **libsumo** where available, TraCI as fallback | libsumo runs in-process and is roughly an order of magnitude faster; TraCI's socket per step would dominate a 3600-step episode |
| D3 | Network definition | Generate `.nod.xml`/`.edg.xml` from the existing `TopologySpec` and run `netconvert` | Hand-written `.net.xml` files would duplicate the topology definition and drift from it |
| D4 | Demand | SUMO `<flow>` elements, rate taken from the existing demand configs | Keeping our own spawner would reintroduce the spawn-side defects SUMO handles natively |
| D5 | AV actuation | `traci.vehicle.setSpeed` on the commanded speed, with the safety layer still producing the command | `setAcceleration` exists but SUMO applies its own safety bound afterwards either way; speed is the more direct expression of what the controller decides |
| D6 | Keep `highway_env` installed | Yes, and leave the old env in place | The parity ledger, the observation audit and 341 passing tests reference it; removing it in the same change would make a regression unattributable |
| D7 | Collision reporting | Record `getCollidingVehiclesNumber` every step and assert it is zero | A silent assumption that SUMO cannot collide is exactly the kind of unmeasured premise this project has been caught by three times |

## Open items for the user

1. **Whether the old `highway_env` path is eventually deleted.** Keeping both is the
   safe choice now, but two simulators is a maintenance cost and a source of
   divergent results. Recommendation if asked: delete it once the SUMO path has
   reproduced a full evaluation, not before.
2. **Whether the sensing model's route-aware machinery stays.** In SUMO a vehicle's
   route is explicit, so the graph walk added under task 67 becomes unnecessary for
   the simulator — but the parity argument for it was about the *deployed* system,
   which is unchanged. Recommendation: keep it, and feed it SUMO's route.

## Steps

| # | step | done when |
|---|---|---|
| 1 | `SumoNetwork`: generate `.nod.xml`/`.edg.xml` from `TopologySpec`, run `netconvert`, expose the edge↔segment mapping | `inverted_tree` builds and `netconvert` reports no errors; a test asserts the node and edge counts against the spec |
| 2 | `SumoTopologyEnv.reset/step` with libsumo, producing `VehicleSnapshot`s | A 600-step run completes and the snapshots pass the same field checks the old env's do |
| 3 | Demand as flows, rate from the demand config | Measured throughput at 1800 veh/h is within 10% of the configured rate |
| 4 | AV actuation and the safety layer through TraCI | An AV commanded to a lower speed measurably slows, verified per step |
| 5 | Metrics from TraCI into the existing `global_metrics` shape | `mean_speed`, `throughput_recent`, `jam_fraction` and `collision_count` all populated; collision count asserted zero |
| 6 | Re-measure the capacity sweep on SUMO | The demand that congests without gridlocking is identified, as task 79 did for the old simulator |
| 7 | Retrain MAPPO on the SUMO env | A checkpoint exists, trained on SUMO, with its config recorded |
| 8 | Evaluate against `no_av` over an hour | Throughput compared with every run completing, since collisions can no longer truncate |

## Risks

- **Speed.** libsumo is fast but a 3600-step episode with 130 vehicles has real cost.
  Step 2 measures it before step 7 depends on it.
- **The observation may change meaning.** SUMO lane positions and our
  `longitudinal_m` are both distance along a lane, but SUMO's lanes are per-edge
  while `highway_env`'s were per-arc. Step 1's mapping is where that is settled, and
  a wrong mapping would silently move every gap the sensing model computes.
- **Any trained checkpoint is invalidated.** The actor's input distribution changes,
  which is the same dependency that put task 9 ahead of task 68. The existing
  checkpoint is a `highway_env` policy and cannot be evaluated on SUMO.
- **`SUMO_HOME` is unset**, so XML validation is disabled. Harmless for generated
  files, but it means a malformed network is caught by `netconvert` rather than by
  a schema.

## Sign-off

- [ ] Steps 1–5 complete, tests passing, collision count asserted zero across a full run
- [ ] Capacity re-measured on SUMO
- [ ] MAPPO trained and evaluated over an hour, with every run completing
- [ ] `validate_dsrc_3` clean


---

# Steps 1-6 measured 2026-09-09

**Steps 1 to 5 are implemented and tested.** 22 new tests across network
generation, the environment, the road view and the control wiring; the simulator
suite is at 369 passed.

## What SUMO changed, measured

| | highway_env | SUMO |
|---|---|---|
| collisions, 600 steps at every demand | 14 to 23 over ten runs | **0** |
| wall time, 600 steps | ~40 s with CPA | **0.2 s** |
| runs completing | 6 or 7 of 10 | **all** |
| congestion reachable | only via crash queueing | **yes, by demand** |

An hour-long episode costs about 1.2 s, so the evaluation that was unmeasurable is
now nearly free.

## Step 6: capacity is junction-limited and much lower than it appeared

1800-step runs, no AVs, insertion keeping up throughout:

| veh/h | mean speed | arrived | departed | ratio |
|---|---|---|---|---|
| 450 | 26.26 | 222 | 228 | 1.01 |
| 600 | 18.11 | 289 | 300 | 1.00 |
| 750 | 20.25 | 361 | 384 | 1.02 |
| **900** | 7.35 | **378** | 456 | 1.01 |
| 1050 | 2.11 | 349 | 528 | 1.01 |
| 1800 | 1.18 | 425 | — | — |
| 3000 | 1.39 | 538 | — | — |

**Capacity is about 900 veh/h**: arrivals are maximal there and fall at 1050 while
insertion still keeps up. `configs/demand/sumo_saturating.yaml` is 1050, past the
knee, with 29 of 378 arrivals lost to congestion as the headroom a controller has.

**It is junction-limited, not lane-limited.** The three-into-two merges at b1 and b2
are unsignalised, so vehicles yield and throughput is throttled well below what the
two-lane trunk could carry. Realistic for an uncontrolled merge, and the reason
SUMO's capacity here is far below the ~1800 the previous simulator appeared to
manage — that figure was measured on a road where vehicles could drive through each
other.

**One instrument caveat.** `jam_fraction` as computed here is the share of vehicles
below `queue_speed_mps` of 5 m/s, and it reads 0.167 even at 450 veh/h where mean
speed is 26 m/s, because vehicles just inserted and vehicles crossing junctions are
momentarily slow. Mean speed and arrivals are the reliable congestion indicators at
this demand range; `jam_fraction` is not, and should not be used as the operating
point criterion.


---

# Steps 7 and 8 complete 2026-09-09: an hour of simulated driving, measured

**The evaluation task 74 asked for has run.** Five seeds per arm, 3600 steps each,
at `sumo_saturating` with 20% AV penetration, on the checkpoint from a 400-update
SUMO training run:

| arm | arrivals | mean speed | completed | collisions |
|---|---|---|---|---|
| `no_av` | 679.0 ± 8.5 | 1.01 ± 0.11 | 5/5 | 0 |
| `mappo` | 685.4 ± 3.0 | 1.06 ± 0.07 | 5/5 | 0 |
| difference | **+6.4 (+0.9%)** | +0.05 | | |

**Every run completed and no run produced a collision.** That has not happened
before in this project: the same comparison on the previous simulator completed 7
of 18 conditions and every figure came from runs selected for not having crashed.
Ten hour-long runs took 2 minutes 18 seconds in total.

**The instrument is now sensitive enough to trust a null.** Arrivals have a standard
deviation of 3.0 and 8.5 on a mean of 680, so the resolution is about 1%. The 52%
effect measured under task 86 would be unmistakable at this resolution. The +0.9%
reported here is therefore a real null and not a measurement failure.

**And the null has a known cause.** Task 86 measured that every action the policy
can express decodes to 20-30 m/s at a 30 m/s limit, while the effect lives at
10 m/s, and that arrivals were identical across `slow`, `nominal`, `fast` and no
action. A policy choosing among equivalent actions cannot produce a difference. So
this evaluation confirms the instrument rather than the policy, which is the useful
thing it can do until the action space is rescaled.

**One caveat on the operating point.** Mean speed is about 1.0 m/s across both
arms, so at 1050 veh/h the network is heavily congested by the end of an hour --
more so than the 900-step measurements that chose that rate showed. An hour-long
evaluation wants a demand chosen against hour-long runs, which is a shorter version
of the capacity sweep at 3600 steps.


## The operating point depends on the run length

`sumo_saturating` at 1050 veh/h was chosen against 600- and 1800-step runs. Over an
hour it is heavily congested: mean speed 4.04 m/s with no AVs over three seeds, and
a 23.9% throughput deficit. It remains the right level for 600-step runs, where a
controller recovers a measurable part of that deficit; the hour-long level is a
separate config for that reason.

Hour-long capacity, 3600 steps after a 300-step warm-up, no AVs, three seeds
(7, 17, 27), with the fleet the demand config declares:

| veh/h | mean speed | arrived | deficit | deficit % |
|---|---|---|---|---|
| 750 | 20.10 | 748.3 +/- 2.1 | 1.7 | 0.2% |
| 780 | 19.76 | 777.7 +/- 1.2 | 2.3 | 0.3% |
| 810 | 19.07 | 807.7 +/- 3.2 | 2.3 | 0.3% |
| 840 | 17.64 | 830.0 +/- 10.6 | 10.0 | 1.2% |
| 870 | 15.45 | 856.0 +/- 12.5 | 14.0 | 1.6% |
| 900 | 12.37 | **858.0 +/- 24.3** | 42.0 | 4.7% — arrivals peak here |
| 930 | 8.44 | 840.3 +/- 15.5 | 89.7 | 9.6% |
| 960 | 6.86 | 833.0 +/- 18.3 | 127.0 | 13.2% |
| 990 | 5.35 | 813.7 +/- 21.1 | 176.3 | 17.8% |
| 1020 | 4.35 | 797.7 +/- 10.3 | 222.3 | 21.8% |
| 1050 | 4.04 | 799.0 +/- 15.7 | 251.0 | 23.9% |
| 1100 | 3.43 | 789.3 +/- 13.6 | 310.7 | 28.2% |
| 1200 | 2.90 | 810.3 +/- 12.7 | 389.7 | 32.5% |

**This table replaces two earlier ones, and the reason is a defect rather than
noise.** `_write_routes` read only `max_mps` from the demand config's speed
distribution and hardcoded the desired-speed spread as a factor on the lane limit,
so every earlier measurement was taken on a fleet desiring about 30 m/s where the
config declares 24. Hour-long capacity was measured at 858 veh/h rather than the
802 recorded before, and the arrival peak moved from 810 to 900 veh/h.

Attributed, at 1050 veh/h over an hour with three seeds:

| route writer | arrived | mean speed |
|---|---|---|
| lane-limit fleet, inserted at rest (the previous behaviour) | 677.0 +/- 7.0 | 2.74 |
| declared fleet, inserted at rest | 798.0 +/- 18.2 | 3.98 |
| declared fleet, inserted at its desired speed (now) | 799.0 +/- 15.7 | 4.04 |

The whole difference is the desired-speed distribution. Adding
`departSpeed="desired"` changes arrivals by 1.0 against a standard deviation of 16,
which is nothing; it is in for fidelity with the other simulator's spawner, not for
an effect.

The direction is worth stating plainly: **a fleet that wants to drive more slowly
gets more vehicles through this junction** — 798 against 677 per hour at the same
demand. That is the same speed-harmonisation effect the controller is meant to
exploit, appearing here in the fleet's own desired speed rather than in a control
action.

840 veh/h remains the choice: it sits past the arrival peak, which is what gives a
controller a deficit to recover, and it is congested while still moving.


## One demand level serves both horizons, now that the fleet is the declared one

Over an hour, with AVs at 20% penetration held at 10 m/s against an uncommanded
fleet, three seeds:

| veh/h | uncommanded | AVs at 10 m/s | gain | deficit recovered |
|---|---|---|---|---|
| 930 | 840.3 +/- 15.5 | 901.3 +/- 29.8 | +7.3% | 61.0 of 89.7 |
| 960 | 833.0 +/- 18.3 | 931.0 +/- 30.1 | +11.8% | 98.0 of 127.0 |
| 1050 | 799.0 +/- 15.7 | **966.3 +/- 31.8** | **+20.9%** | 167.3 of 251.0 |

The effect grows with demand and is larger over an hour than over 600 steps: 20.9%
against 17.6% at the same 1050 veh/h. At that level a controller recovers 67% of the
throughput deficit.

**`configs/demand/sumo_hour.yaml` is therefore removed.** It existed because
`sumo_saturating` at 1050 veh/h appeared to gridlock over an hour — 0.89 m/s in the
first measurement — leaving no deficit a controller could recover. That gridlock was
a property of the undeclared 30 m/s fleet: with the fleet the config actually
declares, 1050 veh/h is congested and moving at both horizons (4.04 m/s uncommanded,
6.11 m/s under control) and carries the largest recoverable deficit of any level
measured. The second level was an artifact of the defect, not a property of the
road. Nothing referenced the file.

## The measured effect depends on the episode length, and 120 steps hides it

Measured 2026-09-09 and re-measured after the demand fix below; these numbers
supersede every earlier figure for a commanded-speed arm. Two defects sat under the
earlier ones: the fairness term's denominator excluded branches that had completed
nothing, and the fleet's desired speed was the lane limit rather than the 24 m/s the
demand config declares.

Every AV commanded to hold one speed for the whole episode, at `sumo_saturating`
(1050 veh/h, 20% penetration), five seeds:

| arm | 120 steps: arrivals | reward | 600 steps: arrivals | reward |
|---|---|---|---|---|
| uncommanded | 27.8 +/- 3.1 | +1.339 | 132.0 +/- 6.4 | +0.823 |
| 5 m/s | 23.2 +/- 3.0 | +0.670 | 123.4 +/- 19.9 | +0.520 |
| 8 m/s | 24.4 +/- 1.9 | +0.773 | 148.8 +/- 9.0 | +0.915 |
| 10 m/s | 26.2 +/- 4.3 | +0.848 | **155.2 +/- 10.5** | +0.930 |
| 12 m/s | 26.8 +/- 3.5 | +0.920 | 150.6 +/- 17.9 | +0.823 |
| 15 m/s | 27.4 +/- 4.0 | +1.132 | 148.0 +/- 17.1 | **+1.034** |
| 20 m/s | 28.4 +/- 4.2 | +1.359 | 138.8 +/- 10.6 | +0.987 |
| 24 m/s | 28.8 +/- 2.9 | +1.359 | 137.0 +/- 4.7 | +0.956 |

Three things follow.

**The effect does not exist at 120 steps.** No commanded arm beats an uncommanded
fleet there: the best is 28.8 +/- 2.9 against 27.8 +/- 3.1, which is noise. At 600
steps holding 10 m/s produces 155.2 +/- 10.5 against 132.0 +/- 6.4, an increase of
17.6% with a standard error of about 5 arrivals.
`configs/training/mappo_sumo.yaml` declared no `duration_steps` and therefore took
the trainer's default of 120, so training would have been run at the one horizon
where the effect it is meant to learn is absent. It now declares 600.

**The 52% figure is superseded twice over.** 167 arrivals against 110 was a single
seed, on the undeclared 30 m/s fleet, with the fairness defect present. The same
comparison is now 155.2 against 132.0. The direction has held through every
correction; the magnitude has fallen from 52% to 17.6%.

**The reward does not rank the arms the way arrivals do.** At 600 steps arrivals are
maximised at 10 m/s and the reward at 15 m/s, and at 120 steps the reward ranks the
fastest arms first — the reverse of the 600-step arrival order. The reward's
`throughput_recent` term is a 60-second window rather than the episode's total, and
`jam_fraction` at weight -2.0 grows as the network fills, so the reward is not a
monotone function of the quantity the experiment is about. Recorded, not fixed: it
bears on task 85 and on any claim that a trained policy maximising this reward
maximises throughput.


## Retraction: the throughput effect was an artefact of the simulation step

Measured 2026-09-09 by `scripts/measure_step_size_convergence.py`, after the step
size was changed to 0.1 s so the deployment's measured 96.7 ms sensing latency
could be represented at all. Five seeds, 600 s episodes after 300 s of warm-up, at
`sumo_saturating`:

| dt | steps | uncommanded | AVs at 10 m/s | gain |
|---|---|---|---|---|
| 1.0 | 600 | 132.0 +/- 6.4 | 155.2 +/- 10.5 | **+17.6%** |
| 0.5 | 1200 | 134.4 +/- 5.9 | 157.8 +/- 16.0 | +17.4% |
| 0.2 | 3000 | 157.6 +/- 12.3 | 160.4 +/- 7.2 | +1.8% |
| 0.1 | 6000 | 168.4 +/- 11.3 | 161.2 +/- 8.6 | **−4.3%** |
| 0.05 | 12000 | 173.0 +/- 8.4 | 161.6 +/- 8.6 | **−6.6%** |

The commanded arm moves by 6.4 arrivals across a twentyfold change in the step
size, which is inside its own standard deviation. The uncommanded arm rises 31%,
from 132.0 to 173.0. The treatment is invariant to the numerical parameter and the
control is not, so what separated them was never a property of the traffic.

**Mechanism.** SUMO's `--step-length` is the physics step, and this migration
passed `dt` straight into it. At dt 1.0 a vehicle travelling 24 m/s advances 24 m
per step, so junction gap acceptance and car following are resolved at 24 m
granularity and an uncommanded fleet loses throughput to the integration. A fleet
held at 10 m/s advances 10 m per step and loses much less of it.

**The other simulator already handled this.** `HighwayTopologyEnv` integrates at
`physics_substeps: 10` and says why at `src/envs/topology_env.py:233-236`: "a
single 1 s Euler step drives IDM vehicles through each other and to negative
speeds". Its decisions are taken once per second while its physics runs at 0.1 s.
The SUMO env had no equivalent, so from the first commit of this migration the
SUMO fleet integrated ten times more coarsely than the fleet it was being compared
against. Every capacity figure in this document above this section was measured on
the coarse integration, including the junction-limited capacity the demand levels
were chosen against.

Full arm table at dt 0.1, five seeds, from `scripts/measure_commanded_arms.py`:

| arm | arrivals | team reward per step | roadblock score |
|---|---|---|---|
| uncommanded | **168.4 +/- 11.3** | +2.159 | 0.002 |
| 8 m/s | 150.0 +/- 8.5 | +1.408 | 0.306 |
| 10 m/s | 161.2 +/- 8.6 | +1.534 | 0.257 |
| 12 m/s | 163.2 +/- 10.8 | +1.642 | 0.206 |
| 15 m/s | 168.0 +/- 11.7 | +2.090 | 0.122 |
| 20 m/s | 172.4 +/- 8.7 | +2.238 | 0.004 |
| 24 m/s | 172.6 +/- 10.8 | +2.323 | 0.002 |

No arm beats an uncommanded fleet by more than one standard error, and the reward
now ranks the arms in the same order as arrivals — which it did not at dt 1.0. The
disagreement that motivated the metering exemption was itself a symptom of the
coarse integration: the roadblock term was firing on AVs whose slowness was
recovering discretisation error.

**Two decision rates now differ between the simulators.** SUMO takes a decision
every 0.1 s and highway_env every 1.0 s. That is deliberate: the deployed loop runs
at about 30 Hz, so 10 Hz is the closer of the two, and it is what makes the
measured sensing latency representable. It does mean a per-step quantity is not
comparable across the two simulators without dividing by the step.


---

# `plan_task_103_local_credit.md`

Moved here on 2026-09-12. The credit-assignment investigation on the local-sensing formulation.

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
8. Record the result, pass or fail, in `plans/implementation_records.md`.

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


---

# `result_fleet_mix_sweep.md`

Moved here on 2026-09-12. A penetration sweep on that formulation.

# Does the network outcome depend on the fleet-wide action mix?

Ankit's objection, and it was the right one: a shared policy can represent the
centralized behaviour, so if a centralized controller helps, the behaviour is in the
search space and the problem is the search rather than the space. Everything measured
before this was about a single agent's single decision. This measures the fleet.

## Why the policy cannot explore this dimension, measured

Each agent samples independently from the shared policy, so the fleet-wide mix
concentrates. Measured over five rollouts at 39 agents:

- per-step fraction choosing `slow`: standard deviation **0.0714**
- rollout-mean mix across five seeds: **0.3073 +/- 0.0082**

**The gradient only ever sees fleet configurations between about 0.291 and 0.324**, a
three-point window, against the 0 to 1 this sweep covers. Each agent's share of a
fleet-level signal is `dJ/dp / N`, which at N = 39 is what sits below the gradient
noise floor. The estimator accounts for parameter sharing correctly -- the gradient is
summed over all agents with respect to the shared parameters -- and it still cannot
explore the fleet dimension, because independent sampling concentrates it.

## The design

At every decision each AV independently issues a command with probability `p` and
otherwise RELEASES to SUMO's car-following, so `p = 0` is exactly the uncommanded
fleet and `p = 1` is every AV commanded at every decision. Without the release a
vehicle commanded once stays commanded and `p` would mean "fraction ever commanded".
Penetration is fixed in every arm, so every arm at a seed sees identical traffic and
they differ only in what is commanded. `sumo_capacity_drop`, 600 s episodes after a
300 s warm-up, dt 0.1, decisions at 1 Hz, five seeds, paired.

## Result: speed metering, and there is no positive region

Reference 184.6 arrivals. The `+/-` is one standard error of the paired difference.

| command | p | arrivals | paired vs p=0 | verdict |
|---|---|---|---|---|
| `shipped`: max(12, allowed - 10), binds on 6% of decisions | 0.25 | 176.2 | -8.4 +/- 6.2 | no effect |
| | 0.50 | 185.2 | +0.6 +/- 14.2 | no effect |
| | 0.75 | 171.4 | -13.2 +/- 12.5 | no effect |
| | 1.00 | 169.2 | -15.4 +/- 14.0 | no effect |
| `binding`: 0.6x the vehicle's own speed, binds on 29% | 0.25 | 128.2 | **-56.4 +/- 15.1** | real |
| | 0.50 | 66.6 | **-118.0 +/- 12.7** | real |
| | 0.75 | 37.8 | **-146.8 +/- 9.6** | real |
| | 1.00 | 45.8 | **-138.8 +/- 11.8** | real |

**A THIRD COMMAND was measured later and sits between these two.** The threshold
sweep in `result_simulation_leg_null.md` issues the SRC config's own `slow` bin, a
fixed 8.33 m/s, on identical episodes with the same Bernoulli draw at the same 1 s
interval -- its p=0 arm reproduces this table's reference seed for seed, 198, 194,
156, 179 and 196. It costs 25 to 37 arrivals where `binding` costs 56 to 147, because
8.33 m/s is a CEILING that SUMO's car-following often already dominates at a network
mean speed of 1.9 to 2.6 m/s, while 0.6 times the current speed always binds and
compounds across decisions. **The three p columns are not comparable to each other**:
the same nominal p means a different intervention in each.

**J(p) is monotone decreasing.** The command that barely bites is flat within noise
with a downward trend; the command that bites destroys throughput in proportion to how
often it is issued. The two definitions are both present because a flat result under
a command that does nothing cannot be told from a flat objective.

**So this is not "findable but not by PPO".** A sweep across the whole fleet-mix
range, which PPO structurally cannot reach, finds no speed-metering configuration that
beats doing nothing.

It also settles the bin-rescale question retracted earlier from the other direction:
bins that bind make things worse, not better.

## What the actions CAN do, since "the actions do nothing" was too strong

Both heads reach the world; the earlier phrasing was wrong.

| head | effect | measured |
|---|---|---|
| speed | acts, but its three values are equivalent on 97% of decisions at this operating point | mean AV speed 7.33 m/s against bins of 20, 27 and 30 |
| headway | acts at any speed. `setTau` reaches W99 although W99 follows on `cc1`, which was not obvious | mean AV gap 78.05 m -> 83.51 m, arrivals 42 -> 40, paired on one seed |

## Still running

`headway` (tau 3.0) and `harmonize` (command the segment's prevailing speed, so a fast
vehicle slows and a slow one speeds up) across the same mix range. Harmonization is
the mechanism the ring-road results damp stop-and-go waves with, and no arm above
tests it: every one of them only ever slows a vehicle.

A first look at harmonization on one seed is neutral on throughput -- 42, 40, 42
arrivals at p = 0, 0.5, 1.0 -- and RAISES jerk from 0.390 to 1.041, because the
command is a hard `setSpeed` to the segment mean once a second and therefore steps
rather than smooths. That is a limitation of this implementation of harmonization, not
a measurement of harmonization.

## Two defects in this instrument, both caught before they reached a result

- **An unrecognised scheme name fell through to the binding command.** `harmonize`
  ran as `binding` under a new label and produced three numbers that looked like a
  measurement of something never executed. An unknown scheme now raises. The guard was
  then run against a control and the first control was inconclusive -- with a 60-step
  warm-up no agent exists, so the command branch is never reached and nothing raises.
- **`harmonize` was missing from `SCHEMES`**, so the queued batch would have skipped
  it silently and the sweep would have reported three schemes where four were
  intended.


---

# `replication_state.json`

Moved here on 2026-09-12. The MAPPO replication's run state.

```json
{
  "schema": 1,
  "updated_at": "2026-09-10T01:57:16Z",
  "stale_after_minutes": 420,
  "blocking": null,
  "awaiting_user": null,
  "next_action": "WAITING on five MAPPO training runs, then execute the pre-registration. Nothing is blocked and no decision is outstanding.\n\nTHE RUNS. Five processes started from scripts/train_policy.py --training mappo_sumo --seed {7,17,27,37,47} --output-root <scratch>/train/runs, 100 updates each, about 3.6 min per update, so roughly six hours from 2026-09-09. Progress: count lines in <run>/training_metrics.csv. All five reported ZERO collisions through update 7, which is the property that invalidated the previous attempt. If the scratch directory is gone, retrain: the runs are reproducible from the committed config and the seeds are fixed.\n\nTHEN, and only then: .venv/bin/python scripts/evaluate_burst_scenario.py --checkpoint-root <scratch>/train/runs. It implements task 93 and nothing else. Do NOT change the metric, the seeds, the comparators or the bar after seeing the numbers; that is the whole point of the pre-registration. Report whatever it says, including a null -- task 92 retracted the evidence that a throughput effect exists here at a converged step size, so a null is the expected outcome.\n\nDO NOT modify src/sumo/env.py or the metric path until the evaluation has run. The policy must be evaluated in the environment it trained in. Three round-3 findings are deliberately parked on that ground and recorded as task 95.",
  "last_verdict": null,
  "section": "C. Simulation replication, task 84 SUMO migration",
  "task": {
    "number": 93,
    "what": "the pre-registered MAPPO run on the burst scenario",
    "stage": "training",
    "stage_started_at": "2026-09-10T01:57:16Z",
    "background_job": {
      "what": "5 MAPPO seeds, 100 updates each",
      "started_at": "2026-09-10T01:57:16Z",
      "expect_by": "about six hours from the start"
    },
    "needs_hardware": false,
    "remaining": [
      95,
      89
    ],
    "stopped_by_user": false
  },
  "carried_forward": [
    "Use .venv/bin/python. System python3 is 3.14 with no pytest.",
    "HEAD 064e688, 475 tests pass, everything pushed to origin/main.",
    "THE GOAL: the paper is a deployment story; the simulation replicates that MAPPO works under the deployment-measured sensing model. Task 89.",
    "TASK 92 IS THE HEADLINE: the 17.6% throughput effect was an artefact of the 1 s physics step. The commanded arm is invariant across a twentyfold change in dt while the uncommanded baseline rises 31%; at a converged step, holding AVs at 10 m/s LOWERS throughput. SUMO's --step-length is the physics step and the migration passed dt into it; HighwayTopologyEnv substeps ten times per decision and says why. Tasks 86 and 91 are retracted and moot respectively.",
    "TASK 94: making the action heads actuate introduced a collision path. hold_lane set SUMO's lane-change mode to 0, which disables collision avoidance rather than lane changes. Every head alone gave 0 collisions and lane+merge together gave 646, so the guard varies the heads together. The first five training runs were killed because of it.",
    "The demand config's burst was accepted and ignored on SUMO. The first fix wrote one <flow> per period and lost half the demand (71 departures against 150); the schedule is now written vehicle by vehicle. The control that caught it is a burst with multiplier 1.0, which must change nothing.",
    "configs/demand/sumo_burst.yaml: 900 veh/h medium-high, doubled for 150 s, then back. Base is 900 because at 1050 the queue grows with no burst at all and recovery is undefined. Profiled with no AVs: speed 19 -> 7.5 m/s, queue peaks at 40, recovers to 17.7 with the queue at 6.",
    "Capacity at dt 0.1 saturates near 900-950 veh/h. Every capacity figure measured at dt 1.0 is superseded.",
    "Measurement scripts are committed: scripts/measure_commanded_arms.py, scripts/measure_step_size_convergence.py, scripts/evaluate_burst_scenario.py. Round 3 could not reproduce the arm tables and had to recover the seed set (3,7,11,19,23) by exhaustive search; quote no figure a committed script does not produce.",
    "Validator rounds 1, 2 and 3 are complete and every finding is either fixed or recorded with a reason. Round 3's remaining three are task 95.",
    "Sensing: position_noise_std 3.2 and speed_noise_std 0.93 cite Song et al. ICRA 2020 (arXiv:2006.04082). latency_s is 0.1, representing the measured 96.7 ms p50, which is why dt is 0.1."
  ],
  "stopped_by_user": false
}
```


---

# `result_task93_burst_evaluation.json`

Moved here on 2026-09-12. The pre-registered burst evaluation's raw result.

```json
{
 "no_av": [
  {
   "seed": 7,
   "arrivals": 248,
   "trough_speed": 7.406476312611605,
   "recovery_s": 486.20000000000005,
   "roadblock_sum": 31.888888888888793,
   "collisions": 0
  },
  {
   "seed": 17,
   "arrivals": 266,
   "trough_speed": 9.97634236991829,
   "recovery_s": 102.60000000000002,
   "roadblock_sum": 31.888888888888797,
   "collisions": 0
  },
  {
   "seed": 27,
   "arrivals": 259,
   "trough_speed": 8.47748295650934,
   "recovery_s": 386.5,
   "roadblock_sum": 28.666666666666583,
   "collisions": 0
  },
  {
   "seed": 37,
   "arrivals": 265,
   "trough_speed": 8.908168910591943,
   "recovery_s": 125.80000000000001,
   "roadblock_sum": 33.77777777777773,
   "collisions": 0
  },
  {
   "seed": 47,
   "arrivals": 261,
   "trough_speed": 9.231528293660858,
   "recovery_s": 221.39999999999998,
   "roadblock_sum": 38.2222222222223,
   "collisions": 0
  }
 ],
 "density_lookup": [
  {
   "seed": 7,
   "arrivals": 232,
   "trough_speed": 7.090468837461658,
   "recovery_s": 533.6,
   "roadblock_sum": 244.66666666667237,
   "collisions": 0
  },
  {
   "seed": 17,
   "arrivals": 265,
   "trough_speed": 9.049838125921932,
   "recovery_s": 232.5,
   "roadblock_sum": 176.00000000000395,
   "collisions": 0
  },
  {
   "seed": 27,
   "arrivals": 258,
   "trough_speed": 8.774728567948928,
   "recovery_s": 260.70000000000005,
   "roadblock_sum": 254.1111111111173,
   "collisions": 0
  },
  {
   "seed": 37,
   "arrivals": 239,
   "trough_speed": 9.017853002800337,
   "recovery_s": null,
   "roadblock_sum": 67.88888888888981,
   "collisions": 0
  },
  {
   "seed": 47,
   "arrivals": 218,
   "trough_speed": 6.572967802694724,
   "recovery_s": null,
   "roadblock_sum": 125.7777777777802,
   "collisions": 0
  }
 ],
 "mappo": [
  {
   "seed": 7,
   "arrivals": 226,
   "trough_speed": 6.2007567560538615,
   "recovery_s": 494.4000000000001,
   "roadblock_sum": 1873.555555555326,
   "collisions": 0
  },
  {
   "seed": 17,
   "arrivals": 222,
   "trough_speed": 5.438965949558433,
   "recovery_s": null,
   "roadblock_sum": 1385.222222222107,
   "collisions": 0
  },
  {
   "seed": 27,
   "arrivals": 258,
   "trough_speed": 11.812363487964344,
   "recovery_s": 32.700000000000045,
   "roadblock_sum": 1195.2222222220942,
   "collisions": 0
  },
  {
   "seed": 37,
   "arrivals": 257,
   "trough_speed": 8.204630819238206,
   "recovery_s": 106.20000000000005,
   "roadblock_sum": 1252.5555555554145,
   "collisions": 0
  },
  {
   "seed": 47,
   "arrivals": 251,
   "trough_speed": 9.688467611901972,
   "recovery_s": 221.0,
   "roadblock_sum": 1119.7777777776423,
   "collisions": 0
  }
 ]
}
```


---

# `result_task99_corrected_road.json`

Moved here on 2026-09-12. The corrected-road fundamental diagram's raw result, on `inverted_tree`.

```json
{
 "no_av": [
  {
   "seed": 57,
   "arrivals": 247,
   "trough_speed": 8.665170766528725,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 14.888888888888852,
   "collisions": 0
  },
  {
   "seed": 67,
   "arrivals": 251,
   "trough_speed": 9.473818986724746,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 9.777777777777759,
   "collisions": 0
  },
  {
   "seed": 77,
   "arrivals": 258,
   "trough_speed": 16.83249841835625,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 0.8888888888888891,
   "collisions": 0
  },
  {
   "seed": 87,
   "arrivals": 228,
   "trough_speed": 7.510798718505416,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 7.888888888888876,
   "collisions": 0
  },
  {
   "seed": 97,
   "arrivals": 255,
   "trough_speed": 9.069060634388993,
   "recovery_s": 245.10000000000002,
   "roadblock_sum": 21.111111111111054,
   "collisions": 0
  },
  {
   "seed": 107,
   "arrivals": 261,
   "trough_speed": 13.462527275860475,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 12.99999999999997,
   "collisions": 0
  },
  {
   "seed": 117,
   "arrivals": 252,
   "trough_speed": 10.04161400465513,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 12.777777777777748,
   "collisions": 0
  },
  {
   "seed": 127,
   "arrivals": 242,
   "trough_speed": 9.484298085131435,
   "recovery_s": 455.80000000000007,
   "roadblock_sum": 3.666666666666668,
   "collisions": 0
  },
  {
   "seed": 137,
   "arrivals": 241,
   "trough_speed": 10.382382982729064,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 18.77777777777773,
   "collisions": 0
  },
  {
   "seed": 147,
   "arrivals": 260,
   "trough_speed": 17.274079760302076,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 1.6666666666666672,
   "collisions": 0
  }
 ],
 "density_lookup": [
  {
   "seed": 57,
   "arrivals": 257,
   "trough_speed": 12.06558043703399,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 24.444444444444375,
   "collisions": 0
  },
  {
   "seed": 67,
   "arrivals": 261,
   "trough_speed": 20.835364994504534,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 0.0,
   "collisions": 0
  },
  {
   "seed": 77,
   "arrivals": 259,
   "trough_speed": 20.591774956610312,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 0.0,
   "collisions": 0
  },
  {
   "seed": 87,
   "arrivals": 248,
   "trough_speed": 7.262257719072921,
   "recovery_s": 441.70000000000005,
   "roadblock_sum": 12.2222222222222,
   "collisions": 0
  },
  {
   "seed": 97,
   "arrivals": 236,
   "trough_speed": 9.910605256464182,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 31.333333333333254,
   "collisions": 0
  },
  {
   "seed": 107,
   "arrivals": 246,
   "trough_speed": 10.02420974988371,
   "recovery_s": 295.5,
   "roadblock_sum": 22.555555555555493,
   "collisions": 0
  },
  {
   "seed": 117,
   "arrivals": 227,
   "trough_speed": 7.3716522768466,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 26.66666666666659,
   "collisions": 0
  },
  {
   "seed": 127,
   "arrivals": 261,
   "trough_speed": 16.030331987448417,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 35.222222222222214,
   "collisions": 0
  },
  {
   "seed": 137,
   "arrivals": 225,
   "trough_speed": 6.627194681594958,
   "recovery_s": null,
   "roadblock_sum": 60.55555555555628,
   "collisions": 0
  },
  {
   "seed": 147,
   "arrivals": 260,
   "trough_speed": 10.765323546054654,
   "recovery_s": 0.10000000000002274,
   "roadblock_sum": 12.99999999999997,
   "collisions": 0
  }
 ],
 "mappo": [
  {
   "seed": 57,
   "policies": 5,
   "arrivals": 243.0,
   "trough_speed": 9.574903309159925,
   "roadblock_sum": 218.9777777777646,
   "collisions": 0.4,
   "recovery_s": 91.22500000000002
  },
  {
   "seed": 67,
   "policies": 5,
   "arrivals": 247.0,
   "trough_speed": 11.72531285629773,
   "roadblock_sum": 326.7111111110932,
   "collisions": 0.0,
   "recovery_s": 99.28000000000002
  },
  {
   "seed": 77,
   "policies": 5,
   "arrivals": 239.2,
   "trough_speed": 12.566456964970342,
   "roadblock_sum": 407.1777777777496,
   "collisions": 0.0,
   "recovery_s": 0.12000000000002728
  },
  {
   "seed": 87,
   "policies": 5,
   "arrivals": 223.4,
   "trough_speed": 7.668105965712536,
   "roadblock_sum": 335.44444444442695,
   "collisions": 1.2,
   "recovery_s": 0.10000000000002274
  },
  {
   "seed": 97,
   "policies": 5,
   "arrivals": 231.8,
   "trough_speed": 7.6031356260688,
   "roadblock_sum": 300.1999999999831,
   "collisions": 0.8,
   "recovery_s": 122.43333333333335
  },
  {
   "seed": 107,
   "policies": 5,
   "arrivals": 245.6,
   "trough_speed": 10.290422367093772,
   "roadblock_sum": 327.466666666646,
   "collisions": 0.4,
   "recovery_s": 97.05000000000003
  },
  {
   "seed": 117,
   "policies": 5,
   "arrivals": 220.6,
   "trough_speed": 7.0929377549400785,
   "roadblock_sum": 241.822222222206,
   "collisions": 0.0,
   "recovery_s": 173.50000000000003
  },
  {
   "seed": 127,
   "policies": 5,
   "arrivals": 238.6,
   "trough_speed": 8.827773886335319,
   "roadblock_sum": 472.3777777777411,
   "collisions": 0.0,
   "recovery_s": 136.57500000000002
  },
  {
   "seed": 137,
   "policies": 5,
   "arrivals": 239.6,
   "trough_speed": 9.983675541540482,
   "roadblock_sum": 419.46666666663333,
   "collisions": 0.8,
   "recovery_s": 82.75000000000003
  },
  {
   "seed": 147,
   "policies": 5,
   "arrivals": 234.0,
   "trough_speed": 9.280400020190445,
   "roadblock_sum": 334.86666666663643,
   "collisions": 0.0,
   "recovery_s": 32.47500000000002
  }
 ]
}
```
