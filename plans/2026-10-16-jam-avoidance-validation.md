# Jam-avoidance program: validation record

Independent validator. This file is the record if the validator is stopped. Each finding has an
id (R<round>-<n>), a bucket (BLOCKS / SHOULD FIX / DOCUMENTED LIMIT / NOT WORTH FIXING), a
location, the evidence and the proposed fix. Findings are appended as soon as there is evidence.

## Round 1 (2026-10-16): commit 706f5b5, steps S1 to S6, S10, block G of S12

Audited from an archive mirror of `706f5b5` at
`scratchpad/dsrc_jam/mirrors/round1` (session `4ea4312b-...`), never the live tree. Diff base
`d699ac2`.

### Coverage table

| attack point | status | method |
|---|---|---|
| G verdict: oracle reads true occupancy, plan arm runs no correction | done | code read; in-process check: the k fed to the oracle's rule equals a brute-force count read vehicle by vehicle from SUMO at 6 ticks (max difference 0.0, 735 to 1,023 egos per tick); plan arm: 0 stepped-down writes, shown = plan in every trace sample |
| G verdict: shared plan, gate, lag D = 99.49 s, seeds | done | both arms built by one `run_jam_episode` path, gate on in both; per-seed replay start index identical across arms; every version switches at ceil(T_k + D) in all 20 episodes |
| G verdict: step-down reaches SUMO, never above plan | done | in-process check on B9 seed 1: 596,763 writes, 0 above the plan recomputed from the stored snapshot, 0 edge mismatches, 0 maxSpeed read-back mismatches against the lane speed limit; 97.7% of stepped-down vehicles at or below their cap 3 s after the write |
| G verdict: jam time and 120 s spell at 19.0 km/h on [900, 3600] s | done | independent recomputation from the 20 traces reproduces every jam time and onset; v_jam 19.0 read from `drain_thresholds.json`, sha256 in the header |
| G verdict: incident vehicle excluded equally in both arms | done | `exit_road` is in no segment, so I_exit segment counts and speeds are untouched in both arms; arrivals exclude incident ids in both; see R1-2 for the arm-dependent block length |
| G verdict: pass rule as section 8.1 | done | recomputed from traces: B9 -26.07 +/- 3.89, I_exit -10.57 +/- 3.38, prevention 4 vs 0; the rule's tests are weak (R1-4) |
| G verdict: re-run episodes on training seeds | done | seed 3, both arms, both scenarios, through the runner in the mirror: flow, arrivals, writes, active share, held, return and every jam metric equal the record; seed 1 oracle B9 also reproduced (3,704.0 veh/h, 3,036 arrivals) |
| Plan attack point 1 (never above the plan) | done | as above; `min` to `max` killed by a unit test and by the engine counter |
| Plan attack point 2 (step-down reaches SUMO) | done | as above; trace shown mean below plan mean on about 2,100 segment-samples per B9 oracle episode |
| Plan attack point 4 (only the oracle reads true density) | done | code read of `_decide`: oracle on `k_true`, rule on `est.k_hat`, take-the-slower on `est.mean_speed`; learned arm's `engine_features` not in this commit (S11, round 2); `sensor` in every record |
| Plan attack point 5 (the lag is injected) | done | switch instants; see R1-5 (the lag acts only through D) |
| Plan attack point 6 (zero collisions, teleports) | done | 0 in all 20 G episodes and in every validator run; `--time-to-teleport -1`, so every counted teleport would be a collision |
| Plan attack point 10 (jam threshold is the speed threshold) | done | code read, test, header sha256; 30 km/h jam time differs on all 20 G episodes; runner wiring untested (R1-4, M46) |
| Plan attack point 11 (equivalence with the record) | done | the test passes (1,936 arrivals, 3,635.56 veh/h; run once as part of its file, seed 16, chooses nothing). It exercises `_env_for`, `run_jam_episode`'s loop and `JamEngine`'s switch-write path; not the replayed process, the version-keyed rule or refresh, which the switch-instant and in-process write checks cover |
| Plan attack point 12 (incidents) | done | SUMO measurement on edge 14 (R1-1); insertion times (R1-2); visibility (R1-3); arrivals and counts by unit tests and mutants M43, M44 (killed) |
| Plan attack point 15 (pairing and clocks) | done | identical seed lists and start indices; start 900 s and cap one number per scenario; the block's own start and length are not (R1-2) |
| S1 demand files | done | departures 7,144, 7,468, 7,594; [0, 900) s at 3,568 veh/h; burst windows 7,488 and 8,976 veh/h; route-file ids equal schedule ids; `mainz.net.xml`, `paper/`, the poller and the analyser untouched; no `yolov8n.pt` tracked |
| Mutation testing of number-producing code | done | 60 mutants, 37 killed, 23 survived; 11 survivors change block G-path numbers or its decision, 2 are equivalent (R1-4) |

### Findings

#### R1-1. SHOULD FIX. The incident correction removes the stopped vehicle from edge 14's speed a second time, so segment 6's speed on I_mid is overstated while the block stands

- Location: `src/sumo/mainz.py:718-734` (`_edge_reading`); test `tests/test_jam_incidents.py:155-168`.
- What the code assumes: SUMO's edge mean speed includes the stopped incident vehicle's zero, so
  it computes `(mean x count - 0) / (count - incidents)`. The unit test builds a fake SUMO whose
  edge mean includes the zero ("SUMO's mean over all eleven, the incident's zero included").
- What SUMO 1.27.1 does: on a multi-lane edge, the lane mean speed already leaves out stopped
  vehicles. Measured on I_mid, seed 1, plan arm, edge 14 (3 lanes, two incident vehicles stopped
  on lanes 0 and 1), at 29 ticks every 10 s on [910, 1190] s
  (`scratchpad/dsrc_jam/val_scripts/check_incident_speed.py`, log `val_logs/check_incident_14.log`):
  on lanes 0 and 1, `lane.getLastStepMeanSpeed` equals the plain mean speed of the lane's other
  vehicles at every tick. SUMO's raw edge mean exceeds the plain mean speed of the non-incident
  vehicles by a mean of 0.009 m/s. The env's corrected reading exceeds that plain mean by a mean of
  0.14 m/s and at most 0.48 m/s (14.50 against 14.03 m/s at 910 s, 3.4%), because it divides by
  `count - 2` a sum that never contained the two zeros.
- What it changes: segment 6's speed in the jam trace (jam time and onset on I_mid) and in the
  feed snapshot the plan acts on, in every arm, on [insertion, 1,200) s of every I_mid episode.
  The bias is upward, so near 19 km/h it can turn a jammed sample into an unjammed one. Block G
  does not use I_mid (its I_exit block stands on `exit_road`, which is in no segment), so the block
  G verdict is unaffected. Blocks T, S and M use I_mid.
- Proposed fix: keep SUMO's own estimator, which every other edge uses: on an edge that holds
  incident vehicles, rebuild SUMO's lane-weighted edge mean from the lanes,
  `sum_l n_l' v_l / sum_l max(n_l', 1)`, with `n_l'` the lane's vehicle count without incidents and
  `v_l` SUMO's lane mean (which already omits stopped vehicles on a multi-lane edge; on a one-lane
  edge, take the incident's zero out of `v_l` as the code does now). The simpler alternative, the
  plain mean of the non-incident vehicles' speeds, differs from SUMO's estimator on edges with
  empty lanes, so the incident edge would be read differently from the others. Replace the fake's
  edge-mean model in the unit test with the measured behaviour, and add to
  `test_a_mid_network_block_is_absent_from_its_segment_counts` an assertion that segment 6's
  speed equals the count-weighted mean of the non-incident vehicles' speeds read from SUMO.

#### R1-2. SHOULD FIX. The incident's duration depends on the arm, so paired arms on I_exit and I_mid face blocks of different length

- Location: `src/sumo/mainz.py:434-451` (`_place_incidents`: removal at `start_s + duration_s`
  whatever the insertion time).
- Evidence: SUMO inserts the stopped vehicle only once its spot is clear, and the spot's traffic
  at 900 s differs between arms on the same seed. In block G (`results/jam/G.jsonl`, I_exit,
  `incidents[0].inserted_s`), the oracle's and the plan's blocks were inserted at 911.5 and 918.0 s
  (seed 1), 906.0 and 909.5 (seed 2), 906.0 and 901.5 (seed 3), 912.5 and 900.5 (seed 4), 911.5 and
  900.0 (seed 5). Every block is removed at 1,200 s, so the oracle's block stood 6.5 and 3.5 s
  longer than the plan's on seeds 1 and 2 and 4.5, 12.0 and 11.5 s shorter on seeds 3 to 5 (mean
  3.6 s shorter). The paired jam-time difference (oracle minus plan) tracks this: across the five
  seeds its correlation with the block-length difference is 0.89, slope 0.40 jammed
  segment-minutes per second of block; at equal block length the fitted difference is -9.1
  segment-minutes against the recorded mean of -10.6. Five points give weak evidence of the
  slope, but the confound is structural: an arm whose traffic leaves the spot occupied at 900 s
  gets a shorter block.
- What it changes: every paired I_exit and I_mid comparison in blocks S, M, L, P and K, and the
  I_exit row of block G. It does not change the block G verdict: the fitted difference at equal
  block length, -9.1, is still more than two standard errors (3.4) below zero, and B9 passes on
  its own.
- Proposed fix: remove each incident vehicle at its own `inserted_s + duration_s` rather than at
  `start_s + duration_s`, so the block stands 300 s in every arm (its start still varies by the 0
  to 18 s SUMO needs); record `removed_s - inserted_s` in each episode record and assert it equals
  the duration in the S2 sanity test. Re-run block G's I_exit pairs after the fix, as the Resumed
  section requires for any fix to code block G uses.

#### R1-3. DOCUMENTED LIMIT. On I_exit no 80 m sensor ever sees the blocking vehicle; the camera arms and the oracle see only its queue

- Location: scenario `I_exit` in `src/sumo/jam_episode.py:120-121`; plan section 4.5 and attack
  point 12 ("present as camera candidates").
- Evidence: the stopped vehicle stands at 100 m on `exit_road`, which follows edge 218
  (219.67 m), so its back is at least 95.5 m from any ego on a segment edge. `incident_candidate_rows`
  is 0 in all 10 block G I_exit episodes (`results/jam/G.jsonl`). On I_mid (edge 14, a segment
  edge) the blockers are candidates: 2,378 candidate rows in a 1,260 s oracle episode on training
  seed 2 (validator run). The implementer's own test records the I_exit geometry
  (`tests/test_jam_episode.py:385-387`).
- What it means: on I_exit, every 80 m arm (oracle, rule, learned, take-the-slower) can react only
  once the queue behind the block reaches within 80 m of a vehicle still on edge 218 or upstream.
  Section 4.5's sentence "They remain visible to the camera and to the oracle as stopped leaders"
  holds for I_mid and for `oracle_wide`, not for I_exit at 80 m.
- Proposed fix: none to the code. State it in the plan's section 4.5 and in the paper's scenario
  description, so the I_exit result is not read as the camera seeing the incident itself.

#### R1-4. SHOULD FIX. 23 of 60 mutants of number-producing code survive the in-scope tests; 11 of them change an episode's numbers or the block G decision

- Method: `scratchpad/dsrc_jam/val_scripts/mutate.py` applied each mutant alone to an archive
  mirror of 706f5b5, ran the listed test files one process each (no `-n`; the two seed-16
  equivalence tests deselected), recorded killed or survived and restored the file. Results:
  `scratchpad/dsrc_jam/val_logs/mutants_r1.jsonl`. 34 killed, 16 survived. For four survivors the
  effect on numbers was measured on one 1,500 s training-seed episode
  (`val_scripts/survivor_effect.py`, B9 seed 1, oracle at `m_on` 0.1; I_exit seed 1 for M42).
- Survivors that change a reported number (the fix is a test that kills each):

| id | mutant | measured or stated effect | test that kills it |
|---|---|---|---|
| M18 | oracle's trend `s_true` multiplied by 0 (`jam_episode.py:289`) | active share 0.259 to 0.199, served flow 3,036 to 3,120 veh/h | engine test on a synthetic snapshot whose density rises 0.01 per tick below the trigger: the oracle must fire only through `tau_l [s]_+` |
| M19 | oracle fed `0.5 k_true` (`jam_episode.py:323`) | active share 0.259 to 0.034, flow 3,036 to 3,402 veh/h | engine test asserting the margin the oracle acts on equals `rule_margin(k_true, s_true)` with `k_true` from a brute-force count (the validator's in-process check found the committed code correct at six ticks: maximum difference 0.0 over 735 to 1,023 egos per tick) |
| M40 | sensing and correction every 10 s instead of 5 s (`jam_episode.py:262`) | active share 0.2587 to 0.2612, flow 3,036 to 3,072 veh/h | assert `sense_passes` and the correction's update instants are every 5 s from 300 s |
| M42 | `oracle_wide` decides on the 80 m `k_true` (`jam_episode.py:288`) | I_exit at R = 200 m: jam time 12.17 to 9.17 segment-min, flow 1,980 to 2,154 veh/h | assert the oracle at R = 200 m is fed `OracleSensor(200).k_true(table)` |
| M31 | pass rule "below by 2 SE" becomes "below zero" (`run_jam_experiments.py:688-689`) | turns a within-noise reduction into "go" | a `g_pass` case with mean difference -1 and 2 SE 3 that must not pass on jam time |
| M34 | 2 SE becomes 1 SE (`jam_metrics.py:297`) | same | a `Paired.of` hand value |
| M32 | prevention margin 2 becomes 1 seed (`run_jam_experiments.py:694`) | "go" on one extra prevented seed | a B9 case with exactly one more prevented seed that must not pass |
| M33 | prevention rule on any scenario (`run_jam_experiments.py:694`) | "go" from I_exit prevention | an I_exit-only prevention case that must not pass |
| M30 | `jam_time_min` counted from 0 s (`jam_metrics.py:274`) | no effect on block G (no jammed sample before 900 s in any of its 20 traces); changes S45, which is jammed before 900 s | the existing record test with an assertion on `jam_time_min` when a spell lies before 900 s |
| M39 | the runner's metrics clock at 300 s instead of the scenario's 900 s (`run_jam_experiments.py:592`) | as M30, and onset can fall before the perturbation | in the `--smoke` test, assert `e["jam"]["start_s"] == SCENARIOS[e["scenario"]].start_s` |
| M46 | the runner computes metrics at 30 km/h (`run_jam_experiments.py:561`) | every jam number at the wrong threshold | in the `--smoke` test, assert `e["jam"]["v_jam"] == 19.0` |

- Survivors that change numbers only once later steps exist: M21 (`k_side` divided by 80 m instead
  of `R_eff`; differs only if the fitted `R_max` is below 80 m) and M25 (field-of-view bound
  0.45 raised to 0.60; the test vehicle at 6 m is outside both bounds). Kill them with a camera
  model whose `R_max` is 60 m, and an adjacent vehicle at 7 m (|y|/d = 0.54).
- Survivors with a small or no effect on reported numbers: M37 (the lag multiplier capped at 1
  inside `FeedSetting.process`) is equivalent, see R1-5. M16 (trend window 60 s) changes a slope
  only after a gap in a vehicle's sensing (a junction crossing), because the 7-sample ring already
  limits the window to 30 s; a test with one skipped tick kills it. M10 (`RuleCorrection` ignores
  its own `tau_l`) changes only the `tau_l` = 30 s cells of block G's fallback grid, which did not
  run; a `tau_l = 30` test with a positive slope kills it.
- Totals over both passes (M01 to M60): 60 mutants, 37 killed, 23 survived. Of the survivors, 11
  (the table) change block G-path numbers or the block G decision, 8 change numbers only in later
  blocks or rarely (M16, M21, M25, M51, M52, M53, M56, M60), 2 change no survival value or median
  (M10 in the current runs, M57) and 2 are equivalent (M37, M58).
- Also in `g_pass`: a decision is computed on any seed subset (`--seeds 3` printed "base setting
  fails: run the grid" on the validator's re-run). Proposed: `summarize_g` gives a decision only
  when every scenario has the block's five seeds, otherwise "incomplete".

R1-4, addendum (ten more mutants, M51 to M60, on `jam_metrics.py`'s lead, ramp, break, KM and
criterion C functions; three killed). Survivors that change block M's `leads.json` and the
Proposition 2 numbers, not block G: M51 (speed precursor at 90% of baseline instead of 80%), M52
(density precursor at 2 SD instead of 3), M53 (precursor hold 20 s instead of 30 s), M56 (`k_b` at
90% of free flow instead of 95%) and M60 (the precursor search not cut at onset, which turns a
missing precursor into a negative lead). The lead and break tests use step changes, so any
threshold between the two levels passes. Proposed: ramp traces in `TestLeads` and `TestTheory`
on which the crossing time, or the bin, moves with the threshold, plus a trace whose only
precursor comes after onset (expected `W_feed` None). M57 (censoring time `cap` instead of
`cap - t_start`) leaves every KM survival value and median unchanged and moves only the censored
point's time; M58 (held minutes at a fixed 10 s) is equivalent at the 10 s trace spacing. Both
NOT WORTH FIXING.

#### R1-5. DOCUMENTED LIMIT. Under the version-keyed rule the lag multiplier acts only through D; the replayed per-version lags never change a switch instant

- Location: `src/sumo/jam_episode.py:155-164` (`FeedSetting.process`), `src/sumo/feed.py:303-321`.
- Evidence: a vehicle fetches version k at its first 10 s query after `T_k + m L_k` and applies it
  at `max(T_k + D, fetch)`. With `D = m L_bar + P_q` and `L_bar` (89.49 s, the session's largest
  lag) at least every replayed lag (the replayed windows' largest is 84.49 s), the fetch always
  precedes `T_k + D`, so every vehicle switches at `T_k + D`. In all 20 block G episodes, every
  version's first and last switch tick equal `ceil(T_k + D)` (55 versions per episode, 0
  exceptions), and the mutant M37, which stops multiplying the per-version lags for m > 1, survives
  every test because it is equivalent.
- What it means: block L's lag levels are levels of one delay shared by the whole fleet: 0 s for
  m = 0 (the synchronous rule switches at `T_k`), and D = 99.5, 189.0 and 368.0 s for m = 1, 2
  and 4. The fleet never
  disagrees on its version. "m = 2 doubles each version's lag" is true of the replayed process
  and has no effect on the vehicles beyond D.
- Proposed fix: none to the code. Describe the lag sweep in the paper and in Figure 3's caption as
  a sweep of D.

#### R1-6. NOT WORTH FIXING. A block header records the feed parameters at m = 1 whatever lags its episodes run

- Location: `scripts/run_jam_experiments.py:968-969` (`context.feed` is built with the default
  m = 1.0) and `header()`.
- Effect: a block L header will carry `m: 1.0, D: 99.49` while its episodes run m = 0, 2 and 4.
  Each episode record carries its own `feed.m` and `feed.D`, which are correct, so no reported
  number depends on the header's values.
- Proposed fix: write the header's feed entry without `m`, `D` and `rule`, or list the block's
  lag levels there.

#### R1-7. DOCUMENTED LIMIT. "Active share" counts vehicle-ticks on which the step-down changes nothing, because the plan is already at 0.5

- Location: `src/sumo/jam_episode.py:337-340` (`active_ticks` counts every active vehicle) and
  `step_down(0.5) = 0.5` (plan J5).
- Evidence: in block G's oracle episodes the active share is 0.358 to 0.380 on B9 and 0.199 to
  0.205 on I_exit, while the share of writes that are below the plan (`writes_stepped_down /
  writes`) is 0.269 to 0.293 on B9 and 0.170 to 0.179 on I_exit. The two are not the same
  denominator (writes include the switch writes between 5 s ticks), but the gap of about 0.09 on B9
  is mostly ticks on segments the plan already holds at 0.5 (segments 5, 7 and 8 sit at 0.5 on 55 to
  67% of decisions, plan appendix A1).
- What it means: the false-alarm measure of section 4.4 and Table 1 ("share of controlled
  vehicle-ticks with the correction active") overstates how often the correction changes a
  vehicle's command, most where the plan is slow.
- Proposed fix: add a counter of controlled vehicle-ticks at which the correction is active and
  the plan's fraction is above 0.5, and report that share beside the active share.

### Round 1 conclusion

- Block G's "go" holds. The 20 recorded episodes reproduce from their traces (jam time, onset,
  held minutes and the pass rule recomputed independently), seed 3 of both arms and both scenarios
  and seed 1 of the oracle on B9 re-run to identical numbers, the oracle's input equals a
  brute-force count from SUMO, and every write is at or below the recomputed plan. B9 passes on
  jam time (-26.07 against a two-standard-error bar of 3.89) and on prevention (4 seeds against 0)
  without I_exit. On I_exit the oracle's advantage is partly an arm-dependent block length
  (R1-2), but the fitted difference at equal block length, -9.1 segment-minutes, still clears
  its bar. The fixes for R1-1 and R1-2 touch code block G runs (`mainz.py`), so the Resumed
  section's rule applies: re-run block G on the fixed commit before block F.
- Counts: BLOCKS 0, SHOULD FIX 3 (R1-1, R1-2, R1-4), DOCUMENTED LIMIT 3 (R1-3, R1-5, R1-7), NOT
  WORTH FIXING 1 (R1-6).
- Not checked this round: the camera error model against the re-processed frames (S7, S8; no
  `camera_error_model.json` at 706f5b5), the learned arm (`src/rl/onset_predictor.py` is not in the
  commit), the camera arms on a fitted camera model, and the runner's blocks T to K beyond the parts
  block G uses. Seeds 16-30: the plan's two seed-16 equivalence tests ran once, as part of running
  `tests/test_jam_episode.py` whole; they reproduce a committed row and choose nothing. Every other
  validator run used training seeds 1 to 3, and the mutation pass deselected those two tests. The
  HERE key: no key-shaped string in the commit's in-scope files (searched by pattern, printing only
  file names and token prefixes).
