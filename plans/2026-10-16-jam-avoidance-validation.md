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

## Round 2 (2026-10-16): commit 5c175ed; S7 to S13 newly in scope, and the round 1 fixes

Audited from an archive mirror of `5c175ed` (`scratchpad/dsrc_jam/mirrors/round2`). Real data:
block F's rebuilt detections at `scratchpad/dsrc_jam/worktrees/F2_91e6a2b/outputs/jam/frames/`
(read only), the camera model of record `results/jam/camera_error_model.json` and the block G
re-run at `0b3c186` (`results/jam/G.jsonl`). Every Python run this round uses
`PYTHONDONTWRITEBYTECODE=1` and `python -B`, and the mutation harness clears `__pycache__` before
each mutant.

### Coverage table (round 2)

| attack point | status | method |
|---|---|---|
| Block S selection: seeds, criterion C, ties, exclusions, the S_summary read path | done | code read; stub records through `summarize_s` and `read_selection` (R2-1, R2-2, R2-8); criterion C, the tie rule and the exclusion behave as section 5.7 on complete data |
| Camera model: tables recomputed from the rebuilt detections | done | validator's own code on `detections_640.jsonl` of the four 0.20 s drives: detection rate, opportunities, EMA and raw sigma equal the artefact in every bin |
| Camera model: the width-prior choice | done | per-track ground-plane / width-prior ratios, by lateral position; deployment config; fit residuals (R2-4, R2-5) |
| Camera model: detection rate against an independent reference | done | 640 px detection of vehicles the 1,280 px tracker holds; first-detection distances (R2-3) |
| Camera model in SUMO: table, R_max, p, sigma, occlusion, b; k_hat against k_true | done | code read; plan-alone B9 seed 2, 1,800 s, passive sensing with the artefact: mean k_hat / k_true 0.861; occlusion off 0.936, detection loss off 1.117, noise off 0.864, b 1.25 0.720, b 0.8 0.972; same trajectory in every variant |
| Predictor: features, labels, seeds, DAgger, theta | done | code read; tests; one block T episode each on B9 seed 3 and I_exit seed 4 with the artefact (feature ranges, labels, censoring); the real smoke test run with the artefact |
| Round 1 fixes' new code (R1-1, R1-2, R1-4, R1-7) | done | I_mid edge 14 re-measured: corrected speed equals the other vehicles' mean at all 29 ticks (difference 0.0 m/s); block G I_exit blocks stand 300.0 s in all 10 episodes, both arms; R1-4 tests kill 19 of 22 round 1 survivors; R1-7 counter by mutant N63 (killed) |
| Mutation pass on newly in-scope computations, with the bytecode fix | done | 44 mutants, 30 killed, 14 survived (R2-8) |
| Round 1 mutant statuses under the bytecode fix | done | 60 re-run on `706f5b5`: one change, M53 now killed (R2-7) |
| S7 frames, S9 drive cases, S13 figures | done | frame counts (18,818 = 18,910 - 92 without a tick), duplicates (87 on `155910`), tick-time gaps (0.20 s and 0.667 s medians), one rotated frame viewed upright with its boxes; drive-case and figure code read and tests run |

### Findings (round 2)

#### R2-1. SHOULD FIX. Block S selects on incomplete data: a seed subset, a missing scenario or a candidate missing from one scenario all yield a selection that blocks M to K accept

- Location: `scripts/run_jam_experiments.py` `summarize_s` (scenarios are those present;
  candidates missing from a scenario are skipped) and `read_selection` (accepts any non-empty
  subset of seeds 11-15).
- Evidence (stub records through the real functions; `scratchpad/dsrc_jam/val_scripts/stub_select.py`,
  log `val_logs/stub_select.log`): `block_specs("S", seeds=[11, 12])` is accepted (156 episodes); `summarize_s` on those
  records selects a best model (`rule/m0.2`) and thresholds, and `read_selection` accepts the
  resulting `S_summary.json` (seeds [11, 12]). With every I_mid record removed, criterion C is the
  mean over four scenarios and a selection is still made. With the rule arm's B5 records
  removed, the three rule candidates silently leave the pool and the best model changes from
  `rule/m0.2` to `learned/th0.6`.
- What it changes: section 5.7 defines C over the five perturbation scenarios and seeds 11-15;
  any partial block S (a `--seeds` subset, a crash followed by `--summarize-only`) would select
  the paper's best model and every threshold from fewer seeds or scenarios without saying so.
  This is the defect R1-4 fixed for block G, left in block S.
- Proposed fix: `summarize_s` selects nothing ("incomplete", with the missing cells listed)
  unless no control, the plan and every candidate have all of seeds 11-15 on B5, B7.5, B9,
  I_exit, I_mid and BG; `read_selection` requires `seeds == [11, 12, 13, 14, 15]` and a
  non-null `best`. Tests: the three stub cases above must give no selection.

#### R2-2. SHOULD FIX. `S_summary.json` records the sha256 of the camera model and predictor files present when it is summarised, not of the ones block S ran on

- Location: `scripts/run_jam_experiments.py:850-852` (`summarize_s`), `:902-907` (`summarize`
  checks only that the block's headers agree with each other).
- Evidence: code read. `camera_model_sha256` and `predictor_sha256` are `_sha256` of
  `context.camera_model` and `context.predictor` at summary time; the identity block S's
  episodes ran under is in the S.jsonl headers (`header_identity`) and is never compared with
  those files. `read_selection` in blocks M to K then compares its own files with the summary's
  hashes, so the check passes whenever the files are unchanged between the summary and block M,
  even if they changed between block S's episodes and the summary (a re-run of
  `fit_camera_model.py` writes a new `generated_wall`, so a new sha256, with the same tables).
- What it changes: blocks M to K could run, and the paper report, a best model and thresholds
  selected under a camera model or predictor other than the one recorded, with every provenance
  check passing.
- Proposed fix: take the three hashes for `S_summary.json` from the S.jsonl header identity, and
  have `summarize` refuse when that identity differs from the files it is given.

#### R2-3. BLOCKS (blocks T to K). The camera model's detection rate at 40 to 70 m is 1.2 to 3 times the rate at which the 640 px detector reports vehicles that the 1,280 px pass tracks

- Location: `src/analysis/camera_error.py:379-414` (`detection_rate_table`), used by
  `CameraModel.p_at`; plan section 2.3 ("`p(d)` is the detection rate of a vehicle that the
  tracker holds").
- What the code does: it implements the plan. A frame counts as an opportunity only between a
  640 px track's first and last detection, so a vehicle the 640 px tracker has not yet
  acquired, or never acquires, is outside the denominator. The validator's own recomputation
  from `detections_640.jsonl` of the four 0.20 s drives reproduces the artefact's table exactly
  (p 0.896, 0.784, 0.748, 0.748, 0.710, 0.688, 0.604 by 10 m bin from 0 to 70 m; opportunities
  5,060 to 159; `val_scripts/recompute_camera.py`, log `val_logs/recompute_camera_640.log`).
- Evidence that this overstates detection: block F ran the same frames at 1,280 px. Taking every
  1,280 px tracked detection as a vehicle known to be there, the share that has a 640 px raw box
  in the same frame (IoU at least 0.3) is, by the 1,280 px width-prior EMA distance: 0.96 (0-10 m),
  0.96 (10-20), 0.85 (20-30), 0.74 (30-40), 0.59 (40-50), 0.38 (50-60), 0.20 (60-70), 0.10
  (70-80) (`val_scripts/cross_resolution_detection.py`, log `val_logs/cross_resolution.log`;
  with IoU at least 0.1 the shares are 0.62, 0.41 and 0.24 at 40-70 m). The 640 px boxes of the
  same vehicles are 6 to 14% wider at 50-70 m, so the 640 px distance of such a vehicle is 6 to
  14% shorter; shifting the comparison by that much leaves the model's rate at 50-70 m about
  twice the cross-resolution rate. The share is an upper bound on the 640 px detector's
  unconditional rate, because vehicles the 1,280 px pass misses are not in its denominator. The
  first-detection distances agree: of 1,034 closing 640 px tracks, 17 (1.6%) were first detected
  beyond 60 m and the median first detection is 32.4 m, while the model detects a visible vehicle
  at 60-70 m with probability 0.60 on every 5 s tick.
- What it changes: in SUMO the camera sees far more vehicles at 40 to 70 m than the drive
  detector does. That raises the adjacent-lane counts `n_hat` within `R_eff` = 70 m, so `k_hat`,
  `W_cam` and every camera arm's response at range. It also changes the learned predictor's
  training features (block T) and every camera arm in blocks S, M, L, P and K. Block G (oracle
  only) is unaffected.
- Proposed fix (an amendment to section 2.3, the orchestrator's decision): build the SUMO
  `p(d)` from the cross-resolution rate (640 px detection of vehicles that the 1,280 px tracker
  holds, binned on the 640 px-equivalent distance), keep the tracker-conditional table as the
  recorded alternative, and recompute `R_max` on the new table. Do this before block T.

#### R2-4. SHOULD FIX. The SUMO camera has no persistent per-vehicle distance error; on the drives such errors are an order of magnitude larger than the frame-to-frame noise the model draws

- Location: `src/sumo/sensing.py` `CameraModel.observe` (independent `N(0, sigma(d)^2)` per tick
  plus one global bias `b`); plan J8.
- Evidence: per track (at least 10 detections with a valid ground-plane distance, box centre
  within 15% of the frame width of the centre column), the median ratio of ground-plane to
  width-prior distance varies between tracks with a relative 1.4826 MAD of 0.29 on `190548`
  (128 tracks, 10th-90th percentile 0.69-1.36) and 0.25 on `173809` (34 tracks). For cars only, binned
  by ground-plane distance, the median ratio rises from 0.93 (0-10 m) to 1.26 (50-60 m) for
  boxes within 10% of the frame width of the centre column, and reaches 1.58 to 1.92 at 30-60 m for boxes at least a quarter of the frame width
  off centre, where the side of the vehicle widens the box and the width prior shortens the
  distance (`val_logs/recompute_camera_640.log`; `val_scripts/lateral_ratio.py`, log
  `val_logs/lateral_ratio.log`). At least one of the two estimators therefore carries persistent per-vehicle and
  distance-dependent scale errors of order 25%. The model's frame-to-frame EMA noise is 2 to 3%
  of distance (0.35 m at 20-30 m, 1.13 m at 50-60 m), and its only scale error is the global
  `b`.
- What it changes: the SUMO camera measures each leader's distance more consistently than
  either real estimator. `k_own = L/(g_hat + L)` and the adjacent counts within `R_eff` carry no
  per-vehicle bias, so camera-arm triggers are cleaner than a device's.
- Proposed fix: draw one multiplicative factor per (ego, leader) pair when the pair first
  appears, log-normal with the spread measured above (or a bracket of 0.1 and 0.25), and apply it
  with `b`. As a lower-cost alternative, run block K at that spread and state the limit. Also
  minor: `sigma_ema` from residuals about a 5-detection line understates the EMA's own error.
  For white raw noise the EMA at alpha 0.45 has 0.54 of the raw sigma; the table's sigma_ema is
  0.44 to 0.47 of `sigma_raw` at 20-60 m.

#### R2-5. DOCUMENTED LIMIT. The width-prior choice: defensible for the scale of the SUMO model, but it models an estimator the deployment is not configured to run, and it shortens the camera window to 70 m while the oracle keeps 80 m

- The case for it holds: the width-prior distance uses only `fx` and the class widths, so its
  scale is the same on every drive, and the `b` sweep covers `fx`.
- Against it, and what to state:
  1. `deployment/jetson/config.yaml` sets `distance.method: ground_plane`, `camera_height_m:
     1.25`, `horizon_y_px: 717`. The modelled device is a width-prior device; the paper, or the
     deployment config, must say so.
  2. The per-drive camera heights that motivated the change are the Theil-Sen slope of the box
     bottom on `w_px / W_class`, i.e. the width prior's own scale written as a height. Their
     spread (0.40 to 1.54 m) shows that the two estimators disagree by drive, not which one is
     wrong. The fit quality differs by drive: residual MAD 32.6 px on `161422` and 14.5 px on
     `155910`, against 7.0 to 8.4 px on `170849`, `173809`, `183538` and `190548`. A phone
     remounted between drives would move its pitch (horizon) more than its height.
  3. Width-prior errors are largest where the SUMO candidates are adjacent-lane vehicles near the
     ego (R2-4).
  4. `R_max` is 70 m on width-prior bins (bin 70-80 m has 19 opportunities, below the 50
     required) and 100 m on ground-plane bins, so `R_eff` = 70 m. The camera arms then work in
     a 70 m window and the oracle in an 80 m one. Open item 4's "isolates the camera's error at
     the same window" no longer holds: the window difference is part of the measured camera
     error.
- Proposed: state items 1 and 4 in the plan and the paper. The orchestrator's choice is not
  reversed by this audit. It stands if R2-3 and R2-4 are addressed, because those two errors
  dominate the choice of estimator.

#### R2-6. BLOCKS (block M's leads and Fig. 1b). Block M's no-control episodes run without the camera, so `W_cam` is never measured

- Location: `scripts/run_jam_experiments.py` `episode_runner` (the camera is loaded only for a
  camera arm or a sampling spec, and passed to later episodes only once loaded) and
  `needs_camera`; plan sections 1.5, 4.3 and J30 (no control senses passively on every vehicle;
  `W_cam` from block M's no-control episodes).
- Evidence: `episode_runner` driven over block M's specs in their run order, with
  `run_jam_episode` stubbed to record whether it received a camera
  (`val_scripts/camera_wiring.py`, log `val_logs/camera_wiring.log`): 0 of the 7 no-control
  scenario groups receive a camera; the first episode given one is `take_the_slower` on B5. Block M
  is arm-major and no control comes first, so in a full run every no-control trace has
  `k_hat_mean` None. `leads()` then returns `W_cam` None for every episode, and Fig. 1(b) shows no
  camera warning. On a resumed run the outcome depends on which arm the process reaches first.
- Proposed fix: load the camera for no-control episodes, so they sense passively as section 1.5
  states (and any arm that should trace `k_hat`), independent of episode order; add no control
  to `needs_camera` so the header records the camera sha256. Test: in the `--smoke` run of
  block M, assert that every no-control trace carries a non-null `k_hat_mean`.

#### R2-7. Round 1 mutation harness: one status was wrong. With caches cleared, M53 is killed, not a survivor

- The implementer's diagnosis is right. A `.pyc` whose recorded source mtime (whole seconds) and
  size match is reused, so a same-size edit written in the same second as the restore can run the
  previous mutant's code. The harness now deletes every `__pycache__` outside `.venv` before
  applying a mutant, before each test file and after restoring. It runs pytest as
  `python -B` with `PYTHONDONTWRITEBYTECODE=1`.
- Re-run of all 60 round 1 mutants on a fresh archive of `706f5b5`
  (`val_logs/mutants_r1_rerun.jsonl`): 38 killed, 22 survived. Against round 1, exactly one
  status changes. M53 (`PRECURSOR_HOLD_S` 30 to 20 s) is killed by
  `TestTheory::test_max_ramp_is_the_largest_30_s_rise_before_onset`, because `max_ramp` takes
  that constant as its default window. R1-4's addendum listed M53 as a survivor; that entry is
  withdrawn. No other round 1 status moves, so R1-4's corrected totals are 38 killed and 22
  survived. 11 survivors change block G-path numbers or the decision (unchanged), and 7 change
  numbers only in later blocks or rarely (M16, M21, M25, M51, M52, M56, M60).

#### R2-8. SHOULD FIX. 14 of 44 round 2 mutants survive. The block S selection, block M's thresholds, the predictor's training and features, and the camera table group have no test that pins them

- Method: `val_scripts/mutants_r2.py` with the fixed harness (R2-7) on an archive of `5c175ed`
  (`val_logs/mutants_r2.jsonl`). 44 mutants, 30 killed, 14 survived. Every survivor changes a
  reported number by construction:

| id | mutant | what it changes | test that kills it |
|---|---|---|---|
| N01 | tie margin 0.02 to 0 (`run_jam_experiments.py:103`) | the best model when two candidates are within 0.02 | stub block S with candidates 0.005 apart, higher BG flow on the one with higher C: that one is best |
| N02 | tie broken by lowest BG flow (`:823`) | the best model in a tie | same stub |
| N03 | plan alone removed from the best-model pool (`:817`) | the best model when every correction is worse than the plan | stub where every candidate has C > 0: best is the plan |
| N04 | a candidate with a collision is not excluded (`:803`) | the best model and thresholds | stub with one collision in the best candidate: it is excluded |
| N05 | writes above the plan no longer exclude (`:787`) | same | same stub with `writes_above_plan` 1 |
| N06 | each grid picks its highest C (`:811`) | every threshold | stub with three rule settings of distinct C: the lowest is selected |
| N09 | block M's oracle runs `m_on` 0.1 whatever the selection (`:202`) | block M's oracle rows | `block_specs("M")` with a selection `m_on` of 0.2: the oracle's params carry 0.2 |
| N10 | block M's learned arm runs theta 0.3 whatever the selection (`:201`) | block M's learned rows | the same with theta 0.6 |
| N30 | `CameraModel.from_artefact` reads the 0.67 s table (`sensing.py:466`) | every camera arm (p, sigma, `R_max` 60 m) | `from_artefact` on an artefact holding both groups, asserting the 0.20 s values |
| N45 | no class weighting (`onset_predictor.py:181`) | the predictor's probabilities, so what each theta means | assert `pos_weight` equals negatives over positives on a 1-in-10 set |
| N46 | no standardisation (`:215`) | the fitted predictor | assert the card's scaler equals the training mean and SD |
| N47 | `plan_fraction_next` reads the own segment (`:94`) | a predictor input | `engine_features` on a fake engine with distinct own and next plan values |
| N48 | the next segment's feed speed and jam factor read the own segment (`:95`) | two predictor inputs | the same fake engine |
| N72 | Table 1's BG "flow minus plan" has its sign reversed (`jam_figures.py:259`) | Table 1's false-alarm cost | assert a BG row with flow 3,000 against the plan's 3,100 gives -100 |

- Block S and the arms of blocks M to K are the path every paper number is chosen on, and only
  their seed and provenance refusals are tested (N07, N08 killed). The validator's stub runs
  (`val_scripts/stub_ties.py`, log `val_logs/stub_ties.log`) show the committed code does what
  section 5.7 says on complete data: C = -0.125 for a rule 5 segment-minutes below a 20-minute
  plan with no control at 40; a learned arm 0.005 above it in C but with BG flow 3,100 against
  3,000 is chosen; with equal BG flow the rule is; a rule setting with one collision is
  excluded and the next rule setting is selected. Those runs are the tests to add.
- Also on 5c175ed: the round 1 survivors are now killed by the R1-4 tests, except M10, M57 and
  M58 (`val_logs/mutants_r1survivors_on_5c175ed.jsonl`: 19 of 22 killed). Those three change no
  reported number in the current runs, or are equivalent (R1-4).

#### R2-9. NOT WORTH FIXING. Smaller items

- Labels near the cap: a sample at `t` > cap - 230 s can see a spell that starts within its
  120 s horizon but does not reach 120 s by the cap. On two training episodes (B9 seed 3, I_exit
  seed 4; `val_scripts/block_t_episode.py`), 0 of 47,519 kept samples were affected.
- `train_onset_predictor.py --smoke` reads `results/jam/T.jsonl` and writes
  `results/jam/onset_predictor*.json` unless `--results-dir` and `--out` are given. A smoke card
  could then replace the predictor of record. The runner's smoke guard is the pattern to copy.
- R1-2's withdrawn request (`never_inserted`) is recorded per episode, but no summary counts it,
  so an I_exit or I_mid episode without its block would enter the means. None occurred in block
  G (every block stood 300.0 s, in both arms of every seed).
- The drive case table (S9) computes `k_hat` on an 80 m window over three lanes, while SUMO
  uses `R_eff` = 70 m and the visible lanes. The table is descriptive (plan section 3).
- `own_leader_reported_speed_mps` is not clipped at zero; its minimum was -2.3 m/s in the B9
  block T episode.

### Round 2 conclusion

- Block G's verdict does not change. B9 is untouched by every round 2 finding, the R1-1 and
  R1-2 fixes are confirmed in SUMO and in the re-run's records, and I_exit passes at -10.47
  against a two-standard-error bar of 1.43.
- Counts: BLOCKS 2 (R2-3, R2-6), SHOULD FIX 4 (R2-1, R2-2, R2-4, R2-8), DOCUMENTED LIMIT 1
  (R2-5), NOT WORTH FIXING 1 (R2-9, five items). R2-7 corrects one round 1 mutant status.
- Before block T: R2-3, because the predictor's training features are camera features, and the
  decision on R2-4. Before block S: R2-1, R2-2 and the R2-8 selection tests. Before block M:
  R2-6.
- Not checked: block T, S, M, L, P and K results (none exist yet); the 0.67 s camera table
  beyond its use as the non-SUMO group; the S7 and S8 smoke tests that need
  `outputs/models/yolov8n.pt` and the S7 smoke frames (skipped in the mirror; the validator's
  own real-data checks replace them). Seeds 16-30: no episode run. A feed replay was drawn for
  seed 16 by `test_a_7200_s_episode_wraps_the_measured_session_and_counts_it` (no simulation),
  and block M's specs were enumerated with a stubbed episode function to trace the camera
  wiring.

## Round 3 (2026-10-16): commit b285f5e; the round 2 fixes and the re-fitted camera model

Audited from an archive mirror of `b285f5e` (`scratchpad/dsrc_jam/mirrors/round3`); every
Python run with `PYTHONDONTWRITEBYTECODE=1` and `python -B`, mutations with `__pycache__`
cleared. For each finding the check that would confirm its fix is stated.

### Coverage table (round 3)

| attack point | status | method and result |
|---|---|---|
| Each round 2 fix's new code | done | diffs read. 17 new mutants (P01-P17) of the pair factor, its spread, the per-spec camera, the oracle window, block K's half spread, the complete-data rule, the header-identity hashes, `read_selection`, the cross-resolution table and the ratio spread: all killed (`val_logs/mutants_r3.jsonl`). All test files pass in the mirror; the block M smoke test also passes when given the camera of record and a stand-in predictor card |
| Camera model of record recomputed from the frames | done | cross-resolution p(d) 0.951, 0.938, 0.788, 0.681, 0.531, 0.318, 0.152 (0-70 m) equals the validator's round 2 tracked-box share in every bin; `R_max` 60 m follows section 1.2's rule (60-70 m has p 0.152 < 0.2); stored pair spread 0.308 reproduced from 201 tracks (`val_scripts/pair_spread.py`) |
| The pooled pair spread, attacked as asked | done, no finding | 0.308 as stored; 0.297 with each drive centred on its own median ratio (log units); 0.329 uncentred in log units; 0.269 to 0.282 without `155910`, whose ground-plane fit is implausible (height 0.70 m, residual MAD 14.5 px). Every alternative lies within 0.04 of 0.308, and block K's half spread (0.154) brackets them all. The stored value is defensible as an upper bound |
| Camera arms, block M oracle and no-control sensing read the model of record | done | block M, L, P and K specs through the real `episode_runner` with `run_jam_episode` stubbed (`val_scripts/wiring_r3.py`). Camera arms and no control get camera sha `fdf527cd408e`, `R_max` and `R_eff` 60 m, pair sigma 0.3083, bias 1.0 (K: 0.8, 1.25, and 1.0 at sigma 0.1542). The oracle has window 60 m (`tau10_m{m_on}_R60`) and `oracle_wide` 200 m. Plan and lookahead get no camera. Thresholds are the selection's |
| Leads non-null on a no-control B9 episode that jams | done | block M's episode function on training seeds 2 and 3, 3,600 s (`val_scripts/leads_r3.py`). Seed 2: onset 1,840 s on segment 9, `W_feed` 270 s, `W_cam` 460 s, `W_true` 460 s. Seed 3: onset 1,750 s, `W_feed` 220 s, `W_cam` 470 s. `k_hat_available` True, camera recorded in the episode, wall 12.9 s per 3,600 s (budget 20 s per 7,200 s) |
| Block S end to end on stub records; M to K read only S_summary.json | done | stub episodes through `run_block`, `header`, `summarize`, `read_selection` and `block_specs("M")` (`val_scripts/block_s_e2e.py`). Complete data: rule/m0.1 has the lowest C (-0.1246); learned/th0.45 (C -0.1171, BG flow 3,100) is within 0.02 and is best; J_th 4, m_on 0.1 and theta 0.45 reach block M, oracle `R60`. One missing cell gives "incomplete", best None, and `read_selection` refuses it. Summarising with another camera file is refused |
| Block T: seeds, features, card camera hash | done | seeds 1-10 enforced by `block_specs` and `split_seeds` (unchanged since round 2); features unchanged; the card's camera hash is recorded but never checked against the camera in use (R3-1) |
| Camera model in SUMO, with the pair factor | done | plan-alone B9 seed 2, 1,800 s, passive sensing (`val_logs/camera_in_sumo_r3.log`). Mean k_hat / k_true (80 m) 0.897; occlusion off 0.926, detection loss off 1.086, b 1.25 0.769, b 0.8 1.026. With no pair factor the share of samples at or above 0.9 k_c falls from 0.0237 to 0.0215 |
| Round 2 survivors | done | N01-N06, N09 (re-targeted at the new oracle line), N10, N30, N45-N48 and N72 all killed. Round 1's M10, M57 and M58 still survive: none changes a reported number in the planned runs (M10 only the unrun `tau_l` 30 s grid; M57 only the censored point's time; M58 equivalent) |

### Findings (round 3)

#### R3-1. SHOULD FIX. Nothing checks that the predictor was trained on the camera model the learned arm runs with

- Location: `scripts/train_onset_predictor.py:124` writes `camera_model_sha256` into the card.
  Nothing reads it: `src/rl/onset_predictor.py` `load`, `scripts/run_jam_experiments.py`
  `episode_runner` and `read_selection` do not, nor does any other code (`grep -rn
  camera_model_sha256 scripts src`).
- Evidence: block S's header identity carries the camera sha256 and the predictor sha256
  separately, and `read_selection` compares each with the files block M is given. The
  predictor's own training camera is never compared with either. Within block T it is held:
  `train_onset_predictor.py --round 2` refuses T and T_dagger headers with two cameras. But a
  camera model re-fitted after block T (one was re-fitted today, between rounds 2 and 3) would
  run the learned arm in blocks S to K on features from a camera it was not trained on, and
  every provenance check would pass.
- Proposed fix: in `episode_runner`, when a spec uses both the camera and the predictor, refuse
  unless `predictor.card["camera_model_sha256"] == camera.source["sha256"]`; the same check in
  block T's DAgger round for the round-1 card.
- Check that confirms it: a unit test that builds a card with `camera_model_sha256` "other",
  runs `episode_runner` on a learned-arm spec with the camera of record and a stubbed
  `run_jam_episode`, and expects `SystemExit`; and a run of block S's smoke with the card of
  record that passes.

### Round 3 conclusion

- Counts: BLOCKS 0, SHOULD FIX 1 (R3-1), DOCUMENTED LIMIT 0, NOT WORTH FIXING 0. Every round 2
  fix does what its finding asked. In particular the camera arms, no control and the block M
  oracle read the camera model of record at 60 m with the 0.308 pair factor, and a no-control
  B9 episode that jams yields non-null `W_feed` and `W_cam`. Block S selects as section 5.7
  says on complete data and refuses incomplete data.
- Nothing found this round changes block G's verdict or any number blocks T to K will
  produce, provided the camera model is not re-fitted between block T and block K (R3-1 guards
  that).
- Still alive, by decision: R1-6 and R2-9 (not fixed), M10, M57, M58 (no reported number).
- Not checked: the experiments themselves (none has run). The validator ran no episode on
  seeds 16-30; block M's specs on those seeds were enumerated only with a stubbed episode
  function.
