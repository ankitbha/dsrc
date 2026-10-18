# Approach roads: independent validation

Validator for plan `plans/2026-10-17-approach-roads-plan.md`. Never staged or committed. Every
mutation and SUMO run happened in a mirror built from `git archive <sha>` in the session
scratchpad (`scratchpad/dsrc_jam/approach_mirrors/round<N>`), with `__pycache__` cleared,
`PYTHONDONTWRITEBYTECODE=1` and `python -B`.

## Round 1: commit 5cea9d3 (diff against 4f4720a)

### Coverage

| # | attack point (plan section 9, brief priorities first) | status | method |
|---|---|---|---|
| B1 / 9.1 | Gate gone on the approach boundary; gated path reproduces block M records | done | 60 Q records and traces; base-vs-new gated runs of 11 settings on training seed 3 (seed 16 forbidden by the brief); mutants JE5, JE8 |
| B2 / 9.2 | Arrivals at entry links reproduce the schedules on real SUMO runs | done | own libsumo script (no repo code) on BG and B9, seeds 1 and 2, plus gated B9 no control for entry 295; far-end departures recomputed for all 5 files; T_A recomputed from raw crossings |
| B3 / 9.3 | Approach super-segments in every metric; network part equals old metric | done | base commit's metric code on 60 Q traces cut to 36/12/24 columns; independent storage and flag recomputation |
| B4 / 9.4 | Q input 12 x 5; approach vehicles shown entry fraction; lookahead, camera, sensing, oracle | done | probe of 2,400 s B9 seed 3 episodes (rule, lookahead, oracle, take-the-slower, learned): model input shapes, write log against plan_row/downstream derived from the route files, snapshot timing, queue_m and on_approach recomputed from raw SUMO positions |
| B5 | Mutation testing of number-producing code | done | 48 mutants in a separate mirror; 34 killed, 14 survive (R1-1 to R1-7) |
| B6 / 9.6 / 9.7 | Collisions, teleports, seeds, HERE key | done | 60 Q records, 8 extra p = 0.75 B9 episodes on seeds 3-6, all validator SUMO runs: 0 collisions, 0 teleports; seeds; grep of the diff and of Q.jsonl |
| 9.5 | pending recorded and zero; queue_m recompute | done | pending 0 at every sample of 60 Q traces; queue_m and on_approach recomputed from raw SUMO positions at 1,320 samples of 5 probe episodes |
| 9.8 | Clock shifted by W only | done | probe, Q records, FeedProcess on seeds 1-10 |
| 9.9 | Original network and models unchanged | done | sha256 and independent XML comparison |
| 9.10 | Regression on the gated boundary | done (on seed 3, not 16) | see B1 |
| 9.11 | Where results go | done | diff touches nothing under results/; Q wrote outputs/jam_approach/run; runner refuses results/jam |
| 9.12 | Provenance headers | done | Q header: commit 5cea9d3, dirty false, W, T_A, every approach and _A sha256; footer tree_unchanged_during_run true |
| 9.13 | Storage measures recompute | done | 60 Q records, independent code |
| 9.14 | One-lane follower test | done | exists, runs SUMO, has a negative control that passes |
| 9.15 | Timing | done | Q per-cell mean walls 1.05 to 1.59 times the gated block M means (21 cells), within a factor of 2; inflated by validator processes running concurrently |
| D | Implementer's reported departures | done | table below |

### Evidence for checks that passed (no finding)

- **B2, arrivals.** An independent libsumo script (`scratchpad/dsrc_jam/approach_val_r1/arrivals_indep.py`,
  same SUMO options as `MainzEnv.reset`) on `mainz_bg_A`: per-entry mean of (first 0.5 s step on the
  entry link minus `entry_time`) was -0.84 to +1.35 s on seed 1 and -1.46 to +1.16 s on seed 2; the
  largest 300 s window difference was 4 and 5 vehicles; 0 vehicles due by 7,140 s missing; insertion
  backlog 0; 0 collisions. These equal the implementer's record. All 7,144 / 7,468 / 7,594 / 12,144 /
  3,124 far-end departures equal `depart + W - T_A(entry)` to within 0.0031 s (rounding to 0.01 s); ids,
  types, routes and order equal the gated schedules, so the controlled sets at every penetration are
  the gated ones; each `_A.rou.xml` holds the same vehicles as its schedule, sorted by departure, with
  `departSpeed="max" departLane="best"`. `T_A` equals the per-entry mean of the stored raw crossings.
- **B2, burst (entry 295 exemption).** B9 no control: entry 295 received 61 of 83 (seed 1) and 65 of 83
  (seed 2) vehicles in [1,140, 1,440) s with an approach queue of 200 m and 183 m. On the gated boundary
  under no control the same entry received 59 of 83 and 56 of 83 vehicles in [900, 1,200) s, with SUMO's
  insertion backlog at 21 and 24 vehicles in that window, and the entry link's mean speed fell to 4.3
  and 3.6 km/h on both boundaries. The shortfall at 295 is therefore present on the gated boundary too;
  the exemption does not hide a boundary artefact. Entry 255's burst lag: 40.7 s (seed 1), 2.9 s (seed 2).
- **B4.** In 2,400 s (gated clock) B9 episodes on training seed 3 for the rule, lookahead, oracle,
  take-the-slower and learned arms (`probe.py`): the Q-network received only (12, 5) inputs; every
  `observe_approaches()` call happened at the SUMO time of an `observe()` call; 127,325 to 133,094
  approach writes per arm, each with `plan` equal to `a_k[plan_row]` where `plan_row` and `downstream`
  were derived independently from the `_A` route files; every lookahead write (489,743, of which 133,094
  on approaches with 66,755 at 0.5 and 3,410 at 0.75) recomputed from the version's 12 + 24 snapshot rows
  and the route-derived chain; approach vehicles stepped down 5,819 (rule), 3,055 (oracle), 3,741
  (take-the-slower) times; 5,375 to 10,257 candidate rows per episode had an ego on `a<e>_2` and a
  candidate on the entry link; `queue_m` and `on_approach` equalled a recomputation from raw SUMO
  positions at all 264 samples; 0 writes above the plan, 0 read-back mismatches, 0 collisions.

- **B1, gated path.** The brief forbids seeds 16-30, so the committed seed-16 regression was not
  re-run. Instead, 11 gated settings (block M no control, lookahead, plan, take-the-slower, rule,
  learned, oracle 60 m on B9; rule on I_exit; no control on I_mid; block P rule p0.5 on B9; block L
  lookahead lag 0 on B9) ran through the runner's episode function on training seed 3 from a mirror of
  the base commit 4f4720a and from a mirror of 5cea9d3 (`gated_equiv.py`, `compare_equiv.py`). Every
  field of every result, including all 720 trace samples, the switch table, incidents and both
  jam-metric dictionaries, was identical; the gated records gained no field.
- **B1, gate gone.** In all 60 block Q records read from the orchestrator's run (read only):
  `gate_entries` False, `admitted` -1, `held_at_entry` 0, `held` 0 at every trace sample,
  `pending` 0 at every sample. Every `_A` route starts with `a<entry>_0`, `_admit` returns before the
  schedule loop when the schedule is empty, so only incidents are added by `_admit`.
- **B3.** For all 60 Q records, every field of `jam`, `jam_network`, `jam_approach` and the three
  `jam_30kmh*` dictionaries (onset, onset segment, prevented, jam time, jam time in [1,140, 3,840],
  30 km/h jam time, held and pending minutes, drain, open spell, start, end, cap) equals the BASE
  commit's `episode_metrics` applied to the trace cut to 36, 12 and 24 columns with every clock
  moved by 240 s (`metrics_check.py`). `jam = jam_network + jam_approach` holds on every record.
  The approach and system excess vehicle-minutes equal an independent recomputation from the trace
  and the `_A` schedule; the flags equal a recomputation from the trace.
- **9.8 clock.** Control start 540 s, first decisions 540, 600, 660 s, flow window 1,140 s, cap
  7,440 s, incident requested at 1,140 s (I_exit Q records); the feed replay at base 300 s and end
  `cap + 420 s` equals the gated replay plus 240 s for training seeds 1-10 at caps 7,200, 3,600 and
  2,500 s (same start index, lags and wraps).
- **9.9.** `mainz.net.xml` 037de791..., `mainz_segments.json` a6f82397..., selection 5bce02db...,
  predictor 3b2256cd..., camera fdf527cd..., checkpoint 941ac210..., feed artefact 8f5430bf...;
  an independent XML comparison found 0 of 414 original edges and 0 of 767 connections changed and
  only the eight `<entry>-begin` junctions changed. `git diff 4f4720a 5cea9d3` touches nothing under
  `results/`, `paper/`, the two excluded scripts or the committed demand files.
- **Existing tests.** In the mirror: test_jam_metrics (28), test_jam_correction (20),
  test_jam_scenarios (18), test_jam_incidents (16), test_jam_predictor (15, 1 skipped),
  test_jam_camera (32), test_jam_runner (53, 1 skipped), test_jam_episode (35; the two seed-16
  episodes deselected) pass; all eight new approach test files pass (the seed-16 regression
  deselected).
- **compare_boundaries real-data path.** Run on a copy of the Q records relabelled as block M seeds
  16 and 17 against the committed `results/jam/M.jsonl` and the gated traces: the gated held plus
  pending mean for rule B9 is 6,836.16 vehicle-minutes (the plan's 6,836) and the gated system
  excess mean, 109,855.28, equals an independent recomputation from the 15 gated traces.

### The implementer's reported departures, judged

| departure | judgement | evidence |
|---|---|---|
| Gated headers keep a 4-element identity; approach headers add the approach net's sha256 as a fifth | Accepted. Equivalent to the plan's "fifth element None on gated" for every comparison the runner makes (identity sets are compared only with identities from the same function), and it keeps each new gated record's `identity` list equal to the committed 4-element lists | `run_block`, `summarize`; 11 gated settings gained no field |
| The approach smoke reads the committed selection | Accepted. Read-only, and the plan's thresholds are the committed ones; sha256 checked | `main` |
| Block Q's no control senses passively | Accepted. Block Q runs "every block M arm", and block M's no control senses passively; it costs wall time only | `PASSIVE_SENSING_BLOCKS` |
| Provisional W of 180 s before the measured 240 s | Accepted. It is rule A9 applied to the provisional T_A of 180 s; the calibration measures time from each vehicle's own far-end departure, which does not depend on W; an independent seed 1 run on the calibrated demand gives per-entry mean travel times within 1.35 s of T_A | `arrivals_indep.py` |
| Approach-only fields only on approach records | Accepted. The plan's "every record carries W" is satisfied on the approach boundary; adding W = 0 to gated records would only break field-for-field equality with committed records | `compare_equiv.py` |
| MainzEnv refuses `gate_entries=True` with an approach file | Accepted; it is a guard the plan's attack point 1 asks a mutant to trip | test_the_gate_cannot_be_on_with_approach_roads |
| `on_engine` hook in `run_jam_episode` | Accepted. Called after `env.close()`; it reads only | code |
| mean speed, space-mean speed and mean vehicles network-only; running and system excess include approach vehicles | Accepted. On the gated boundary these three were already computed over the 12 super-segments' edges only, so they stay comparable across boundaries; system excess is defined in section 5.3 with `running`, which must include approach vehicles for N_ff to cancel | `_accumulate`, `_per_edge` |
| approach-short threshold = length minus one edge (2,000 m) | Accepted. Equals the plan's 2,000 m at 3 km and keeps "reaches the insertion super-segment" at 5 km (4,000 m) | `approach_metrics` |
| Approach edges use netconvert's default lane spread; entry links use center spread | Accepted, no effect on any number. The connectors at the entry nodes are 1.75 to 5.63 m long with speed 16.67 m/s (no curvature limit), so they add a few metres absorbed in T_A; the camera reads lane widths, not coordinates (`EdgeGeometry`) | net XML; `sensing.py` |
| Entry 295 receives 22-27% fewer burst vehicles and is exempted | Accepted. The same shortfall exists on the gated boundary under no control (59 and 56 of 83 vehicles, with an insertion backlog of 21 and 24 in the window), so the exemption does not hide a boundary artefact | `analyze_arrivals.py` |
| Entry 255's burst is 40.7 s late on average | Reproduced (40.7 s on seed 1; 2.9 s on seed 2). Already open item 3 of the plan | `analyze_arrivals.py` |

### Findings

No finding is a defect in the code that produced the block Q numbers: every surviving mutant below
was checked against the real Q records, and the committed code computes each quantity correctly.
The finished block Q (60 episodes, seeds 1 and 2) passed its rule: no episode flagged, the
longest approach queue 832 m, pending 0, 0 collisions, 0 writes above the plan, 0 read-back
mismatches; the header carries commit 5cea9d3 with `dirty: false` and the footer
`tree_unchanged_during_run: true`. Eight extra approach episodes at penetration 0.75 on B9
(plan and rule, training seeds 3-6, `mixed_fleet.py`) had 0 collisions, 0 teleports, pending 0,
longest queue 488 m.

Each finding is a test that would pass against a broken version of a computation that produces a
reported number. Mutation method: `scratchpad/dsrc_jam/approach_val_r1/mutate.py` applies one
textual mutant at a time to a separate mirror of 5cea9d3 (`approach_mirrors/round1_mut`), clears
`__pycache__`, and runs the eight approach test files (`-x`, one process per file, the seed-16
regression deselected) until one fails; 48 mutants (44 in `mutants.py`, 4 in `mutants2.json`),
results in `mutation_results.json` and `mutation_results2.json`: 34 killed, 14 survived. The effect sizes come from `survivor_effects.py` on the 48 Q records
available at the time.

**R1-1. SHOULD FIX. The W shift of the flow window and of the incident request is untested.**
- Location: `src/sumo/jam_episode.py:707` (`env.mark_window()` when `env.time_s() >= window_s`) and
  `src/sumo/jam_episode.py:696` (`schedule_incident(..., start + W, ...)`).
- Evidence: mutant JE1 (`>= window_s` to `>= window_start_s`) and mutant JE2 (`start + W` to
  `start`) survive all eight approach test files. The committed code is right: in 48 of 48 Q
  records `flow` equals the arrivals between the 1,140 s trace sample and the last sample, per hour,
  and the I_exit records show the incident requested at 1,140 s. Under JE1 the reported flow of the
  40 B9 and I_exit Q records would be 28.5 to 62.9 veh/h lower (window from 900 s, inside the
  extra 240 s of fill). Under JE2 the I_exit and I_mid incidents would stand from 900 s, 240 s
  before the scenario clock starts.
- Fix: in `tests/test_approach_engines.py` add to the 840 s smoke an assertion that `out["flow"]`
  equals `(arrived(840) - arrived(540)) * 3600 / 300` from the trace (the smoke's window is
  300 + 240 s), and add one short approach I_exit smoke (cap 1,000 s on the gated clock, no
  sensing) asserting `out["incidents"][0]["requested_s"] == 900 + 240`.
- Check: re-run mutants JE1 and JE2; both must be killed.

**R1-2. SHOULD FIX. The W shift of the selection window and of the approach-excess window is untested.**
- Location: `src/analysis/jam_metrics.py:303` (`window = (SELECTION_WINDOW_S[0] + offset_s,
  min(SELECTION_WINDOW_S[1] + offset_s, cap))`) and `src/analysis/jam_metrics.py:364`
  (`_window_integral(trace, excess, start_s + offset_s, cap + offset_s)`).
- Evidence: mutants JM2 (window start not shifted), JM3 (window end not shifted), JM6 (excess window
  start not shifted) and JM7 (excess window end not shifted) survive. On the Q records JM3 changes
  32 of 96 `jam_time_min_900_3600` values (all, network and approach parts) by up to
  10.3 segment-minutes; JM6 and JM7 change all 32 B9 and I_exit approach-excess values, by up to
  108.7 and 841.8 vehicle-minutes. JM2 changes none of the Q values, because no Q episode jams
  before 1,140 s. The tests place the only jam inside [1,140, 3,600) and the only excess inside
  [1,140, 1,740], so a window boundary that moves by W changes nothing they check.
- Fix: in `tests/test_approach_metrics.py`, extend `test_the_clock_moves_by_the_offset` with spells in
  [900, 1,140) and [3,600, 3,840) and assert `jam_time_min_900_3600` counts only the second; extend
  `test_approach_excess_is_on_approach_minus_free_flow_occupancy` with excess in [900, 1,140) and in
  (7,200, 7,440] and assert only the second is counted.
- Check: re-run JM2, JM3, JM6, JM7; all must be killed.

**R1-3. SHOULD FIX. The runner's storage arguments and the leads baseline shift are untested.**
- Location: `scripts/run_jam_experiments.py:785-790` (`approach_excess_minutes(..., offset_s=boundary.W)`
  and `system_excess_minutes(..., rows=rows, travel_s=meta["T_A"])`) and
  `scripts/run_jam_experiments.py:1178` (`base = (base[0] + offset_s, base[1] + offset_s)`).
- Evidence: mutants RJ3 (approach excess with `offset_s=0.0`), RJ4 (system excess with `rows=None`,
  so N_ff is not subtracted) and RJ5 (leads baseline not shifted) survive.
  `test_the_approach_record_carries_three_parts_storage_and_flags` checks only the storage keys,
  and no test calls `write_leads` with an offset except the smoke, which checks that
  `leads.json` exists. RJ4 would add the integral of the free-flow approach occupancy N_ff to every
  system excess: 20,501 vehicle-minutes on B9, 19,500 on I_exit and 5,976 on S45 (mean N_ff of
  195, 185 and 223 vehicles over the window), computed from the `_A` schedules. RJ5 changes real leads: on the Q no-control traces, B9 seed 1's `W_cam` would read
  1,060 s instead of 710 s and `W_true` 1,060 s instead of 740 s.
- Fix: in the runner test, give the fake trace a non-zero `on_approach` and assert the two storage
  values equal `jm.approach_excess_minutes(trace, rows, T_A, start_s=900, cap=7200, offset_s=240)`
  and `jm.system_excess_minutes(..., rows=rows, travel_s=T_A)` with the `_A` schedule; add a
  `write_leads` test with `offset_s=240` on a trace whose baseline differs between [600, 900) and
  [840, 1,140).
- Check: re-run RJ3, RJ4, RJ5; all must be killed.

**R1-4. SHOULD FIX. The comparison's network difference is not distinguished from the all-segment difference.**
- Location: `scripts/compare_boundaries.py:135` (`"network_jam_time_min": pair_rows(g_jam, a_net)`).
- Evidence: mutant CB1 (`a_net` to `a_all`) survives `tests/test_compare_boundaries.py`; the stub
  approach records give the same value to `jam` and `jam_network`, so the comparison's main
  number, the network jam time against the gated jam time, is not tested.
- Fix: give the stub approach records a non-zero `jam_approach` time so that `jam` differs from
  `jam_network`, and assert `approach_minus_gated.network_jam_time_min.differences`.
- Check: re-run CB1; it must be killed.

**R1-5. SHOULD FIX. The queue's speed threshold and the 2,000 m flag limit used by the runner are untested.**
- Location: `src/sumo/mainz.py:977` (`>= v_queue_kmh`) and `scripts/run_jam_experiments.py:792`
  (`limit_m = length_m - segment_m`).
- Evidence: mutant MZ3 (19 km/h to 30 km/h) survives: no fake vehicle in `TestApproachReport` drives
  between 19 and 30 km/h, and controlled approach vehicles are capped at 30 km/h, so the mutant
  would count part of the slowed platoons as a queue in `queue_m`, `queue_max_m` and the
  approach-short flag. Mutant RJ1 (flag limit 3,000 m) survives: the runner test's queue is 150 m.
  The block Q pass rule, which decides whether block M starts, depends on this limit. The Q
  records' largest queue is 816 m, so neither mutant changes a Q flag today.
- Fix: add to `test_queue_from_the_entry_with_a_100_m_gap` a vehicle at 25 km/h 50 m beyond the
  queue's end and assert it is not counted; add a runner test whose fake trace has `queue_m` 2,500 m
  and assert `approach_short` is True.
- Check: re-run MZ3 and RJ1; both must be killed.

**R1-6. NOT WORTH FIXING. The `assert_no_gate` wiring in `run_jam_episode` is untested end to end.**
- Location: `src/sumo/jam_episode.py:690`.
- Evidence: mutant JE6 (`assert_no_gate=False`) survives. It changes no number: on the approach
  boundary `_load_schedule` leaves the queues empty, so `held_at_entry()` is 0 whether or not the
  assertion runs, and the engine-level test `test_a_held_vehicle_on_the_approach_boundary_stops_the_episode`
  covers the assertion itself.
- Fix (optional): assert `engine.assert_no_gate is True` in the smoke fixture.

**R1-7. SHOULD FIX. Take-the-slower's re-run row for an approach vehicle is untested.**
- Location: `src/sumo/jam_episode.py:426` (`rows = np.array([self._row(int(segments[k])) for k in index], ...)`).
- Evidence: mutant JE9 (`self._row(...)` to `min(segment, 11)`, so an approach vehicle's re-run
  replaces and reads row 11 instead of its entry segment's row) survives. The smoke test checks only
  `shown <= plan` for this arm, which the mutant also satisfies. The committed code passes the
  `plan_row` row, as decision A15 specifies. The arm is not one of the paper's three (open item 5),
  but its block M and Q numbers are reported.
- Fix: a unit test that drives `JamEngine._decide` for the take-the-slower arm with a stub
  correction recording its `segments` argument and one ego on `a140_1` (index 25), asserting the
  row passed is 5.
- Check: re-run JE9; it must be killed.

The four mutants of the second batch: JE8 (gate on in `_env_for` on the approach boundary) killed by
`test_every_arm_runs_without_the_gate_on_the_full_route_file`, which satisfies attack point 1's
mutant requirement; OP2 (learned arm's own row clipped to 11) killed; JE10 (approach report not
recorded in the trace) killed; JE9 survived (R1-7).


### Round 1 counts

BLOCKS 0, SHOULD FIX 6 (R1-1 to R1-5, R1-7), DOCUMENTED LIMIT 0, NOT WORTH FIXING 1 (R1-6).

### Not checked in round 1

- The committed seed-16 regression records (`M/rule/m0/B9/16`, `M/no_control/-/B9/16`): the brief
  forbids running seeds 16-30. The base-versus-new comparison on training seed 3 stands in for it.
- The gated block P collision case (plan at penetration 0.75, B9, seed 23) on the approach boundary:
  seed 23 is forbidden; seeds 3-6 at 0.75 had no collision.
- Blocks M, L, P and K have not been run, so their records, flags and comparison file do not exist.
- The 5,000 m fallback build was not exercised beyond its existing unit test.
- Clean wall times: block Q ran while the validator's SUMO and pytest processes were running, so its
  walls (1.05 to 1.59 times the gated block M means) include contention and are not a measurement
  for the plan's 20% overrun rule.
- The HERE key was checked only by searching the diff and the Q outputs, not the whole disk.
