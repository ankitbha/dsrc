# Task 142 — a paired-seed evaluator for the committed Mainz checkpoints

## Short version

Written under `plan_dsrc_rec`: every decision below was taken by recommendation, without
asking. **None carries user sign-off.** The open items are the ones that would have been
questions.

**The defect.** Four documents assert `3,765 / 3,759 / 3,535 veh/h` with paired gains
`+230 +/- 44` and `+224 +/- 61` over seeds 16-30. `scripts/train_mainz_src.py:63` reads
`TEST_SEEDS = tuple(range(16, 21))` — five seeds — and the committed
`results/checkpoints/mainz_{here,src}_result.json` carry that five-seed read and nothing
else: `3,774.7 / 3,756.4 / 3,568.0`. `git log -S "range(16, 31)" --all` returns no commit.
Nothing in the repository produces the fifteen-seed figure.

**What this task builds.** `scripts/evaluate_mainz_checkpoints.py`: loads a committed
checkpoint, runs seeds 16-30 for both trained arms and the no-control baseline, pairs on
seed, and writes a per-seed table with its own two-standard-error bar to
`results/evaluation/`. Then one run, and then either the quoted figures reproduce or every
document carrying them is corrected.

**The configuration is no longer unknown.** It was not recorded as a command line
anywhere, so it was recovered by measurement for this plan: `--step-length 0.5
--duration-s 2500 --window-start-s 900`, warmup 300 s. Under it the committed
`no_control` block reproduces on all seven of its metrics to within 1e-6 (section 1.3).
`train_mainz_src.py`'s own defaults are `1.0` and `0.0` and do **not** reproduce it: they
give 3,809.78 veh/h against the stored 3,568.00.

**One finding changes what "reproduce" can mean.** The two trained arms do **not**
reproduce at HEAD: they read `3,783.11` and `3,764.44` against the stored `3,774.67` and
`3,756.44`. Commit `eb86e72` replaced `SPEED_ACTIONS_KMH = (30.0, 45.0, 60.0)` with
`SPEED_ACTION_FRACTIONS = (0.5, 0.75, 1.0)` of each edge's own limit, and every controlled
edge on Mainz is limited to **60.012** km/h, not 60.000. Restoring the old mapping
reproduces all seven stored metrics exactly (section 1.4), so a 0.02% change in commanded
speed moved the five-seed mean flow by 8.4 veh/h. `plans/mainz_src_port.md:1054` says
"Mainz is unchanged" by that generalisation; that sentence is wrong.

**One stated fact in the task brief needs correcting.** The per-seed values behind the
*five*-seed bars do exist, at `plans/mainz_src_port.md:930-934`, and all fifteen reproduce
to the rounding shown. It is the *fifteen*-seed table that exists nowhere. That matters:
the five-seed table settles the pairing formula by measurement (section 1.6).

**Cost.** 45 episodes at a measured 6.3 s each: about 5 minutes of simulation, under 15
minutes for the whole experiment stage including the falsification arm. Retraining, which
this task explicitly avoids, costs 18.7 minutes per arm by the committed logs.

### Scope boundary

| In | Out |
|---|---|
| One evaluator over committed checkpoints | Retraining, or re-selecting a checkpoint |
| Seeds 16-30, three arms, paired on seed | A demand sweep — one demand level only, still |
| The per-seed table and its two-standard-error bar | A second network (`inverted_tree_bottleneck`) |
| A `results/evaluation/` artifact | A penetration sweep |
| Correcting the five documents that carry the figure | Changing `DENSITY_CRITICAL`, the fleet or the exit geometry |
| A falsification run proving the instrument can fail | Rewriting the paper's argument around whatever lands |
| Recording the run configuration in the artifact | Task 143's golden vectors, task 145's device runtime |

### Decisions, all taken by recommendation and none signed off

| # | Question | Taken |
|---|---|---|
| 1 | New script or an extension of `train_mainz_src.py` | **New** `scripts/evaluate_mainz_checkpoints.py`. The driver's docstring commits to reading the test seeds "once at the end"; re-reading them on demand is a different job, and retraining to re-read costs 18.7 min/arm |
| 2 | What it emits, and where | `results/evaluation/mainz_paired_seeds.jsonl` (one line per episode, flushed as it completes) and `results/evaluation/mainz_paired_seeds.json` (config, per-seed table, means, paired gains, bars), written whole-then-rename |
| 3 | How the bar is computed | `2 * stdev(per-seed differences) / sqrt(n)`, sample standard deviation, `n-1`. **Not** `2 * sqrt(se_A^2 + se_B^2)`. Settled by measurement against the published five-seed bars (section 1.6) |
| 4 | Is the baseline recomputed in the same script | **Yes**, per seed, in the same process. The gate is ON for both trained arms and NEVER for no control, and the artifact records `held_at_entry` per episode so the asymmetry is visible rather than asserted |
| 5 | Step length, duration, window start | `0.5 s`, `2500 s`, `900 s`, warmup `300 s`. Evidenced twice: `plans/mainz_src_port.md:918-920`, and an exact reproduction of the stored `no_control` block |
| 6 | Which action mapping the number is quoted under | **HEAD's** `SPEED_ACTION_FRACTIONS`. The stored `test` blocks become superseded artifacts of an action encoding the repository no longer contains |
| 7 | Proving the instrument can fail | A zeroed output head — a constant half-the-limit advisory — measured at 2,844.00 veh/h on seeds 16-20, 939 below the trained arm and 724 below no control (section 1.7) |
| 8 | Checkpoint / feature-set binding | One `--arm {here,src}` flag selects the checkpoint path **and** the feature tuple together, and the script asserts the loaded input width equals `12 * len(features)` |
| 9 | Repeat runs per (arm, seed) | **One.** Evaluation is deterministic: `epsilon=0.0`, `generator=None`, SUMO seeded per episode. Proved by exact cross-process reproduction. The bar is a seed spread and has no run-to-run component |
| 10 | Resume support for the run | **No `--resume`.** The run is 5 minutes. The incremental JSONL is written for the record, not for restart |

### Open items flagged for the user

1. **Decision 6 reverses my first instinct and should be read before the run.** The
   obvious goal was "reproduce the stored numbers". Reading the code showed the stored
   numbers cannot be reproduced by the code that is in the tree, because the action
   mapping changed after the run. Quoting HEAD means the paper's number moves for a second
   reason on top of the seed extension. The alternative — pinning the old mapping in the
   evaluator — would keep the published number but would make the repository carry two
   action sets, one of them only to serve a result. Recommended HEAD; this is the decision
   most likely to be reversed.
2. **`plans/mainz_src_port.md:1054` is a claim, not a rounding.** It states Mainz is
   unchanged by the action-set generalisation. Measured, the five-seed mean flow moves
   from 3,774.67 to 3,783.11. Whether that sentence is corrected or deleted is a wording
   call this plan does not take.
3. **What `results/evaluation/` is for.** This creates a third directory under `results/`.
   The alternative is to put the artifact in `results/checkpoints/` beside the checkpoints
   it reads. Recommended a new directory, because the artifact is a run over both arms and
   is not a checkpoint, but nothing turns on it.
4. **Whether the five-seed result stays in the documents alongside the fifteen-seed one.**
   `plans/mainz_src_port.md:918-935` records the five-seed result correctly, with its
   per-seed table, and that section is the evidence for the "five seeds was not enough"
   argument. Recommended keep it and mark it as the five-seed read; deleting it would
   remove the only per-seed table the repository has.
5. **Whether task 142 closing requires the paper's prose to change.** If the fifteen-seed
   gain lands unresolved, the paper's headline claim changes in kind, not in digits. That
   is named as out of scope here and would be its own task.

---

## 1. Facts, each re-verified for this plan

Verified at `975b7a2`, branch `mainz-src-port`, with `.venv/bin/python` (3.12.14, torch
2.13.0, `libsumo`). Every measurement below was run for this plan.

### 1.1 What the repository asserts against what it stores

`scripts/train_mainz_src.py:63`:

```
TEST_SEEDS = tuple(range(16, 21))
```

The committed result files carry that five-seed read:

| file | `test.flow` | `no_control.flow` | difference |
|---|---|---|---|
| `results/checkpoints/mainz_here_result.json` | 3774.666667 | 3568.0 | +5.8% |
| `results/checkpoints/mainz_src_result.json` | 3756.444444 | 3568.0 | +5.3% |

The fifteen-seed figure is asserted in **five** places, not four:

| location | what it says |
|---|---|
| `plans/paper_deploying_self_regulating_cars.md:312` | "16-30 evaluate" |
| `plans/paper_deploying_self_regulating_cars.md:316-318` | 3,765 / 3,759 / 3,535 with +230 +/- 44 and +224 +/- 61 |
| `plans/paper_deploying_self_regulating_cars.md:598` | "`train_mainz_src.py` — train on 1-10, select on 11-15, read 16-30 once" |
| `plans/mainz_src_port.md:15-17` | the same three-row table |
| `plans/mainz_src_port.md:19` | "Fifteen evaluation seeds, 16 to 30" |
| `results/README.md:10` | "16-30 evaluate" |
| `results/README.md:21-22` | "+/- 44" and "+/- 61" |
| `plans/implementation_records.md:94` | the same gains |
| `plans/implementation_records.md:2331` | "16-30 evaluate" |

`plans/paper_deploying_self_regulating_cars.md:598` is the sharpest of these: it is a
claim about what the script does, and the script does the opposite.

`plans/mainz_src_port.md` heads its result table "Held-out seeds 16-20" at line 10 and
says "Fifteen evaluation seeds, 16 to 30" at line 19. One of the two is wrong whatever the
run produces.

### 1.2 No evaluator over seeds 16-30 ever existed

```
git log -S "range(16, 31)" --all   -> no commits
git log -S "16 to 30"     --all   -> eb86e72 only
```

`eb86e72` ("Train both arms on inverted_tree_bottleneck: no resolved gain, and Mainz
re-checked") is where the fifteen-seed numbers entered the documents. Its diff touches
five files — `.gitignore`, `plans/mainz_src_port.md`, `scripts/build_tree_scenario.py`,
`scripts/train_mainz_src.py`, `src/sumo/mainz.py` — and leaves `TEST_SEEDS` at
`range(16, 21)`. The script that produced the fifteen-seed read was never committed.
`scripts/build_tree_scenario.py` was itself deleted in `6b538f2`.

### 1.3 The run configuration, recovered

No command line is recorded in `results/checkpoints/*_training.log` or anywhere else. The
configuration was recovered two ways, which agree.

**From the document.** `plans/mainz_src_port.md:918-920`: "80 episodes, 2,500 s, 100%
penetration, 0.5 s steps, flow counted from 900 s". Note that the *same document* records
a different configuration at line 643 — "0.5 s steps, throughput counted from 600 s",
with `DENSITY_CRITICAL` at 0.095 and the exit meter at 0.85 green — for a superseded run
whose flows are near 2,150 veh/h. A reader taking the first configuration they find gets
the wrong one.

**From measurement.** `evaluate_no_control((16..20), duration_s=2500, step_length=0.5,
window_start_s=900)` at HEAD, in a fresh process:

| metric | measured | stored in both result files | |
|---|---|---|---|
| `flow` | 3568.00000000 | 3568.00000000 | match |
| `arrived` | 1950.80000000 | 1950.80000000 | match |
| `return` | 55.06611054 | 55.06611054 | match |
| `mean_speed_kmh` | 45.44194183 | 45.44194183 | match |
| `space_mean_speed_kmh` | 42.67574065 | 42.67574065 | match |
| `mean_vehicles` | 858.03703704 | 858.03703704 | match |
| `held_at_entry` | 0.00000000 | 0.00000000 | match |

All seven agree to within 1e-6. For contrast, the script's own defaults
(`--step-length 1.0 --window-start-s 0.0`) give `flow` 3809.78, `arrived` 2058.8,
`mean_vehicles` 817.56; `--step-length 0.1` gives 3228.89, 1697.4, 985.03. Neither is the
committed run.

The window start was independently pinned before those runs, from the stored number
itself: `mean_vehicles = 858.0370370370371` is exactly `115835/135 = 115835/(5 x 27)`, so
each episode contributed 27 `_accumulate` samples after `mark_window`. With warmup 300 s,
a 60 s decision interval and a 2,500 s episode there are 37 decisions, so the window was
opened after the tenth, at t = 900 s.

`scripts/measure_mainz_fundamental_diagram.py:143-145` carries the same three values as
its defaults (`--step 0.5`, `--duration-s 2500.0`, `--window-start 900.0`), and
`results/gates/mainz_eidm_lane_drop.log` heads its table "dt 0.5".

### 1.4 The trained arms do not reproduce at HEAD, and the cause is identified

Same configuration, both committed checkpoints, seeds 16-20, at HEAD:

| arm | metric | HEAD | stored |
|---|---|---|---|
| here | flow | 3783.11111111 | 3774.66666667 |
| here | return | 178.69452515 | 182.47228088 |
| here | held_at_entry | 50.6 | 51.2 |
| src | flow | 3764.44444444 | 3756.44444444 |
| src | return | 210.88234253 | 193.78722534 |
| src | held_at_entry | 54.2 | 49.8 |

The no-control arm matches exactly, which rules out the network, the route file, the
schedule and the seeding. The only code path that differs between the stored run and HEAD
for a trained arm is `MainzEnv._command`. Commit `eb86e72` changed it from

```
target_mps = SPEED_ACTIONS_KMH[int(actions[index])] / 3.6
```

to

```
fraction = SPEED_ACTION_FRACTIONS[int(actions[index])]
target_mps = fraction * self._static[edge]["free_flow_kmh"] / 3.6
```

Measured: all 188 controlled edges across the 12 super-segments have a free-flow speed of
**60.012** km/h. So action index 2 commands 16.6700 m/s at HEAD against 16.6667 m/s
before — a 0.02% change.

Re-running the `here` arm with the pre-`eb86e72` mapping monkeypatched in reproduces all
seven stored metrics exactly (flow 3774.66666667, return 182.47228088, arrived 1996.6,
mean_speed 42.08332214, space-mean 40.81188567, mean_vehicles 892.71111111,
held_at_entry 51.2). The stored numbers are therefore correct for the code that produced
them and unreachable from the code that is in the tree.

Two consequences carry into the plan. First, the reproduction gate in step 5 cannot demand
that the trained arms match the stored JSON; it demands the values in the table above.
Second, a 0.02% perturbation in commanded speed moves a five-seed mean by 0.22%, so any
future edit to `_command` invalidates every stored trained-arm number, and the evaluator
must record the commit it ran at.

### 1.5 The five-seed per-seed table exists, and reproduces

`plans/mainz_src_port.md:930-934` records it. Re-measured for this plan under the
pre-`eb86e72` mapping, which is the mapping it was produced under:

| seed | 16 | 17 | 18 | 19 | 20 | mean |
|---|---|---|---|---|---|---|
| no control, measured | 3722.22 | 3553.33 | 3564.44 | 3555.56 | 3444.44 | 3568.00 |
| no control, document | 3,722 | 3,553 | 3,564 | 3,556 | 3,444 | 3,568 |
| here, measured | 3760.00 | 3755.56 | 3793.33 | 3811.11 | 3753.33 | 3774.67 |
| here, document | 3,760 | 3,756 | 3,793 | 3,811 | 3,753 | 3,775 |
| src, measured | 3848.89 | 3706.67 | 3564.44 | 3826.67 | 3835.56 | 3756.44 |
| src, document | 3,849 | 3,707 | 3,564 | 3,827 | 3,836 | 3,756 |

Every entry agrees to the rounding shown. The five-seed pipeline was real, paired and
correct; what is missing is the fifteen-seed one.

### 1.6 The published bars are the paired formula, measured

From the per-seed table above, the `here` differences are
`[37.8, 202.2, 228.9, 255.6, 308.9]` and the `src` differences are
`[126.7, 153.3, 0.0, 271.1, 391.1]`.

| arm | paired gain | paired 2se | independent 2se | published |
|---|---|---|---|---|
| here | +206.67 | 91.5 | 91.8 | +207 +/- 92 |
| src | +188.44 | 133.0 | 140.4 | +188 +/- 133 |

The `here` arm does not discriminate the two formulas — its baseline variance dominates,
so pairing barely helps. The `src` arm does: 133.0 against 140.4, and the published figure
is 133. The published bars are the paired formula, and the evaluator must keep it.

### 1.7 An obviously wrong policy reads obviously wrong

Copying `mainz_here_best.pt` and zeroing `stack[2].weight` and `stack[2].bias` makes
`greedy_actions` return index 0 for all twelve segments on any input — a standing order to
drive at half the limit. Measured on seeds 16-20 at the run configuration: **2844.00
veh/h**, which is 939 below the trained arm's HEAD read and 724 below no control. The
instrument separates a wrong policy from a right one by an order of magnitude more than
the bar it is being asked to resolve.

### 1.8 Cost per episode, and the cost of the alternative

| thing | measured |
|---|---|
| no control, seeds 16-20 | 31.6 s for 5 episodes (6.3 s/ep) |
| here trained, seeds 16-20 | 31.4 s for 5 episodes (6.3 s/ep) |
| src trained, seeds 16-20 | 30.7 s for 5 episodes (6.1 s/ep) |
| no control, seeds 25 and 30 | 12.8 s for 2 episodes (6.4 s/ep) |
| zeroed-head arm, seeds 16-20 | 39.6 s for 5 episodes (7.9 s/ep) |
| `mainz_here_training.log`, summed | 1120 s (18.7 min) for 80 episodes |
| `mainz_src_training.log`, summed | 1128 s (18.8 min) for 80 episodes |

The training logs' bracketed per-episode times have median 7 s, consistent with the 6.3
s/ep measured here at the same configuration; the extra second is the TD backward pass.

45 episodes at 6.5 s is **293 s**. The zeroed-head arm over 15 seeds adds about 120 s.

### 1.9 Seeds 21-30 run, and point the right way

The scenario carries no per-seed data: `data/mainz/mainz_schedule.json` is one fixed list
of 3,124 departures over 2,498.4 s across 8 entry links, each at its own constant
headway; summing `3600 / headway` over all eight gives exactly 4,500.0 veh/h (R2-6),
and the seed is passed to SUMO only. Seeds 21-30 therefore need no rebuild.

Smoke-measured: seeds 25 and 30 under no control give a mean of **3450.00 veh/h**, below
the 3568.00 of seeds 16-20. The asserted fifteen-seed baseline of 3,535 requires the mean
over seeds 21-30 to be about 3,518.5, so two seeds point in the required direction. Two
seeds are not a test and this is recorded as weak corroboration, not evidence.

---

## 2. Decision 1 — where the evaluator lives

**A new `scripts/evaluate_mainz_checkpoints.py`.**

`scripts/train_mainz_src.py`'s module docstring states that the test seeds "are read on
the final line and nowhere else". That is a property worth keeping: it is what makes the
test read a single read rather than a thing that can be repeated until it agrees. An
evaluator that can be pointed at a committed checkpoint any number of times is the
opposite kind of tool, and merging the two would either destroy the driver's property or
bolt a second mode onto it that the docstring then has to qualify.

The cost argument is separate and also decides it: re-reading the test seeds by re-running
the driver means retraining, at 18.7 minutes per arm by the committed logs, and would
produce a *different* checkpoint, because selection is by validation return over a run
that is not seeded identically end to end. The committed checkpoints are the artifacts the
paper cites, and a separate evaluator reads exactly those.

The repository convention agrees: `scripts/` holds the CLIs, `src/` stays import-only, and
`scripts/evaluate_*.py` already existed as a naming pattern before the cleanup.

The evaluator imports `run_episode`, `evaluate_no_control` and `SCENARIOS` from
`train_mainz_src` rather than copying them. That is deliberate: the two must not drift
apart, and the reproduction gate in step 5 would not catch a divergence that affected both
identically.

## 3. Decision 2 — what it emits, and where

Three files, each with one job.

**`results/evaluation/mainz_paired_seeds.jsonl`** — append-only, one JSON object per
(arm, seed) episode, flushed as the episode completes. Fields: `arm`, `seed`, `flow`,
`arrived`, `return`, `mean_speed_kmh`, `space_mean_speed_kmh`, `mean_vehicles`,
`held_at_entry`. This is the record that is missing today: the per-seed values behind the
bar. Write the whole line and flush; never a partial line.

**`results/evaluation/mainz_paired_seeds.json`** — the summary, written whole-then-rename
at the end. Structure:

```
{
  "commit": "<git rev-parse HEAD>",
  "config": {"seeds": [16..30], "duration_s": 2500.0, "step_length_s": 0.5,
             "window_start_s": 900.0, "warmup_s": 300.0, "demand_veh_per_h": 4500.0,
             "scenario": "mainz", "action_set": "SPEED_ACTION_FRACTIONS",
             "checkpoints": {"here": "...", "src": "..."},
             "gate_entries": {"here": true, "src": true, "no_control": false}},
  "per_seed": {"no_control": {...}, "here": {...}, "src": {...}},
  "arms": {"here": {"flow_mean": ..., "flow_2se": ...}, ...},
  "paired": {"here": {"differences": [...], "gain": ..., "two_se": ...,
                      "resolved": true, "percent": ...}, "src": {...}}
}
```

`commit` and the whole `config` block are mandatory, and are the direct response to what
made this task necessary: nothing recorded the command line, so the configuration had to
be reverse-engineered from a float's denominator.

**`results/evaluation/mainz_paired_seeds.log`** — redirected stdout, matching the
`results/checkpoints/*_training.log` precedent, which are redirected stdout and not
written by the script.

`outputs/` is gitignored and `results/` is not; `results/README.md` states that what is
under `results/` is what the paper cites, kept small. These three files are a few
kilobytes.

## 4. Decision 3 — the pairing formula, stated

For arm `A` against the no-control baseline `B`, over seeds `s in S`, `n = |S| = 15`:

```
d_s  = flow_A(s) - flow_B(s)                      per-seed difference
g    = (1/n) * sum_s d_s                          paired gain
s_d  = sqrt( sum_s (d_s - g)^2 / (n - 1) )        SAMPLE standard deviation
bar  = 2 * s_d / sqrt(n)                          the reported +/- figure
```

The gain is resolved when `|g| > bar`.

**This is not the same quantity as** `2 * sqrt(s_A^2/n + s_B^2/n)`, the two-independent-
means bar. Each seed is one traffic realisation and both arms ran it, so the seed's
contribution to the level of flow is common to both and cancels in `d_s`. The independent
formula does not cancel it and reports a wider interval that is measuring the wrong thing.
Section 1.6 shows the difference is 133.0 against 140.4 on the `src` arm, and that the
published figure is the paired one.

Three consequences the implementer must honour:

- **`statistics.stdev`, not `statistics.pstdev`.** The `n-1` denominator is what the
  published bars used; on a two-element fixture the two differ by 41%.
- **Compute `d_s` first, then the statistic.** Computing `g` as a difference of means and
  `bar` from the arms' separate spreads is the independent formula wearing the paired
  formula's name.
- **The bar is a seed spread, not a noise estimate.** Evaluation is deterministic
  (decision 9), so repeating an episode adds nothing. The interval describes variation
  across traffic realisations and nothing else, and the artifact should say so.

Per-arm `flow_2se` is also emitted (`2 * stdev(flow_A) / sqrt(n)`) because the documents
quote it — `plans/mainz_src_port.md:932-934` gives 23 for DSRC and 109 for SRC — but it is
labelled separately and is never the paired bar.

## 5. Decision 4 — the baseline, recomputed here, with the gate asymmetry preserved

**The no-control baseline is recomputed per seed inside the same script**, for three
reasons. It does not exist for seeds 21-30 anywhere. Pairing is only defined against the
same seed, so the baseline has to be produced by the same loop that produces the arms.
And a baseline read out of a stored file would have been produced under an unrecorded
configuration, which is the failure this whole task is repairing.

**The entry-gate asymmetry is load-bearing and must be preserved exactly.**

- A trained arm runs with `gate_entries=True`. `MainzEnv` then loads
  `data/mainz/mainz_routes.rou.xml` (routes and vehicle types, no vehicles) plus
  `mainz_schedule.json`, and `_admit` releases at most one vehicle per entry per step,
  only while that entry link is under `DENSITY_CRITICAL`.
- No control runs with `gate_entries=False`. `MainzEnv` then loads
  `data/mainz/mainz.rou.xml`, which contains the vehicles, and SUMO inserts them itself.

The reasoning is in two docstrings and is not a detail: `src/sumo/mainz.py:206`
(`_admit` — "THE GATE BELONGS TO THE POLICY, NOT TO THE NETWORK, and callers must not
apply it to an uncontrolled run") and `scripts/train_mainz_src.py:238`
(`evaluate_no_control` — "NO CONTROL RUNS WITHOUT THE ENTRY GATE, and that is deliberate
rather than an omission"). The gate stands in for the policy acting on the link above each
entry link, which the scenario does not simulate. Gating an uncontrolled run would credit
it with a control action it is not taking; not gating a controlled run would remove the
boundary condition the policy was trained under. Either error invalidates the comparison.

The evaluator makes this checkable rather than assumed, three ways:

1. `gate_entries` is derived from the arm name in one place, never passed by the caller.
2. Every episode record carries `held_at_entry`. In the committed five-seed read it is
   51.2 and 49.8 for the trained arms and exactly 0.0 for no control.
3. The script asserts, after the run, that every `no_control` episode has
   `held_at_entry == 0` and that at least one trained episode has `held_at_entry > 0`. An
   inverted gate fails this immediately.

The exit meter is the opposite case and applies to every arm, because it is part of the
network. It is in the built `mainz.net.xml` and needs no handling.

## 6. Decision 5 — step length, episode duration, throughput window

| parameter | value | evidence |
|---|---|---|
| `--step-length` | 0.5 s | `plans/mainz_src_port.md:919`; `measure_mainz_fundamental_diagram.py:143`; `results/gates/mainz_eidm_lane_drop.log` header; exact reproduction (1.3) |
| `--duration-s` | 2500 s | same; `mainz_schedule.json` last departure 2498.4 s |
| `--window-start-s` | 900 s | `plans/mainz_src_port.md:919`; `measure_mainz_fundamental_diagram.py:145`; the 27-sample denominator (1.3); exact reproduction |
| `warmup_s` | 300 s | `MainzEnv.__init__` default, unchanged; consistent with the 27-sample count |
| `--episodes` at training | 80 | `results/checkpoints/*_training.log`, best at 79 (here) and 44 (src) |

These become the evaluator's **defaults**, so the correct configuration is what a bare
invocation produces. They are printed at start-up and written into the artifact's `config`
block.

**Where the record disagrees with itself**, stated flatly rather than reconciled:

- `scripts/train_mainz_src.py`'s defaults are `--step-length 1.0` and `--window-start-s
  0.0`. Neither is the configuration the committed checkpoints were produced under. A
  reader running the script as documented gets 3,809.78 veh/h for no control, not 3,568.
- `plans/mainz_src_port.md` records two configurations. Line 643 gives "0.5 s steps,
  throughput counted from 600 s" with `DENSITY_CRITICAL` 0.095 and a 0.85-green exit
  meter — a superseded run whose flows are near 2,150 veh/h. Line 919 gives the final one.
  Only the second describes the committed checkpoints.
- `results/checkpoints/*_training.log` record no command line at all. They are the primary
  artifact for the run and they do not say what was run.

The third of these is the general lesson and belongs in the artifact design, not just in
this plan: the evaluator writes its full configuration and its commit into the output, so
the next person does not have to recover it from a denominator.

## 7. Decision 6 — which action mapping the number is quoted under

**HEAD's `SPEED_ACTION_FRACTIONS`.** The evaluator runs the code that is in the tree,
unmodified.

The argument: `tests/test_mainz_port.py::TestTheActionSet` pins the fraction form, and
`6b538f2` records it as one of the four defects that test exists to pin. A result the
repository cannot reproduce with its own code is precisely the defect task 142 names. The
stored `test` blocks in `results/checkpoints/*_result.json` become artifacts of a
superseded encoding; the new artifact supersedes them and `results/README.md` says so.

The argument against, recorded because it is real: the published five-seed numbers move
for a second reason on top of the seed extension, so a reader comparing the old and new
tables sees two changes and one explanation. This is why it is open item 1.

The evaluator does **not** carry a `--legacy-action-set` flag. Adding one would put a
second action set in the tree whose only purpose is to reproduce a superseded number, and
the correct home for that comparison is this plan's section 1.4, where it now is.

## 8. Decision 7 — proving the evaluator can report a wrong answer

A result from an instrument never seen to fail is not evidence. Two guards, one
structural and one numeric, both required to run before the real numbers are believed.

**Structural: the checkpoint cannot be silently mismatched.** `--arm here` binds
`mainz_here_best.pt` to `HERE_FEATURES` (5 features, input width 60); `--arm src` binds
`mainz_src_best.pt` to `SRC_FEATURES` (6 features, width 72). The script constructs
`SrcQNetwork(12, len(features), 3)` and calls `load_state_dict` without `strict=False`, so
a crossed pair raises on the first layer's shape. Step 6 runs the crossed pair once and
records that it raises. This is the failure mode
`feedback_a_field_built_and_never_passed` describes: a mismatch that reads as a number
rather than as an error.

**Numeric: an obviously wrong policy must read as wrong.** Take
`results/checkpoints/mainz_here_best.pt`, zero `stack[2].weight` and `stack[2].bias`, and
evaluate it as a fourth arm. Every Q-value is then zero, `argmax` returns index 0 for all
twelve segments on every input, and the policy is a standing order to drive at half the
limit. Measured on seeds 16-20 (section 1.7): **2844.00 veh/h**.

> **2026-09-12 amendment (validator round 1).** The acceptance criterion below
> originally had a second bullet requiring the zeroed-head arm's flow to fall below
> the *trained* arm's flow by more than ten times the paired bar. Both the validator
> and an independent check agreed that bullet is mis-specified and replace it below;
> bullet 1 is unchanged. Two defects in the original bullet 2: the only paired bar
> this evaluator computes is the here-vs-no-control difference -- the uncertainty of
> a comparison the bullet was not making, since the bullet compared the zeroed arm
> against the *trained* arm, not against no-control. And the multiplier ten has no
> derivation; it is the observed five-seed ratio (939 / a bar of order 90) rounded,
> and it gets **looser** as more seeds are added, because the bar it is multiplying
> shrinks with `n` while the margin it is being compared against is not defined to
> shrink for the same reason -- the wrong direction for an acceptance gate. Bullet 2
> is replaced with the same paired statistic this evaluator already reports for
> every trained arm, applied instead to the zeroed-head arm against no-control, which
> is now runnable via `--arms zeroed_head` (round-1 fix F9). Bullet 1 is retained
> unchanged and is the test that actually carries the falsification claim: on five
> seeds the zeroed-head arm read 724.00 veh/h below the no-control mean, against
> no-control's own two-standard-error bar of 88.8071 -- a ratio of 8.15, not a
> borderline call.

The acceptance criterion is stated numerically, not as "differs":

- the zeroed-head arm's flow must be below the no-control mean by more than the
  no-control arm's own two-standard-error bar, and
- its **paired** gain against no-control (the same `paired_statistics` figure this
  evaluator already reports for every trained arm) must be negative and resolved by
  its own paired bar -- the same statistic, applied to an arm known to be wrong, with
  no free multiplier, and it gets harder to satisfy as arms converge, not easier.

**The statistics are tested separately, on fixtures chosen to discriminate.** A fixture
whose two candidate formulas agree proves nothing (`feedback_a_fix_is_new_code`: equal
values hide a swap). Three fixtures, in `tests/test_mainz_paired_evaluation.py`, with no
SUMO import so they run in milliseconds:

| fixture | baseline | arm | paired bar | what it catches |
|---|---|---|---|---|
| perfectly correlated | 1000, 2000, 3000, 4000, 5000 | each +100 | **0.0** | the independent formula, which gives 2000.0 |
| two seeds | 0, 0 | 0, 100 | **100.0** | `pstdev` in place of `stdev`, which gives 70.7 |
| arm below baseline | 3000, 3000 | 2900, 2900 | 0.0, gain **-100** | a sign convention reversed |

Each must be seen to fail before it is trusted: run the test file against a deliberately
wrong implementation (independent formula, `pstdev`, reversed subtraction) and record that
each fixture rejects it. A guard that has only been seen to pass is not a guard.

## 9. The steps

Steps 1-7 are `implement_dsrc`. Step 8 is `experiment_dsrc` and spends the compute. Steps
9-11 are the document correction and are not started until step 8's numbers are in hand.

| # | Step | Touches | Done when |
|---|---|---|---|
| 1 | CLI and configuration constants: `--arms`, `--seeds` (default 16..30), `--step-length 0.5`, `--duration-s 2500`, `--window-start-s 900`, `--out results/evaluation`, `--checkpoint-dir results/checkpoints` | `scripts/evaluate_mainz_checkpoints.py` | `--help` runs; defaults print |
| 2 | Arm binding: `here`/`src` -> (checkpoint path, feature tuple, `gate_entries=True`); `no_control` -> `gate_entries=False`. Input-width assertion after `load_state_dict` | same | crossed pair raises |
| 3 | The episode loop, sequential, importing `run_episode` and `evaluate_no_control`'s body from `train_mainz_src`; one JSONL line flushed per episode | same | a 2-seed run writes 6 lines |
| 4 | Paired statistics per section 4, plus per-arm `flow_2se`; the gate-asymmetry assertion from section 5 | same | fixtures in step 7 pass |
| 5 | Summary writer: `config` + `commit` + `per_seed` + `arms` + `paired`, whole-then-rename | same | JSON parses; `config` complete |
| 6 | **Reproduction gate.** `--seeds 16 17 18 19 20`. no_control must match `results/checkpoints/mainz_here_result.json`'s `no_control` block on all seven metrics to 1e-6; `here` must read flow 3783.111111 and `src` 3764.444444 (section 1.4). Run the crossed-checkpoint pair once and record that it raises | none | all three match; ~95 s |
| 7 | **Falsification gate.** The zeroed-head arm over the same five seeds, criterion in section 8; plus `tests/test_mainz_paired_evaluation.py` with the three fixtures, each seen to fail against a wrong implementation | `tests/test_mainz_paired_evaluation.py` | zeroed head reads ~2844; each fixture rejects its wrong implementation |
| 8 | **The run.** 15 seeds x 3 arms, defaults, stdout redirected to `results/evaluation/mainz_paired_seeds.log` | `results/evaluation/*` | 45 JSONL lines; summary written; ~5 min |
| 9 | Branch on the outcome (section 10) and correct every location in section 1.1 | the five documents | no location asserts a number no artifact carries |
| 10 | `plans/mainz_src_port.md:10` vs `:19` — the heading and the seed count made consistent; `:1054` "Mainz is unchanged" corrected per open item 2 | `plans/mainz_src_port.md` | one seed count in the document |
| 11 | `results/README.md` gains a `evaluation/` section; `plans/implementation_records.md` task 142 closed with what the run produced | `results/README.md`, `plans/implementation_records.md` | task 142 states the outcome, not the intent |

## 10. Decision — what happens to the documents, in both outcomes

The run has two outcomes and neither is a failure. Both are written out now so the choice
is not made after seeing the number.

**Outcome A — the quoted figures reproduce.** The fifteen-seed means land near 3,765 /
3,759 / 3,535 with paired gains near +230 +/- 44 and +224 +/- 61, both resolved. Then:

- the numbers stay, with the +0.22% action-mapping shift from section 1.4 applied and
  noted once;
- `plans/mainz_src_port.md:10` changes from "Held-out seeds 16-20" to the fifteen-seed
  wording, and the five-seed section at 918-935 is retitled so the two coexist;
- `scripts/train_mainz_src.py:63` stays at `range(16, 21)` — the driver's job is still to
  read the pre-registered five once — and `paper_deploying_self_regulating_cars.md:598`
  is corrected to say so, with the evaluator named as what reads 16-30;
- `results/README.md` cites `results/evaluation/mainz_paired_seeds.json` as the artifact.

**Outcome B — they do not reproduce.** Any of: a different mean, a wider bar, or a gain
that is no longer resolved. Then:

- every location in section 1.1 is corrected to what the artifact says, with no averaging
  of old and new and no retained "approximately";
- if a gain stops being resolved, that is reported as the result. The commit message of
  `eb86e72` already records the same thing happening on `inverted_tree_bottleneck`, where
  a five-seed +185 +/- 211 fell to a fifteen-seed +44 +/- 111. Mainz surviving that test
  is a claim, and this run is the first time it is checked in the repository;
- the paper's argument may then need to change in kind rather than in digits. That is
  named out of scope here and becomes its own task.

In both outcomes the same sentence is added to `results/README.md`: which script produced
the figure, at which commit, under which configuration.

## 11. Risks

1. **`libsumo` is process-global.** `src/sumo/binding.py` states it: one environment may
   be live at a time, and whoever opened it closes it. The evaluator is strictly
   sequential and every episode closes its env in a `finally`. Parallelising across seeds
   would need separate processes and is not worth 5 minutes; it is explicitly not done.
   The failure mode if this is ignored is not an exception but a second `start()` against
   a live simulation, which produces numbers.
2. **Episode count and wall clock.** 45 episodes, measured 6.3 s each: **about 5 minutes**
   for step 8, plus about 2 minutes for the fifteen-seed falsification arm if it is run at
   full width, plus about 95 s for the step 6 reproduction gate. Under 15 minutes for the
   whole experiment stage. Seeds 21-30 may be more congested than 16-20, and wall clock
   scales with vehicle count, so budget 8 minutes for step 8 rather than 5.
3. **The figures may not reproduce.** Section 10 outcome B. This is an outcome, not a
   failure, and the plan is written so that discovering it requires no new decision.
4. **The result is chaotically sensitive at the 0.2% level.** A 0.012 km/h change in the
   commanded speed — 0.02% — moved the five-seed mean flow by 8.4 veh/h. So "reproduces"
   means bit-exact against a named commit, any edit to `_command` or to the scenario
   invalidates every stored trained-arm number, and the artifact must carry its commit.
   It also means the paired bar of ~44 veh/h is only about five times this sensitivity,
   which is worth stating when the bar is quoted.
5. **Nothing recorded the run configuration, and that is why this task exists.** The
   configuration was recovered from a float's denominator and a reproduction test. If the
   evaluator repeats the omission, the next person repeats the recovery. The `config` and
   `commit` blocks are mandatory, not decorative.
6. **Selection is not contaminated by the extension.** Checkpoint selection used
   validation seeds 11-15 only, and the checkpoints are committed and read read-only.
   Adding seeds 21-30 to the evaluation set cannot leak into selection. This is worth
   stating because "we enlarged the evaluation set until it agreed" is the obvious
   objection to the whole exercise, and the answer is that selection is frozen in a `.pt`
   file that predates the extension.
7. **The evaluator shares code with the driver, so a defect in the shared path is
   invisible to the reproduction gate.** `run_episode` is imported rather than copied
   (section 2), which is right for drift, but it means step 6 cannot catch a bug that
   affects the stored run and the new run identically. What step 6 does catch is a wrong
   configuration and a wrong checkpoint, which are the two failure modes that actually
   occurred here.
8. **Two concurrent agents are writing plan files in this tree.** `plan_task143`,
   `plan_task144` and `plan_task145` are untracked at the time of writing. Step 9's
   document edits touch `plans/implementation_records.md`, which those tasks also touch.
   The implementer should re-read that file immediately before editing it rather than
   working from a stale copy.

---

## 12. Sign-off

- [ ] Open item 1: quoting the number under HEAD's action set, accepting that the
      published five-seed figures then move by +0.22% for a reason unrelated to seeds.
- [ ] Open item 2: what happens to `plans/mainz_src_port.md:1054`, "Mainz is unchanged".
- [ ] Open item 3: `results/evaluation/` as a new directory, against
      `results/checkpoints/`.
- [ ] Open item 4: keeping the five-seed section at `plans/mainz_src_port.md:918-935`
      alongside the fifteen-seed result.
- [ ] Open item 5: whether an unresolved fifteen-seed gain opens a separate task for the
      paper's prose.
- [ ] Decision 1 confirmed: a separate evaluator rather than an extension of the driver.
- [ ] Decision 3 confirmed: the paired formula, with the independent formula named and
      rejected.
- [ ] Decision 7 confirmed: the zeroed-head falsification arm is run and recorded before
      the real numbers are quoted.
- [ ] Section 10 confirmed: both outcomes are pre-committed, and outcome B is reported as
      a result rather than retried.
