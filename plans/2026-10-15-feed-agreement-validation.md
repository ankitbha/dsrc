# Feed agreement: validation record

Independent validator's findings against `plans/2026-10-15-feed-agreement-plan.md`. One
section per round. Every finding has an id, a bucket, a location, the evidence and a
proposed fix. Buckets: BLOCKS, SHOULD FIX, DOCUMENTED LIMIT, NOT WORTH FIXING.

## Round 1: commit d56ba5b (program commits d32187d..d56ba5b)

Mirror: `scratchpad/dsrc_feed/mirrors/round1`, built by `git archive d56ba5b`, with the
repository's `.venv` linked in. The checkpoint in the archive has the same sha256 as the
live one (`941ac210...`).

### Coverage

| # | attack point | status | method |
|---|---|---|---|
| 1 | published mode equals committed `here` rows, seeds 16-18 | done: holds | run `run_feed_episode` in the mirror, compare with `results/evaluation/mainz_paired_seeds.jsonl` |
| 2 | injection changes commands; realised vs configured fraction | done: R1-2, R1-3, R1-11 | instrumented `inject_delta` runs; per-vehicle `setSpeed` comparison against `published` |
| 3 | staleness injected | done: holds | recompute `greedy_actions` on stored ring snapshots |
| 4 | rule arms use the measured process; placeholder refused | done: holds; R1-10 | read `load_feed_parameters`, run without the flag |
| 5 | lemma conditions checked | done: holds (block A table) | recompute from `feed_process.json` and the drive logs |
| 6 | held and pending counted | done: counted; definition R1-4 | read `drain_time`; force pending; block D pilot traces |
| 7 | jam threshold is not the reward's; threshold well defined | done: R1-4, R1-5, R1-6 | block D pilot episodes (seed 16, both scenarios, both arms) in the mirror; refit |
| 8 | censored runs kept | done: holds | `drain_summary` and KM on constructed inputs |
| 9 | one clock start per scenario | done: holds | read `DRAIN` and `summarize_d` |
| 10 | confidence-bound direction | done: holds; pulse signal R1-8 | numeric checks of `t_quantile`, `confidence_bounds`, `rho_upper_bound` |
| 11 | pulse size | done: R1-9 | pulsed vs unperturbed accounting in the mirror |
| 12 | penetration is what it says | done: holds (count form) | count commanded ids at p = 0.5 |
| 13 | key never reaches disk | done: holds (no polling run yet) | grep tracked files and outputs with the key held in a shell variable |
| 14 | pairing | done: holds | read `block_specs` and `_paired` |
| 15 | figures carry artefact numbers | done: read; FG mutants killed | read `feed_figures.py` |
| 16 | timing | done: holds | wall times of the mirror runs |
| M | mutation sweep on number-producing code | done: 50/73 killed; R1-14 | mutants in the mirror, one test file per process |
| X | collisions in the controlled arm (found while checking 11) | done: R1-7 | collision and teleport counts per run; `--collision.action warn`; `actionStepLength` 0.5 s |

### Findings

**R1-1. SHOULD FIX (process, no reported number).** Location: commit `d32187d` (the plan
commit), `paper/architecture.tex`. The plan commit also commits the deletion of
`paper/architecture.tex` (18 lines), which was one of the author's uncommitted `paper/`
edits; the implementer and planning briefs both say the `paper/` edits stay unstaged.
Evidence: `git show --stat d32187d` lists `paper/architecture.tex | 18 -` beside the plan;
`git log main..d56ba5b -- paper/` returns only `d32187d`. Fix: before the branch is
pushed or merged, rewrite `d32187d` so that it adds only the plan (for example
`git rebase -i` dropping the path from that commit, or a follow-up commit that restores
`paper/architecture.tex` from `main` in the index while leaving the working-tree deletion
as the author's uncommitted edit). The orchestrator decides which; nothing downstream
reads the file.

**Attack point 1 holds.** `published` mode run from the mirror reproduces the committed
`here` rows exactly on seeds 16, 17 and 18 (flow 3,751.11 / 3,795.56 / 3,771.11 veh/h,
arrivals 1,989 / 2,013 / 1,997, held 56 / 51 / 44), and `no_control` reproduces the committed
`no_control` rows (3,722.22 / 3,553.33 / 3,564.44 veh/h). `inject_delta(0)` also reproduces
`published` on seed 16 exactly. Wall time 6.1 to 7.1 s per 2,500 s episode.
Driver: `scratchpad/dsrc_feed/val/run_modes.py`.

**R1-2. BLOCKS (block B, Fig. 2(a)).** Location: `src/sumo/feed_episode.py:316-326`
(`_decide_injected`). A late vehicle is written only when `target != before`, where
`before` is inferred as `k-1` for any vehicle not in the previous late set. That
inference is false for a vehicle that was not on a segment edge at the previous decision
(inserted during the interval, or on a junction): it was never written `a_{k-1}`, and
being late it is not written now, so it drives uncommanded for the whole interval
instead of applying `a_{k-1}`. Separately, a late vehicle that was written `a_{k-1}` at
the previous decision keeps that absolute speed, which was computed for the edge it was
on 60 s earlier, not `a_{k-1}[e] x free-flow(current edge)`. So the treatment the plan
describes ("applies `a_{k-1}` for the whole interval", plan 3.3) is mostly not what
SUMO receives. Evidence (`val/inject_probe.py`, seed 16, delta 0.1, 2,500 s, decisions 2
to 37, 2,894 late vehicle-decisions): 67 (2.3%) are on a segment whose policy fraction
changed between `a_{k-1}` and `a_k`; 269 (9.3%) were never written and drive uncommanded;
872 (30.1%) keep a speed that equals neither `a_k` nor `a_{k-1}` on their current edge.
The share of late vehicles whose SUMO speed command differs from the on-time command is
therefore about 40%, of which 2.3 points come from the snapshot. Block B's flow loss
against delta would measure mostly "command not refreshed for 120 s" and "uncommanded",
and Fig. 2(a)'s reading (a disagreeing vehicle acts like a non-compliant one) would be
partly built in by the 9.3% that are in fact uncommanded. Fix: at each decision write
every scanned vehicle `fraction(target, e) x free-flow(edge)`, with `target = k-1` if
late else `k` (as `step()` writes every scanned vehicle). Then `inject_delta(0)` still
equals `published`, and a late vehicle differs from an on-time one only where
`a_{k-1}[e] != a_k[e]`. Tests: (i) a fake-env unit test that at a decision `k >= 2` with
delta 1.0 every scanned vehicle, including one absent from the previous scan, is written
`a_{k-1}[e] x free-flow(edge)`; it fails on d56ba5b for the absent vehicle. (ii) An
end-to-end equality `inject_delta(1.0) == inject_stale(60)` does NOT hold even after the
fix, because at 300 s `inject_stale(60)` writes `pi(x(240))` while `inject_delta` has no
`a_0` and writes nothing; measured on d56ba5b, seed 16: `inject_stale(60)` 3,784.44 veh/h,
`inject_delta(1.0)` 3,760.00 veh/h, `inject_stale(0)` 3,751.11 (equal to `published`).

**R1-3. SHOULD FIX (block B x-axis, Fig. 2(a), F1 slope).** Location:
`scripts/run_feed_experiments.py:534-553` (`summarize_b`), `scripts/figures/feed_figures.py:127-153`
(`fig_flow_consequence`), `DELTAS = (0.1, 0.25, 0.5, 0.75)`. The x-axis of the delta fit
and of Fig. 2(a) is the configured delta, labelled "injected disagreement". Under the
plan's own definition of disagreement (D2, fraction not on the plurality version), a
late fraction above 0.5 is not a disagreement of that size: the late vehicles are the
plurality. Measured on seed 16: `inject_delta(1.0)` records a time-averaged disagreement
of 0.085, and `inject_delta(0.1)` 0.091; at 0.75 the decision-instant disagreement is
0.25. Table 1 reports block A's plurality disagreement, so a slope fitted on the
configured delta (with the 0.75 point at x = 0.75) cannot be multiplied by a Table 1
delta to predict a rule's flow cost. The configured and realised values also differ
below 0.5 (0.1 configured, 0.091 realised on seed 16 over 2,500 s; 0.5 configured, 0.456
realised), because vehicles that reach a segment after the draw are counted as on time
and late vehicles leave; the draw itself is exact (`round(delta n)` of the scanned
vehicles). This answers the implementer's question 4: 0.087 against 0.1 is the
definition (time average over the interval), not a draw defect. Fix: name the x-axis
"fraction on the previous snapshot", record that fraction per second beside the
plurality disagreement (for delta <= 0.5 they coincide at the decision instant), and
either restrict DELTAS to values at or below 0.5 or plot both quantities; fit the F1 line
on the measured per-second fraction, not on the label.

**Block D pilot, run in the mirror (seed 16, both scenarios, both arms; driver
`val/run_modes.py`, analysis inline in the session).** Wall time 31 to 41 s per episode
(budget 60 s). Scenario (b) `mainz_burst`, cap 12,000 s: `no_control` served 3,501 veh/h
over [900, 12,000] s and ended with 154 pending; `here` served 3,667 veh/h and ended with
0 held and 0 pending. Scenario (a) `mainz_surge`, cap 9,000 s: both arms delivered all
6,000 vehicles. The no-control arm has `held = 0` on every sample (attack point 6 holds
for held); pending is positive after 4,800 s in scenario (b).

**R1-4. BLOCKS (Fig. 3, scenario (b)).** Location: `src/sumo/drain.py:101-121`
(`drain_time`). The drain instant is the first sample `t_d >= t_0 + H` with no jam in
`[t_d - H, t_d]` and nothing waiting at `t_d`. When the jam forms after the clock starts,
this returns the minimum possible drain time for a run that never drains. Evidence: on
the `mainz_burst` pilot, `no_control` has no jammed segment at `t_0 = 1,500 s`, jams from
about 1,680 s (segments 10 and 11 at 29 and 20 km/h at 1,680 to 1,740 s), stays jammed on
2 to 5 segments to the cap, and accumulates pending from 4,800 s to 154 at 12,000 s, yet
`drain_time` returns `(120.0, True)` under all three thresholds (pooled 22.0 km/h, per
segment, fixed 30 km/h). `here` returns `(130.0, True)` under the pooled threshold. Fig. 3
would show both arms drained within about 2 minutes on this seed. The plan's definition
(3.5) is implemented literally; the defect is in the definition, which assumes the
network is jammed at `t_0`. Fix: count a run as drained at `t_d` only if, in addition,
no later sample up to the cap starts a jam spell or a waiting spell (held + pending > 0)
lasting at least `H`; otherwise censor at the cap. Measured on the pilot with that rule:
`no_control` (b) is censored under every threshold; `here` (b) drains at 9,730 s after
`t_0` (thresholds 20.25 and 22.0 km/h) or 10,410 s (30 km/h), because the exit segment
under a 3,650 veh/h background keeps dipping below the threshold for more than 120 s; for
scenario (a) the rule changes nothing at 20.25 or 30 km/h (no_control 5,450 / 5,460 s, here
4,810 / 4,850 s). A stricter absorbing rule (nothing jammed or waiting from `t_d - H` to
the cap) censors both arms of (b) because of single-sample exit dips, so it is not
recommended. Add a test with a trace that is clear at `t_0`, jams at `t_0 + 200 s` and
holds pending to the cap; it must be censored.

**R1-5. BLOCKS (Fig. 3, scenario (a); answers the implementer's question 1).** Location:
`src/sumo/drain.py:207-239` (`thresholds_from_traces`), `scripts/run_feed_experiments.py:642`.
The fit itself is correct: on the surge pool and on a bimodal 15/50 km/h fixture the
vectorised grid search returns the same knots and residual as a brute-force
least-squares search (3.5/25.0, residual 0.040291; 23.0/42.5, residual 0.403407). The
plan's S5 expectation `s1` in [12, 22] for that fixture was a wrong expectation of what a
three-piece fit does to a bimodal CDF, not a code defect. The defect is the pool. The
2023 paper fits per segment over two years of hourly data, in which normal operation
dominates, and justifies `(s1 + s2)/4` by `s2 <= 3 s1` so that the threshold lies below
`s1`. The plan pools only the scenario's no-control samples from `t_0 - 600 s`, and
empty segments are excluded, so in scenario (a), whose demand stops at 1,200 s, the pool
is the jam. Evidence (seed 16): surge pool median 14.7 km/h, fit `s1 = 3.5`, `s2 = 25.0`,
`v_jam = 7.12 km/h`, so `s2/s1 = 7.1` and `v_jam > s1`, which fails the 2023 paper's own
condition. Burst pool: `v_jam = 22.0`. Sustained-demand `mainz` episodes (both arms,
seeds 16 to 18, 300 to 2,500 s): `s1 = 30.0`, `s2 = 51.0`, `v_jam = 20.25`. The primary
threshold differs 3-fold between the two drain scenarios on the same network, and in (a)
it reverses the arm ordering: under 7.12 km/h `here` drains 610 s after `no_control`
(4,700 against 4,090 s); under 20.25 km/h `here` drains 640 s before (4,810 against 5,450 s);
under 30 km/h 610 s before (4,850 against 5,460 s). Per-segment fits on the surge pool give
`v_jam` of 4.4 to 11.5 km/h, so the per-segment sensitivity shares the defect. Fix: fit
one network-level threshold on a normal-operation pool, the sustained `mainz` scenario's
`published` and `no_control` episodes on seeds 16 to 30 from 300 s to 2,500 s (block A
already traces `published`; 15 `no_control` sustained episodes cost about 100 s), and
apply it to both drain scenarios; refuse any fit (pooled or per segment) with
`s2 >= 3 s1`, the 2023 paper's condition, recording the refusal; keep the per-scenario
no-control fit and 30 km/h as reported sensitivities. Add the guard as a unit test on the
surge-like pool (median 14.7, knots 3.5/25.0), which must be refused.

**R1-6. SHOULD FIX (plan-level, block D pilot rule).** Location:
`scripts/run_feed_experiments.py:580-612` (`pilot_decision`), plan section 5. The pilot
rule's breakdown test (served flow over [1,500, 3,000] s below 3,650 veh/h AND pending at
3,000 s above pending at 1,500 s) is implemented as written, but on seed 16 it returns
"does not break down" (served 3,537.6 veh/h, pending 0 at both instants) and instructs a
rebuild with a 6,000 veh/h burst, although no control does break down on that seed: the
jam starts at about 1,680 s and pending first appears at 4,800 s and reaches 154 by the
cap. The pending criterion at 3,000 s lags the breakdown by about 1,800 s. Fix: replace
the pending criterion by a jam criterion under the threshold of R1-5 (a jam spell of at
least `H` after `t_0`) or by pending at the cap; with either, seed 16 keeps the 5,000
veh/h burst. This is the orchestrator's call, because it decides whether the scenario is
rebuilt.

**R1-7. BLOCKS (paper-level and every block; the cause predates this branch).**
Location: `src/sumo/mainz.py:183-192` (SUMO options; no `--collision.action`, so SUMO's
default `teleport` applies), `data/mainz/mainz.rou.xml:2-3` (`carFollowModel="EIDM"`,
`actionStepLength="1.0"` against a 0.5 s step), `MainzEnv._command` / `command_vehicles`
(`setSpeed`). Controlled vehicles rear-end stopped vehicles, and SUMO resolves each
collision by teleporting the follower downstream. No uncontrolled run collides. Evidence,
seed 16, in the mirror at d56ba5b (`val/missing_probe.py` and inline probes):

| run | collisions (colliding-vehicle records / 2) | collision teleports |
|---|---|---|
| sustained `published` (`here`), 2,500 s | 76 | 76 |
| sustained `no_control` | 0 | 0 |
| `mainz_surge` `here`, 9,000 s (6,000 vehicles) | 3,962 | 3,962 |
| `mainz_surge` `no_control` | 0 | 0 |
| `mainz_burst` `here`, 12,000 s | 706 | 706 |
| `mainz_burst` `no_control` | 0 | 0 |

Collision records: 70 of 76 are lane collisions, 6 junction; the victim's speed is 0 in
the sampled records and the collider's 0.2 to 5.6 m/s; 17 + 11 + 6 on the exit link 218
and 31 on edge 31; both vehicles are type `av` in all 76. Re-running the sustained
`published` arm with `--collision.action warn` (collisions logged, nobody teleported, so
no vehicle jumps the queue) lowers its flow by 144, 193 and 144 veh/h on seeds 16, 17 and
18 (3,606.67 / 3,602.22 / 3,626.67 against the committed 3,751.11 / 3,795.56 / 3,771.11),
and leaves `no_control` unchanged; the paired gain of `here` over `no_control` on these
three seeds goes from a mean of +159 veh/h to a mean of -2 veh/h. Setting both vehicle
types' `actionStepLength` to 0.5 s (equal to the step) removes every collision on seed
16 (`here` 3,791.11, `no_control` 3,846.67 veh/h), so the likely cause is `setSpeed`
under EIDM with a 1.0 s action step; it is not established on more than one seed. What
this does to the program: block D's drain comparison sets an arm whose jammed vehicles
are teleported out of the queue (3,962 teleports for 6,000 vehicles in scenario (a))
against an arm with none, so drain times and the R1-4/R1-5 numbers above are not
comparable between arms; block E's deviations include teleport jumps; and every block's
reference arm (`published`, the committed `here` rows) carries the effect. Fix, in
order: (1) record per episode the colliding-vehicle count and the collision-teleport
count (`simulation.getCollidingVehiclesNumber`, `getStartingTeleportNumber`) in every
JSONL record and in the 10 s trace, so no reported number silently contains them;
(2) the author decides the remedy (`actionStepLength` equal to the step for both types,
or a speed mode that keeps EIDM's safe speed binding under `setSpeed`, or
`--collision.action warn` with the overlap reported) and the published `here` row is
re-run under it before any feed number is stated relative to it. This is outside
d32187d..d56ba5b but it decides whether the program's comparisons mean what they say.

**R1-8. BLOCKS (block E, the certificate).** Location: `src/sumo/mainz.py:495-535`
(`inject_pulse`), called from `src/sumo/feed_episode.py:512-513` right after the warm-up.
`inject_pulse` hands SUMO all `n` pulse vehicles at 300 s with future departure times
(`depart="1200.50"`, ...), which changes the simulation from that instant on, 900 s before
the first pulse vehicle departs. So `y = |rho_pulse - rho_base|` over [1,200, 1,800] s is
mostly the divergence of two different traffic realisations, not the response to the
pulse. Evidence (`val/pulse_probe.py`, seed 16): the pulsed and unperturbed traces first
differ at 310 s, the first sample after the call, both at `p = 1` (site 7) and at
`p = 0.5` (site 0); the largest pre-pulse deviation over [900, 1,190] s is 0.0195,
against a peak self-deviation at the pulsed segment of 0.026 (at 1,290 s), and the mean
deviation per segment over [1,800, 2,500] s is 0.000 to 0.016 (0.007 on most segments).
Deferring each `vehicle.add` to the tick of its departure (`depart="now"`), with
`route.add` still at 300 s, makes the two runs identical through 1,200 s; they first
differ at 1,210 s (`val/pulse_defer_probe.py`). Fix: insert pulse vehicles from the tick
callback at their departure instants; add a SUMO test that the pulsed and unperturbed
traces are equal on every sample before `start_s` (it fails on d56ba5b at 310 s); and,
because a single inserted vehicle also changes the realisation after it departs, report
a noise floor beside the gains (for example the mean `y` over [1,800, 2,500] s, or a
control pulse of one vehicle at the exit's last edge) and state whether the pulse
response is resolved above it. The plan's regression window includes the 60 s injection
itself, during which the increments are driven by the input rather than by `y`; on the
site-7 run the diagonal estimate moves from `a_7 = 0.0217` (window from 1,200 s) to
`0.0333` (from 1,290 s). Starting the regression at `start_s + duration_s` removes the
input from the fit.

**R1-9. SHOULD FIX (attack point 11 check).** Location:
`scripts/run_feed_experiments.py:748-773` (`_accounted`, `_inserted_differences`).
`arrived + running + pending + held` omits vehicles that are mid-teleport at the end
(R1-7), so the pulsed-minus-unperturbed difference is not the pulse size even when the
pulse was inserted correctly. Evidence: seed 16, `p = 1`, site 7: difference 56, not 60;
the unperturbed run accounts for 3,123 of 3,124 vehicles (`v293_2737` in teleport at the
end) and the pulsed one misses 5 (`v140_382`, `v182_878`, `v182_897`, `v289_2219`,
`v295_3015`, all teleports started and not ended). At `p = 0.5`, site 0, the difference is
60. Fix: add the count of vehicles whose teleport has started and not ended (tracked from
`getStartingTeleportIDList` and `getEndingTeleportIDList`) to `_accounted`, or account
from the loaded count; after R1-7's remedy the term should be zero, and the check then
tests the pulse.

**Block A rules on the drive stamps (brief attack point 2).** Pure computation with
`src/sumo/feed.py` on the replayed placeholder (session `drive_run_20260908_170849`,
`L_bar = 67.08 s`, `g_min = 26 s`, seeds 16 to 18, 2,000 vehicles with uniform phases,
time average over [900, 2,500] s), and the same rules run in SUMO on seed 16
(`val/rule_probe.py`):

| rule | pure, plurality disagreement | SUMO seed 16 | Lemma 1 (b) cycle average, g = 26 s / 60 s |
|---|---|---|---|
| synchronous | 0 | 0 | - |
| fixed phase, psi* = 46 | 0 | 0 | - |
| no rule, P_q = 60 s | 0.280 | 0.281 | - |
| no rule, P_q = 10 s | 0.047 | 0.046 | - |
| version-keyed, D = L_bar - 20 | 0.032 | 0.031 | 0.931 / 0.417 |
| version-keyed, D = L_bar - 5 | 0.011 | - | 0.385 / 0.167 |
| version-keyed, D = L_bar | 0.0046 | 0.0046 | 0.192 / 0.083 |
| version-keyed, D = L_bar + 5 | 0.0008 | - | 0.048 / 0.021 |
| version-keyed, D = L_bar + 10 | 0 | 0 | 0 / 0 |

The measured disagreement is below the bound at every `D`, and the SUMO engine's
time-averaged disagreement equals the pure rule computation to within 0.001. The bound
is not tight (0.0046 against 0.083 at `D = L_bar`), because most replayed lags are below
`L_bar` (33.8 to 67.1 s) and most gaps are 60 s. `P_q <= g_min - J` is false on the
placeholder (J = 49.9 s measured, 33.3 s replayed) and the summary records
`zero_disagreement_claimed: false` (attack point 5 holds); zero disagreement at
`D = L_bar + 10` is still expected, because the proof of part (a) needs only
`A_{k+1} - A_k >= P_q`, and the replay's smallest availability gap is 16.7 s. Switches
consumed without a write (vehicle on a junction or not yet scanned at its switch tick):
11.9% in `published`, 12.4% to 13.7% in the rule arms, so deviation 3 has the published
arm's semantics. Per-tick episodes take 7.6 to 7.8 s (budget 10 s; attack point 16 holds).

**R1-10. SHOULD FIX (latent; block A fixed-phase arm with the polling artefact).**
Location: `src/sumo/feed.py:110-122` (`FeedProcess.from_artefact`). Only the first stamp
is aligned to its recorded second of the minute. When the replay wraps from the last
`(gap, lag)` pair to the first, the next stamp is placed at `T + gap` of the last pair,
whose second of the minute is that of the recorded version after the last, not that of
the first pair's version whose lag is now applied. Every stamp after the wrap is
shifted, so the simulated availability phases `(T_k + L_k) mod 60` are not the measured
support from which `psi*` was chosen. Evidence: on the placeholder (29 pairs, about 50
versions per episode) every seed 16 to 30 wraps, and 22 to 49 of the 48 to 51 simulated
stamps per seed carry a second of the minute different from the recorded one (a 26 s
shift). On the placeholder the shifted availability phases still miss the window
[46, 47] s, so `fixed_phase` disagreement stays 0 on all 15 seeds; with the polling
artefact (about 120 versions per area-session, about 45 needed) roughly a third of seeds
would wrap and the fixed-phase arm's disagreement could change. Fix: draw the start
index from `[0, n - needed]` so the horizon never wraps, and refuse a session shorter
than the horizon (the placeholder is already refused without `--allow-placeholder`);
or, at a wrap, insert a gap that restores the recorded second of the minute. Add a test
that every simulated stamp's `T mod 60` equals its pair's recorded phase.

**R1-11. DOCUMENTED LIMIT (answers the implementer's question 7; plan risk 4 is
realised).** Command-level disagreement is about 20 times smaller than version-level
disagreement because the policy's action vector rarely changes. Seed 16, 2,500 s: the
action vector changes on 8 of 37 decisions (14 of 444 segment-decisions) in
`inject_delta(0.1)` and on 13 of 37 (16 of 444) in `inject_delta(0.5)` and `(1.0)`;
`no_rule_60` version disagreement 0.281 against command disagreement 0.0133;
`version_keyed` at `D = L_bar` 0.0046 against 0.00025. A vehicle on a different snapshot
commands a different speed fraction only when its segment's entry changed between the
two snapshots, which on seed 16 is about 2.3% of late vehicle-decisions (R1-2). The paper
must state this beside any flow consequence of disagreement: on this policy the rule
matters on few decisions, and block A's flow differences between rules (seed 16: 3,751
to 3,824 veh/h across `published` and the six rule arms) are of the size of
seed-to-seed variation.

**R1-12. DOCUMENTED LIMIT (deviation 2).** Location: `scripts/analyze_feed_process.py:203-228`.
The drive logs have no send times (`here_index.jsonl` carries `received_t_mono` only), so
the lower lag bound uses the previous response's receipt, which is later than that
query's send by one round trip; the clip at zero is valid (a version cannot be returned
before its stamp). `L_low` is therefore too high by up to one round trip and `J` too low
by the same amount, on the placeholder only; the polling records carry `t_send_wall` and
use it. No reported number changes once polling exists. State the 1.2 s clock
uncertainty and the round trip together when the drive numbers are quoted.

**R1-13. NOT WORTH FIXING (deviation 6, staleness 300 s).** `inject_stale(300)` has no
snapshot 300 s before the 300 s decision (the ring starts at 10 s), so nothing is written
over [300, 360) s and controlled vehicles drive uncommanded for that interval; every
other staleness level has its snapshot (270, 240, 180, 120 s). The interval ends 540 s
before the flow window opens. Record it in the block B notes. `inject_stale(0)` equals
`published` exactly (seed 16), and the stale vectors are computed from the stored ring
snapshot by construction (attack point 3 holds by reading; the snapshots carry `x` and
`actions` for a recomputation).

**Attack points that hold (round 1), with the check made.**

- 3 (staleness injected): on `inject_stale(60)`, seed 16, all 37 stored snapshots give
  `greedy_actions(model, x)` equal to the recorded action vector, each applied exactly 60 s
  after the snapshot instant; `inject_stale(0)` equals `published` exactly.
- 4 (measured process; placeholder refused): `load_feed_parameters` raises `SystemExit`
  on the committed placeholder without `--allow-placeholder`; the header and summary
  carry the artefact sha256, the session and the flag. Re-analysing the real drive logs
  with `analyze_drive_logs` reproduces the committed artefact's sequences and `L_bar`
  exactly; the 6 drive-log tests that skip in a bare `git archive` mirror (the data path
  is relative to the repository's parent) pass when the data is present (14 of 14).
- 5 (lemma conditions): see the block A table above.
- 6 (held and pending counted): `drain_time` reads `held + pending`; `no_control` has
  `held = 0` on every sample and pending positive from 4,800 s in scenario (b). The
  definition itself is R1-4.
- 7 (threshold not the reward's): `src/sumo/drain.py` contains neither name; the 30 km/h
  threshold changes `here`'s scenario (b) drain time from 130 to 390 s. The pool is R1-5.
- 8 (censored kept): `drain_summary` keeps every seed in its arm's denominator; KM
  matches the textbook values on Kleinbaum's example (0.8571, 0.8067, 0.7529, 0.6902,
  0.6275, 0.5378, 0.4482; median 23) and returns `median None` when nothing drains.
- 9 (one clock start per scenario): `DRAIN` holds one `t_0` per scenario and
  `summarize_d` uses it for both arms.
- 10 (bound direction): `t_quantile` matches table values to 6 decimals (df 1, 2, 5, 10,
  14, 1000; levels 0.95 to 0.9995) and the Bonferroni quantile 4.3283 (df 14) matches an
  independent numerical integration (4.32834); in 500 random instances `rho_UCB >=
  rho_hat` always; `fit_gains` recovers a known 3-segment system exactly with
  `B[e, j]` the effect of `j` on `e`, divided row-wise by `a_e`.
- 12 (penetration): the floor rule controls exactly 2,343 / 1,562 / 781 / 312 of 3,124
  at p = 0.75 / 0.5 / 0.25 / 0.1, with consecutive controlled positions at most
  `ceil(1/p)` apart; writes go only to `controlled_on_segments()`. As the implementer
  says, the plan's exact count of commanded ids cannot hold: at p = 0.5, seed 16, 1,460
  of 1,562 controlled vehicles are ever commanded (63 are still held at the end, the
  rest never on a segment edge at a write tick).
- 13 (key): with the key held in a shell variable and never printed, `git grep` over
  d56ba5b, `git log -p main..d56ba5b`, the working tree's `results`, `outputs`, `tests`,
  `scripts`, `src`, `plans`, and the whole session scratchpad return no match;
  `phone/local.properties` is gitignored. No polling run has happened, so the polling
  outputs could not be checked.
- 14 (pairing): every block's arms are built over the same seed list; `paired_statistics`
  is arm minus reference with the sample standard deviation.
- 16 (timing): 6.1 to 7.4 s per published or injected 2,500 s episode, 7.6 to 7.8 s per
  rule episode, 31 to 41 s per drain episode, all within the budgets.
- Scenario files: `mainz_burst` departures run at 3,648 / 5,004 / 3,648 veh/h over
  [0, 900) / [900, 1,500) / [1,500, 3,000) s and 3,650 to 12,000 s; `mainz_surge` at
  18,000 veh/h for 1,200 s (6,000 vehicles), as the plan states.
- Baseline tests in the mirror: 179 passed, 6 skipped (the drive-log tests, which pass
  with the data present), one file per process.

**Mutation sweep (round 1).** 73 mutants on the number-producing code of `feed.py`,
`feed_episode.py`, `drain.py`, `gains.py`, `run_feed_experiments.py`,
`analyze_feed_process.py`, `feed_figures.py` and `mainz.py`, each run against the module's
own test file and the runner or figure tests, one file per process, in a separate
`git archive` mirror (`val/mutate.py`, results `val/mutants.jsonl`). 50 killed, 23
survived. Seven survivors are equivalent on the data the program uses, or change a
boundary that does not occur in practice: F5 (no query before t = 0; no version exists
before 60 s), E2 (floor instead of round for the late-set size; at most one vehicle), E5
(age measured from the snapshot tick instead of the stamp; stamps are whole seconds),
D4 (`<=` instead of `<` against a continuous speed), D11 (onset excludes a sample
exactly at the burst start), D12 (left-continuous CDF on a 0.5 km/h grid), A2 (psi*
margin inside the window negative instead of zero; the arg-max is unchanged).

**R1-14. BLOCKS (by the program's exit criterion: surviving mutants that change a
reported number or let a wrong result through).** Each line is the mutant, what it
would change, and the test that kills it.

- G4, `src/analysis/gains.py:131`, `a = |diag(coef)|/dt` instead of `-diag(coef)/dt`: a
  segment whose own deviation grows (coef_ee > 0) gets a positive `a_e`, and a
  certificate is issued where none is due. Test: a synthetic seed set with one growing
  self-response must give `a_hat_e < 0` and "no certificate".
- G7, `gains.py:184`, Bonferroni bound computed at the 95% level: the reported
  Bonferroni `rho_UCB` equals the 95% one. Test: `t` of the Bonferroni bounds equals
  `t_quantile(1 - 0.05/144, n - 1)` (4.3283 at n = 15) and `rho_UCB_bonferroni >= rho_UCB`.
- R1, `scripts/run_feed_experiments.py:545`, loss sign flipped in `summarize_b`: Fig. 2's
  F1 slopes change sign. R4, `:482`, slope `sum(y)/sum(x)` instead of
  `sum(xy)/sum(x^2)`: the slope changes on noisy data. Test: `summarize_b` on synthetic
  rows with known noisy losses gives the closed-form least-squares slope with a positive
  sign when flow falls.
- M4, `src/sumo/mainz.py:349`, `pending()` returns 0: the drain condition loses its
  pending term silently. Test: in the `no_control` SUMO run on seed 16 the trace's last
  `pending` is 32 (the committed row) and positive on the last samples.
- A3, `scripts/analyze_feed_process.py:121`, previous receipt instead of previous send;
  A6, `:112`, `L_up` from the send time: both move `L_bar`, `L_low` and `J`, hence the
  `D` values and the lemma condition, by one round trip. The synthetic records in the
  tests have zero round trip. Test: records with a 0.4 s round trip give `L_up` from the
  receipt and `L_lo` from the previous send, to the decimal.
- A4, `analyze_feed_process.py:133`, `min_availability_gap` without the lag change:
  changes `no_skip_in_replay`. Test: a two-version sequence with lags 40 and 30 s and a
  26 s gap gives 16 s.
- F9, `src/sumo/feed.py:113`, first stamp not aligned to its recorded second: moves
  every simulated availability phase relative to `psi*` and changes the fixed-phase arm.
  The test proposed under R1-10 (every simulated `T mod 60` equals its pair's recorded
  phase) kills it.
- E4, `src/sumo/feed_episode.py:386`, measured age shifted by 1 s in the per-second
  modes: Table 1's mean-age column. Test: `version_keyed(P_q=10, D=20)` on the fake env
  with the synthetic 60 s process gives a time-averaged age of 49.5 s.
- R5, `run_feed_experiments.py:658`, every hold sensitivity computed with the primary
  120 s: the 60 s and 300 s rows equal the primary. Test: a trace whose drain time
  differs between H = 60 and H = 300.
- R6, `:70`, scenario (b) clock at 900 s: Test pinning `DRAIN` to the plan (t_0 1,200 and
  1,500 s, caps 9,000 and 12,000 s).
- R7, `:771`, pulse check `>` instead of `!=`: a pulse that inserts fewer vehicles passes.
  Test: a pulsed record with 59 accounted against 60 is listed as mismatched.
- R8, `:567`, penetration loss at `p` instead of `1 - p` in the C summary. Test on
  `loss_points_one_minus_p`.
- R10, `:593`, pilot `and` replaced by `or`. Test: served below 3,650 with pending flat
  must not count as breakdown under the plan's rule (or replace the rule per R1-6).
- D13, `src/sumo/drain.py:34`, separation 1.0 instead of 1.5: changes which segments fall
  back. Test with `s2/s1 = 1.2` that must fall back (and the `s2 >= 3 s1` refusal of R1-5).

The surviving mutants were run against the test files listed in `val/mutate.py`; a
mutant could in principle be killed by another test file, but every listed file is the
one that tests the mutated function.

### Round 1 summary

BLOCKS 6 (R1-2, R1-4, R1-5, R1-7, R1-8, R1-14); SHOULD FIX 5 (R1-1, R1-3, R1-6, R1-9,
R1-10); DOCUMENTED LIMIT 2 (R1-11, R1-12); NOT WORTH FIXING 1 (R1-13), plus 7 equivalent
mutants.

Not checked in round 1: any polling output (no polling run exists, so the measured lag,
jitter and the key check on polling files wait for it); the experiments on more than
seed 16 (seeds 17 and 18 for attack point 1 and R1-7); block E end to end over 12 sites
and 15 seeds (only single-site probes); R1-7's cause on more than one seed; the Mainz
coordinates (plan open item 8); figures rendered from real summaries. The lemma's proof
was read and checked numerically on the drive stamps, not proved independently.

## Round 2: commits 4fe4710..750de8a (Amendment 1), mirror at 750de8a

Mirror: `scratchpad/dsrc_feed/mirrors/round2` (`git archive 750de8a`, the repository's
`.venv` linked); mutations in `mirrors/round2_mut`. The committed checkpoint
`mainz_here_best.pt` is used wherever the plan names the cap checkpoint, which does not
exist yet. Probes in `scratchpad/dsrc_feed/val/r2_*`.

### Coverage

| # | attack point | status | method |
|---|---|---|---|
| R1-1 | paper diff empty | done: R2-1 | `git diff main..750de8a -- paper`, `git log -- paper/` |
| a | no setSpeed under the default actuation; release | done: holds | grep of every actuation call; call counts per mode in SUMO |
| b | counters summed every step, every path | done: holds | env counters against an independent per-step count |
| c | A3 drain rule and A4 threshold on real traces | done: rules behave as the traces show; R2-5 | the runner's own `--block D --pilot` under the cap |
| d | B and C pair against the cap reference and refuse otherwise | done: R2-2 | read `summarize`/`main`; key paths against the evaluator header |
| e | retraining and evaluation CLIs | done: R2-4 | read guards and result fields; retraining tests |
| f | A8 noise floor and resolved flag | done: R2-3 | one-vehicle null pulse under the cap; runner's `_noise_floor` |
| i1-i4 | the implementer's four choices | done: all four accepted | reading plus the pilot |
| M | R1-14 survivors, implementer's 7, new mutants | done: 16/16, 7/7, 16/19 killed | `val/mutate2.py` in `round2_mut` |

### Findings

**R2-1. SHOULD FIX (process, no reported number; R1-1 is not resolved).** Location:
commit `4fe4710` (the Amendment 1 commit). `8582cb7` restores `paper/architecture.tex`,
and `4fe4710`, made 58 s later, deletes it again, so `git diff main..750de8a -- paper`
still shows `paper/architecture.tex | 18 -` and the file is absent from `750de8a`'s tree
(`git ls-tree 750de8a -- paper/architecture.tex` is empty; `main` has blob `68d48f5`).
`git status` shows no change for the path, so the working tree also lacks it: the
restore went into the index while the author's working-tree deletion remained, and the
next commit staged that deletion again. Fix: restore the file on the branch with a
commit that stages only that path (`git checkout main -- paper/architecture.tex`, which
also writes it back to the working tree, then commit that path alone). The author can
then delete it in the working tree again as an uncommitted edit; every later commit
must stage explicit paths.

**Attack points (a) and (b) hold.** Seed 16 to 18 under the cap with the committed
checkpoint (`val/r2_actuation_probe.py`): the per-tick `published` path and the `step()`
path (`train_mainz_src.run_episode`) give identical flow and arrivals (3,635.56 / 3,451.11
/ 3,606.67 veh/h; 1,936 / 1,855 / 1,919), 0 colliding vehicles and 0 teleports, and only
`setMaxSpeed` calls (29,605 / 30,142 / 29,840); `inject_delta(0.25)` and `inject_stale(60)`
also call only `setMaxSpeed` and collide 0 times; `no_control` makes no actuation call.
Legacy `set_speed` reproduces 3,751.11 veh/h with 152 colliding vehicles and 76
teleports, and the env's counters equal an independent count taken on every one of the
5,040 steps. `simulationStep` is called only in `MainzEnv._step`, which both `_advance`
and `advance_ticks` use, so warm-up, `step()`, the per-tick path and training all count.
`setSpeed` remains only under `actuation="set_speed"` and in standalone
fundamental-diagram scripts outside the program. Controlled vehicles are never released
in any controlled arm (as before with `setSpeed`); `release` restores the maximum speed
read before the first cap and is reached only from no-control paths, where nothing is
capped. Pulse vehicles: identical to the unperturbed run through 1,200 s (first
difference at 1,210 s), pulse accounting exactly 60, no vehicle in teleport.

**R2-2. BLOCKS (attack point d).** Location: `scripts/run_feed_experiments.py:443-473`
(`committed_references`, `_reference_check`), `:927-935`, `main`. Blocks B and C read
`mainz_paired_seeds_cap.jsonl`, but nothing refuses to run or to summarise when it is
missing or was produced under another checkpoint or actuation: `main` checks only that
the block's checkpoint exists, and the summary records `exists`, `same_checkpoint` and
`same_actuation` as flags. With the file missing, every `paired_vs_published` is `None`,
`summarize_b` has no loss points and `fig_flow_consequence` silently drops every row;
with a file produced from the old checkpoint (for example `--out-name
mainz_paired_seeds_cap` without `--checkpoint-suffix _cap`), the pairing is computed
against the wrong arm and only a flag says so, which neither the figure nor Table 1
reads. The key path the check reads matches the evaluator's header
(`config.checkpoints.here.sha256`), so the flag itself works. Fix: before running block B
or C, and again in `summarize`, raise unless the reference file exists, its header's
`actuation` equals the block's, and its `here` checkpoint sha256 equals the block
checkpoint's; have `fig_flow_consequence` refuse a summary whose `reference_check` is not
all true. Tests: a missing file, a set_speed file and a different-sha file each refuse.

**R2-3. SHOULD FIX (attack point f).** Location: `scripts/run_feed_experiments.py:861-877`
(`_noise_floor`), `:815-821`, `src/analysis/gains.py:193`. The resolved flag gates nothing:
`certificate`, `rho_by_level`, `p_min` and Table 1 depend on `rho_UCB < 1` alone. And the
flag is not a like-for-like null: it compares the mean of maxima over about 54 samples
at each pulsed segment with a single-sample 95th percentile pooled over all 12 segments,
whose noise levels differ (the exit segments are the noisiest). Evidence, seed 16, p = 1,
cap (`val/r2_pulse_null.py`): a ONE-vehicle pulse at site 11 has self-peak 0.0227 against
its pooled p95 0.0146 (ratio 1.55) and is "resolved" by the runner's rule, the same ratio
as a 60-vehicle pulse at site 7 (0.0354 against 0.0227, 1.56; site 3: 0.0288 against
0.0211). Against a per-segment null the 60-vehicle pulses do stand out: the one-vehicle
run's maximum at segment 7 is 0.0085 and at segment 3 0.0060, 4.2 and 4.8 times below the
pulses' peaks. Noise alone did not produce a certificate here (on the one-vehicle run all
12 `a_e` are positive but `rho_hat = 2.8`), so no wrong certificate was observed. Fix:
per site `e`, compare the pulse's peak self-deviation with the same statistic (the
maximum over the same window length at segment `e`) from a null pair per seed (a
one-vehicle pulse at the exit's last edge against the unperturbed run); call a site
resolved when its peak exceeds the null's 95th percentile across seeds; mark the
certificate unavailable at a level where fewer than a stated share of sites (for
example 10 of 12) are resolved, and carry that into Table 1.

**R2-4. SHOULD FIX (attack point e, minor).** Location: `scripts/train_mainz_src.py:273-284`.
The retraining cannot overwrite a committed checkpoint (`--save-as` refuses if either
output exists, tested; without it the output goes under `outputs/`), and the evaluator
refuses to replace an existing `OUT_NAME` without `--overwrite` (tested, and the committed
`mainz_paired_seeds` is refused by default). The evaluator's header records `actuation`
and `checkpoint_suffix` with each checkpoint's sha256. The training `result.json` records
the actuation, the command line and the configuration, but not the train, validation and
test seed lists (module constants 1-10, 11-15, 16-20), the git commit or the saved
checkpoint's sha256; the published protocol's seed split is therefore not in the
artefact that claims it. Fix: add `train_seeds`, `validation_seeds`, `test_seeds`,
`_provenance()` and the checkpoint sha256 to `result.json`. Two further gaps the CLIs
cannot close themselves: the docstring's `> results/checkpoints/mainz_here_cap_training.log`
redirection truncates that log before `--save-as` refuses, and the evaluator accepts
`--out-name mainz_paired_seeds_cap` with the old checkpoints (`--checkpoint-suffix` empty);
R2-2's refusal is what catches the second.

**Attack point (c): the A3 drain rule and the A4 threshold on real traces.** Run through
the runner itself (`run_feed_experiments.py --block D --pilot`, then `--block D --seeds 16
--resume` for scenario (a)), cap actuation, committed `here` checkpoint, results under
`val/r2_pilot/`. 34 episodes, 0 collisions, 0 teleports, actuation `max_speed` throughout.

- Network threshold (A4), 30 sustained episodes (both arms, seeds 16-30, 300-2,500 s,
  79,534 samples): `s1 = 25.5`, `s2 = 50.5`, `v_jam = 19.0 km/h`, not refused. Per segment:
  segments 0-4 and 6 fall back (`s2 < 1.5 s1`); own thresholds 20.1, 19.8, 17.4, 10.8,
  13.6 and 13.8 km/h at segments 5, 7, 8, 9, 10 and 11. The surge scenario's own
  no-control fit (knots 3.5 and 25.0) is refused with the recorded reason; the burst
  scenario's own fit gives 22.0 km/h.
- Scenario (b), burst, seed 16: both arms censored under every rule (network, network
  per segment, scenario, 30 km/h) and every hold (60, 120, 300 s). The traces agree:
  `no_control` is jammed on 90.6% of samples after `t_0`, with a jam spell of 3,690 s and
  pending from 4,610 s to 154 at the cap; `here` serves 3,574.7 veh/h, below the 3,650
  veh/h background, and its gate queue is non-zero from 7,140 s to the cap (63 held at
  the end). Onset 1,710 s (no control) and 2,130 s (`here`); jammed segment-minutes 285
  and 115 under 19.0 km/h. Pilot decision: no control breaks down (jam spell of 3,690 s
  >= 120 s), the controlled arm does not drain, so "rebuild with a 3,600 veh/h
  background". A3 and A5 give the answer the traces show.
- Scenario (a), surge, seed 16 (network threshold, H = 120 s): `no_control` drains at
  5,450 s, `here` at 5,390 s (60 s earlier); H = 60 s: 5,390 and 5,330; H = 300 s: 5,630
  and 5,570; 30 km/h: 5,460 and 5,420. Under the network per-segment thresholds the order
  reverses: `no_control` 5,010 s, `here` 5,280 s. In round 1, under `setSpeed` with collision
  teleports and a 20.25 km/h threshold, `here` drained 640 s before no control on the same
  seed; with the cap the difference is 60 s, and its sign depends on the threshold rule.
  One seed, the old checkpoint: no claim follows; the paper must report the sign under
  each rule.

**R2-5. SHOULD FIX (plan-level; the orchestrator decides).** Location: plan D13 and the
section 5 pilot rule, `scripts/build_mainz_scenario.py` scenario (b) parameters,
`scripts/run_feed_experiments.py:pilot_decision`. Scenario (b)'s background (3,650
veh/h) and fall-back (3,600 veh/h) sit in a band computed from the `setSpeed` served
flows (no control 3,535, controlled 3,765 veh/h). Amendment 1 changes the controlled arm
and does not revise the band. Under the cap, the controlled arm with the committed
checkpoint serves 3,635.6 / 3,451.1 / 3,606.7 veh/h on seeds 16-18 against no control's
3,722.2 / 3,553.3 / 3,564.4, and 3,574.7 veh/h on the burst scenario. The band may not
exist under A1, and the pilot's fall-back of 3,600 veh/h lies above two of these four
controlled flows (3,451.1 and 3,574.7). Fix: after the cap retraining and `mainz_paired_seeds_cap`,
recompute the band from that evaluation's mean served flows and set scenario (b)'s
background inside it. If the cap evaluation puts the controlled arm's served flow at or
below no control's, there is no band and scenario (b) cannot test "the controlled arm
drains and no control does not"; record that as a result, not a pilot failure. Run the
pilot only with the cap checkpoint.

**The implementer's four choices.**

1. Placeholder wrap: correct and safe. `allow_wrap` is true only for a placeholder under
   `--allow-placeholder`; without it the placeholder is refused (its session spans
   1,526 s of gaps). With it, seeds 16-30 wrap once or twice, and no replayed stamp's
   second of the minute differs from its recorded one (0 mismatches). A synthetic
   130-pair non-placeholder session never wraps (87 covering starts). Wrapped replays
   are counted in block A's summary.
2. A spell open at the cap counts with its observed length: a defensible reading of A3.
   The trace ends at the cap, so a longer spell cannot be observed, and counting an open
   spell as long would censor runs for a single-sample dip at the final sample (the
   round-1 burst `here` run was below threshold at 12,000 s). It did not decide any pilot
   run: both burst runs end inside spells far longer than H. Record in the drain
   artefact which drained runs end inside an open spell, so a reader can see the cases.
3. "Previous snapshot": correct for the injections, and consistent across modes. Under
   `inject_delta` the newest version any scanned vehicle applies is the decision's own
   whenever one on-time vehicle is scanned, so the two definitions coincide there. In the
   rule arms it is the share not on the newest applied version, which is at least the
   plurality disagreement; it is not used on Fig. 2's axis.
4. Block D's own 30 sustained episodes: a correct reading of A4 (block A traces only
   `published`; `no_control` sustained episodes exist in no other block). They are
   excluded from the drain summaries (the drain loop runs over `DRAIN`'s scenarios only)
   and keyed separately (`variant = "mainz"`).

**Round-1 fixes re-checked.** R1-2: on seed 16, delta 0.1 under the cap, 2,976 late
vehicle-decisions, of which 77 differ from the on-time command and all 77 because
`a_{k-1}[e] != a_k[e]`; none uncommanded, none on another edge's speed. R1-3: DELTAS
(0.05, 0.1, 0.25, 0.5); the fit and Fig. 2(a) use the measured `previous_fraction_mean`.
R1-4, R1-5, R1-6: above. R1-8 and R1-9: pulses identical through 1,200 s, accounting
exactly 60 with the in-teleport term. R1-10: above (choice 1).

**Mutation sweep (round 2).** In `mirrors/round2_mut` (`val/mutate2.py`, results
`val/mutants2.jsonl`): all 16 round-1 survivors (R1-14), adapted to the new code, are
killed; the implementer's 7 extra mutants (`impl_mut/mutate_extra.py`), re-run in my
mirror, are killed; of 19 new mutants on the Amendment 1 code (actuation defaults in the
env, the episode and the training CLI; per-step counter summation; the spell length and
the waiting spells in A3; the network window and arms in A4; the overwrite guards; the
recorded actuation; `setSpeed` in the cap path; the in-teleport term; the A6 reference
version) 16 are killed. The 3 survivors are N10 (`same_actuation` forced true), N11 (an
evaluation file without an actuation read as `max_speed` instead of `set_speed`) and N13
(the noise floor at the 50th instead of the 95th percentile): each lives in a flag that
gates nothing, and each is killed by the tests R2-2 and R2-3 propose.

### Round 2 summary

BLOCKS 1 (R2-2); SHOULD FIX 4 (R2-1, R2-3, R2-4, R2-5); DOCUMENTED LIMIT 0; NOT WORTH
FIXING 0. Tests at 750de8a in the mirror: 301 passed, 0 skipped, one file per process
(the pre-existing `test_mainz_port.py` and `test_mainz_paired_evaluation.py` included).
The round-1 code findings are fixed and their mutants killed; the remaining items are a
missing refusal (R2-2), a flag that is computed but gates nothing and is not a
like-for-like null (R2-3), provenance fields (R2-4), a plan parameter that Amendment 1
made stale (R2-5), and the paper file again (R2-1).

Not checked in round 2: anything with the cap checkpoints (they do not exist); the
80-episode retraining (not run, as instructed); the cap evaluation file; blocks A, B, C
and E beyond seed 16 probes; any polling output; scenario (a) and (b) beyond seed 16.

## Round 3: commits f1dafe9..c7dd09d (Amendment 2), mirror at c7dd09d

Mirror: `scratchpad/dsrc_feed/mirrors/round3` (`git archive c7dd09d`, the repository's
`.venv` linked); mutations in `mirrors/round3_mut`. The committed `here` checkpoint stands
in for the cap checkpoint, which does not exist yet. Probes in `scratchpad/dsrc_feed/val/r3_*`.

### Coverage

| # | attack point | status | method |
|---|---|---|---|
| R2-1 | paper diff empty | done: resolved | `git diff main..c7dd09d -- paper` is empty; `ls-tree` has blob `68d48f5`; the deletion is unstaged (` D`) |
| a | B2 null and gate on the right samples; gate against pooled regression | done: R3-2, R3-3 | 12 sites x {60, 1} pulses + base, seed 16, p = 1, cap; the runner's own `site_resolution`; `fit_gains` with and without unresolved sites |
| b | B1 bypasses | done: R3-1 | synthetic block and reference files through the runner's `summarize` and `main` |
| c | B3 provenance | done: holds | a 2-episode training under `--save-as` and `--log` in the mirror |
| B5 | open spells | done: holds | read; mutants V7 and B5-open |
| d | anything that makes a reported number wrong | done: R3-1, R3-3, R3-4 | whole-program read of the paths each block's numbers take |
| M | mutation kills, re-verified | done: R3-4, R3-5 | all harnesses re-run with bytecode caching off, without `-x`, in a git mirror |

### Findings

**R2-1 is resolved.** `git diff main..c7dd09d -- paper` is empty, `c7dd09d`'s tree has
`paper/architecture.tex` (blob `68d48f5`, as on `main`), and `git status` shows the author's
deletion as an unstaged working-tree change. `f1dafe9` touches only the plan.

**Attack point (c) holds (B3).** A 2-episode training run in the mirror
(`--save-as <scratch>/t --log <scratch>/t.log`, 600 s episodes) writes `result.json` with
`train_seeds` 1-10, `validation_seeds` 11-15 and `test_seeds` 16-20, which are the constants the
loops read (the log shows episodes on seeds 1 and 2); `checkpoint_sha256` equals the sha256 of
the saved `t_best.pt`; the log keeps a pre-existing line and is appended to; a second run with
the same `--save-as` is refused and leaves the log unchanged (17 lines before and after).
`commit` and `dirty` are `None` only because a `git archive` mirror has no `.git`. The
provenance is sampled once at the start of training, not compared at the end as the
experiment runner does; NOT WORTH FIXING for an 80-episode run made from a committed tree.

**B5 holds.** `open_spell_at_cap` returns the jam or waiting spell whose last sample is the
trace's last sample at or before the cap, and `summarize_d` lists drained runs that end inside
one; mutants V7 (spell length) and B5-open are killed.

**R3-1. SHOULD FIX (attack point b; B1 is implemented as Amendment 2 words it).** Location:
`scripts/run_feed_experiments.py:472-516` (`_reference_check`, `reference_identity`,
`reference_problems`), `:1043-1060` (`summarize`), `run_block`. Tested with synthetic block and
reference files through the runner's own `summarize` and `main` (`val/r3_b1_probe.py`):

| case | refused? | what the summary pairs |
|---|---|---|
| `main --summarize-only`, reference missing | yes | - |
| block resumed under a different checkpoint: two episodes from sha A, a later header with sha B = the reference's | no | all three seeds, two of them run on A |
| reference header matches; its rows are for seeds 1-3, the block's 16-18 | no | nothing (`paired_vs_published` None), so Fig. 2 would draw an empty panel |
| reference header matches; it was produced at `--duration-s 1200` | no | the block's 2,500 s flows against 1,200 s flows |
| `--smoke` on a real block, reference missing | no, by design | the summary records the failing check and the figure refuses it |

`_reference_check` reads only the block's last header, and episode records do not carry the
checkpoint, so a `--resume` under another `--checkpoint` or `--actuation` mixes episodes without
notice (in every block, not only B and C). The identity check compares actuation and checkpoint
but not the reference's seeds, duration, window or step, which the evaluator's header records.
The evaluator's defaults equal the blocks' (2,500 s, 900 s, 0.5 s, seeds 16-30), so the last two
rows need a non-default evaluator run. Fix: (i) `run_block` refuses to resume when the new
header's checkpoint sha256 or actuation differs from any earlier header in the JSONL, and
`summarize` refuses a JSONL whose headers disagree; (ii) `reference_problems` also requires the
reference's `seeds` to contain the block's seeds and its `duration_s`, `window_start_s` and
`step_length_s` to equal the block's.

**R3-2. SHOULD FIX (attack point a; block E).** Location: `scripts/run_feed_experiments.py`
`summarize_e` (`baseline_trace`). Block E reuses its unperturbed baselines from block A's
`published` traces and block C's `penetration_p*` traces without checking that those blocks
ran the same checkpoint and actuation; nothing compares them. Since A8, a pulsed run and its
own baseline are identical on every sample before the pulse starts: on seed 16, p = 1, under
the cap, all 24 pulsed runs (60- and one-vehicle, 12 sites) equal the baseline on every sample
before 1,200 s. That is an exact, free identity test. Fix: in `summarize_e`, refuse a
(level, seed) whose pulsed and baseline traces differ on any sample before the pulse start, and
record the count of such pairs checked.

**Attack point (a): B2's null is like for like, and the gate can pass while the regression pools
unresolved sites.** `peak_self` computes the same statistic for the 60-vehicle pulse and its
null: the maximum of the deviation at the pulsed segment from the pulse start to 1,800 s,
against the same baseline (mutants V1, null at the next segment, and V2, both statistics taken
from the noise window, are killed). On seed 16, p = 1, cap, one seed each
(`val/r3_sites_probe.py`): 10 of 12 sites are resolved (mean pulse peak against null p95,
which with one null seed is the null's own peak); the two unresolved sites are the exit segments
10 (0.0084 against 0.0084) and 11 (0.0162 against 0.0227). The gate passes at exactly 10. The
regression then still stacks all 12 sites' runs: `rho_hat` is 2.34 with all 12 and 2.60 with
the 10 resolved sites only, so pooling the unresolved runs lowered `rho_hat` by about 10%,
toward a certificate. Also, for segments 2 to 11, the run pulsed at that segment supplies only
3% to 31% of the sum of squares that determines its own `a_e` (96% and 67% at segments 0 and 1);
the rest comes from pulses elsewhere passing through and from divergence. On the 12 one-vehicle
nulls the same regression gives `a_e` of 0.015 to 0.081 and `rho_hat` 1.24, against 0.002 to 0.053
and 2.34 on the 60-vehicle pulses.

**R3-3. SHOULD FIX (attack point a).** Location: `summarize_e`, the `resolution["available"]`
branch. Fix: when the gate passes with fewer than 12 sites resolved, also fit the gains on the
resolved sites' runs only, report both `rho_UCB`, and issue the certificate only when both are
below 1 (on the seed-16 probe the two differ by about 10%). State in the paper that the gains
are a pooled regression over all pulses, not per-site impulse responses; for most segments the
self-gain is estimated mostly from runs pulsed elsewhere. One seed: the size of the difference
is indicative only.

**Mutation verification (round 3), and two hazards in how kills were counted.** All harnesses
were re-run in `mirrors/round3_mut` with Python's bytecode cache disabled
(`PYTHONDONTWRITEBYTECODE=1`, every `__pycache__` removed), without `-x` so every failing test
is listed, and with the mirror made a git repository for the training test. Results:

| set | killed | survived |
|---|---|---|
| implementer's 17 (Amendment 2, incl. N10, N11, N13) | 17 | 0 |
| round-1 survivors (R1-14), implementer's harness | 16 | 0 |
| my round-2 list (35 applicable at c7dd09d) | 35 | 0 |
| implementer's 7 round-2 extras | 7 | 0 |
| my 8 new (V1-V8: null at the wrong segment, null in the noise window, max for mean, p_min ignoring the gate, reference existence, the equal-x guard, open-spell length, sha of the wrong file) | 7 | 1 (V3) |

Every kill is by a test that names the mutated behaviour.

**R3-4. BLOCKS (exit criterion: a surviving mutant that changes a reported number).**
Location: `scripts/run_feed_experiments.py:844` (`site_resolution`). Mutant V3, `mean =
float(np.max(pulses))` in place of `np.mean`, survives every test. It decides which sites are
resolved, hence whether a level's certificate is reported and whether that level can be `p_min`;
with 15 pulse seeds the maximum is at or above the mean, so the mutant makes the gate easier to
pass. Fix: a unit test with a site whose 60-vehicle peaks have a mean below the null's 95th
percentile and a maximum above it; the site must be unresolved.

**R3-5. SHOULD FIX (process; how mutation results are produced).** Two ways a harness reported
kills that were not kills:
1. Stale bytecode. Python reuses a `.pyc` when the source's size and whole-second modification
   time match. A same-size mutant (for example `B5-open`, `==` to `!=` in `drain.py`) whose test
   file runs in under a second is restored within the same second, so the mutated bytecode stays
   in place and later runs that import the file execute the mutant. In my first round-3 pass this
   made V1-V6 all fail a block D test, and V3 was reported killed; with the cache off V3
   survives. A false survival is possible the same way.
2. A test that fails without a mutation. `test_one_600_s_training_episode_under_the_cap` asserts
   a 40-character commit, so in a `git archive` mirror it fails unmutated and "kills" every
   mutant listed against it (B3-sha, B3-seeds, B3-log, V8, N3, N17 in my first pass). With a git
   repository in the mirror the test passes unmutated and the six kills are real.
Fix: every harness sets `PYTHONDONTWRITEBYTECODE=1` and removes `__pycache__` before the sweep,
runs the listed test files unmutated first and refuses to start if any fails, and records every
failing test, not only the first. The implementer's reported kills could carry either hazard;
the table above is the re-verified count.

**Tests at c7dd09d.** 327 tests in 14 files, one file per process, with the bytecode cache off:
all pass, given a git repository (in a bare `git archive` mirror the one training test above
fails on its commit assertion, an environment artefact). The pre-existing
`test_mainz_port.py` (12) and `test_mainz_paired_evaluation.py` (55) pass unchanged.

### Round 3 summary and the exit criterion

BLOCKS 1 (R3-4); SHOULD FIX 4 (R3-1, R3-2, R3-3, R3-5); DOCUMENTED LIMIT 0 (the pooled-regression
reading is folded into R3-3); NOT WORTH FIXING 1 (training provenance sampled only at the start).
R2-1 is resolved; B1, B2, B3 and B5 are implemented as Amendment 2 states them; B4 has no code.

**The exit criterion is not met, by one item.** The criterion is: no surviving mutation that
changes a reported number or produces a plausible-but-wrong result, and no plan requirement
unimplemented. Status:

- Plan requirements: none unimplemented in code. Steps S1-S11, Amendment 1 (A1-A10; A11 is paper
  text) and Amendment 2 (B1, B2, B3, B5) are implemented and tested. What remains is the
  experiment stage, which is not code: the polling run (budget confirmation, plan open item 1),
  the cap retraining and `mainz_paired_seeds_cap`, B4's band, then blocks A-E.
- Surviving mutations that change a reported number: one, V3 (R3-4). One unit test kills it.
- Plausible-but-wrong results without a surviving mutant: R3-1 (a resumed block mixing
  checkpoints or actuations, and a reference from another protocol, are accepted), R3-2 (block E
  baselines from another checkpoint are accepted) and R3-3 (the 10-of-12 gate lets two unresolved
  sites into the regression, about 10% lower `rho_hat` on the seed-16 probe). R3-1 and R3-2 need an
  operator to resume or reuse across configurations; R3-3 is in the design as Amendment 2 states
  it. None is a mutation, so none fails the criterion's text, but each is a way a wrong number
  could reach the paper without an error message. The fixes for R3-1 and R3-2 are refusals of a
  few lines each.

Exactly what remains for exit: the R3-4 test. Recommended before the experiments run: R3-1 (i)
and (ii), R3-2, R3-3, and R3-5 for any further mutation sweep.

Not checked in round 3: anything that needs the cap checkpoints or the cap evaluation; the
retraining itself; block E over 15 seeds and 5 levels (one seed at p = 1 only); any polling
output.
