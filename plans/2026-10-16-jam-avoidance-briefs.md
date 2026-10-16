# Jam-avoidance program: agent briefs and paper outline

Copied out of the session scratchpad on 2026-10-16 at the author's `pause`, so the loop can resume without it. The implementer and validator briefs are the exact text the agents were (or will be) given; the scratchpad paths inside them are where mirrors go and need not exist in advance.

---

## Implementer brief: dsrc jam-avoidance program

You implement the plan at `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc/plans/2026-10-16-jam-avoidance-plan.md`.
Read it in full first. It is authoritative: implement what it says, and if something in it cannot be built as
written, stop and report the conflict instead of choosing a different design. You do not argue validator
findings; the orchestrator adjudicates them and sends you confirmed fixes.

## Repository and environment

- Repository: `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc`, branch `jam-avoidance`. Work in place.
- Environment: `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc/.venv/bin/python` (Python 3.12, sumo,
  traci, pytest). Never build another environment.
- Tests: `.venv/bin/python -m pytest <one file> -q -p no:cacheprovider`. One file per process. NEVER pass
  `-n`; parallel test workers hung this laptop on 2026-10-12. Check `memory_pressure` before a SUMO run.
- The `paper/` directory has uncommitted author edits. Never read them as your spec, never edit them, never
  stage them.

## Program-specific rules

- Do not edit `scripts/poll_here_feed.py` or `scripts/analyze_feed_process.py`: a polling session scheduled for
  22:00 tonight runs the first, and the orchestrator runs the second.
- Every controlled arm must record colliding-vehicle and collision-teleport counts per episode, and a smoke run of
  each controlled arm must show zero. Advice reaches SUMO only through `setMaxSpeed` (`_apply` in
  `src/sumo/mainz.py`), never `setSpeed`.
- Drive data for the camera error model: `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908/run_*/`
  (`video.avi`, `video_index.jsonl`, `metadata.jsonl`). It exists. Read it, never write into it. If the plan has you
  fetch the official `yolov8n.pt` release weights, record the SHA-256, keep the file out of git (gitignored path),
  and fetch it once.
- Never select a model, a threshold or a "best arm" on seeds 16-30.

## Test-driven

Every module gets unit tests (concrete input and output) and sanity tests (expected behaviour over a range of
inputs). Write the test, see it fail for the right reason, then implement.

## Experiments

You do NOT run the experiments. Smoke runs on a tiny slice (one seed, a short episode, a few polls if the plan
calls for a polling smoke) are allowed and expected, to prove each entry point runs end to end on real inputs.
Every real-data entry point (the drive logs at `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908/`,
the checkpoints in `results/checkpoints/`) must have one real smoke run before you report done; a hand-built
fixture is not enough.

## Secrets

The HERE API key is in `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc/phone/local.properties`
(property `here.apiKey`). Read it at run time only. Never print it, never write it to a file, log, saved URL,
test fixture or commit. Strip it from any URL before the URL is stored, and test that.

## Committing

- Commit as soon as each step's tests pass; do not accumulate a large uncommitted diff.
- Stage explicit paths only (`git add <paths>`). NEVER `git add -A`, `git add .` or `git commit -a`: the paper
  edits must stay unstaged, and a validator may be reading the tree.
- Every commit message ends with this exact line:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- Do not push.

## Mutation and destructive checks

Never run mutation sweeps or anything that edits source files in the live repository. If you need one, build a
mirror: `mkdir -p $M && git -C /Users/ankit_nash/Desktop/ankit_summer_2026/dsrc archive HEAD | tar -x -C $M`
with `$M` under
`/private/tmp/claude-502/-Users-ankit-nash-Desktop-ankit-summer-2026/4ea4312b-6fea-47ac-a8f1-821301808d0b/scratchpad/dsrc_jam/mirrors/`,
and symlink `.venv`, `data` and `results/checkpoints` into it if needed.

## Task list

If you keep a task list with `~/.claude/hooks/todo_gate.py`, pass `"role":"implementer"` and
`"session_id":"4ea4312b-6fea-47ac-a8f1-821301808d0b"`; never `"main"`.

## Report

Reply with: the commits (SHA and one line each), the test files and their pass counts, the smoke runs done and
what each showed, anything in the plan you could not implement as written, and anything you are unsure about.

---

## Validator brief: dsrc jam-avoidance program

You are the independent validator for one research program. You will be kept alive for up to three rounds;
each round I will tell you what changed since the last one. Your job is to find what is wrong: code that does
not do what the plan says, a reported number that would come out wrong, a test that would pass against a
broken implementation, a real-data path that was never exercised. For every finding, propose the fix.

## What you audit against

- The plan (authoritative): `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc/plans/2026-10-16-jam-avoidance-plan.md`.
- The code on branch `jam-avoidance` of `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc`, from the commit
  I name in each round's message. Compare with `main` to see what the program added
  (`git -C <repo> diff main...<sha> --stat` is fine to run; it is read-only).
- The real references, never your own re-implementation of them: the drive data at
  `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908/run_*/` (video.avi, video_index.jsonl, here/*.json),
  the checkpoints in `results/checkpoints/`, SUMO itself. Self-consistency is not correctness: a computation
  that agrees with a second computation you wrote proves little.

## Isolation

Never write to, and never run git commands that change state in, the live repository. Build a mirror for
anything that edits or mutates code:
`M=/private/tmp/claude-502/-Users-ankit-nash-Desktop-ankit-summer-2026/4ea4312b-6fea-47ac-a8f1-821301808d0b/scratchpad/dsrc_jam/mirrors/round<N>;
mkdir -p $M && git -C /Users/ankit_nash/Desktop/ankit_summer_2026/dsrc archive <sha> | tar -x -C $M`, then
symlink `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc/.venv`, `data` and `results/checkpoints` into it as
needed. Run tests with the repo's venv: `<repo>/.venv/bin/python -m pytest <one file> -q -p no:cacheprovider`.
One file per process; NEVER `-n` (parallel workers hung this laptop). Check `memory_pressure` before a SUMO run.

Never read, print or store the HERE API key. If you need to test the polling client, test it with the network
call stubbed; do not make HERE requests.

## What to attack first (the checks most likely to catch a wrong reported number)

The plan's own validator attack points section is the full list; these come first:

1. The correction never raises a vehicle's speed above the plan's speed, in any arm, on any tick; and the
   one-step-down output actually reaches SUMO (through `setMaxSpeed`) and changes vehicle speeds.
2. The camera model is applied: range limit, detection probability by distance and distance noise change the
   local density estimate the non-oracle arms see; the oracle arm reads true density and the others never do.
   Check the error-model parameters against the re-processed drive frames themselves, not against the code's
   own summary of them.
3. The feed lag is actually injected into the plan and into the network-wide lookahead, and the lag sweep's
   levels are what the runner records.
4. Zero colliding vehicles and zero collision teleports in every controlled arm (a nonzero count created a
   false +230 veh/h result in the previous program).
5. Selection of the learned correction's model, of every tuned threshold and of the "best model" for the paper
   never reads seeds 16-30.
6. The jam metrics: the jam threshold is the speed threshold, not the reward's density threshold; vehicles held
   at the entry gate are counted; the clock starts at the same instant in every arm; runs that never jam or
   never drain are censored and kept, never dropped.
7. The tests: which assertions would still pass against a broken implementation? Run mutations in your mirror on
   the computations that produce reported numbers, and report surviving mutants that would change a number.
8. The HERE API key never reaches disk, logs, saved URLs, fixtures or commits.

## Findings

Write your findings to `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc/plans/2026-10-16-jam-avoidance-validation.md`.
Create it at the start of each round with a coverage table (one row per attack point: not started / in progress /
done, and the method). Append each finding the moment you have evidence for it. This file is the record if you
are stopped. Each finding has: id (R<round>-<n>), bucket (BLOCKS / SHOULD FIX / DOCUMENTED LIMIT / NOT WORTH
FIXING), location (file:line), evidence (command and output, or mutant and result), and the proposed fix.

Exit criterion for the program: no surviving mutation that changes a reported number or produces a
plausible-but-wrong result, and no plan requirement unimplemented. Do not manufacture findings to fill a round; a
short round is a fine outcome.

If you keep a task list with `~/.claude/hooks/todo_gate.py`, pass `"role":"validator"` and
`"session_id":"4ea4312b-6fea-47ac-a8f1-821301808d0b"`; never `"main"`.

Reply each round with: counts per bucket, one line per finding, the coverage table, and what you could not check.

---

## Outline: the jam-avoidance rewrite (6-page RA-L letter)

Numbers marked [M] come from the experiments; nothing below is a result yet.

## The story, one line per step
1. A free-flow network can jam from a burst or a blocked lane, and under a background demand the
   jammed network serves less than arrives, so the jam does not drain (point queue, Prop. 3).
2. Preventing the jam is therefore worth more than draining it, and prevention needs a warning
   before onset.
3. A vehicle can read a public traffic feed for the whole network, and SRC's policy turns that feed
   into a speed plan with no coordinator. The feed reports speed, and late: 34-110 s publication to
   receipt on the road, [polling: L_low-L_bar], plus a 60 s publication cycle.
4. Speed stays near free flow while density climbs, so the feed's warning is short; if it is shorter
   than the feed's delay plus its cycle, no feed-only controller can act before onset (Prop. 1).
5. A dashboard camera measures headway and leaders in range, which is local density, and density
   rises earlier [M: W_cam vs W_feed].
6. So each vehicle keeps the shared feed plan and may lower its own shown speed by one step from its
   camera (bounded, never above the plan, so the shared plan keeps vehicles in agreement).
7. On the Mainz network, against no control and a network-wide feed lookahead [M], with the
   camera's errors measured on the drive frames.
8. Limits: simulation for flow; camera scale unmeasured (bias swept); one network.

## Sections
- **Abstract.** Steps 1-7 with the three headline numbers [M].
- **I. Introduction** (funnel, C1): phenomenon -> sudden traffic jam named -> controllers that slow
  vehicles -> SRC (mechanism, number, venue) -> its coordinator and, without one, a lagged speed feed
  -> question: can a commodity vehicle avoid jams the feed reports too late? -> answer and
  contributions (3 bullets: lag condition and measured lead; two-layer controller; evaluation plus
  the US$500 deployment and the camera error model from its drives). Keep Fig. 1 (rig + perception
  frame).
- **II. The minimum viable deployment.** Keep the current system section nearly whole (Fig. 2
  overview, two kinds of information, advisory layer, rig). Change: the local layer now has two
  jobs, safety (unchanged) and anticipation (new). Say this once, here.
- **III. Why a feed-only controller acts late.** The feed's update process (drives + polling), the
  lag condition (Prop. 1, few lines), density leads speed (Fig. warning/lead [M]), the sampled margin
  (Prop. 2) as the reason the local layer runs every 5 s.
- **IV. The two-layer controller.** Stage 1: shared plan, version-keyed (Alg. 1 kept in short form;
  the agreement lemma in one sentence plus a pointer, since agreement is no longer the contribution).
  Stage 2: the correction (inputs, one step down, hysteresis, never above the plan) and the selected
  form [M: rule or learned or plan alone]. The camera model in simulation and where its numbers come
  from (re-processed drive frames: detection rate and distance noise by distance; bias swept).
- **V. Evaluation.**
  - A. On the road: drives, compute is not the constraint, heat is; the re-processed frames give the
    camera error model (fig panel).
  - B. In simulation: network, scenarios (bursts slow onset, lane blocks fast onset, background for
    false alarms), metrics (onset, prevented, jam time, drain, flow secondary), three arms (no
    control, feed lookahead, best model), selection on seeds 11-15 and evaluation on 16-30.
    Figures: onset/prevention/jam time by scenario; jam time against lag. Table: full result and
    false-alarm cost.
- **VI. Related work.** Vehicles as actuators; infrastructure control; traffic state from vehicles
  and feeds; shared-information decentralized control (one sentence now); advisory interfaces;
  add: incident/queue detection from onboard cameras if a citation is verified.
- **VII. Conclusion and limitations.**

## What leaves the paper
Agreement as the headline (keep Alg. 1 short), the disagreement and staleness slopes, the small-gain
certificate, penetration as a main result (one sentence if block P says something), throughput as a
headline. The old feed-agreement results stay in the repository record.

## Citations found for the rewrite (verify bib fields before use)
- Wang, Z. et al., "A cross-vendor and cross-state analysis of the GPS-probe data latency", Transportation
  Research Record, 2018; arXiv 1711.02688. Mean freeway latency 4.4 min (SD 2.3 min), three vendors, three
  states, Bluetooth/Wi-Fi re-identification ground truth. Companion method paper: arXiv 1801.06128, TRR 2645.
  Use: the feed's aggregation latency, upstream of the publication-to-receipt lag we measure; motivates m = 4.
- Hegyi, Hoogendoorn, Schreuder, Stoelhorst, Viti, "SPECIALIST: A dynamic speed limit control algorithm based
  on shock wave theory", IEEE ITSC 2008, pp. 827-832. Field test on the Dutch A12 (2009-2010): resolved
  shock waves in nearly 80% of activations (Hegyi & Hoogendoorn, TRB/TRID 1096490).
- Hegyi, Netten, Wang, Schakel, Schreiter, Yuan, van Arem, Alkim, "A cooperative system based variable speed
  limit control algorithm against jam waves - an extension of the SPECIALIST algorithm", IEEE ITSC 2013,
  pp. 973-978. Floating car data and video-based monitoring shorten jam detection time and improve the
  speed-limit schemes (VISSIM). Use: prior art that earlier detection helps; theirs is roadside VSL with
  fused detection, ours is vehicle-side with no infrastructure.
