# Task 39 — Session summary generator

> Written by `plan_dsrc_rec`: every decision below was taken by recommendation,
> without putting the question to the user. The decisions table says what was
> chosen and why; **none of it is user-approved**, and the "Open items" section
> holds every point where the recommendation was weak or where the code
> contradicted the first reading. This plan is the fixed target a validator
> audits the implementation against.

> **Tree state every line number below was verified against, and a warning about
> it.** `HEAD` is `9f8bc1e`. The working tree was **not** clean, and **it changed
> twice while this plan was being written, by another process** — the plan itself
> wrote nothing to the repository. At the moment every citation was re-resolved,
> `deployment/jetson/eval_run.py` was +29 lines and
> `deployment/jetson/logio/failure_log.py` +94 lines against `HEAD`, and
> `sha256(git diff)` began `0fc0fd8c8c4f73ce`; an earlier read of
> `failure_log.py` and a later one disagreed by 74 lines, which is how the drift
> was noticed. By the time this plan was closed the fingerprint had already moved
> to `7f92a7b77782b8cc` with `tests/test_eval_run.py` and
> `tests/test_failure_log.py` newly dirty as well. **So: re-resolve every line
> number before trusting it.** Nothing in this plan depends on one. The counts in
> §5.2 were derived by parsing the source, not by reading a line, and were
> re-derived after the change landed; the findings are about which reader exists
> for which field, which no line number carries.

## The short version

Task 39 (plans/task_list.md:1424) asks for a session summary generator carrying
latency percentiles, achieved versus commanded rates, API calls made, trigger
counts and failure counts.

**The generator already exists. It is `eval_run.py`, and it renders two of the
five things the task names.** Latency percentiles are in `## Latency (full run)`
and `## Per-stage timings`; failure counts are in `## Failures`. The other three
are not rendered by anything:

- **Trigger counts are measured and unread.** `SensingLoop.to_record()` keeps
  `decisions_by_trigger`, `rules_by_status` and `inputs_by_source`
  (deployment/jetson/policy/sensing_loop.py:335-339) and writes them to
  `summary["sensing"]` (run_demo.py:682). A repository-wide grep for those three
  names outside `policy/sensing_loop.py` and its own tests returns nothing.
- **Achieved-versus-commanded rates exist as two halves that are never put side
  by side.** Commanded is `sensing.rates` on every tick; achieved is
  `sensing.reference.achieved` on every tick. `eval_run.py` reads neither.
  `score_shadow.py` computes `reference_witness.achieved_mean` (:345-349) and
  `vs_incumbent.rates.incumbent_mean_commanded` (:515-516) — the first only when
  `replay_identity` holds, the second only when a `--candidate` is supplied, and
  never in the same table.
- **API calls made never reaches disk at all.** The phone counts every HERE HTTP
  call it places (`HerePipeline.calls`, HerePipeline.kt:113) and sends it on
  every 1 Hz telemetry frame as `here_calls` (transport/messages.py:600, :644,
  :709). On the Jetson `here_calls` has **zero non-test readers**: it is decoded
  into `PhoneTelemetry` and dropped. `summary["phone"]["here"]
  ["responses_received"]` counts something else — responses that reached the
  Jetson — and is itself rendered by nothing.

**So this is a small task, and the plan scopes it to the gap.** It adds two
fields to one existing per-tick record builder, and everything else is in
`eval_run.py`, which runs offline on a finished directory. **It creates no
seventh file.**

**The defect this task exists to close is in the first line of the artifact
everyone reads.** `report.md`'s verdict is `**Overall: PASS**`, computed from
five gates — latency, throughput, GPS freshness, GPS speed RMSE, perception
coverage (eval_run.py:801-839) — none of which touches the thermal sampler, the
failure sampler, the sensing controller, the HERE feed or the provenance map. A
drive on which four of those six instruments produced nothing prints the same
first line as a drive on which all six worked. Task 37's drive is the proof:
751 ticks, 0 thermal samples, every tick `absent` / `sampler_stopped`, and the
only thing that stopped it reading as healthy was a reader who got as far as
`## Thermal`.

**What the summary does about it, in one rule.** The report gains a
`## Session summary` section placed **above `## Gates`**, listing seven
instrument axes. Each axis reports `answered of attempted` — two counts taken
independently from the records — plus, when they differ, the census of the
axis's own reason words. **There is no percentage anywhere in the section, and
no scalar health field in `report.json`.** The count of answering axes is never
rendered without the enumeration of the ones that did not answer, on the same
lines. `**Overall:**` gains a clause naming how many axes did not answer; the
`overall_pass` value and the exit code are unchanged.

**How the two named drives look different from a healthy one, in the summary
itself.**

| drive | first lines of `## Session summary` |
|---|---|
| task 37's (sampler died on pass 1) | `thermal: 0 of 751 ticks answered — absent, sampler_stopped 751` |
| task 38's (camera blind 40 times) | `failures: RECONCILIATION FAILED — sources["camera.blind_ticks"].total is 0 and blind_ticks is 40, on the same record; this drive's failure counts are not usable` |
| healthy | `7 axes, 7 answered. 9 reconciliations, 9 held.` |

**No new vocabulary.** The section states no word about a measurement that is
not already a member of one of the five: task 33's `measured` / `converted` /
`absent` / `instant`, task 34's `fired` / `quiet` / `not_evaluable` with
`missing`, task 36's eleven provenance classes, task 37's six absence reasons,
task 38's `recovered` / `open_at_end` / `unobservable` and its `(source,
reason)` pair. The only words the summary adds are about its own arithmetic —
`attempted`, `answered`, `of`. Every axis carries the module path of the
vocabulary its reason keys come from, and a reason word that is not a member of
that vocabulary is counted verbatim and listed in `vocabulary_violations` rather
than absorbed.

**Does it change behaviour? In zero places.** Two fields are added to
`reference_from`'s returned dict (policy/sensing_loop.py:188-217) — publishing
two integers the tick loop already holds. No predicate is added, no rate moves,
no message gains a field, no channel changes, and `SensingController.decide` and
`Decision.to_record()` are byte-identical. §"Behaviour changes" states what
asserts each half of that.

**Scope boundary.** In: `policy/sensing_loop.py` (two fields on `reference_from`,
both branches); `eval_run.py` (a `session_summary` builder, a `sensing` result
block, two report sections, one qualifier on the `Overall` line, and a `main()`
branch so a zero-tick drive still produces a summary); tests on the Python side;
pins in `scripts/remutate.py`; `ARCHITECTURE.md` §9. Out: `summary.json`'s shape,
the wire, the controller, the phone, every gate, and any new artifact file.

**Open items, in one line each** (details at the bottom): `here_calls` is
cumulative for the phone's *service* run and `PhoneLink._telemetry` is cleared on
every redial (phone_link.py:479), so calls placed before a session's first
observed report are uncounted and only bounded; the telemetry window is measured
from arrival instants rather than the phone's own window, so a lost telemetry
frame widens a weight silently; `ModeHolder.flip_to` has no caller in
`run_demo.py`, so every real drive is entirely shadow or entirely live and the
mode is a fact rather than a series; three of the seven axes are promotions of
numbers `report.md` already prints, so their value is placement rather than
measurement; and a drive that never reached teardown has no `summary.json`, so
four of the nine reconciliations are unavailable on exactly the drive that most
needs them.

---

## Section 1 — The inventory: what each existing surface already answers

**"Reader" means a non-test reader.** `scripts/remutate.py` hits are mutation
source strings, not readers. Line numbers are from the tree state stamped at the
top of this plan, not from `9f8bc1e` alone: two files are dirty in it.

### 1.1 `report.md` / `report.json` — `eval_run.py`, written at :1500 and :1502

The largest surface, and the only one with a verdict. Sections, in order:

| section | what it answers | source |
|---|---|---|
| header | tick count, duration, median rate, camera frames dropped | ticks + `summary["camera_dropped_frames"]` |
| `## Gates` | five PASS/FAIL gates + `log complete`, and `**Overall:**` | `analyze` :801-839, rendered at :1313 |
| `## Latency (full run)` | n/min/mean/p50/p95/max for 8 named series | :630-638, `render_markdown` :1277 |
| `## Per-stage timings` | 14 stages x {measured, converted, absent, instant} with a reason census | `stage_timings` :404-441 |
| `## Perception` | tracked-vehicle coverage, leader gap, distance methods, track lifetimes | `analyze` :641-670 |
| `## Observation quality` | missingness mean + spread + distinct count, provenance coverage, `by_source`, `derived_empty` fields | `analyze` :673-729 |
| `## Thermal` | jetson zone p50/p95/max, phone status census, headroom absence, throttle events per device, `ticks_by_basis` | `thermal_result` :443-483, `_thermal_lines` :974 |
| `## Failures` | scan cadence, ticks seen per pass, episodes by outcome, all 30 sources with their three-word status, backwards counters, blind ticks, pipeline exception, `log_health`, the phone's own failure lines | `failures_result` :520-570, `_failure_lines` :1140 |
| `## GPS` | fresh-fix fraction, scripted-dropout RMSE | `analyze` :731-763 |
| `## Advisory` | head distributions, switches per minute, confidence labels | `analyze` :766-793 |
| `## Phone join` | advisories the phone saw, matched/unmatched, `return`/`render` coverage | `join_phone_log` :300-352 |

**What it does not answer.** Grepping `eval_run.py` for `sensing`, `rates`,
`trigger`, `commanded`, `achieved`, `here_`, `rules_fired`, `attribution`,
`quota` returns exactly one hit: the import of `RULE_FIRED` and
`RULE_NOT_EVALUABLE` at :43, used only by the thermal and failure sections. The
tool reads five keys of `summary.json` — `thermal` (:479), `failures` (:559),
`ticks` (:611), `policy_trained` (:788), `camera_dropped_frames` (:868) — and
none of the other five.

### 1.2 `summary.json` — `MetadataLogger.write_summary`, one call site, run_demo.py:708

Ten top-level keys, five unconditional and five conditional:

| key | condition | read by |
|---|---|---|
| `ticks` | always | `eval_run` (:611) |
| `stats` | always | **nobody** |
| `camera_dropped_frames` | always | `eval_run` (:868) |
| `camera_file_recoveries` | always | **nobody** |
| `policy_trained` | always | `eval_run` (:788) |
| `network` | `--phone` | **nobody** |
| `sensing` | `--phone` | `score_shadow`, two sub-keys only: `mode` (:230) and `ticks` (:272) |
| `thermal` | `logio.thermal` | `eval_run` (:479) |
| `failures` | `logio.failures` | `eval_run` (:559) |
| `phone` | `--phone` | **nobody** |

Two facts about it matter to this task.

**`summary["stats"]` is not a whole-drive distribution.** Every series is a
`RollingStats(window=300)` (pipeline.py:39-41, :171-188), so its p50/p95 are over
the last 300 ticks and `n` saturates at 300. It is the only latency figure the
end-of-run console line prints (`summary_line`, run_demo.py:829-845), and
`eval_run` legitimately disagrees with it because `eval_run` recomputes over
every tick.

**`summary["sensing"]` already holds the trigger counts this task needs**, and
nothing reads them:

```
sensing.ticks, sends_by_reason, rate_commands_sent, heartbeat_s,
sensing.mode.{mode, flips, flip_count, shadow_predicts, structurally_absent,
              feed_possible_from_mono, reference_rates_hold},
sensing.decisions_by_trigger   # 6 Trigger words -> count
sensing.rules_by_status        # 4 rules -> {fired, quiet, not_evaluable} -> count
sensing.inputs_by_source       # 3 fields -> 11 provenance classes -> count
```

### 1.3 `shadow_score.json` — `score_shadow.py`, written at :708

Answers one question: **can this log reproduce its own incumbent's decisions, and
if so how does a candidate policy differ.** It refuses a log by name
(`no_metadata_jsonl`, `no_tick_records`, `phoneless_run`, `decision_inputs_absent`,
`decision_inputs_schema`, :86-95) and, when `replay_identity.status != "ok"`,
returns before computing `rules_never_exercised`, `input_provenance` and
`candidates` at all (:588-591).

Two of its blocks are the closest thing that exists to this task's rate axis, and
both sit behind that gate:

- `reference_witness.achieved_mean` (:345-349) — per `RATE_KEYS`, the mean of
  `reference.achieved` over **ticks** whose report is fresh. Ticks, not reports:
  a 5 Hz drive against 1 Hz telemetry echoes each report about five times, and
  the echo count varies with tick jitter, so the mean is weighted by how many
  ticks happened to observe each report.
- `candidates.<label>.vs_incumbent.rates.incumbent_mean_commanded` (:516) — the
  incumbent's mean commanded rate per key, present only when `--candidate` was
  passed.

It is also the only tool that names a rule never exercised
(`rules_never_exercised`, :413-417), and its stdout table renders neither
`activity` nor `candidate_evaluated_where_incumbent_could_not`.

### 1.4 `log_health.json` — run_demo.py:264-284, written after `close()`

Eight keys: `t_wall`, `t_mono`, `dropped_records`, `writer_failure`,
`queue_depth`, `thread_alive_at_close`, `path`, `bytes_on_disk`. `eval_run`
reads it (`failures_result`, :532) and renders exactly two of them in one bullet
(`_log_health_lines`, :1128). The other six reach `report.json` and are
never rendered.

### 1.5 The phone's own session log — `SessionLog.kt`

Four line shapes, discriminated by a `dir` key that the outbound shape does not
carry: the bare outbound frame header (:100-107), `{"dir":"in"}` (:119-129),
`{"dir":"shown"}` (:140-149), `{"dir":"fail"}` (:169-210). `eval_run` reads three
of them, and only when `--phone-log` is supplied (`load_phone_log`, eval_run.py:140). `SessionLog.Stats`
(:302-323 — `written`, `bytes`, `droppedQueueFull`, `droppedNotRunning`,
`droppedAtCap`, `failures`, `failuresSuppressed`, `complete`) goes to logcat and
nowhere else.

### 1.6 What a reader still cannot answer after reading all six

1. **What rates were commanded, and did the phone deliver them?** Both halves are
   on every tick and neither is aggregated into anything a reader opens.
2. **How many HERE API calls did this drive place?** Not recorded anywhere on
   disk. The counter crosses the wire 180 times on a 180 s drive and is dropped.
3. **Which rules fired, how often, and which never got the chance?** Counted into
   `summary["sensing"]` and read by nothing.
4. **Did this drive measure anything?** No surface answers this. `report.md`'s
   verdict answers a narrower question — five gates — and prints `PASS` above a
   thermal section reading `absent 751`.
5. **A drive that produced no ticks produces no report at all.** `analyze` raises
   `SystemExit` at :589-599, so the drive with the most failures — the camera
   never delivered a frame — has no artifact, only the exit message.

**Items 1-3 are the task's own words. Items 4-5 are what a summary is for.** The
honest scope is: build 1-3, build 4 as a section rather than a file, and fix 5
because a session summary that cannot be produced for the worst session is the
section's defect class turned on the section's last task.

---

## Section 2 — Which of the five vocabularies this is

The section has built five and the brief forbids a sixth. **This task builds
none.** It aggregates, and aggregation is where a third state gets destroyed.

**The destruction to avoid, precisely.** Task 34 gives a rule three states:
`fired`, `quiet`, `not_evaluable`. Any function of the form
`healthy_fraction = quiet / total` collapses `not_evaluable` into whichever side
the author picked, and every such fraction reads the same on a drive that
observed nothing as on a drive that observed nothing happening. The same holds
for task 33's `absent`, task 37's six reasons, and task 38's `unobservable`.

**The rule this plan applies to every aggregate it emits:**

> An axis reports two independently counted integers, `answered` and
> `attempted`, and — whenever they differ — the full census of the axis's own
> reason words, keyed by the word itself. It reports no ratio, no percentage,
> and no single word summarising the axis. `answered + sum(census.values()) ==
> attempted` is checked, not used to derive any of the three.

Three consequences worth naming.

**`0 of 0` is not "answered".** An axis whose instrument never attempted a
reading has `attempted == 0`, and it appears in the headline list alongside the
axes that attempted and failed. A ratio would have made it 100 per cent, or
undefined and quietly dropped.

**A word outside the declared vocabulary is counted verbatim.** Each axis
carries `vocabulary`, the dotted path of the constant its reason keys come from
(`sensors.thermal.ABSENT_REASONS`, `policy.sensing_controller.RULE_*`,
`perception.provenance.SOURCES`, `logio.failure_log.MISSING_*`,
`sensors.time_sync.STAGE_BASES`). A record carrying a word that is not a member
is counted under that word and the word is listed in `vocabulary_violations`, so
the identity still holds and the drift is visible. Absorbing it into an "other"
bucket would be the sixth vocabulary arriving by the back door.

**The five vocabularies keep their own owners.** The summary never re-derives a
status. `thermal.jetson.basis` is read, not recomputed; `rules[*].status` is
read, not recomputed; `sources[*].status` is read, not recomputed. Where the
summary disagrees with a producer it says so and reports both numbers — it never
picks one. That is §5's reconciliation list.

---

## Section 3 — What makes an aggregate real

Two aggregates have shipped in this section and been removed: a `stale` count in
`summary["thermal"]["jetson"]["basis_counts"]` that nothing could increment
(the dict is seeded with `measured` and `absent` only, sensors/thermal.py:412,
and `THERMAL_BASIS_STALE` is assigned only on the per-tick path at :719), and a
`basis_counts` literal in the failure scan record. **A third is live right now
and this plan does not remove it:** `summary["camera_file_recoveries"]` reads
`camera.file_recoveries`, which on a `--phone` run is set to `0` at
sensors/phone_source.py:357 and incremented by nothing — the only `+= 1` in the
tree is `CameraStream`'s at sensors/camera_stream.py:192. The same field feeds
`failures.sources["camera.file_recoveries"]`, which therefore reports `quiet` /
`total: 0` for the life of every phone drive.

**Four conditions, and every aggregate this plan emits is held to all four.**

1. **It is computed by reading records.** Never a literal, and never derived by
   subtracting one aggregate from another. `answered` counts the records that
   answered; the census counts the records that did not; `attempted` counts all
   of them. Three passes over one list, and the identity in §2 checks them
   against each other rather than defining one from the others.
2. **There exists an input under which it takes a different value, and a test
   exhibits that input.** This is what the `stale` count failed. Each aggregate
   below names, in §7's table, the fixture that moves it.
3. **Its denominator is counted independently of its numerator.** An axis whose
   `attempted` is derived from the same records as its `answered` is `n of n` by
   construction. Where a natural independent denominator does not exist, the
   axis says so rather than manufacturing one — see the failures axis, whose
   pass-count sub-line is buildable only when a cadence can be established from
   at least two records.
4. **Deleting the producer moves it.** Every aggregate has a pin in
   `scripts/remutate.py` that removes or neuters the thing it measures and
   asserts a named test fails.

**One aggregate this plan deliberately does not build**, because it would fail
condition 2: a count or a series over `sensing.mode.flips`. `ModeHolder.flip_to`
(policy/shadow_mode.py:136-149) has no caller in `run_demo.py`; the mode is fixed
at construction from `--live-rates` (run_demo.py:506) and never changes. The only
non-test caller in the tree is `scripts/run_phone_drive.py:185`, a harness with a
simulated peer that writes neither `metadata.jsonl` nor `summary.json`. So on
every drive that produces a run directory, `flips` is `[]` and `flip_count` is
`0`. The summary states the mode as a **fact about the drive**, once, and counts
nothing.

---

## Section 4 — Decisions taken (by recommendation — not signed off by the user)

| # | Question | Options | Taken | Why |
|---|---|---|---|---|
| D1 | Does task 39 create a new artifact | (a) `session_summary.json` + its own CLI; (b) sections inside `report.md` / `report.json` | **(b)** | Six surfaces exist and the brief's first hazard is a seventh that restates them. `report.md` is the artifact a reader opens and it already carries a verdict; the defect is in that verdict's neighbourhood, so the fix belongs there. A seventh file would also need its own answer to "what if it is absent", which (b) gets for free by living in a file that is written or not written as a whole. |
| D2 | Where the summary sits in `report.md` | (a) at the end, as a recap; (b) first, above `## Gates` | **(b)** | A reader who reads one thing reads the first thing, and `**Overall: PASS**` currently is that thing. Placing the axis list below it means the reader has already formed the verdict. |
| D3 | Does the summary change `overall_pass` or add a gate | (a) gate on all axes answering; (b) no gate, but qualify the `Overall` line | **(b)** | Task 38's D17 verbatim, and this section has two proofs it is right: task 34's drive had `source_disagreement` not evaluable on 899 of 899 ticks because HERE is unconfigured, and task 37's first drive had thermal absent on 751 of 751. Gating would fail every drive for the normal condition. The qualifier makes `PASS` unreadable as unqualified without changing `overall_pass`, `report.json`, or the exit code. |
| D4 | How the summary aggregates a three-state vocabulary | (a) a health percentage; (b) a single summary word per axis; (c) `answered of attempted` plus the reason census | **(c)** | (a) is the destruction §2 names. (b) is a sixth vocabulary. (c) adds no word about a measurement and keeps every third state visible, because the census keys *are* the third states. |
| D5 | Whether the summary emits a scalar health figure in `report.json` | (a) `axes_answered: 4`; (b) only the per-axis records | **(b)** | A scalar in the JSON is the thing a later dashboard plots on its own, and once plotted the enumeration is gone. The markdown counts the list for the reader, on the same lines as the list. |
| D6 | How "API calls made" reaches disk | (a) add `here_calls` to `phone_link`'s telemetry record in `summary.json`; (b) add it to the per-tick `reference` block | **(b)** | (a) records the last report's value only, has no time axis, and is absent entirely on a drive that did not reach teardown — which is the drive most likely to have a HERE problem. (b) gives a monotone series, survives a run that died up to the log buffer, and lands in the block whose stated job is "what the phone says it is actually running, witnessed rather than assumed" (sensing_loop.py:188-190). Cost is 36 B/tick. |
| D7 | What "achieved" is aggregated over | (a) ticks, as `score_shadow` does; (b) distinct telemetry reports, deduplicated on `reference.at_mono` | **(b)** | `PhoneLink.telemetry` holds the newest report and every tick echoes it, so (a) weights each report by how many ticks happened to observe it — a function of tick jitter, not of the phone. `score_shadow`'s own docstring (:304-312) already established that `reports` must be counted on distinct `at_mono` for exactly this reason; its `achieved_mean` is the one place it did not apply that. Deduplicating is a strict improvement and costs one `set`. |
| D8 | Whether achieved is compared to commanded unconditionally | (a) always report the ratio; (b) only when the commands were live, and say why not otherwise | **(b)** | On a shadow drive the phone was never told to run the commanded rates: `ConfigApplier.apply` returns on the shadow branch before touching any rate (ConfigApplier.kt:61-63). A ratio would report the mode working correctly as an 80 per cent shortfall. `summary["sensing"]["mode"]["mode"]` and the per-tick `sensing.shadow` flag both name it, and the second survives a missing summary. |
| D9 | What to do about a modality whose commanded period exceeds the telemetry window | (a) report percentiles anyway; (b) suppress percentiles below one expected delivery per window | **(b)** | `achieved` is deliveries in one ~1 s window divided by the window (TelemetryReporter.kt, `report()`). At the idle `here_hz` of 0.05 Hz that is one delivery per 20 windows, so 19 of 20 windows report exactly `0.0` and the median is `0.0` against a commanded `0.05`. The mean is right and the percentiles are a statement about the window. The rule is stated on the record as `lambda_per_window = commanded_hz x observed_window_s` and the suppression is named, not silent. |
| D10 | Where the telemetry window length comes from | (a) mirror the phone's `PERIOD_MS = 1000`; (b) measure it from the arrival instants | **(b)** | (a) is a cross-device constant with no compile-time link, and the section has already paid for one hand-ported mirror (`PHONE_FAILURE_LIFETIME_CAP`, eval_run.py:1092). (b) is the median of `diff(sorted(distinct at_mono))`, is reported on the record, and degrades to a named refusal when fewer than two reports exist. |
| D11 | How HERE calls are totalled across a redial | (a) `last - first`; (b) per `session_id`, summing `last - first`, with `sum(first)` reported as an uncounted prefix | **(b)** | `PhoneLink._rebind` clears `self._telemetry` (phone_link.py:479) and a different handset's `HerePipeline.calls` starts at zero, so `last - first` across a redial can be negative. (b) is exact within a session and states its own bound: calls placed before a session's first observed report are not counted, and `first` is how many they were. `session_id` is on every tick record (run_demo.py:590). |
| D12 | Whether the summary re-derives any producer's status | (a) recompute from the underlying counters; (b) read the producer's word, and report disagreement | **(b)** | Task 37's D5: two accounts of one machine whose relationship is unknown is worse than one. Where the summary can check a producer against itself it does, and a failed check is reported with both numbers rather than resolved — §5's reconciliation list. |
| D13 | Whether a zero-tick drive gets a summary | (a) leave task 38's open item 1 open; (b) build the summary before `analyze` and emit it even when `analyze` cannot run | **(b)** | A session summary that is absent for the worst session is this section's defect class turned on its last task. `analyze` is untouched — `main()` gains a branch that builds the session summary from `LoadedRecords`, `summary.json` and `log_health.json`, writes a `report.md` containing only that section plus the reason the rest is absent, and returns 2. Every metric block still indexes `ticks` and none of them runs. |
| D14 | Whether the failures axis uses the per-pass scan records or the per-tick block | (a) `failure_scan` records; (b) the per-tick `failures` block | **(b)**, with (a) as a sub-line | (a)'s natural ratio is `sources_readable / sources_n`, which reads `15 of 30` on every pass of a phone-less drive and is correct rather than a fault; folding that into an axis verdict would make a local-camera drive look broken. (b) uses task 38's own three-state basis and gives `0 of N` for a dead sampler, which is the failure the axis exists to catch. The never-readable sources are enumerated beside it, never folded in. |
| D15 | Whether the summary reads `run_config.yaml` for sampler intervals | (a) yes; (b) derive the cadence from the records | **(b)** | The configured interval is a statement of intent; the observed median is a measurement, and the two disagree exactly when something is wrong. It also keeps `eval_run` free of a YAML dependency it does not have today. When fewer than two records exist there is no cadence and the sub-line says so. |
| D16 | Whether `summary["stats"]`'s 300-tick window is reconciled against the report's whole-drive percentiles | (a) yes, as a reconciliation; (b) name it once and move on | **(b)** | They measure different things and are both correct. A reconciliation that is expected to fail is not a check. The `## Session summary` names the window once so a reader who has seen the console line knows why it differs. |
| D17 | Whether the three promoted axes (latency, thermal, provenance) restate their sections' numbers | (a) restate the key figures; (b) one line and a pointer | **(b)** | Restating is how the seventh surface gets built inside the sixth. Each promoted axis prints `answered of attempted`, the census when they differ, and the section name. Nothing else. |

---

## Section 5 — The record, exactly

### 5.1 `policy/sensing_loop.py` — two fields on `reference_from`

The only change to a live path. Both branches, and **null rather than zero** on
the absent branch, because a phone never heard from has not placed zero calls.

```python
def reference_from(phone, *, now: float) -> dict[str, Any]:
    telemetry = getattr(phone, "telemetry", None)
    if telemetry is None:
        return {"achieved": None, "dropped": None, "age_s": None, "at_mono": None,
                "here_calls": None, "here_errors": None, "absent": "no_telemetry"}
    telemetry_at = getattr(phone, "telemetry_at_mono", None)
    return {
        "achieved": {key: telemetry.achieved[key] for key in RATE_KEYS},
        "dropped": {key: telemetry.dropped[key] for key in DROP_KEYS},
        "age_s": None if telemetry_at is None else now - telemetry_at,
        "at_mono": telemetry_at,
        # The phone's own count of HERE HTTP calls placed and non-2xx responses,
        # cumulative for its service run. It has crossed the wire on every
        # telemetry frame since task 27 and reached no reader; the drive artifact
        # could not say how many calls a drive made.
        "here_calls": telemetry.here_calls,
        "here_errors": telemetry.here_errors,
        "absent": None,
    }
```

`PhoneTelemetry.here_calls` and `.here_errors` are required non-null ints
(transport/messages.py:600-601, decoded with `check_count` at :709-710), so the
present branch cannot produce a null and the null branch cannot produce an int.
That is the shape rule the validator pins.

### 5.2 `eval_run.py` — the session summary builder

New module-level constant and one function. `AXES` is a fixed tuple so an axis
cannot go missing:

```python
#: The seven instruments a drive runs, in the order the summary lists them.
#: Fixed: an axis that cannot be built appears with `attempted: null` and a
#: named missing input, never by being absent from the list.
AXES = ("latency", "rates", "api_calls", "triggers", "failures", "thermal",
        "provenance")
```

One axis record:

```json
{"axis": "thermal",
 "attempted": 751,
 "answered": 0,
 "attempted_is": "ticks carrying a thermal block",
 "answered_is": "ticks whose thermal.jetson.basis is measured",
 "unanswered_by_reason": {"sampler_stopped": 751},
 "vocabulary": "sensors.thermal.ABSENT_REASONS",
 "vocabulary_violations": {},
 "unbuildable": null,
 "section": "## Thermal"}
```

and an axis that could not be built at all:

```json
{"axis": "rates", "attempted": null, "answered": null,
 "attempted_is": "distinct telemetry reports observed (sensing.reference.at_mono)",
 "answered_is": "reports fresh by the controller's own staleness predicate",
 "unanswered_by_reason": {}, "vocabulary": "policy.sensing_loop.reference_from",
 "vocabulary_violations": {},
 "unbuildable": "no tick carries a sensing block (not a phone run)",
 "section": "## Sensing"}
```

The seven axes, with both counts taken from records:

| axis | `attempted` | `answered` | census keys drawn from |
|---|---|---|---|
| `latency` | ticks | ticks with `jetson_ms` non-null | `{"absent from this run; e2e used": n}`, the wording `latency["jetson_ms_source"]` already uses |
| `rates` | distinct `sensing.reference.at_mono` where `absent is None` | those the controller's own `abs(age_s) > MAX_TELEMETRY_AGE_S` predicate calls fresh | `{"stale": n}` plus `{"no_telemetry": n}` counted over ticks, both named on the record |
| `api_calls` | ticks carrying a `sensing.reference` block | ticks with `here_calls` non-null | `{"no_telemetry": n}` |
| `triggers` | ticks carrying a `sensing` block | ticks where all four `RULES` carry a status in `{fired, quiet, not_evaluable}` | `{rule: {missing_field: n}}` for every rule that was `not_evaluable` |
| `failures` | ticks carrying a `failures` block | ticks with `basis == "measured"` | `{"stale": n}` plus task 38's `{"sampler_stopped": n, "no_sample_yet": n}` |
| `thermal` | ticks carrying a `thermal` block | ticks with `thermal.jetson.basis == "measured"` | `sensors.thermal.ABSENT_REASONS`, six members, plus `{"stale": n}` |
| `provenance` | ticks | ticks whose `field_sources` key set equals `sim_contract.encoded_slot_names()` | `{"provenance_fields_mixed": n}` and `{"short: <size>": n}` keyed on the map size found |

**The counts these definitions rest on, and where each was verified.** Every one
was checked against the code in this working tree, not taken from a previous
plan:

| count | value | verified by |
|---|---|---|
| encoder slots | **39** | `sim_contract.encoded_slot_names()` returned 39 and `local_obs_dim()` returned 39, executed under the repo venv |
| provenance classes | **11** | `perception.provenance.SOURCES` has 11 members; 4 of them are in `SUBSTITUTED` |
| controller rules | **4** | `sensing_controller.RULES` = `(EVENT, NARROW_MARGIN, DISAGREEMENT, THERMAL)`, :151 |
| trigger words | **6** | `Trigger.ALL` = `{idle, event_from_free_tier, advisory_margin_narrow, source_disagreement, thermal_backoff, holding_after_event}`, :129-137 |
| rule statuses | **3** | `RULE_FIRED`, `RULE_QUIET`, `RULE_NOT_EVALUABLE`, :143-146 |
| rate keys / drop keys | **4** / **4** | `transport/messages.py:61-62` |
| failure sources | **30** | AST parse of `build_registry`'s return tuple, `logio/failure_log.py:830-909` |
| — with `event_records` | **17** | the same parse, `Source` field index 6 |
| — without | **13** | the same parse |
| — device `jetson` / `phone` | **28** / **2** | the same parse, `Source` field index 7. **Task 38's plan said "twenty-eight sources" throughout and derived every byte figure from it; 28 is the count of `jetson` rows, and the registry has 30.** |
| stages in `STAGE_ORDER` | **14** | regex parse of `eval_run.py:397-401` |
| thermal absent reasons | **6** | `sensors/thermal.ABSENT_REASONS`, :62-67 |
| episode outcomes | **3** | `logio/failure_log.py:75-78` |
| phone failure kinds | **10** | `FailureKinds.kt:18-27` |
| `summary["stats"]` series / window | **13** / **300** | `pipeline.py:190-205`, `RollingStats(window=300)` at :40 |
| top-level `summary.json` keys | **10** (5 unconditional) | `run_demo.py:649-697`, one `write_summary` call site at :708 |

### 5.3 `eval_run.py` — the `sensing` result block

Built from tick records, cross-checked against `summary["sensing"]` where it
exists. This is the detail for the three axes that have no existing surface.

```json
"sensing": {
  "present": true,
  "ticks": 899,
  "mode": "shadow",
  "mode_source": "summary[sensing].mode.mode",
  "ever_live": false,
  "structurally_absent": ["feed_congestion", "source_disagreement"],
  "commands": {"reached_the_wire": 41, "by_reason": {"first": 1, "changed": 12,
                                                     "query_moved": 0, "heartbeat": 28},
               "summary_agrees": true},
  "telemetry_window": {"observed_median_s": 1.002, "reports": 178,
                       "expected_from_span": 180, "windows_not_observed": 2},
  "rates": {
    "camera_hz": {"commanded_by_value": {"1.0": 719, "5.0": 180},
                  "commanded_distinct": 2,
                  "commanded_time_mean": 1.8,
                  "clamped_ticks": 0, "thermal_scaled_ticks": 0,
                  "achieved_time_mean": null,
                  "lambda_per_window": 1.8,
                  "percentiles": {"p50": null, "p95": null},
                  "percentiles_suppressed": null,
                  "comparable": false,
                  "not_comparable_because": "mode shadow on 899 of 899 decisions"}
  },
  "triggers": {
    "decisions_by_trigger": {"idle": 0, "event_from_free_tier": 0,
                             "advisory_margin_narrow": 899, "source_disagreement": 0,
                             "thermal_backoff": 0, "holding_after_event": 0},
    "rules_by_status": {"advisory_margin_narrow": {"fired": 899, "quiet": 0,
                                                   "not_evaluable": 0},
                        "source_disagreement": {"fired": 0, "quiet": 0,
                                                "not_evaluable": 899}},
    "rules_missing": {"source_disagreement": {"feed_congestion": 899}},
    "summary_agrees": true
  },
  "here": {
    "by_session": [{"session_id": "a1b2c3d4", "first": 0, "last": 0,
                    "observations": 178}],
    "calls_total": 0,
    "uncounted_prefix": 0,
    "errors_total": 0,
    "expected_from_commanded": 0.0,
    "responses_received": 0,
    "zero_calls_because": "mode shadow; a shadow command never reaches setHereQuery"
  }
}
```

Three fields carry the whole honesty of the rates axis.

**`comparable` / `not_comparable_because`.** False whenever `ever_live` is false,
whenever no fresh telemetry report exists, or whenever fewer than two reports
make the window unmeasurable. The reason is a sentence naming the count that
made it so — never an empty comparison rendered as zero shortfall.

**`lambda_per_window` and `percentiles_suppressed`.** `lambda = commanded_time_mean
x observed_median_s`. Below 1.0 the percentiles are `null` and
`percentiles_suppressed` reads
`"commanded 0.05 Hz over a 1.002 s window is one delivery per 20 windows; a
median over per-window rates would be 0.0 and is not a statement about the rate"`.
At `lambda == 1.0` exactly — the ordinary `gps_hz` case — the percentiles are
reported and the field names the quantisation: values are multiples of
`1 / observed_median_s`.

**`windows_not_observed`.** `expected_from_span - reports`, where
`expected_from_span = drive_span_s / observed_median_s`. Non-zero means the
time-weighted mean is over the windows that arrived, and the weight is the
arrival interval rather than the phone's own window. Stated on the record, not in
a comment.

### 5.4 `report.md` — the two new sections and the one changed line

`## Session summary`, placed after the tick-count line and **before** `## Gates`:

```markdown
## Session summary

7 axes. 4 answered on every observation. 3 did not, and 0 could not be built:

- **thermal**: 0 of 751 ticks answered — absent: sampler_stopped 751. See ## Thermal.
- **triggers**: 899 of 899 ticks answered; source_disagreement was not_evaluable
  on 899 of them, missing feed_congestion. See ## Sensing.
- **rates**: 178 of 180 reports answered — stale 2; no_telemetry on 0 ticks.
  Commanded rates were never applied (mode shadow on 899 of 899 decisions), so
  achieved is not a shortfall against them. See ## Sensing.

9 reconciliations, 8 held. 1 failed:

- **failures**: sources["camera.blind_ticks"].total is 0 and blind_ticks is 40,
  on the same record. This drive's failure counts are not usable.

Inputs: metadata.jsonl yes, summary.json yes, log_health.json yes, phone log no.
Latency in ## Latency is over all 899 ticks; the console line at the end of the
run is over the last 300.
```

`## Sensing`, placed between `## Advisory` and `## Phone join`, rendering the
`sensing` block: the rate table (one row per key: commanded value census,
commanded time mean, achieved time mean or the reason there is none, and the
comparison or the reason there is none), the trigger census (one row per rule
with all three states, one row per trigger word), and the HERE line (calls per
session, total, expected from commanded, responses received, and the reason for
zero when it is zero).

**The one changed line.** `**Overall: PASS**` becomes

```
**Overall: PASS** — the five gates. 3 of 7 instrument axes did not answer;
see ## Session summary.
```

and, when every axis answered,

```
**Overall: PASS** — the five gates; all 7 instrument axes answered.
```

`report.json["overall_pass"]`, the gate table, and `main()`'s exit code are
unchanged.

### 5.5 `eval_run.main()` — the zero-tick branch

```python
loaded = load_records(run_dir / "metadata.jsonl")
summary = _read_summary(run_dir)
health = _read_log_health(run_dir)
session = session_summary(loaded, summary, health)
if not loaded.ticks:
    md = render_session_summary_only(run_dir, session, loaded)
    (run_dir / "report.md").write_text(md)
    (run_dir / "report.json").write_text(json.dumps(
        {"run_dir": str(run_dir), "n_ticks": 0, "session_summary": session,
         "analysis": None,
         "analysis_absent": "no tick records; every metric block indexes ticks"},
        indent=2))
    print(md)
    return 2
result = analyze(run_dir, phone_log, loaded=loaded)
```

`analyze` gains one optional keyword, `loaded=None`, and loads the records itself
when it is not given. Its `SystemExit` at :589-599 stays exactly as it is — it is
now unreachable through `main()` and remains the contract for a direct caller.

---

## Section 6 — Reading rules a validator audits against

Each rule says whether it is checkable **within one record** or **across
records**, and every cross-record rule names its join and what happens when one
side is absent. Task 38's plan specified `sum(episode.n) == the source's run
total`, which relates a field on `failure_event` close records to a field in
`summary["failures"]` — two files, and unmeetable for the 13 sources that write
no event records at all; the implementation replaced it with the within-row
identity `kept_total + suppressed + below_episode_threshold == total`
(logio/failure_log.py:1522-1528). The rules below were written against that
lesson.

**Within one axis record:**

1. `attempted` and `answered` are non-negative integers with
   `0 <= answered <= attempted`, or both are `null` with `unbuildable` naming a
   missing input. `unbuildable` is non-null if and only if `attempted` is null.
2. `answered + sum(unanswered_by_reason.values()) == attempted`. Every one of the
   three is counted by reading records; none is derived from the other two.
3. Every key of `unanswered_by_reason` that is not a member of the constant named
   by `vocabulary` also appears in `vocabulary_violations` with the same count.
4. `unanswered_by_reason` is empty if and only if `answered == attempted`.
5. `attempted == 0` implies `answered == 0` and implies the axis appears in the
   markdown's did-not-answer list. **A `0 of 0` axis is never counted as
   answering.**

**Within `session_summary`:**

6. `axes` has exactly `len(AXES)` entries, one per member of `AXES`, in that
   order. There is no path by which an axis is omitted.
7. `session_summary` contains no field whose value is a ratio, a percentage, or a
   single word describing an axis's health. (Checkable by asserting the key set.)

**Within one tick record:**

8. `sensing.reference.absent is None` if and only if `achieved is not None`, if
   and only if `dropped is not None`, if and only if `here_calls is not None`,
   if and only if `here_errors is not None`. All five fields are on the same
   object. This is the rule that makes a phone never heard from unable to report
   zero API calls.
9. `set(sensing.attribution.rules) == RULES` and every `status` is one of the
   three constants. (Already enforced by `score_shadow._has_valid_attribution`
   :438-445; restated because the triggers axis reads these fields.)
10. `sensing.rates` has exactly the four `RATE_KEYS`, and for each key
    `attribution.per_sensor[key]["hz"] == rates[key]`.

**Within `summary["failures"]` (one record, when `summary.json` exists):**

11. `sources["camera.blind_ticks"]["total"] == blind_ticks`.
12. For every source row: `status == "fired"` if and only if `total > 0`.
13. For every source row: `kept_total + suppressed + below_episode_threshold ==
    total`.
14. For every source row: `passes_readable <= passes_attempted`.
15. `len(sources) == scan["sources_n"] == 30`.

**Across records, join named:**

16. *Join: the run directory.* `sum(row["events_written"])` over
    `summary["failures"]["sources"]` equals the number of `failure_event` records
    in `metadata.jsonl` with `phase == "open"`. When `summary.json` is absent the
    reconciliation is reported as `unavailable: summary.json not written`, never
    as held. A failure here detects a truncated log and is reported as a failure
    of the pair, not of either side.
17. *Join: the run directory.* `summary["thermal"]["jetson"]["samples"] - (count
    of `thermal_sample` records)` is 0 or 1 — never negative. The 1 is task 37's
    own documented race between counting a pass and writing its line
    (run_demo.py:246-255). Unavailable when `summary.json` is absent.
18. *Join: the run directory.* The triggers axis's `rules_by_status` census,
    computed over tick records, equals `summary["sensing"]["rules_by_status"]`;
    likewise `decisions_by_trigger`. A disagreement is reported with both maps
    and resolves nothing. Unavailable when `summary.json` is absent or carries no
    `sensing` key.
19. *Join: the run directory.* `here.calls_total >= summary["phone"]["here"]
    ["responses_received"]`. Every response the Jetson decoded came from a call
    the phone placed; the reverse does not hold, because a call whose response
    the phone's sink refused, or whose frame the wire dropped, is counted only on
    the phone. Unavailable when `summary.json` is absent.

**Rendering rules, checkable by reading `report.md`:**

20. Every `report.md` this tool writes contains the literal string
    `## Session summary`, including one written for a zero-tick drive.
21. The `**Overall:` line always carries the axis clause, in both the
    all-answered and the not-all-answered form.
22. The count of answering axes never appears without the enumeration of the ones
    that did not, in the same section.

---

## Section 7 — What each aggregate is, and what moves it

Condition 2 of §3 in table form. Every row names the fixture that changes the
value, which is what a test asserts and what the pin removes.

| aggregate | what moves it | fixture |
|---|---|---|
| `thermal.answered` | one tick whose `thermal.jetson.basis` is `stale` instead of `measured` | a two-tick log, one of each basis; asserts `1 of 2` and `{"stale": 1}` |
| `thermal.unanswered_by_reason` | changing the reason word on the absent tick | two absent ticks with different reasons; asserts both keys present, neither merged |
| `thermal.vocabulary_violations` | a tick carrying `"basis": "absent", "reason": "gremlins"` | asserts the key is counted verbatim and listed, and rule 2 still holds |
| `rates.attempted` | two ticks echoing one `at_mono` versus two ticks with distinct `at_mono` | asserts 1 and 2 respectively — the D7 deduplication |
| `rates.answered` | one report with `age_s` of 11.0 against `MAX_TELEMETRY_AGE_S = 10.0` | asserts `1 of 2` and `{"stale": 1}` |
| `telemetry_window.observed_median_s` | three reports 1.0 s apart versus 2.0 s apart | asserts 1.0 and 2.0; asserts `null` and the axis unbuildable with one report |
| `percentiles_suppressed` | commanded `here_hz` 0.05 against commanded `camera_hz` 5.0, same window | asserts suppressed for the first and present for the second |
| `comparable` | the same log with `sensing.shadow` true and false | asserts False with the mode named, and True |
| `api_calls.calls_total` | `here_calls` 0 -> 9 within one session | asserts 9; asserts 0 when every value is 0 |
| `api_calls.by_session` | the same series split across two `session_id`s, the second starting at 0 | asserts two rows and no negative total — the D11 redial case |
| `api_calls.answered` | one tick with `absent: "no_telemetry"` | asserts `here_calls` is null, the tick is counted under `no_telemetry`, and `calls_total` does not move |
| `expected_from_commanded` | doubling `sensing.rates["here_hz"]` across the log | asserts the integral doubles |
| `triggers.rules_missing` | a rule `not_evaluable` with `missing: ["policy_margin"]` versus `["feed_congestion"]` | asserts the field name reaches the summary, not just the count |
| `commanded_by_value` | a log whose `camera_hz` takes 1.0 on 3 ticks and 5.0 on 1 | asserts `{"1.0": 3, "5.0": 1}` and `commanded_distinct == 2`; asserts the time mean is 2.0 and is labelled a time mean |
| `failures.answered` | a tick with `failures.basis == "absent", reason "sampler_stopped"` | asserts `0 of 1` and the reason key |
| reconciliation 11 | `blind_ticks` 40 with `sources["camera.blind_ticks"]["total"]` 0 | asserts the reconciliation fails, both numbers are printed, and the axis is named in the headline |
| reconciliation 16 | one `failure_event` open record deleted from the log | asserts the reconciliation fails rather than the summary silently agreeing |
| reconciliation 18 | `summary["sensing"]["rules_by_status"]` altered by one | asserts both maps are printed and neither is preferred |
| the zero-tick branch | a `metadata.jsonl` with only `failure_scan` lines | asserts `report.md` exists, contains `## Session summary`, names `no tick records`, and the exit code is 2 |

**Two aggregates deliberately absent, and why**, so a validator does not report
them as omissions: a count or series over `sensing.mode.flips` (§3 — `flip_to`
has no caller that writes a run directory, so it is constant by construction);
and any reconciliation between `summary["stats"]`'s percentiles and the report's
whole-drive percentiles (D16 — they measure different windows and are both
correct, so a check that is expected to fail is not a check).

---

## Section 8 — Behaviour changes

**Zero.** Nothing this task adds evaluates a predicate the code does not
evaluate, moves a rate, changes a decision, or puts a byte on the wire. There is
one record change and one rendering change, and each is fenced.

**Record change: `reference_from` returns two more keys.** The values are read
off `PhoneTelemetry`, an object the tick loop already holds; no new call, no new
lock, no new thread, no new failure mode. `TickOutcome.reference` is consumed by
`run_demo` (written into the tick record) and by `score_shadow._reference_witness`
(:325-349), which indexes `absent`, `age_s`, `at_mono`, `achieved` and `dropped`
by name — so two additional keys are inert there.

*Fenced by:* a test asserting `Decision.to_record()` is byte-identical over a
scripted corpus; the golden-frames test asserting `frame_sha256` is unchanged for
every message type (`specs/transport_golden_frames.json`); a test asserting
`TickOutcome.to_record()` differs from the pre-task shape in exactly the two keys
and no other; and `score_shadow`'s `replay_identity` reporting 0 mismatched on a
recorded fixture, which it does because `reference` is not part of the replay
comparison at all (:181 compares `Decision.to_record()`'s keys).

**Rendering change: the `Overall` line gains a clause.** `report.md` only.
`report.json["overall_pass"]`, `report.json["gates"]`, and `main()`'s return
value are untouched.

*Fenced by:* a test asserting `report.json["overall_pass"]` and the exit code for
a fixture that both passes every gate and has an unanswered axis; and a test
asserting the clause is present in both forms.

**What does not change, asserted rather than argued.** `Inputs` keeps its
seventeen fields. `SensingController` is not edited. `summary.json` gains and
loses nothing — its ten keys and every nested shape are as they are today, and a
test asserts the key set of a summary written by a scripted run is unchanged. The
39-slot coverage identity and the missingness denominator are untouched, because
no encoder slot and no `field_sources` entry is added. No gate is added and no
gate threshold moves.

---

## Section 9 — Byte cost

Measured with `json.dumps` and the default separators `MetadataLogger.write`
uses — `", "` and `": "`, which are not compact, and omitting that is a way to be
wrong by about ten per cent. `report.json` is written with `indent=2`
(eval_run.py:1502) and is sized that way. **Every field is enumerated**, because
three estimates in this section have missed: task 36's by 26 per cent low
(one field omitted entirely), task 37's by 39 per cent high (3 cooling devices
assumed against 13 on the hardware), and task 38's by 18 per cent high.

**Base tick record**, and how much of it has ever been measured. Task 35 measured
means of 8,511 and 8,552 B on two drives. Task 36 added a **measured** 852 B,
giving 9,363 B. Task 37 added an **estimated** 409 B whose field measurement was
never reported. Task 38 added an **estimated** 131 B. So the base is **9,903 B,
of which 540 B has never been measured.**

**Per tick — two fields on `sensing.reference`,** computed field by field:

| variant | added bytes, inline including the leading `", "` |
|---|---|
| `"here_calls": 36, "here_errors": 0` (two digits) | **36** |
| `"here_calls": 180, "here_errors": 1` (three digits) | **37** |
| `"here_calls": 54000, "here_errors": 12` (five digits) | **40** |
| `"here_calls": null, "here_errors": null` (no telemetry) | **41** |

The whole `reference` block goes from 190 B to 226 B on the present branch and
from 93 B to 134 B on the absent branch. **The absent branch is the more
expensive one** — `null` is four characters where a two-digit integer is two —
which is counter-intuitive and is the same direction task 38's degraded scan
record went.

36 B on 9,903 B is **0.36 per cent**. Over 900 ticks: **32,400 B**. On a drive
with no phone the `sensing` block does not exist at all, so the per-tick cost is
**0 B**.

**`report.json`**, sized by constructing the blocks with realistic content and
`indent=2`:

| block | bytes |
|---|---|
| `session_summary` (7 axes, 9 reconciliations, inputs) | **4,487** |
| `sensing` (4 rate rows, 6 triggers, 4 rules, 1 HERE session) | **3,346** |
| total added to `report.json` | **7,833** |

**`report.md`**: `## Session summary` is 8 lines healthy and about 20 with three
unanswered axes and one failed reconciliation; `## Sensing` is about 24 lines.
Roughly **2,500 B**, an estimate, because no `report.md` exists in the tree to
measure a base against — generated artefacts are not committed.

**`summary.json`: 0 B. On the wire: 0 B.**

**Drive total**, 900 ticks over 180 s: 32,400 + 7,833 + 2,500 = **42,733 B, about
42 KiB**, against a metadata log of 900 x 9,903 = 8,912,700 B, about 8.5 MiB —
**0.48 per cent**.

**What this estimate is most sensitive to, named in advance so the experiment
checks it rather than discovering it. The dominant term is the tick count**, and
it is the one thing this plan cannot bound: `loop.target_hz` is `0` in
`config.yaml:111` — run as fast as the detector allows — so the tick rate is set
by the hardware. 900 ticks over 180 s is 5 Hz, task 33's measured rate. At 30 Hz
the same drive is 5,400 ticks and the per-tick term is 194,400 B, at which point
it is 95 per cent of the addition and `report.json` is noise.

The rest, in order:

1. **Whether the drive has a phone at all.** No phone, no `sensing` block, no
   per-tick cost; the whole addition is then the ~10 kB of report blocks.
2. **How many ticks have no telemetry.** The absent branch costs 41 B against 36,
   so a drive whose phone never reported pays 14 per cent more per tick, not
   less.
3. **The digit width of `here_calls`.** 36 B at two digits to 40 B at five. Five
   digits needs 10,000 calls, which at the active `here_hz` of 0.2 Hz is about
   14 hours of driving. 36-38 B covers every drive this project will run.
4. **The number of distinct commanded rate values.** `commanded_by_value` is a
   census. Today `camera_hz` takes two values, but `THERMAL_SCALE`
   (sensing_controller.py:47-58) has **four distinct multipliers** — 1.0, 0.6,
   0.3, 0.15 — across its eight status keys, and only `camera_hz` and `here_hz`
   are scaled. Enumerating the products: `camera_hz` can take **8** distinct
   values, `here_hz` **7** (0.03 is reachable two ways), `gps_hz` and `imu_hz`
   **1** each, for at most **17** across the four keys against 4 today. At about
   24 B per extra value that is a ceiling of about **310 B** on `report.json`,
   not the kilobytes a careless reading of "eight statuses" would give.
5. **Reconciliation violations.** A drive where a per-source identity fails lists
   the offending rows; 30 sources at about 60 B each is 1.8 kB worst case.
6. **The base denominator carries 540 B that was never measured** (task 37's 409
   plus task 38's 131), so the 0.36 per cent ratio is against a base with an
   unverified term. **The absolute 36 B is the number to compare against.**

---

## Section 10 — The work

1. `policy/sensing_loop.py`: two keys on both branches of `reference_from`, with
   the docstring saying what they are and that they are cumulative for the
   phone's service run.
2. `eval_run.py`: `_read_summary` and `_read_log_health` extracted from their
   current inline sites so `main()` can call them before `analyze`; `analyze`
   gains `loaded=None`.
3. `eval_run.py`: `AXES`, one builder per axis, `session_summary()`, and
   `reconciliations()`.
4. `eval_run.py`: `sensing_result()` — the rate, trigger and HERE block from tick
   records, cross-checked against `summary["sensing"]`.
5. `eval_run.py`: `_session_summary_lines()`, `_sensing_lines()`, the `Overall`
   clause, and `render_session_summary_only()` for the zero-tick branch.
6. `eval_run.py`: the `main()` branch (§5.5).
7. Tests: §7's table, one per row, plus the six shape rules (§6 rules 1-5, 8) and
   the nine reconciliations.
8. `scripts/remutate.py`: the pins in §11.
9. `ARCHITECTURE.md` §9: two sentences — that `report.md` opens with a per-axis
   session summary, and that `here_calls`/`here_errors` now ride on the per-tick
   `reference` block.

**Not in the work: running the drive.** Every number in §9 is an estimate until
`experiment_dsrc` runs, and this plan does not run it.

---

## Section 11 — Mutations to pin in `scripts/remutate.py`

Each names a defect a specific test is supposed to catch. Anchors are the exact
source text at the time of writing and will be re-anchored during implementation.

1. **"summary: a zero-answering axis is dropped from the headline list"** — the
   rendering filter changed from `answered != attempted or attempted == 0` to
   `answered != attempted`, so an axis whose instrument never ran vanishes from
   the list. Caught by the `0 of 0` test (§6 rule 5).
2. **"summary: an axis's census is derived by subtraction"** —
   `unanswered_by_reason` replaced by `{"unanswered": attempted - answered}`.
   Caught by the two-different-reasons fixture, which asserts both keys survive.
3. **"sensing_loop: a phone never heard from reports zero API calls"** — the
   absent branch's `"here_calls": None` changed to `0`. Caught by §6 rule 8's
   test.
4. **"summary: achieved is averaged over ticks rather than reports"** — the
   `set` over `at_mono` removed. Caught by the two-ticks-one-report fixture.
5. **"summary: percentiles are reported for a modality slower than the window"** —
   the `lambda_per_window` guard removed. Caught by the `here_hz` 0.05 fixture.
6. **"summary: achieved is compared to commanded on a shadow drive"** — the
   `ever_live` guard on `comparable` removed. Caught by the shadow/live pair.
7. **"summary: a HERE counter that restarts at zero is differenced across the
   redial"** — the per-`session_id` split removed. Caught by the two-session
   fixture, which asserts no negative total.
8. **"summary: the blind-tick reconciliation is not checked"** — reconciliation
   11 removed. Caught by the 40-versus-0 fixture.
9. **"summary: a missing summary.json is reported as a held reconciliation"** —
   the `unavailable` arm changed to `held: True`. Caught by the no-summary
   fixture.
10. **"eval_run: a zero-tick drive produces no report"** — the `main()` branch
    removed so `analyze`'s `SystemExit` runs. Caught by the failure-scans-only
    fixture.
11. **"eval_run: the Overall line drops its axis clause"** — the clause removed.
    Caught by §6 rule 21's test.
12. **"summary: a vocabulary violation is absorbed into a known key"** — the
    verbatim-count arm replaced by a fallback to the first vocabulary member.
    Caught by the `"gremlins"` fixture.

---

## Section 12 — Open items

They are allowed to stay open. Unclosed open items are not defects.

1. **Calls placed before a session's first observed report are not counted.**
   `PhoneLink._rebind` clears `_telemetry` (phone_link.py:479) and a different
   handset's `HerePipeline.calls` starts at zero, so the summary counts
   `last - first` per session and reports `sum(first)` as `uncounted_prefix`.
   That is a bound, not a measurement. Closing it needs the phone to stamp its
   counter with a service-run identity, which is a wire change.
2. **The telemetry window is measured from arrival instants, not from the
   phone's own window.** A telemetry frame lost in transport widens one arrival
   interval, and the time-weighted achieved mean weights that window as if it
   were longer. `windows_not_observed` bounds how often it happened;
   distinguishing "the phone did not send" from "the wire lost it" is not
   possible from this side, because the phone's `TelemetryReporter.Stats`
   (`reports`, `skipped`, `refusedBySink`) reaches logcat and nothing else.
3. **Three of the seven axes are promotions.** `latency`, `thermal` and
   `provenance` count numbers `report.md` already prints in their own sections.
   Their value is placement — a reader who stops after the first section sees
   them — not measurement. If a later reading finds the placement does not
   change behaviour, they can be cut to a pointer without touching the other
   four.
4. **Four of the nine reconciliations need `summary.json`.** A drive that did
   not reach teardown has none (run_demo.py:708 is the last statement before
   `close()` and everything from :698 onward is unguarded), so reconciliations
   16-19 report `unavailable` on exactly the drive whose records most need
   checking. The tick-record path still builds every axis, which is why the
   summary is computed from ticks and cross-checked against the summary rather
   than the other way round.
5. **`lambda_per_window` is a rule of thumb with a defensible boundary and no
   measured justification.** Below 1.0 the median is provably 0 for a Poisson
   arrival process; between 1 and about 10 the percentiles are noisy and this
   plan reports them anyway, naming the quantisation. Where exactly the
   percentiles become useful cannot be settled without a drive.
6. **`summary["camera_file_recoveries"]` is constant by construction on every
   phone drive** (§3), and so is `failures.sources["camera.file_recoveries"]`.
   This plan names it and does not remove it: deleting a summary key changes an
   artifact shape that `eval_run` and `score_shadow` both tolerate but that
   nothing has re-read, and the source row is task 38's to own.
7. **`summary["sensing"]["inputs_by_source"]` stays unrendered.** It is a third
   account of provenance beside the per-tick `field_sources` and
   `score_shadow`'s own recomputation (score_shadow.py:428-435), over three
   fields rather than 39. The provenance axis reads `field_sources`, which is
   the encoder's own map; rendering both would be the two-accounts-of-one-machine
   problem task 37's D5 refuses. Named so a validator does not report it as an
   omission.
8. **Nothing verifies that `here_calls` is monotone within a session.** The
   phone increments it under a lock (HerePipeline.kt:112-116) and the wire
   refuses a negative (`check_count`), but a decrease within one `session_id`
   would mean the service restarted without the session doing so. The summary
   counts only positive deltas and reports a decrease under
   `counter_went_backwards` — task 38's own word — rather than clamping. Whether
   it can happen is unknown.
9. **The zero-tick report has no plots and no gates**, so a reader who diffs two
   `report.json` files will find `gates` present in one and absent in the other.
   `analysis_absent` names it. Making the two shapes uniform means emitting an
   empty gate table, which reads as five gates that passed.
10. **No drive has been run against any of this.** Every count in §5.2's table is
    verified against the code; every byte figure in §9 is an estimate; and the
    claim that the summary makes task 37's and task 38's drives look different
    from healthy ones is a claim about rendering that has been reasoned, not
    observed.

---

## Section 13 — Scope boundary: what this task does not do

- **It creates no new file.** No `session_summary.json`, no new CLI. The session
  summary is a section of `report.md` and a key of `report.json`.
- **It changes no commanded sensor rate, in either direction, on either device.**
  `Inputs` keeps its seventeen fields, `SensingController` is not edited, and
  `Decision.to_record()` is byte-identical.
- **It changes not one byte on the wire.** No message gains a field, no channel
  is added, removed or re-policied, and the golden frames are unchanged.
- **It adds no encoder slot and no `field_sources` entry**, so the 39-slot
  coverage identity and the missingness denominator are unchanged.
- **It adds no gate and does not change `overall_pass` or the exit code.** The
  `Overall` line gains a clause; the value behind it does not move.
- **It changes no key of `summary.json`.** Everything it needs from a live path
  is two fields on one per-tick record builder.
- **It does not touch the phone.** No Kotlin file is edited, `SessionLog` gains
  no line shape, and `TelemetryReporter` is unchanged. `here_calls` has been on
  the wire since task 27; this task reads it.
- **It does not re-derive any producer's status.** Where the summary and a
  producer disagree it prints both and resolves nothing.
- **It does not merge the six existing surfaces.** `summary.json`,
  `shadow_score.json`, `log_health.json` and the phone's session log keep their
  shapes, their writers and their readers.
- **It does not make `analyze` tolerate a zero-tick drive.** `analyze`'s
  `SystemExit` stays; `main()` gains a branch that never reaches it.
- **It does not gate, retry, alarm or respond to anything.** It reports what the
  instruments answered and what they did not.
- **It does not run the drive.** The experiment is `experiment_dsrc`, separately,
  and every number in §9 is an estimate until it does.
