# Task 38 — Failure event log

> Written by `plan_dsrc_rec`: every decision below was taken by recommendation,
> without putting the question to the user. The decisions table says what was
> chosen and why; **none of it is user-approved**, and the "Open items" section
> holds every point where the recommendation was weak or where the code
> contradicted the first reading. This plan is the fixed target a validator
> audits the implementation against.

## The short version

Task 38 (plans/task_list.md:1422) asks for a failure event log covering GPS
dropout, HERE failure or quota exhaustion, dropped frames and transport stalls,
with recovery outcome.

**Almost every failure the task names is already detected, and almost all of it
is already counted.** Reading the code produced 98 distinct detected failures on
the phone and 88 on the Jetson (the inventory below). GPS dropout is detected in
three separate places; HERE quota exhaustion arrives at the Jetson as
`refused_by_reason["http_error:status 429"]`
(deployment/jetson/sensors/here_feed.py:414-415, :431-434); dropped frames are
counted in fifteen distinct places on the phone and four on the Jetson; and a
transport stall is a closed enum member, `SessionEndReason.STALLED`, raised by a
watchdog at deployment/jetson/transport/session.py:667-668. **So this task does
not build detection. It builds a time axis, an episode, and a reader.**

**Three properties are missing from every one of those records, and they are the
same three each time.**

1. **No time.** Of the 186 detections, four carry an instant:
   `TcpAcceptor.first_accept_error_mono_ns` / `last_accept_error_mono_ns`
   (transport/tcp.py:237-238), the phone's `thermal_change_at_mono_ns`
   (task 37), and the phone's `advisory_shown` line. One carries a duration:
   `rebinds[].down_s` (sensors/phone_link.py:509). Everything else is an integer
   that says how many and never when. `here_errors` moving from 3 to 4 says
   nothing about which second it moved in, so no failure can be aligned with a
   tick, with another failure, or with the thermal series task 37 built.
2. **No episode.** A counter says how many occurrences; nothing says when a
   condition began, when it ended, or whether it ended. "With recovery outcome"
   is the part of the task that no existing record can answer at all.
3. **No reader.** `MetadataLogger.writer_failure` and `.dropped_records`
   (logio/metadata_logger.py:57, :61) have zero non-test readers.
   `ChannelStats.seq_gaps` / `missing_seqs` (transport/session.py:590-592) — the
   only cross-device loss evidence in the system — are computed, serialised by
   `ChannelStats.to_record`, and then omitted by hand from
   `phone_link._wire_record` (:778-805), so a drive summary never carries them.
   `TcpAcceptor.stats()` (tcp.py:258-288) is called only by a bench script.
   `GpsDiagnostics` (sensors/gps_reader.py:54-72) reaches disk through nothing at
   all. And `{"type": "system_error"}` (logio/metadata_logger.py:208) is the
   repository's only existing failure *record*, written by one line and read by
   none — `eval_run.load_records` (:70-105) has no branch for it.

**The two failures that are not detected anywhere, and that the log's own
honesty depends on.** `run_demo`'s tick loop `continue`s at :515-519 whenever
the camera yields no frame for a full second, and nothing counts it: a drive
blind for 110 of 120 seconds and one blind for 0 write byte-identical artifacts
apart from a smaller tick count. And `_tick_loop` has no `except` at all
(:497-576, wrapped by `worker()` at :483-495, which is `try/finally` with no
`except`), so an exception out of `pipeline.step` or `sensing.on_tick` prints a
traceback to stderr, sets `stop`, and produces a summary indistinguishable from
a clean short run.

**Does it change behaviour?** In three places, each stated precisely in
§"Behaviour changes". Two are on the Jetson: `run_demo.worker()` gains an
`except BaseException` that records the exception and **re-raises**, and the
tick loop increments a counter on the `frame is None` branch it already takes.
One is on the phone: `SessionLog` gains a fourth line shape. **No commanded
sensor rate moves, no controller input is added, no decision changes, and not
one byte changes on the wire.** The rule this plan applies throughout: counting
an outcome on a branch the code already takes is not a behaviour change;
evaluating a predicate the code does not evaluate today is. Only the `except`
comes close to the second, and it changes no control flow.

**The channel decision is that there is no channel decision.** Nothing new goes
on the wire in either direction. The eight channels, their priorities, depths
and overflow policies (specs/transport_protocol.md:120-129,
transport/channels.py:67-79, Channels.kt:49-58) are untouched, and
`PhoneTelemetry` gains no field. The reason is in the task itself: the phone
failure that matters most is the link being down, and it cannot be sent over the
link. So the phone's own failures are written to `SessionLog`, which already
survives the drive on the handset, already spans every redial, and already has a
`dir`-tagged wrapper convention the Jetson-side reader skips unless it
recognises the shape (log/SessionLog.kt:95-125, eval_run.py:120-131).

**Scope boundary.** In: new `deployment/jetson/logio/failure_log.py`;
`run_demo.py` (construct/start/stop the sampler, one tick-record key, one
summary key, the worker `except`, the blind-tick counter, one post-close write);
`eval_run.py` (`load_records` gains a record type and becomes a dataclass, one
result block, one report section); `score_shadow.py` (the same unpack, one
line); `phone_link.py` (`_wire_record` gains `seq_gaps`/`missing_seqs`, and
`to_record` gains the acceptor's stats); `config.yaml` (two keys); Kotlin
`log/SessionLog.kt` and `SensingService.kt` plus the pipelines that report a
failure into it; tests on both sides; pins in `scripts/remutate.py`;
`ARCHITECTURE.md` §9. Out: the transport protocol and every channel; any
`PhoneTelemetry` field; `Inputs` and `sensing_controller.py`; the thermal
records (task 37 already gives thermal a time axis, an event stream and a
reader, and duplicating it would create the second source of truth task 37's D5
refuses); the session summary generator (task 39); and any gate on the failure
counts.

**Open items, in one line each** (details at the bottom): a drive that produces
zero ticks still cannot be reported, because `eval_run` raises at :467; a
model-loading failure happens before the run directory exists, so it has nowhere
to be recorded; the sampler's 1 Hz cadence bounds every instant it records to
±1 s and that bound is stated rather than removed; two of the sources it reads
reset on a redial and the plan's handling of that is argued, not measured; the
phone's failures reach the Jetson only offline; and the failure log cannot
record its own log's death, which is why `log_health.json` exists as a separate
file.

## Which of the four vocabularies this is

The section has built four and the brief forbids a fifth. This task needs three
different things named, and only the third is new.

**Two of them already have vocabularies, and this task imports both rather than
restating them.**

- **"Was failure detection running?"** is task 34's question, word for word.
  A source that was watched and never moved is `quiet`; a source that was watched
  and moved is `fired`; a source that could not be read is `not_evaluable` with
  `missing` naming what was missing. `count == 0` on both `quiet` and
  `not_evaluable`, which is exactly why the status word carries the distinction
  and the count never does. `failure_log.py` imports `RULE_FIRED`, `RULE_QUIET`
  and `RULE_NOT_EVALUABLE` from `policy.sensing_controller` (:144-146), the way
  `sensors/thermal.py:33` already does.
- **"When did we learn it, and how well?"** is task 33's question. Every instant
  the log records is either `measured` — the sampler observed the counter move
  at this instant, bounded by one sampler period — or `converted` — the peer
  stamped it and it crossed a clock — or `absent` with a named reason.
  `failure_log.py` re-exports `STAGE_BASIS_MEASURED`, `STAGE_BASIS_CONVERTED`
  and `STAGE_BASIS_ABSENT` from `sensors.time_sync` (:108-111) rather than
  retyping the strings, following `thermal.py:38-39` exactly. The absence
  reasons are `thermal.py`'s own, re-exported for the same reason:
  `ABSENT_NO_SAMPLE_YET` and `ABSENT_SAMPLER_STOPPED` mean here precisely what
  they mean there.

**The third thing is what names the failure, and this is where a fifth
vocabulary would get built by accident.** The answer this plan takes is that
**the failure log invents no names for failures.** A failure is identified by a
pair, `(source, reason)`, where:

- `source` is a member of a **registry of accounting sites that already exist**
  — `here.refused`, `wire.decode_errors`, `link.session_end`, and so on. The
  registry is an enumeration of places in this repository that already count
  something, not a description language. Adding a source means pointing at an
  existing counter.
- `reason` is drawn from **that source's own existing closed set**, and from
  nowhere else:

  | source | its own vocabulary | where it is defined |
  |---|---|---|
  | `here.refused` | `Outcome` (9 members) plus an HTTP status detail | sensors/here_feed.py, `class Outcome` |
  | `wire.decode_errors`, `wire.send_rejected` | `REASONS` (9 members) | transport/messages.py:78-98 |
  | `link.session_end` | `SessionEndReason` (6 members) | transport/session.py:64-70 |
  | `acceptor.accept_errors` | `errno` integers | transport/tcp.py:234 |
  | `clock.proxied` | the proxy-reason strings | sensors/phone_source.py:145 |
  | `phone.dropped` | `DROP_KEYS` (4 members) | transport/messages.py:62 |
  | `gps.fresh`, `camera.*`, `link.down` | one reason each, stated in the registry | new, one word per source |

  A test asserts that every `reason` a run emits is a member of the vocabulary
  its source declares, so a new string cannot appear in the log without a
  registry change appearing in the diff.

**Task 36's eleven-member provenance vocabulary is deliberately not extended**,
for the reason task 37 gave: the identity `set(field_sources) ==
set(encoded_slot_names())` is pinned on every tick and the missingness
denominator is a number already quoted in the paper. A failure is not an encoder
slot.

**One closed set is genuinely new, and it is three members long.** An episode's
`outcome`:

- **`recovered`** — the condition cleared while the source was still readable.
- **`open_at_end`** — the run ended with the condition still true.
- **`unobservable`** — the source stopped being readable while the episode was
  open, so this drive cannot say whether it recovered.

The third member is the one the section exists for. Without it, an episode that
lost its instrument and an episode that recovered are both "an episode with no
further occurrences", and the log would report a recovery that was never
observed. `unobservable` is to `recovered` what task 34's `not_evaluable` is to
`quiet`, and it is spelled differently only because "the rule could not be
evaluated" and "the episode's end could not be observed" are different facts.

## The inventory: what is already detected, where it is recorded, who reads it

This is the section the brief asked for first. It is grouped by where the
detection lives. **"Reader" means a non-test reader**; `scripts/remutate.py`
hits are mutation source strings, not readers.

Three abbreviations: **C** = a cumulative count, **L** = a latest value that is
overwritten, **E** = one record per occurrence. **T** = the record carries an
instant.

### Jetson: sensors

| failure | where detected | recorded as | kind | reader |
|---|---|---|---|---|
| Serial port will not open | gps_reader.py:125-131 | raises `RuntimeError` | — | `selfcheck` only (run_demo.py:756) |
| UBX rate config failed, GPS silently stays at 1 Hz | gps_reader.py:158-159 | `diagnostics.last_error`, `rate_configured` False | L | `rate_configured` in `selfcheck`; `last_error` **none** |
| Serial read error mid-drive | gps_reader.py:168-170 | `last_error`, `continue` | L | **none** |
| NMEA parse failure | gps_reader.py:181-182 | `diagnostics.parse_errors += 1` | C | **none** |
| Ingest raised (thread-death guard) | gps_reader.py:186-193 | `ingest_errors += 1`, `last_error` | C+L | **none** |
| RMC status not 'A' — fix invalid | gps_reader.py:200, :227-243 | `GpsFix.valid=False`; lat/lon/speed **hold their previous values** while `t_mono`/`t_wall` are refreshed | L, T | observation_builder.py:193-197; beacon.py:119; here_feed.py:465; dashboard.py:114 |
| Fix stale / GPS dropout | gps_reader.py:270-272 `is_stale()` | a bool | — | **`is_stale` has no non-test caller.** The live gate is `observation_builder.py:168, :193-197` → `obs_diagnostics.gps_age_s` / `gps_fresh` |
| Scripted GPS dropout (sim) | gps_sim.py:213-226, :304-305 | **publishes nothing**; no counter, no reason | — | offline, from the declared profile in the `scenario` record |
| Camera `read()` returned False | camera_stream.py:155-156, :164-165 | `sleep(0.05); continue` | — | **none** |
| File decoder poisoned, reopened | camera_stream.py:183-194 | `_file_recoveries += 1` | C | `summary["camera_file_recoveries"]` (run_demo.py:610). `eval_run` never reads it |
| End of file / recovery budget exhausted | camera_stream.py:159-163, :187-188 | `end_of_stream = True`; the three causes are not distinguished | L | run_demo.py:517 |
| Frame dropped unconsumed (local) | camera_stream.py:173-174 | `_drop_counter += 1` | C | `summary["camera_dropped_frames"]`; eval_run.py:724 |
| Capture thread death, any other exception | camera_stream.py:147-181 — **no try/except in `_loop`** | thread dies, `end_of_stream` stays False, consumer blocks forever | — | **none** |
| JPEG decode failure (phone camera) | phone_source.py:428-434 | `decode_failures += 1`, **reset per session** at :395 | C | `summary["phone"]["camera"]` |
| Frame dropped unconsumed (phone camera) | phone_source.py:446-447 | `_drop_counter += 1`, reset per session at :394 | C | same |
| Phone stage stamps out of order | phone_source.py:74-75, :485-490 | `out_of_order_phone_stages`, plus a `StageTiming.absent` on the frame | C+E | same, and eval_run's stage table |
| Reader thread died | phone_source.py:253-266 | `self.failure = "..."`, re-raises | L | `health()` → `summary["phone"]` |
| Rebind aborted, old reader would not stop | phone_source.py:228-242 | `self.failure = "reader did not stop in time for the rebind"` | L | same |
| Timebase conversion refused → proxy stamp | phone_source.py:129-154 | `proxied += 1`, `proxy_reasons[reason] += 1` | C | `summary["phone"]["clock"]` |
| **HERE HTTP error — this is the quota path.** 429, 500 and the phone's `status 0` timeout all land here | here_feed.py:414-415 → :431-434 | `refused_by_reason["http_error:status 429"] += 1`, `_last_refusal` | C+L | `summary["phone"]["here"]`. **`eval_run` does not read it** |
| HERE body unparseable | here_feed.py:303-314, :417-419 | `refused_by_reason["unparseable"]` | C+L | same |
| HERE numeric field out of range / `OverflowError` | here_feed.py:240-257 | field becomes `None`, **uncounted** | — | — |
| Response stale (>30 s) | here_feed.py:461-463 | `Outcome.STALE` with `response_age_s` | L, T | feed_fusion.py:143; `last_outcome` |
| Fix unusable / stale for HERE | here_feed.py:465-493 | `Outcome.UNUSABLE_FIX` / `STALE_FIX`; the age is **only inside the detail string** | L | feed_fusion.py:143 |
| HERE reader thread raised | phone_link.py:588-595 | `here_failure` (L) + `here_failures` (C) | C+L | `summary["phone"]["here"]` |
| Thermal: six absence reasons, throttle events | thermal.py:45-66, :649-663, :689-696 | `{"type":"thermal_event"}` and `{"type":"thermal_sample"}`, both with `t_wall`/`t_mono` | E, T | eval_run.py:101-104; `## Thermal` |
| Thermal sample pass raised | thermal.py:440-450 | bare `pass` — **counted nowhere** | — | inferable only from `sample_gaps_s` |

### Jetson: transport

| failure | where detected | recorded as | kind | reader |
|---|---|---|---|---|
| Outbound queue overflow, oldest evicted | session.py:438-440 | `dropped_outbound += 1` | C | `summary["phone"]["wire"]` |
| Inbound queue overflow | session.py:607-609 | `dropped_inbound += 1` | C | same |
| **Sequence gap — the peer dropped something** | session.py:590-592 | `seq_gaps += 1`, `missing_seqs += n` | C | **`_wire_record` (:778-805) omits both.** Only scripts/run_loopback_pipeline.py:652 reads them |
| Outbound queued but never sent | session.py:346-347 | `abandoned_outbound` | C | `summary["phone"]["wire"]` |
| Write failed, peer gone / our end | session.py:501-518 | `abandoned_outbound += 1` + `_shutdown(PEER_CLOSED\|TRANSPORT_ERROR)` | C+E | `end_reason` |
| Frame decode refusal (framing) | session.py:575-576 | `_shutdown(FRAMING_ERROR)` — **the exception text is discarded** | E | `end_reason` only |
| **Stall timeout** — no read progress for 5 s | session.py:667-668 | `_shutdown(SessionEndReason.STALLED)` | E | `end_reason` in `sessions[]` / `rebinds[].previous_end_reason` |
| 23 distinct framing refusals on encode and decode | frames.py:93-324 | all raise `FramingError` | — | all collapse to one `end_reason == "framing_error"` |
| 9 handshake refusals incl. protocol version mismatch | handshake.py:106-155 | raise `HandshakeError` | — | via `SessionRefused` below |
| Handshake rejection / timeout | endpoint.py:237, :248-255 | `refused += 1`, `SessionRefused(peer, error)` | C+E | `phone_link.refusals[]` capped at 50, plus `refusals_not_kept` |
| Handshake worker leaked | endpoint.py:230-236 | `handshake_workers_leaked += 1` | C | `summary["phone"]` |
| Displacement — a second device took the session | endpoint.py:270-272 | `displaced += 1` | C | `summary["phone"]` |
| `SessionEnded` events | endpoint.py:309-315 | put on the event queue | E | **nothing drains it during a live session** |
| Accept loop died | endpoint.py:183-186 | `return` — no counter, no event, no log | — | **none** |
| **Transient accept failure** | tcp.py:350-378 | `transient_accept_errors`, `accept_errors_by_errno`, `max_consecutive_accept_errors`, **`first_accept_error_mono_ns` / `last_accept_error_mono_ns`** | C, **T** | `TcpAcceptor.stats()` is read only by scripts/run_transport_listener.py. **`PhoneLink` never calls it** |
| Peer message failed typed decode | messages.py:1282-1288 | `decode_errors`, `errors_by_reason[reason]`, `last_error` | C+L | `summary["phone"]["wire"]["messages"]` |
| We built a message our own decoder refuses | messages.py:1218-1223 | `send_rejected`, `rejected_by_reason`, raises `InvalidMessage` | C | same |
| Send with no session | phone_link.py:664-666 | `sends_without_a_session += 1` | C | `summary["phone"]["sent"]` |
| Send refused by the session | phone_link.py:673-675 | `sends_refused += 1`. Its own docstring (:726-732) records that eviction-and-return-True means this can only ever see a closed link, never a backed-up one | C | same |
| **Redial** | phone_link.py:508-513 | `rebinds.append({down_s, previous_end_reason, peer_device_id, session_id})` | E, **duration** | `summary["phone"]["rebinds"]`. `eval_run` never reads `summary["phone"]` |
| Gave up on the redial after 120 s | phone_link.py:439-445 (`rebind_timeout_s = 120.0`, :245) | `supervisor_ended = "gave_up_after_120s"` | L | `summary["phone"]` |
| Link went down | phone_link.py:428-431 | `logging.warning` | — | the log file |

### Jetson: run loop and logging

| failure | where detected | recorded as | kind | reader |
|---|---|---|---|---|
| **Camera yielded no frame for a full second** | run_demo.py:515-519 `continue` | **nothing** | — | — (scripts/run_phone_drive.py:198-201 counts it as `blind`; the production loop does not) |
| **Exception out of `pipeline.step` or `sensing.on_tick`** | run_demo.py:483-495 — `try/finally`, **no `except`** | traceback to stderr; `stop.set()`; a summary that reads like a clean short run | — | — |
| Tick overran its period | run_demo.py:573-576 — a negative `budget` is discarded by `if budget > 0` | **nothing** | — | — |
| GPS failed to start, run continued without it | run_demo.py:423-428 | `print()` to stdout, `gps = None` | — | **not in the summary** |
| Model or engine failed to load | detector.py:126-129, :154-155, :181-182; actor_runtime.py:75-127 | raise | — | `selfcheck` only. In `run_live`, `build_components` is at :386, **before `make_run_dir` at :404**, so there is no run directory to record it in |
| Metadata record refused, queue full | metadata_logger.py:69-70 | `dropped_records += 1` | C | **none** |
| **Metadata writer thread died** (full card, read-only mount) | metadata_logger.py:114-120 | `writer_failure = "..."`, thread returns | L | **none** |
| Records still queued at close | metadata_logger.py:105 | added to `dropped_records` — and this happens **after** `write_summary` at run_demo.py:660 | C | **none** |
| jtop sampler raised | metadata_logger.py:207-208 | `{"type": "system_error"}` | E, T | **none.** `load_records` has no branch for it |
| Unparseable line in metadata.jsonl | eval_run.py:88-93 | `unparseable += 1` | C | `log_integrity`, ANDed into `overall_pass` at :735. **Rendered only when it fails** (:967-974) |
| Unparseable line in the phone log | eval_run.py:120-125 | `continue` | — | **silently dropped** |
| Unparseable line on replay | replay_demo.py:37-42 | `continue` | — | **silently dropped** |
| Log shorter than the run claimed | eval_run.py:479-490 | `missing_ticks`, folded into `log_complete` | C | `log_integrity` |

### Phone

The full 98-row table is not reproduced here; the six facts that decide this
plan's design are.

1. **The wire carries six failure numbers and nothing else.**
   `PhoneTelemetry.dropped` (4 integers, `DROP_KEYS`), `here_calls` and
   `here_errors` (transport/messages.py:599-601). That is the entire failure
   surface reaching the Jetson.
2. **`dropped["camera"]` sums six distinct loss modes into one integer**
   (SensingService.kt:516-540): buffer displacement, encoder abandonment,
   encode failures, pack failures, sender refusals, and the transport channel's
   own drops — and the last term is per-session, so **the reported camera drop
   count decreases across a redial**.
3. **`here_calls` and `here_errors` are decoded on the Jetson and read by
   nothing.** Grep over `deployment/` and `scripts/` outside tests and golden-frame
   generators returns only `messages.py` itself.
4. **The phone has no quota handling.** `HttpHereClient` is "one attempt, no
   retry" (HereClient.kt:36-41); a 429, a 500 and an `IOException` timeout
   (`status = NO_RESPONSE = 0`) are all `errors++` at HerePipeline.kt:114. The
   distinguishing information is the raw `status`, which is forwarded to the
   Jetson on the `here` frame and lands in `refused_by_reason` keyed by status.
5. **The failures that matter most never reach the Jetson at all**: the IMU
   timebase mismatch that tears down IMU capture for the drive
   (ImuSource.kt:291-301), no IMU hardware (ImuSource.kt:155-160), the HERE key
   being absent (SensingService.kt:403-408), every dial failure and session end
   (SessionHolder.kt:198-235), permission revocation (SensingService.kt:198-205),
   come-up failure (:228-233), teardown failures (:858-865) and the
   `resourcesHeldAfterTeardown` census (:842-847). All of them reach exactly one
   place: a logcat line at teardown, SensingService.kt:761-811.
6. **`SessionLog` already is a durable, per-run, redial-spanning JSONL file with
   a `dir`-tagged wrapper convention.** Three line shapes today
   (log/SessionLog.kt:83, :95-105, :116-125); the Jetson-side reader
   (eval_run.py:120-131) recognises two of them and skips the rest. It is
   bounded (256 MiB, `MAX_BYTES` at :234), lossy-and-counted (queue depth 128 at
   :243; `droppedQueueFull`, `droppedNotRunning`, `droppedAtCap`, `failures` at
   :52-57), and retrieved out of band by scripts/run_device_session.py:198-201.

### The three facts the inventory establishes

- **Detection is not the gap.** 186 detections exist. The task's four named
  failures are all among them.
- **Time is the gap.** Four detections out of 186 carry an instant and one
  carries a duration.
- **Readership is the gap.** The single existing failure record type,
  `system_error`, is written by one line and read by none — which is precisely
  the outcome this task must not repeat, and the reason §"The record, exactly"
  ends with a `report.md` section rather than with a record shape.

## Unifying, or adding?

**Adding one stream, and the existing records cannot carry it.** The
justification is specific rather than general:

- **A counter cannot carry a time.** Adding a timestamp to
  `HereFeed.refused_by_reason` means changing it from `dict[str, int]` to a list
  of stamped occurrences, on a field whose 30 s window is queried on every tick.
  Doing that to twenty-eight counters is twenty-eight changes to live code paths
  in service of a logging task, and each one is a chance to change a value that
  a decision reads. The existing counters stay exactly as they are.
- **A counter cannot carry an episode.** An episode has two endpoints and an
  outcome. No existing field has a second endpoint, and `rebinds[].down_s` — the
  one duration in the system — is written only on a *successful* rebind, so an
  outage that never ended has no record of its length.
- **The failures with no home at all** — a blind tick, a tick-loop exception,
  the metadata writer's own death — have nowhere to be added *to*.
- **The stream must survive a drive that produces no ticks.** The drive with the
  most failures is the one where the camera never delivered a frame, and that
  drive writes no tick records at all (run_demo.py:515-519). A failure record
  that lives on the tick record is silent exactly when it matters. This is task
  37's D3 argument, and it applies here with more force, because thermal at
  least kept sampling; a failure log on the tick path would be recording the
  absence of the very thing that stopped.

**But the stream is a projection, not a second detector.** Every source in the
registry points at a counter or field that already exists; the sampler reads
them and derives episodes. Three consequences the plan holds itself to:

1. No source may be added without naming the existing site it reads. A test
   enumerates the registry and asserts every accessor resolves on a real object.
2. The log never contradicts the counter it reads. A test asserts, at teardown,
   that `sum(episode.n)` for a source equals the counter's own run total, or
   records the disagreement by name (`counter_went_backwards`) rather than
   clamping it — the phone silently clamps exactly this case at
   TelemetryReporter.kt:147, and the resulting `achieved` value is wrong with
   nothing saying so.
3. Where a counter and the log disagree, **the counter is right** and the log
   says it is wrong. The log is the second source of truth here, and it is the
   one that yields.

## Behaviour changes

Three, each bounded and each with the test that fences it.

**1. `run_demo.worker()` gains an `except BaseException` that records and
re-raises.**

```python
def worker() -> None:
    nonlocal last_print
    try:
        _tick_loop()
    except BaseException as exc:                       # noqa: BLE001
        if failures is not None:
            failures.note_pipeline_exception(exc)      # writes one failure_event
        raise
    finally:
        stop.set()
```

What changes: one record is written and one summary field is set. What does not
change: the exception still propagates, `threading`'s excepthook still prints
the traceback to stderr, `stop.set()` still runs in the `finally`, and the
teardown sequence is byte-identical. `BaseException` rather than `Exception`
because `KeyboardInterrupt` reaching this thread is itself a fact worth
recording, and the `raise` means catching it costs nothing.
**Fenced by:** a test that raises inside `pipeline.step`, asserts the exception
still reaches the caller, asserts `stop.is_set()`, and asserts the record; and a
mutation that removes the `raise`.

**2. The tick loop counts the branch it already takes.**

```python
frame = camera.wait_for_fresh(timeout=1.0)
if frame is None:
    if failures is not None:
        failures.note_no_frame(end_of_stream=camera.end_of_stream)
    if camera.end_of_stream:
        break
    continue
```

No predicate is added; `frame is None` and `camera.end_of_stream` are both
already evaluated on this path. The counter increments and, at the second
consecutive occurrence, an episode opens.
**Fenced by:** a test that drives the loop with a camera returning `None` and
asserts the count; and the no-rate-change replay below.

**3. The phone's `SessionLog` gains a fourth line shape.**

`{"dir": "fail", ...}` beside the bare header, `{"dir": "in"}` and
`{"dir": "shown"}`. It is a write to an existing bounded file on an existing
thread. The cost is that a failure line can displace a frame-header line when
the 128-deep queue is full, which is why D11 caps the rate.
**Fenced by:** a test asserting that with the failure rate cap saturated, the
number of header lines written is unchanged; and the teardown census test
asserting `resourcesHeldAfterTeardown == 0`, since no new resource is held.

**What does not change, asserted rather than argued.** `Inputs` keeps its
seventeen fields; `SensingController.decide` is byte-identical; not one byte
moves on the wire. The contract is closed the way task 34 closed its own:
**replay 15,000 randomized `Inputs` through the pre-task and post-task
controller trees and assert byte-identical `Decision.to_record()` and the same
md5**, plus a golden-frames test asserting `frame_sha256` for every message type
is unchanged.

## Decisions taken (by recommendation — not signed off by the user)

| # | Question | Options | Taken | Why |
|---|----------|---------|-------|-----|
| D1 | Does this task add detection, or read what is detected | (a) instrument each site to emit its own event; (b) one sampler that reads the existing counters | **(b)** | (a) is twenty-eight edits to live code paths, several of which a decision reads, in service of a logging task; task 36's plan claimed a scope boundary it did not have and moved `policy_margin` by accident. (b) touches no site's own logic. The cost is that (b) sees a counter's *movement*, not each occurrence's instant, which D9 bounds and states. |
| D2 | Where the sampler runs | (a) on the tick path; (b) its own thread at 1 Hz | **(b)** | The tick loop `continue`s without a record whenever the camera yields no frame (run_demo.py:515-519), so a failure log on that path goes silent exactly when the camera has failed, when the link is down for up to 120 s, and when the loop has stalled. Task 37's D3, and the 54.98 s stall it observed was visible only because its sampler was independent. 1 Hz also matches `ThermalSampler.interval_s` and `TelemetryReporter.PERIOD_MS`, so the three series share a cadence. |
| D3 | Where the module lives | (a) `sensors/`; (b) `logio/failure_log.py` | **(b)** | It reads no device. `sensors/` is for things with a `latest()` over a physical input; this reads other modules' counters and writes records, which is `logio`'s job. It imports the three status words from `policy.sensing_controller` and the three basis words from `sensors.time_sync`, both acyclic (neither imports `logio`). |
| D4 | How the sampler reaches a counter | (a) dotted string paths resolved with `getattr`; (b) one small typed accessor function per source | **(b)** | (a) turns a rename into a silent zero, which is this section's own defect class, and `ty` cannot see it. (b) makes a rename a lint failure. It also lets each accessor return `None` for "not readable now" explicitly, which is what `not_evaluable` is built from. |
| D5 | Whether an episode is opened for every source | (a) yes; (b) only for the sources the task names, others counted | **(a) with one flag** | One mechanism, not two. Every source gets episodes; the registry's `event_records` flag decides only whether a `failure_event` line is written per episode or the episode is rolled up in `summary["failures"]` alone. The task's four named failures plus the three self-honesty sources carry `event_records=True`; the rest are summary-only. That keeps the line count bounded without a second code path that could diverge. |
| D6 | What an episode is, for a counter | (a) one episode per increment; (b) one episode per unbroken run of passes in which the counter moved | **(b)** | (a) makes a 5 Hz failure 900 records. (b) gives the task's own words — a beginning, an end, and a recovery outcome — and the count of occurrences is carried inside the episode. What a reader loses is stated in D9. |
| D7 | When an episode closes | (a) the first pass with no movement; (b) after `quiet_passes_to_close` consecutive still passes | **(b)**, default 3 | (a) turns one flapping link into dozens of episodes. Three passes is 3 s at 1 Hz and is **derived** from `interval_s` rather than typed, following `MAX_EVIDENCE_GAP_S`'s precedent (sensing_controller.py:117-122). Stated on every episode as `close_after_s`, so a reader can undo it. |
| D8 | What happens when a source stops being readable while an episode is open | (a) leave it open and close it at teardown as `open_at_end`; (b) close it as `unobservable` | **(b)** | This is the whole task in one decision. A link that goes down while GPS is in a dropout takes the phone's counters with it; reporting "open at end" claims the failure persisted, and reporting "recovered" claims it stopped. Neither was observed. `unobservable` is the only true answer and is the third member of the outcome set. |
| D9 | How a failure repeating at 5 Hz is kept from flooding the log | (a) one record per occurrence; (b) deduplicate to one record; (c) an episode carrying a count | **(c)** | **What a reader loses, stated exactly**: within an episode, the individual occurrence instants. The record says `n` occurrences happened between `opened_t_mono` and `last_t_mono` and nothing finer. Two further bounds travel with it: the sampler's period, so every instant is a *first observation* accurate to ±`interval_s` and never the occurrence itself, carried as `bound_s`; and `first_pass_n`, the counter's movement on the opening pass, which is the only occurrence count that can be attributed to a single second. A reader wanting per-occurrence timing must go to the source's own instrumentation, which does not have it either — so nothing is lost that exists today. |
| D10 | The episode cap | (a) unbounded; (b) `MAX_EPISODES_PER_SOURCE` with the overflow counted | **(b)**, 100 | `phone_link.refusals`' precedent verbatim (`MAX_REFUSALS = 50`, :279-285, plus `refusals_not_kept`): the first ones are the diagnosis, the rest are a count. A peer failing to handshake once a second put 10,800 entries and half a megabyte into one summary before that cap existed. 100 rather than 50 because an episode is a coarser object than a refusal string. `episodes_not_kept` is per source and is in the summary. |
| D11 | Where a phone failure goes | (a) new `PhoneTelemetry` fields; (b) a new channel; (c) `SessionLog` lines, joined offline | **(c)** | The failure that matters most is the link being down, and (a) and (b) both require the link. (a) also grows a 1 Hz frame with a variable-length list, and this repository's own rule (task 37 D9) is that a list on a periodic frame is unbounded. (c) needs no protocol change, no channel decision, no golden-frame regeneration, and it survives a drive whose link never came up at all. **The channel table, priorities, depths and overflow policies are therefore untouched.** The cost: the phone's failures reach an analysis only when the session log is pulled off the handset, which is already true of the `advisory_shown` line the phone-join depends on. |
| D12 | The phone's rate cap | (a) none; (b) one line per kind per second plus a per-kind cap | **(b)** | `SessionLog`'s queue is 128 deep and a full queue drops the *frame header*, which is the file's reason to exist. One line per kind per second, `MAX_LINES_PER_KIND = 64`, and a per-kind `suppressed` count carried on the next line and in the teardown census. A reader loses the individual instants of suppressed occurrences within a second, and the `suppressed` count says how many. |
| D13 | Whether the phone's counters are read on the Jetson too | (a) no, the phone's log is the record; (b) yes, and the periods the link was down are `not_evaluable` | **(b)** | `dropped`, `here_calls` and `here_errors` arrive at 1 Hz and `here_calls`/`here_errors` are read by nothing today. Reading them gives the Jetson a live view while the link is up. The two records are not in conflict: the Jetson's is `not_evaluable` for exactly the window the phone's covers alone, and a test asserts the Jetson never reports `quiet` for a phone source over a window in which no telemetry arrived. |
| D14 | How a source that resets on a redial is handled | (a) diff it and accept the negative; (b) baseline keyed on `session_id`; (c) drop those sources | **(b)** | `_rebind` (phone_link.py:451-519) resets `telemetry_received`, `pings_answered`, `imu_received` and the whole `HereFeed`; every Kotlin `SessionStats` counter resets on every redial. Task 37's own experiment lost a drive's throttle count to exactly this. Each source declares `run_cumulative` or `session_scoped`; a session-scoped source's baseline resets to 0 on a `session_id` change, its open episodes close as `unobservable` (D8), and the sampler keeps its own run total by accumulating deltas. A *decrease* in a `run_cumulative` source is recorded as `counter_went_backwards`, never clamped. |
| D15 | Torn reads across threads | (a) read each counter field individually; (b) one snapshot call per source per pass, and one `session_id` for the pass | **(b)** | `_wire_record`'s own docstring (phone_link.py:745-755) records what individual reads cost: "two loads of `self.session` let a rebind land between them and pair one handset's peer address with another handset's channel counters". The pass takes `session.stats()`, `router.to_record()`, `here.to_record()` and so on once each, stamps the pass with one `session_id`, and any source whose `session_id` disagrees with the pass's is `not_evaluable` for that pass. |
| D16 | Whether the metadata log records its own failure | (a) in `summary["failures"]`; (b) a separate `log_health.json` written after `close()` | **(b)** | `run_demo` calls `write_summary` at :660 and `close()` at :661, and `dropped_records` is finalised inside `close()` at metadata_logger.py:105 — so a summary can never carry the logger's final drop count. Rewriting `summary.json` after `close()` would truncate a good summary if the second write failed, because `write_summary` opens `"w"`. A separate 211-byte file has no such failure mode, and `eval_run` reporting `null` for a run that predates it is the `jetson_ms_source` precedent (eval_run.py:447-451). |
| D17 | Whether the failure log gets a gate | (a) gate on zero failures; (b) gate on the sampler having run; (c) no gate | **(c)** | (a) would fail every drive with a redial, which is the normal condition this system is built for. (b) is close to right and is what open item 6 names, but it needs one drive to establish that every registered source is readable on real hardware, and this task is what produces that. `report.md` states the three-word status per source in words instead. `log_complete` (eval_run.py:489) already covers the one failure that genuinely invalidates a drive. |
| D18 | `load_records`' growing tuple | (a) add a seventh element; (b) return a frozen dataclass | **(b)** | The tuple is six wide and unpacked positionally in two places (eval_run.py:463, score_shadow.py:547). Three of its members are `list[dict]` and a fourth would be — adjacent, same-typed, and a mis-ordered unpack would type-check and pass most tests. This repository has already paid for "equal values hide a swap". One mechanical change, two non-test call sites, and future growth costs nothing. |
| D19 | Whether the two omitted transport counters are wired up | (a) leave them; (b) add `seq_gaps`/`missing_seqs` to `_wire_record` and `TcpAcceptor.stats()` to `to_record` | **(b)** | `seq_gaps` is the only evidence in the system that the peer dropped something, and `TcpAcceptor`'s pair is the only timestamped failure record in the transport. Both are computed today and thrown away by the record builder. Two small additions to `summary["phone"]`, and they are what makes `wire.seq_gaps` and `acceptor.accept_errors` legitimate registry sources rather than sources reading fields the drive artifact does not carry. |
| D20 | Whether thermal appears in the failure log | (a) yes, as sources; (b) no | **(b)** | Task 37 already gives thermal an event stream with instants, a 1 Hz record, a per-tick block, a summary rollup and a `report.md` section. Re-recording it would create the second source of truth task 37's D5 refuses, and the two could disagree. `report.md`'s failure section names `## Thermal` as the place to look instead. |

## The record, exactly

### `deployment/jetson/logio/failure_log.py` (new)

```python
from policy.sensing_controller import RULE_FIRED, RULE_NOT_EVALUABLE, RULE_QUIET
from sensors.time_sync import (
    STAGE_BASIS_ABSENT, STAGE_BASIS_CONVERTED, STAGE_BASIS_MEASURED)
from sensors.thermal import ABSENT_NO_SAMPLE_YET, ABSENT_SAMPLER_STOPPED

FAILURE_BASIS_MEASURED  = STAGE_BASIS_MEASURED    # re-export, not a new string
FAILURE_BASIS_CONVERTED = STAGE_BASIS_CONVERTED   # re-export, not a new string
FAILURE_BASIS_ABSENT    = STAGE_BASIS_ABSENT      # re-export, not a new string

#: The one closed set this task adds. Three members; see "Which of the four
#: vocabularies this is" for why the third exists.
OUTCOME_RECOVERED    = "recovered"
OUTCOME_OPEN_AT_END  = "open_at_end"
OUTCOME_UNOBSERVABLE = "unobservable"
OUTCOMES = frozenset({OUTCOME_RECOVERED, OUTCOME_OPEN_AT_END, OUTCOME_UNOBSERVABLE})

#: Why a source could not be read this pass. Closed.
MISSING_NO_PHONE      = "phone"           # no PhoneLink was constructed
MISSING_NO_SESSION    = "session"         # a link exists, no session is bound
MISSING_NO_TELEMETRY  = "telemetry"       # spelled as sensing_loop.reference_from's is
MISSING_SESSION_MOVED = "session_changed" # the pass straddled a rebind (D15)
MISSING_NO_SOURCE     = "source"          # the object itself is absent this run
MISSING = frozenset({...the five...})

#: The instrument disagreeing with the counter it reads. Never clamped.
BACKWARDS = "counter_went_backwards"
```

`Source` (frozen): `name`, `read` (the typed accessor, D4), `vocabulary`
(a frozenset or `None` for a single-reason source), `reason` (for
single-reason sources), `scope` (`"run"` or `"session"`, D14),
`event_records: bool` (D5), `device` (`"jetson"` or `"phone"`).

`Episode`: `episode_id`, `source`, `reason`, `opened_t_mono`, `opened_t_wall`,
`last_t_mono`, `n`, `first_pass_n`, `session_id`, `tick_id`, `basis`, `bound_s`,
`closed_t_mono`, `outcome`.

`FailureSampler(sink, sources, interval_s=1.0, clock=time.monotonic)` — the
1 Hz thread, the same shape as `ThermalSampler`. `start()` / `stop()`,
`latest(now)` for the tick block, `to_record()` for the summary, plus the two
direct-notification entry points the tick loop calls
(`note_no_frame`, `note_pipeline_exception`), which do not wait for a pass.

### The registry

Twenty-eight sources. Each row names the site it reads; the `E` column is D5's
`event_records` flag.

| source | reads | scope | E | reasons |
|---|---|---|---|---|
| `camera.blind_ticks` | the `frame is None` branch, run_demo.py:517 | run | **E** | `no_frame` |
| `camera.dropped_unconsumed` | `camera.dropped_frames` | run/session | | `unconsumed` |
| `camera.decode_failures` | `PhoneCameraStream.decode_failures` | session | **E** | `jpeg_decode` |
| `camera.file_recoveries` | `camera.file_recoveries` | run | | `decoder_poisoned` |
| `camera.end_of_stream` | `camera.end_of_stream` | run | **E** | `end_of_stream` |
| `camera.reader_failure` | `_PhoneSource.failure` | session | **E** | the string itself, capped |
| `gps.not_fresh` | `GpsReader.is_stale()` / `PhoneGpsReader.is_stale()` | run | **E** | `stale`, `invalid`, `absent` |
| `gps.parse_errors` | `GpsDiagnostics.parse_errors` | run | | `nmea_parse` |
| `gps.ingest_errors` | `GpsDiagnostics.ingest_errors` | run | | `ingest_raised` |
| `gps.last_error` | `GpsDiagnostics.last_error` | run | | the string, capped |
| `gps.rate_unconfigured` | `GpsDiagnostics.rate_configured` | run | | `ubx_rate_config_failed` |
| `here.refused` | `HereFeed.refused_by_reason` | session | **E** | `Outcome` members + status detail |
| `here.reader_failures` | `phone_link.here_failures` / `here_failure` | run | **E** | the string, capped |
| `here.proxied_stamps` | `HereFeed.proxied_stamps` | session | | `proxy` |
| `phone.dropped` | `PhoneTelemetry.dropped` | mixed (D14) | **E** | `DROP_KEYS` |
| `phone.here_errors` | `PhoneTelemetry.here_errors` | session | **E** | `here_error` |
| `link.down` | `phone_link.session is None` or closed | run | **E** | `no_session` |
| `link.session_end` | `_end_reason_of(session)`, `rebinds[].previous_end_reason` | run | **E** | `SessionEndReason` (6) |
| `link.refusals` | `_listener.refused`, `refusals[]` | run | | the refusal string, capped |
| `link.displaced` | `_listener.displaced` | run | **E** | `displaced` |
| `link.workers_leaked` | `_listener.handshake_workers_leaked` | run | | `handshake_worker_leaked` |
| `link.sends_lost` | `sends_without_a_session`, `sends_refused` | run | | `no_session`, `refused` |
| `link.supervisor_ended` | `phone_link.supervisor_ended` | run | **E** | the four strings |
| `wire.dropped` | `ChannelStats.dropped_outbound` / `dropped_inbound` | session | **E** | `outbound`, `inbound`, with `channel` |
| `wire.seq_gaps` | `ChannelStats.seq_gaps` / `missing_seqs` (D19) | session | **E** | `seq_gap`, with `channel` |
| `wire.decode_errors` | `ChannelMessageStats.errors_by_reason` | session | **E** | `REASONS` (9), with `channel` |
| `wire.send_rejected` | `ChannelMessageStats.rejected_by_reason` | session | | `REASONS` (9) |
| `acceptor.accept_errors` | `TcpAcceptor.stats()` (D19) | run | | errno integers |
| `clock.proxied` | `PhoneClockAdapter.proxy_reasons` | session | | the proxy-reason strings |
| `pipeline.exception` | the `worker()` `except` | run | **E** | the exception type name |

Two things this table decides. **`link.down` is the source that makes every
phone-side source honest**: while it has an open episode, every source whose
accessor needs a session returns "not readable", so those sources record
`not_evaluable` for exactly that window rather than `quiet`. And **the
`event_records` flag is the only difference between a task-named failure and the
rest** — the mechanism is one mechanism.

### Per pass, at 1 Hz: `{"type": "failure_scan"}`

Written every pass whether or not anything failed. This is the record that makes
"nothing failed" different from "nothing was watching", and it exists on a drive
with no ticks at all.

```json
{"type": "failure_scan", "seq": 180, "t_wall": 1756700000.123, "t_mono": 1234.567,
 "session_id": "a1b2c3d4", "ticks_seen": 5, "sources_n": 28,
 "sources_readable": 28, "unreadable": [], "open": []}
```

and on a drive whose phone has dropped — the shape that must not be confusable
with the one above:

```json
{"type": "failure_scan", "seq": 47, "t_wall": 1756699890.4, "t_mono": 1124.8,
 "session_id": null, "ticks_seen": 0, "sources_n": 28, "sources_readable": 19,
 "unreadable": ["phone.dropped", "phone.here_errors", "wire.seq_gaps",
                "wire.dropped_outbound", "clock.proxied"],
 "open": ["link.down", "gps.not_fresh"]}
```

`ticks_seen` is the number of tick records produced since the previous pass,
read from `pipeline._tick_counter`. It is a measurement, not a detection: no
threshold is applied and no stall event is raised. It is how a reader learns
that the tick loop stalled, which is how task 37's 54.98 s stall became visible,
and it is why `sample_gaps_s` had to be inferred there instead of read.

### Per episode, two records: `{"type": "failure_event"}`

Two records, not one, because a record written only at close does not exist for
an episode the run never closed. The open record is written the moment the
episode opens.

```json
{"type": "failure_event", "phase": "open", "episode_id": 7,
 "source": "gps.not_fresh", "reason": "stale", "device": "jetson",
 "t_wall": 1756700000.123, "t_mono": 1234.567,
 "basis": "measured", "bound_s": 1.0, "session_id": "a1b2c3d4",
 "tick_id": 431, "channel": null, "value": 4.213, "first_pass_n": 1,
 "detail": "gps_age_s 4.213"}
```

```json
{"type": "failure_event", "phase": "close", "episode_id": 7,
 "source": "gps.not_fresh", "device": "jetson",
 "t_wall": 1756700046.9, "t_mono": 1280.401, "outcome": "recovered",
 "duration_s": 45.834, "n": 229, "last_t_mono": 1279.398,
 "close_after_s": 3.0, "basis": "measured", "bound_s": 1.0,
 "session_id": "a1b2c3d4"}
```

An episode closed as `unobservable` carries the same shape with
`"outcome": "unobservable"` and `"duration_s"` measured to the last pass on
which the source was readable — never to teardown, because the interval after
that was not observed.

Reading rules a validator audits against:

- `basis` is a member of `{measured, converted, absent}`; `bound_s` is
  non-null if and only if `basis != "absent"`; for a counter-derived episode
  `basis` is `measured` and `bound_s == interval_s`, and the instant is the
  sampler's **first observation**, never the occurrence.
- Every `open` record has exactly zero or one `close` record with the same
  `episode_id`. An `open` with no `close` and no `open_at_end` entry in the
  summary means the log was truncated.
- `outcome` is a member of `OUTCOMES`; `duration_s == closed_t_mono -
  opened_t_mono` for `recovered` and `open_at_end`, and
  `duration_s == last_readable_t_mono - opened_t_mono` for `unobservable`.
- `reason` is a member of its source's declared vocabulary.
- `n >= first_pass_n >= 1`.
- For every source, `sum(episode.n)` equals the source's own run total, or
  `summary["failures"]["counter_went_backwards"]` names the source and the
  discrepancy.

### Per tick, on the tick record, key `"failures"`

Written in `run_demo`'s tick loop beside `record["thermal"]` (:553-554). Small
on purpose: the tick record already carries task 36's `field_sources`, task 34's
rule statuses and task 35's `reference.dropped`, so this block's only job is to
name what was open at decision time and to say how fresh that view is.

```json
"failures": {"open": [], "open_n": 0, "episodes": 0, "scan_age_s": 0.412,
             "basis": "measured", "unreadable_n": 0, "reason": null}
```

```json
"failures": {"open": ["link.down", "gps.not_fresh"], "open_n": 2, "episodes": 6,
             "scan_age_s": 0.412, "basis": "measured", "unreadable_n": 5,
             "reason": null}
```

and with the sampler not running — which must not read like the first:

```json
"failures": {"open": null, "open_n": null, "episodes": null, "scan_age_s": null,
             "basis": "absent", "unreadable_n": null, "reason": "sampler_stopped"}
```

`basis` is `stale` when `scan_age_s > 2 x interval_s`, with the bound
**derived** from `interval_s` rather than typed — task 37's D4, and the
`MAX_EVIDENCE_GAP_S` precedent. The block never depends on `sensing`, which can
be `None` (run_demo.py:471).

### `summary["failures"]`

Written from `FailureSampler.to_record()` in `run_demo`'s `finally`, immediately
after `summary["thermal"]` (:643) and before the `if phone is not None` block at
:644, so it is unconditional. It follows the sibling convention: a `to_record()`
of an owning object, not an inline dict.

```json
"failures": {
  "scan": {"passes": 180, "seq_last": 180,
           "interval_s": {"p50": 1.001, "p95": 1.008, "max": 1.012},
           "basis_counts": {"measured": 180, "stale": 0, "absent": 0},
           "absent_reasons": {}, "sources_n": 28},
  "sources": {
    "gps.not_fresh": {"status": "fired", "passes_attempted": 180,
                      "passes_readable": 180, "episodes": 2, "total": 229,
                      "by_reason": {"stale": 229},
                      "first_t_mono": 1234.567, "last_t_mono": 1280.401,
                      "events_written": 4, "episodes_not_kept": 0},
    "phone.dropped": {"status": "not_evaluable", "passes_attempted": 180,
                      "passes_readable": 131, "episodes": 0, "total": 0,
                      "by_reason": {}, "missing": ["telemetry"],
                      "first_t_mono": null, "last_t_mono": null,
                      "events_written": 0, "episodes_not_kept": 0},
    "here.refused": {"status": "quiet", "passes_attempted": 180,
                     "passes_readable": 180, "episodes": 0, "total": 0, ...}
  },
  "outcomes": {"recovered": 5, "open_at_end": 1, "unobservable": 2},
  "counter_went_backwards": {},
  "blind_ticks": 0, "pipeline_exception": null
}
```

`passes_attempted` and `passes_readable` are the pair that carries the evidence
for the `quiet` claim, the way task 37's confirmation drive prints "241 of 241
passes fully readable" rather than asserting "readable throughout". `missing` is
present only on a `not_evaluable` row, the way `RuleCheck.to_record` emits it.

### `log_health.json`

Written by `run_demo` after `logger.close()` at :661, from the logger's own
final state (D16). One file, one instant, no duplication of anything in
`summary.json`.

```json
{"t_wall": 1756700180.9, "t_mono": 1414.9, "dropped_records": 0,
 "writer_failure": null, "queue_depth": 50000, "thread_alive_at_close": true,
 "path": "metadata.jsonl", "bytes_on_disk": 8402113}
```

### The phone: a fourth `SessionLog` line shape

No wire change. `SessionLog.offerFailure(kind, atMonoNs, atWallNs, n, detail)`
writes:

```json
{"dir":"fail","at_mono_ns":547121961348739,"at_wall_ns":1787698281076000000,
 "kind":"link.dial_failed","n":3,"detail":"ConnectException: failed to connect"}
```

`at_mono_ns` is `SystemClock.elapsedRealtimeNanos`, the clock every other phone
stamp uses, and it is **not converted** — the same reason task 37 gave for a
thermal transition: converting attaches an estimate-dependent bound to a number
nobody differences. `kind` is drawn from a closed Kotlin `FailureKinds` object
whose members are asserted against the Python registry's phone-device rows by
`InteropTest`, the mechanism `scripts/refusal_reasons.py` already uses so that
"this matches Python" is executed rather than asserted by hand.

The kinds reported are the ones §"Phone" fact 5 lists as unreachable from the
Jetson: `link.dial_failed`, `link.session_ended`, `imu.no_hardware`,
`imu.timebase_mismatched`, `here.unconfigured`, `service.come_up_failed`,
`service.permission_revoked`, `service.teardown_failed`,
`service.resources_held`, `log.self`. Each is emitted from the site that already
detects it, at the point it already increments its counter.

`eval_run.load_phone_log` gains a third recognised shape and returns the failure
lines alongside the two it reads today; the join surfaces them in
`## Failures` under a `phone (offline)` heading, and a phone log that was not
supplied produces the line "phone-side failures: not read (no --phone-log)"
rather than nothing.

### `eval_run.py`

`load_records` becomes a frozen dataclass (D18) with a `failure_scans` and a
`failure_events` field alongside the five it has. `result["failures"]` carries
`summary["failures"]` when `summary.json` has one, plus what only the records
can say: `scan_gaps_s` (p50/p95/max of the interval between consecutive
`failure_scan` records, which is how a stalled sampler becomes visible),
`ticks_seen_per_pass` (p50/min, which is how a stalled *tick loop* becomes
visible), `ticks_by_basis` from the tick blocks, and `episodes` as the joined
open/close pairs. A run recorded before this task reports `null` and is **not** a
failed drive.

`report.md` gains, between `## Thermal` (spliced at :1091) and `## GPS` (:1094):

```
## Failures

- 28 sources scanned on 180 passes; scan interval p50 1.001 s, max 1.012 s
- ticks seen per pass: p50 5, min 5 -- the tick loop did not stall
- 8 episodes: 5 recovered, 1 open at end, 2 unobservable
- gps.not_fresh: FIRED -- 2 episodes, 229 occurrences, longest 45.8 s, recovered
- link.down: FIRED -- 1 episode, 12.4 s, recovered (rebind on session a1b2c3d4)
- here.refused: quiet -- readable on 180 of 180 passes, 0 refusals
- phone.dropped: NOT EVALUABLE on 49 of 180 passes -- missing telemetry;
  this drive says nothing about the phone's drops during those 49 s
- blind ticks: 0; pipeline exception: none
- log: 0 records dropped, writer healthy
- thermal failures are recorded separately -- see ## Thermal
```

and, when the sampler did not run, the line that must not read like the ones
above:

```
- failure sampling: NOT EVALUABLE -- sampler_stopped on 900 of 900 ticks;
  this drive says nothing about whether anything failed
```

### `config.yaml`

```yaml
logio:
  failures: true                     # failure episode log + 1 Hz scan records
  failure_interval_s: 1.0            # freshness bound on the tick block is 2x this
```

## The work

1. **`logio/failure_log.py`** — the constants, `Source`, `Episode`,
   `FailureSampler`, and the registry as a module-level tuple of `Source`. The
   three status words imported from `policy.sensing_controller`, the three basis
   words and the two absence reasons re-exported from `sensors.time_sync` and
   `sensors.thermal`.
2. **`run_demo.py`** — construct and `start()` the sampler under
   `config["logio"]["failures"]`, beside `SystemStatsSampler` and the thermal
   sampler; `record["failures"] = failures.latest()` beside
   `record["thermal"]` (:553-554); the `worker()` `except` (behaviour change 1);
   the `note_no_frame` call (behaviour change 2); `failures.stop()` in the
   `finally` before `logger.close()` so its last records flush;
   `summary["failures"]` after :643; the `log_health.json` write after :661.
3. **`config.yaml`** — the two keys.
4. **`phone_link.py`** — `_wire_record` gains `seq_gaps` and `missing_seqs` per
   channel; `to_record` gains `acceptor` from `TcpAcceptor.stats()` (D19). Two
   additions to a record builder; no counter's own logic changes.
5. **`eval_run.py`** — `load_records` to a dataclass (D18) with the two new
   record types; `load_phone_log`'s third shape; `result["failures"]`;
   `_failure_lines`; the `SystemExit` at :467 gains the counts of failure
   records present, so a zero-tick drive says what it does hold.
6. **`score_shadow.py`** — the `load_records` call site (:547), one line.
7. **Kotlin `log/SessionLog.kt`** — `offerFailure`, the rate cap and its
   `suppressed` counters, and the `Stats` fields for them.
8. **Kotlin `FailureKinds.kt`** (new, in `log/`) — the closed kind set.
9. **Kotlin `SensingService.kt`** and the pipelines — one `offerFailure` call at
   each site that already detects and counts the condition. No new detection.
10. **`ARCHITECTURE.md` §9** — the paragraph gains `type: failure_scan`,
    `type: failure_event`, the per-tick `failures` block and `log_health.json`.
11. **Tests and pins** (below).

## Tests, and what each one proves

**Task 37's lesson is carried into this section explicitly: a fixture's failure
mode has to be the field's.** There, a deleted file and a denied permission both
raised `OSError`, the real device raised `TypeError`, every test passed, and the
feature did nothing on the only machine it was written for. This task's
equivalent hazard is sharper, because every source reads through an accessor
into a real object: **a fake counter object cannot fail the way the real one
does.** Tests 1 and 2 exist for that and nothing else.

Python, `deployment/jetson/tests/test_failure_log.py` (new) unless stated:

1. **Every registry accessor resolves against the real object.** Construct a
   real `PhoneLink`, `HereFeed`, `MetadataLogger`, `CameraStream`,
   `GpsReader` and `MessageRouter` — not fakes — and assert every `Source.read`
   returns an `int`, a `str` or `None`, and never raises. **A renamed counter
   fails here and nowhere else.** This is the test that would have caught the
   task-37 class of defect at plan time.
2. **A source whose object is absent is `not_evaluable`, not `quiet`.** The same
   registry with `phone=None`: every phone-device source reports
   `not_evaluable` with `missing: ["phone"]`, and `passes_readable` is 0 while
   `passes_attempted` is the pass count. Asserts both halves in one test so
   deleting either is visible.
3. **`quiet` and `not_evaluable` are different records.** A source readable
   throughout that never moved gives `{"status": "quiet", "episodes": 0}` with
   no `missing`; a source that could not be read gives
   `{"status": "not_evaluable", "episodes": 0, "missing": [...]}`. **This is the
   task's headline assertion** and is written as one test with both halves.
4. **A counter that moves opens exactly one episode, not one per increment.**
   A counter advancing by 5, 3 and 7 over three passes closes as one episode
   with `n == 15` and `first_pass_n == 5`. Pins D6 and D9.
5. **An episode closes only after the configured still passes.** A counter that
   moves, is still for two passes, then moves again is one episode, not two.
   Constructed at `quiet_passes_to_close=3`.
6. **The close threshold is derived from `interval_s`, not typed.** Construct at
   `interval_s=0.2`; the still window is 0.6 s, not 3.0 s. The
   `MAX_EVIDENCE_GAP_S` failure mode.
7. **A source that becomes unreadable closes its episode as `unobservable`.**
   Not `recovered`, not `open_at_end`. `duration_s` measures to the last
   readable pass, not to teardown. **Pins D8, the decision the third outcome
   member exists for.**
8. **All three outcomes are reachable and distinct.** One test producing
   `recovered`, `open_at_end` and `unobservable`, asserting three different
   strings and that all three are in `OUTCOMES`.
9. **A session-scoped counter is not diffed across a redial.** A source at 40
   before a rebind and 3 after produces no episode of size -37 and no episode of
   size 3 attributed to the old session; the run total is 43. **Pins D14, which
   is the defect task 37's own experiment paid for.**
10. **A run-cumulative counter that decreases is recorded, not clamped.**
    `summary["failures"]["counter_went_backwards"]` names the source and the
    step. Pins that the instrument reports its own disagreement.
11. **A pass that straddles a rebind is `not_evaluable`, not a fabricated
    delta.** The pass's `session_id` disagrees with the source's; the source
    records `missing: ["session_changed"]`. Pins D15.
12. **The scan record is written on a drive with no ticks at all.** The sampler
    is driven with no pipeline; the sink receives `failure_scan` lines with
    `ticks_seen: 0`. Pins D2's reason for the independent thread.
13. **`ticks_seen` falls to zero across a stalled tick loop and recovers.** The
    measurement task 37's stall analysis had to infer. Asserts no threshold is
    applied and no event is emitted — it is data, not a detection.
14. **The tick block does not depend on `sensing`.** Build a tick record with
    `sensing = None`; `record["failures"]` is present and complete.
15. **A stale scan is stale, and is never collapsed to absent.** Advance 600 s;
    `basis == "stale"` with `scan_age_s == 600`, not `absent`. Task 37's D4
    divergence, applied here.
16. **The status words are the controller's own objects.**
    `assert failure_log.RULE_QUIET is sensing_controller.RULE_QUIET` — identity,
    not equality, because equal strings hide a copy.
17. **The basis words are `time_sync`'s own objects.** Same, with `is`.
18. **Every emitted reason is a member of its source's declared vocabulary.**
    Drive the sampler over a corpus exercising every source and assert
    membership on every episode. This is the test that stops a fifth vocabulary
    being built by accident.
19. **The episode cap counts what it does not keep.** 150 episodes on one source
    with `MAX_EPISODES_PER_SOURCE = 100` gives 100 kept and
    `episodes_not_kept: 50`. `phone_link.refusals`' precedent.
20. **`Inputs` is unchanged.** `[f.name for f in fields(Inputs)]` equals the
    seventeen names, in order. Pins that every task-35 and task-36 log stays
    scoreable.
21. **`score_shadow` still scores a task-36 log after this task.** Run the tool
    over a recorded fixture and assert `replay_identity` mismatched is 0 and no
    refusal. The second half of 20, from the reader's side, and the test that
    catches D18's dataclass conversion breaking :547.
22. **The `load_records` dataclass carries the right list in the right field.**
    A fixture with one `thermal_sample`, one `thermal_event`, one
    `failure_scan` and one `failure_event` asserts each lands in its own field.
    Written because four same-typed adjacent members is exactly where a swap
    survives (D18).
23. **`eval_run` prints the failure section, on real record shapes.** Task 33
    and task 36 both shipped a measurement with no surface; the test builds a
    `metadata.jsonl` with ticks, scans and two episodes, runs `analyze` and
    `render_markdown`, and asserts the `## Failures` heading, an episode line
    with a duration, and the words `NOT EVALUABLE` on a fixture where a source
    was unreadable.
24. **A pre-task-38 run does not crash and does not fail.** `analyze` over a log
    with no failure records reports `result["failures"] is None` and
    `overall_pass` is unchanged.
25. **`log_health.json` carries the writer's final state.** A `MetadataLogger`
    whose writer thread died mid-run produces `writer_failure` non-null and
    `dropped_records > 0` in the file, and the report says so. Written with a
    file handle that raises `OSError` on write — the *field's* failure mode
    (a full card, a read-only mount), which is what
    `metadata_logger.py:114-120` names.
26. **The tick-loop exception is recorded and still propagates.** Raise inside
    `pipeline.step`; assert the `failure_event` exists, assert the exception
    reached the caller, assert `stop.is_set()`. **Both halves in one test**,
    because recording it and swallowing it are different things.
27. **A blind tick is counted and a second one opens an episode.** Camera
    returns `None` for four consecutive polls; `blind_ticks == 4` and one
    episode with `n == 4`.
28. **`_wire_record` carries `seq_gaps` and `missing_seqs`.** Build a session
    with a real sequence gap and assert both appear in
    `summary["phone"]["wire"]["channels"][ch]`. Pins D19's first half.
29. **`to_record` carries the acceptor's stats including its two instants.**
    Pins D19's second half, and gives `first_accept_error_mono_ns` its first
    reader.

Kotlin, JVM (`phone/app/src/test/.../log/`):

30. **A failure line is written and is a distinct shape.** `offerFailure`
    produces a line with `dir == "fail"`, and the three existing shapes are
    unchanged byte for byte.
31. **The rate cap holds and counts what it suppressed.** 100 failures of one
    kind inside one second write one line with `suppressed: 99` on the next.
    Pins D12's stated loss.
32. **A saturated failure rate does not displace frame headers.** With the cap
    saturated for the whole test, the number of header lines written is
    identical to a run with no failures. **This is the direction test for
    behaviour change 3.**
33. **Every Kotlin kind is a member of the Python registry's phone rows.**
    Executed, not asserted by hand: `InteropTest` spawns Python the way
    `scripts/refusal_reasons.py` is already used, so a kind added on one side
    and not the other fails.
34. **The teardown census is unchanged.** `resourcesHeldAfterTeardown == 0`
    after a come-up-and-destroy, because no new resource is held.
35. **A failure detected while the link is down is still written.** Stop the
    session, raise a dial failure, assert the line is in the file. The property
    D11 exists for.

### What asserts the boundary of the three behaviour changes

1. Test 26 — the exception still propagates, so no control flow changed.
2. Test 32 — a saturated failure rate writes the same header lines, so the
   phone's primary record is not displaced.
3. Test 20 — `Inputs` has seventeen fields, so nothing new can reach `decide`.
4. The 15,000-decision replay — the controller's output is byte-identical.
5. The golden-frames test — every message type's `frame_sha256` is unchanged,
   so not one byte moved on the wire.

### Mutations to pin in `scripts/remutate.py`

Each names a defect a specific test above is supposed to catch. Anchors are
single, unambiguous lines.

- `failures: an unreadable source is reported as quiet` — the `not_evaluable`
  arm returns `RULE_QUIET` with an empty `missing`. **The task's own defect
  class turned on the task's own output**, in the shape task 34 used. Caught by
  3 and only by 3.
- `failures: a source that lost its instrument closes as recovered` — the
  `unobservable` arm returns `OUTCOME_RECOVERED`. Caught by 7 and 8.
- `failures: a session-scoped counter is diffed across a redial` — drop the
  `session_id` baseline reset. Caught by 9, and by 11.
- `failures: a counter that went backwards is clamped to zero` — `max(0, delta)`.
  Caught by 10.
- `failures: the scan record is written only when something is open` — guard
  `_write_scan` on `self._open`. **This is the mutation that turns the whole
  task back into its own defect class**, and it is caught by 12.
- `failures: the still window is a typed 3.0 rather than 3 x interval` — caught
  by 6 and only by 6.
- `failures: the episode count is derived by subtraction rather than counted` —
  `n = passes_open - still_passes`. Task 34's `superseded = received - shown -
  expired`, and the replay pilot's rule that outcomes are counted from records.
  Caught by 4.
- `failures: the tick block reports a stale scan as measured` — drop the age
  comparison in `latest()`. Caught by 15.
- `failures: the tick-loop exception is swallowed` — remove the `raise`. Caught
  by 26, and **not** by the record half alone, which is why 26 is two
  assertions.
- `failures: an emitted reason is not checked against its source's vocabulary` —
  Caught by 18.
- `eval_run: the failure section is computed and not rendered` — return `[]` from
  `_failure_lines` unconditionally. The exact defect task 33 and task 36 both
  shipped. Caught by 23.
- `eval_run: the record lists are swapped in the dataclass` — assign
  `failure_scans` from the `thermal_samples` list. Caught by 22 and by nothing
  else, which is D18's whole argument.
- `phone: a suppressed failure is not counted` — drop `suppressed`. Caught by 31.
- `phone: the failure line displaces a header line` — remove the rate cap.
  Caught by 32.

A note the validator should hold this plan to: **a fix is new code.** Every
round of this section has found its defects inside the previous round's fix, so
each fix gets its own mutation, not a re-run of the mutation that found it.

## Byte cost

Measured with `json.dumps` and the default separators `MetadataLogger.write`
uses (`", "` and `": "` — they are not compact, and omitting that is a way to be
wrong by about 10 per cent), and with canonical compact JSON for the phone's
`SessionLog`, which is what `Json.encode` writes. **Every field is enumerated,
including the ones that feel too small to count**, because task 36's estimate
came out 26 per cent below the measurement by omitting one field entirely and
task 37's sample line came out 39 per cent over by assuming 3 cooling devices
where the hardware had 13.

**Base record.** Task 35 measured a mean tick record of 8,511 and 8,552 B; task
36 added a measured 852 B, giving 9,363 B measured. Task 37 added an estimated
409 B that **was never measured in the field** — its experiment reported the
wire cost exactly and the sample-record overrun, and did not report the tick
block's size. So the base is **9,772 B, of which the last 409 B is an estimate**.

**Per tick — the `failures` block, 7 named fields** (`open`, `open_n`,
`episodes`, `scan_age_s`, `basis`, `unreadable_n`, `reason`) plus the block key
itself:

| variant | bytes, inline including the leading `, ` |
|---|---|
| healthy, nothing open | **131** |
| two open episodes, five sources unreadable | **159** |
| sampler not running | **152** |

131 B on 9,772 B is **1.34 per cent**. Over 900 ticks: **117,900 B**.

**Per 1 Hz scan record**: **197 B** healthy, **320 B** with five unreadable
sources and two open episodes named. A 180 s drive writes 180 of them:
**35,460 B** healthy, **57,600 B** degraded.

**Per episode, two records**: open **334 B**, close **317 B**, so **651 B** per
episode. Eight episodes is **5,208 B**. At the cap
(`MAX_EPISODES_PER_SOURCE = 100` × 28 sources, which no real drive approaches)
it is 1.82 MB, which is why the cap exists.

**`summary["failures"]`**: **7,899 B** as `write_summary` writes it
(`indent=2`), 5,961 B compact. It is 28 source rows at about 280 B each plus a
190 B `scan` block. This is 7.3 times task 37's `summary["thermal"]` of
1,086 B, and the registry size is the only lever on it.

**`log_health.json`**: **211 B**, once per run.

**Per phone `SessionLog` line**: **156 B** with a detail string, **130 B**
without. At the D12 cap of one line per kind per second and ten kinds, the
worst case is 1,560 B/s against `SessionLog`'s roughly 20 kB/s of headers —
**7.8 per cent**, and the realistic case is a handful of lines per drive.

**On the wire: 0 B.** No field is added to any message and no channel changes.

**Drive total**, 900 ticks over 180 s with 8 episodes:
117,900 + 35,460 + 5,208 + 7,899 + 211 = **166,678 B, about 163 KiB**, against
a metadata log of about 8.5 MiB — **1.9 per cent**.

**What this estimate is sensitive to, named in advance so the experiment checks
it rather than discovering it:**

1. **The number of sources unreadable at once.** The degraded scan record is
   62 per cent larger than the healthy one, and on a drive with no phone every
   phone-device source is unreadable on every pass. A phone-less desk drive
   therefore pays 320 B per pass, not 197 — 57,600 B rather than 35,460 B. I do
   not know how many of the 28 sources go unreadable together on a real drive;
   the experiment's first check is the distribution of `sources_readable`.
2. **The registry size.** `summary["failures"]` is linear in it at about 280 B
   per source, and 28 is a count taken from this plan's own table. If the
   implementation finds two more sites worth registering, the summary grows by
   560 B and nothing else moves.
3. **The episode count.** 8 is a guess. A drive with a flapping link produces
   one `link.down` episode per outage plus one `link.session_end` episode plus
   the session-scoped sources closing as `unobservable` each time — so a single
   redial is roughly six episodes, not one, and a drive with ten redials is
   sixty episodes and 39 kB. The experiment should report episodes per redial.
4. **The `detail` string length.** The event records above assume detail strings
   of 20 to 35 characters. `gps_reader`'s `last_error` is
   `f"{type(exc).__name__}: {exc}"` and a `serial.SerialException` message can
   exceed 200 characters. The implementation caps `detail` at a stated length
   and counts truncations; if the cap is 200 the open record grows to about
   500 B.
5. **The tick rate.** 900 ticks over 180 s is 5 Hz. `loop.target_hz` is 0 in
   `config.yaml` (run as fast as the detector allows), and task 33's drive ran
   at 5 Hz. At 30 Hz the tick block alone is 707 kB and becomes the dominant
   term.
6. **The base denominator has moved four times in this section**, so the
   1.34 per cent figure is a ratio against a moving base and **the absolute
   131 B is the number to compare against**. The 409 B of it contributed by task
   37 has never been measured, so the denominator itself carries an unverified
   term.

## Open items

They are allowed to stay open. Unclosed open items are not defects.

1. **A drive that produces zero ticks still produces no report.**
   `eval_run.analyze` raises `SystemExit` at :467 when there are no tick
   records, and that is the drive with the most failures — the camera never
   delivered a frame, so `_tick_loop` `continue`d for the whole run. This task
   makes the failure records exist and readable with `json.load`, and changes
   the exit message to name how many are present, but it does not make
   `analyze` tolerate zero ticks: every metric block below :467 indexes
   `ticks`, and rewriting them is a larger change than this task. Named, not
   built.
2. **A model-loading failure has nowhere to be recorded.** `build_components` is
   called at run_demo.py:386 and `make_run_dir` at :404, so a missing engine or
   a contract-fingerprint mismatch kills the process before a run directory
   exists. `selfcheck` catches these and reports them; a live run does not. The
   fix is to move the run directory's creation earlier, which changes what a
   failed start leaves on disk, and that is its own decision.
3. **Every instant in this log is a first observation, bounded by one sampler
   period.** `bound_s` carries it and D9 states it, but it means a 100 ms outage
   and a 900 ms outage inside the same second are the same record. Lowering
   `interval_s` costs one snapshot call per source per pass and is a
   configuration change, not a code change; whether it is worth it cannot be
   decided before a drive shows how often sub-second episodes occur.
4. **D14's handling of session-scoped counters is argued and tested, not
   measured.** Test 9 proves the sampler does not fabricate a delta across a
   simulated rebind. Whether the run totals it accumulates agree with the phone's
   own view over a drive with real redials is exactly what the experiment must
   check, and task 37's redial defect is the reason to expect a surprise.
5. **The phone's failures reach an analysis only offline.** D11 accepts this. A
   drive whose session log was not pulled off the handset reports "phone-side
   failures: not read", which is correct and is also the state the analysis will
   most often be in. Making them live means a wire field and a channel decision,
   which this task refuses.
6. **No gate.** D17. What would have to be true to gate it later: one drive
   establishing that every registered source is readable on real hardware,
   after which `failure_sources_observable` becomes a legitimate gate whose
   failure means the drive cannot answer the question. `log_complete` already
   covers the one failure that invalidates a drive outright.
7. **The failure log cannot record its own log's death.** A `failure_event` line
   written after the writer thread died goes into the queue and is counted in
   `dropped_records`. `log_health.json` (D16) is the answer and it is a separate
   file for that reason, but it is written by the same process, so a run killed
   with `SIGKILL` leaves neither. Nothing in this repository survives that, and
   this task does not change it.
8. **`camera.dropped_unconsumed` has two scopes depending on the camera.**
   `CameraStream._drop_counter` is run-cumulative; `PhoneCameraStream`'s resets
   per session (phone_source.py:394). The registry declares the source's scope
   at construction, from which camera was built, which is one more place a
   wrong wiring is silent. Test 1 checks the accessor resolves; nothing checks
   the scope is right. A drive with a redial and a local camera would catch it,
   and that configuration does not occur.
9. **Nothing reconciles the two GPS staleness predicates.**
   `GpsReader.is_stale` (gps_reader.py:270-272) gets its first caller here;
   `observation_builder`'s `gps_fresh` (:193-198) is what the decision uses.
   They should agree. Task 35 found its witness and the controller disagreeing
   on `NaN`, where the tick landed in neither partition. `result["failures"]`
   reports the count of ticks where they disagree, and the expected value is
   zero; a non-zero value is a finding, not a bug in the report.
10. **`system` and `system_error` records remain unread.** This task adds a
    reader for its own two record types and does not adopt `SystemStatsSampler`'s,
    for the same reason task 37 left it alone: merging two accounts of the same
    machine whose relationship is unknown. `system_error` is named in the
    inventory as the precedent to avoid, and it stays unread.

## Scope boundary — what this task does not do

- It does not change any commanded sensor rate, in either direction, on either
  device. `Inputs` keeps its seventeen fields, `SensingController` is untouched,
  and the 15,000-decision replay asserts the controller is byte-identical.
- **It does not change one byte on the wire.** No message gains a field, no
  channel is added, removed or re-policied, and the golden frames are
  unchanged. The eight-channel table, its priorities, depths and overflow
  policies stay exactly as specs/transport_protocol.md:120-129 records them.
- It adds no encoder slot and no `field_sources` entry, so the 39-slot coverage
  identity and the missingness denominator are unchanged.
- It adds no detection of any condition the code does not already evaluate. The
  three behaviour changes are enumerated above and each is fenced by a named
  test.
- It does not change the logic of any counter it reads. `HereFeed`,
  `ChannelStats`, `ChannelMessageStats`, `GpsDiagnostics`, `CameraStream` and
  `SessionStats` keep their fields, their types and their reset points. The only
  edits to existing accounting are two additions to record *builders*
  (`_wire_record`, `to_record`) that publish counters already computed.
- It does not record thermal failures (D20). Task 37 owns them and already gives
  them everything this task provides.
- It does not add a gate (D17, open item 6) and does not change `overall_pass`.
- It does not make `eval_run` tolerate a zero-tick drive (open item 1).
- It does not move `make_run_dir` earlier, so a model-loading failure still has
  no artifact (open item 2).
- It does not add retry, backoff, quota budgeting or any other *response* to a
  failure. HERE gets no 429 branch; the redial backoff is unchanged; nothing
  recovers anything. It records what happened and what happened next.
- It does not prune `filesDir/sessions` on the phone, which
  log/SessionLog.kt:32-38 already names as a retention decision rather than an
  engineering one.
- It is not the session summary generator (task 39). Latency percentiles,
  achieved-versus-commanded rates, API call counts and trigger counts are that
  task's; `summary["failures"]["sources"][*]["total"]` is available to it and is
  not itself a session summary.
- It does not run the drive. The experiment is `experiment_dsrc`, separately,
  and every number in §"Byte cost" is an estimate until it does.
