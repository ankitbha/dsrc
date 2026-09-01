# Task 33 — Per-stage timestamps across the loop

## PAUSED 2026-08-31 19:35 — read this first

### The blocking unknown

**This plan has not been read or checked by anyone.** A `plan_dsrc_rec` agent wrote
it start to finish and returned; the session paused at that moment, so no human and
no other agent has verified a single line of it. That matters more than usual here
because of what the plan is for: it is the fixed target an independent validator
audits the implementation against, so an error in it becomes an error the validator
certifies rather than catches.

Three specific things to settle before any code is written, in this order:

1. **Verify the `file:line` citations.** The agent states it read every one directly
   rather than assuming. That claim is exactly the kind this project has been wrong
   about before, and it is cheap to check: the load-bearing ones are
   `deployment/jetson/transport/timebase.py:338` (a round-trip estimator the plan
   says already ships and is unused) and `timebase.py:959` (a docstring the plan
   quotes as calling the one-way estimate "unfit for attributing latency"). If the
   round-trip estimator does not exist as described, the plan's first two steps
   collapse and the sequencing has to be redone.
2. **Decide the two naming questions the agent could not settle**, listed as open
   items 1 and 2. It read "decode" in the task's stage list as *advisory* decode on
   the phone, from the position of the word in the list, while its own first
   instinct was JPEG decode on the Jetson. It instrumented both, so only the name
   needs settling — but the name is what the record will carry. Same for "fuse",
   read as the feed-fusion/peer-merge sub-segment of `builder.build`.
3. **Accept or reject the wire-format cost**, open item 7: about 70 bytes per frame
   of permanent header growth on the camera channel, which is the busiest one. The
   last measured run carried 561 camera frames in 120 s, and the phone already
   evicts frames on that channel when the socket cannot drain (61 of 655 on the
   task-32 run), so added header bytes land on the channel with the least headroom.

### Exactly where the work stopped

- Repo `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc`, branch `main` at
  `4ef3a3c`, level with `origin/main`. Working tree clean except this file, which is
  **untracked and uncommitted**.
- Section G (tasks 33–39) was just started. Task 33 is at stage `plan`, validator
  round 0 of a 3-round budget. Nothing has been implemented.
- Pipeline per task: `plan_dsrc_rec | implement_dsrc | validate_dsrc_3 |
  experiment_dsrc`, then commit, push, strike the task in `plans/task_list.md` and
  move to the next. Models per stage: plan `fable`, implement `sonnet`, validate
  `opus`.
- Section-level state lives in `/Users/ankit_nash/Desktop/ankit_summer_2026/.pipeline/state.json`
  (`task.number`, `task.stage`, `task.validator_round`, `task.remaining`). The
  watcher that reads it is `.pipeline/watch.py`.

### What moved underneath, which a resuming reader cannot notice

- **The watcher was rewritten today** to walk a task list instead of watching one
  task. Its `closed` verdict now returns `NEXT_TASK` when `task.remaining` is
  non-empty and `ALL_DONE` only when it is empty; the previous version would have
  ended the seven-task section after the first task and reported success. It also
  no longer demands the phone and Jetson at `implement` and `validate`, only at
  `experiment` or when a task sets `needs_hardware`.
- **The 5-minute watcher cron was cancelled at the pause** and is not armed. Nothing
  will wake this work on its own. Re-arm it on resume.
- **Task 32 closed earlier today** at `4ef3a3c`, and its last act changed a test this
  plan's area touches: `ImuWireTest`'s rate assertion moved off the peer-received
  rate and onto `ImuPipeline.seen`, because the socket's drain rate bounded the wire
  measurement. That is the same measurement hazard task 33 is about.

### What was learned that would otherwise be re-derived

- Gate: `ANDROID_SERIAL=ZY227VV4XC ./scripts/check.sh`. **Do not pipe it into
  `tail`** — the exit status then comes from `tail` and a failed build reports 0.
  Redirect to a file and check `$?`.
- Handset `ZY227VV4XC` (moto g power, Android 11). Jetson at ssh alias `jetson`,
  user `edge`, tailnet `100.90.108.88`; the deployment is a file copy at
  `~/dsrc-task32`, and its separate checkout at `~/projects/dsrc-deployment/dsrc`
  tracks a different remote and must not be touched. System python 3.10.12 there has
  every dependency; no virtualenv is needed.
- `pkill -f run_demo.py` over ssh matches its own shell command string and kills the
  session. Use `pkill -f "[r]un_demo.py"`.
- macOS `crontab` writes are blocked here by a privacy restriction; the watcher is
  armed through the harness scheduler instead.
- Mutation harness: `scripts/remutate.py`, now takes `--name=SUBSTR`. Score on the
  pytest summary line **and** the failing test name, never the exit code.
- The phone dials and the Jetson listens, because the Jetson's Tegra kernel has
  `CONFIG_NF_CONNTRACK_MARK` unset and cannot originate to tailnet peers.

### Task order to resume

1. Settle the three items under "The blocking unknown".
2. Commit the plan (it is untracked).
3. Re-arm the watcher cron and set `.pipeline/state.json` to task 33, stage
   `implement`.
4. `implement_dsrc` on a `sonnet` agent, against this plan.
5. `validate_dsrc_3` on an `opus` validator, in a throwaway `git archive` mirror,
   kept alive across the three rounds.
6. `experiment_dsrc`, then the gate, commit, push, strike task 33, move to task 34.

### Open decisions not yet made

The seven items in "Open items flagged for the user" at the end of this plan, none
of which has been put to the user. Items 1, 2 and 7 are the ones that change the
work; see the blocking unknown above.


> Written by `plan_dsrc_rec`: every decision below was taken by recommendation,
> without putting the question to the user. The decisions table says what was
> chosen and why; **none of it is user-approved**, and the "Open items flagged
> for the user" section holds every point where the recommendation was weak or
> where the code contradicted the first reading. This plan is the fixed target
> a validator audits the implementation against.

## Resolved on resume, 2026-08-31 — supersedes the three items above

**1. The citations hold, and so does the characterisation.** `TimebaseEstimator`
(timebase.py:338) consumes `TimeSyncSample`, which carries four stamps across two
clocks (timebase.py:153-174), so it is a round-trip estimator and not a renamed
one-way one. `TimeSyncInitiator` (timebase.py:653) owns one. Neither is constructed
anywhere in live Jetson code outside `timebase.py` and its tests: the live path
builds `OneWayEstimator` at phone_link.py:115 and phone_link.py:428. The text the
plan quotes, "fit for a 2 s freshness threshold and unfit for attributing latency",
is at timebase.py:959. The plan's first two steps stand as written.

**2. The two names are taken as the agent read them, on the evidence of the task's
own stage order.** The list is "capture, encode, transport, detect, track, fuse,
infer, decode, return, render". `decode` sits after `infer` and before `return` and
`render`, which are the return path, so it is the phone decoding the advisory, not
JPEG decode — that would sit between `transport` and `detect`. The agent's first
instinct was JPEG and the position of the word contradicts it. `fuse` sits between
`track` and `infer`, which matches the feed-fusion step before the observation is
built. Both are instrumented either way, so this settles what the record calls them,
not what it measures. **Taken by recommendation, not user-approved.**

**3. The wire cost is accepted, and it is smaller than the thing it was feared to
worsen.** About 70 bytes per frame at 5 Hz is 350 bytes per second, against a JPEG
of roughly 30-60 kB, so 0.1 to 0.2 per cent of the frame. The camera channel does
evict — 61 of 655 frames on the task-32 run — but that eviction was the socket's
drain rate, about 50 frames per second, not the byte count; a 0.2 per cent increase
does not move it. **Taken by recommendation, not user-approved.**

The remaining open items (3, 4, 5, 6 in the list at the end of this plan) do not
block implementation and stay open for the user.

## The short version

Task 33 (plans/task_list.md:1067) asks for per-stage timestamps across the
whole loop: capture, encode, transport, detect, track, fuse, infer, decode,
return, render. Four of those already exist on the Jetson
(`stage_ms` in deployment/jetson/pipeline.py:249-255: `detect`,
`track_distance`, `observe`, `policy_advisory`, plus `capture_to_start`,
`e2e_ms`, `jetson_ms`, `link_ms`). **Six do not**: capture, encode, return and
render happen on the phone, transport is today a single capture→arrival lump
converted through a one-way clock estimate whose own docstring calls it
"unfit for attributing latency" (deployment/jetson/transport/timebase.py:959),
and fuse/infer/decode are buried inside two undivided segments.

The plan, in one paragraph: (1) land the `t4`-on-next-ping wire field that
three earlier plans name as the prerequisite, so the Jetson forms full
four-stamp time-sync samples and runs the round-trip `TimebaseEstimator` it
already ships but never feeds — that is what makes any cross-device segment a
bounded measurement instead of a one-sided guess; (2) carry the phone's
capture/encode instants on the camera frame header as same-clock stamps, so
every phone-side duration is exact and needs no clock conversion at all;
(3) split the Jetson tick into the task's named stages and publish a per-tick
`stages` object in which every entry says whether it was **measured** on one
clock, **converted** across clocks with a stated bound, or is **absent with a
named reason** — never a zero, never a bare number; (4) log advisory arrival
and first-display on the phone (return and render are facts only the phone
can witness), and extend `eval_run.py` to join the two logs into the
ten-stage table offline.

**Scope boundary.** In: the `t4` wire field and the Jetson-side four-stamp
estimator; encode/wire stamps on camera frames; wire stamp on advisories;
JPEG-decode, fuse, infer and advisory-decode timing; the per-tick `stages`
record and `timebase_estimate` log lines; phone-side inbound-advisory logging
and render stamps; the offline join in `eval_run.py`; tests and spec/golden
updates for all of it. Out: trigger attribution (task 34), the shadow decision
log (task 35), field provenance (task 36 — `field_sources` already exists),
thermal/failure logs (tasks 37, 38), the full session summary generator
(task 39), any change to what the controller decides, and the known
`achieved["camera_hz"]` overstatement (carried open item from task 32 — it
needs `Session.send` to surface the displacement, a transport API change this
task does not make).

**Open decisions flagged for the user** (details at the bottom): the reading
of "decode" as *advisory* decode (list order says so; my first instinct said
JPEG — both get instrumented, only the name is at stake); the reading of
"fuse" as the feed-fusion/peer-merge sub-segment of `builder.build`; "capture"
being an instant with an unmeasurable shutter→callback bias; render quantised
to the UI's 250 ms tick; "return" computed offline rather than live; and how
long the one-way estimator survives as a fallback.

## What already exists (verified by reading the code)

- **Jetson tick timing.** `pipeline.step` stamps `t0..t4` with
  `time.monotonic()` and publishes `stage_ms` = detect / track_distance /
  observe / policy_advisory / capture_to_start (pipeline.py:213-255).
  `e2e_ms` is capture→advisory, `jetson_ms` is arrival→advisory, `link_ms` is
  capture→arrival and is `None` for a local camera or a proxied stamp
  (pipeline.py:240-248, sensors/time_sync.py:50-67). Rolling summaries in
  `PipelineStats.snapshot` (pipeline.py:160-169) are the
  `e2e_ms/link_ms/jetson_ms/detect_ms/track_ms/observe_ms/policy_ms` keys seen
  in run summaries. `eval_run.py` percentiles every `stage_ms` key generically
  (eval_run.py:172-173) and collects only `type=="tick"` records
  (eval_run.py:74), so both new stage keys and new record types are absorbed.
- **Reference clocks.** All Jetson stages are `time.monotonic()` on the
  Jetson. `frame.t_mono` for a phone frame is the phone's capture stamp
  *converted* by `PhoneClockAdapter.stamp` (sensors/phone_source.py:61-98),
  which falls back to arrival ("proxy") and counts why. Conversion runs on
  `OneWayEstimator` (transport/timebase.py:886-1047): offset from arrivals
  alone, error one-sided by the delay floor, bound = delay *spread* —
  explicitly "fit for a 2 s freshness threshold and unfit for attributing
  latency" (timebase.py:955-960). The round-trip `TimebaseEstimator` with
  rtt/2 + drift bounds and a skew fit exists in the same file (timebase.py:338)
  and is fed by nothing on the Jetson: the spec makes the phone initiate and
  the Jetson only answer (phone_link.py:13-22), and a responder never learns
  t4 (timebase.py:847-861).
- **The wire already half-carries what task 33 needs.** Every frame header has
  `t_mono_ns` (sender enqueue) and `t_wall_ns`; frames that ask get
  `t_wire_mono_ns` stamped by the writer at departure (transport/frames.py:47-68,
  session.py:398-452; Kotlin mirror Session.kt:301-338, 436-446). Today only
  time-sync messages ask (Session.kt:572, 593). Camera frames carry only
  `captureMonoNs` (CameraFrameMessage.kt:12-20, messages.py:359-398) — no
  encode stamps. Both `from_wire` decoders read named keys and tolerate
  unknown ones, and the phone's inbound path deliberately does not apply the
  reserved-key rule (Messages.kt:357-379), so additive fields and a wire stamp
  on new channels are non-breaking; golden frames
  (specs/transport_golden_frames.json, ProtocolSpecTest/InteropTest,
  scripts/generate_transport_golden_frames.py) pin the canonical bytes and
  must gain cases.
- **Phone side.** Capture stamp is the CameraX analyzer callback on
  `elapsedRealtimeNanos` — deliberately the same clock as the header's enqueue
  stamp so their difference is a valid subtraction, with a known
  shutter→callback bias (CapturedFrame.kt:16-24). Encode runs on a
  single-thread executor (CameraPipeline.kt:107-139), untimed. JPEG decode on
  the Jetson runs untimed inside the camera reader
  (phone_source.py:359-367). `SessionLog` writes **outbound** headers verbatim
  (SessionLog.kt:9-41; wired at SensingService.kt:353) — nothing logs inbound.
  Advisories are stamped at *delivery*, not at the reader's receipt
  (SensingService.kt:584-590), though the reader's stamps exist on `Received`
  (Queues.kt:241-267, 313-314) and the repo's own doctrine is to prefer them
  (phone_source.py:236-240, timebase.py:816-823). `AdvisoryHolder` keeps
  arrival and expiry (AdvisoryHolder.kt:45-78); the UI polls `current()` every
  250 ms (MainActivity.kt:37-45, 255). The phone sends pings
  (TimeSyncDriver.kt) but matches no pongs and holds no estimator — there is
  no Kotlin clock conversion at all.
- **The advisory joins to its frame exactly.** `AdvisoryMessage` carries the
  frame's capture stamp on the Jetson's clock (sensing_loop.py:146-156,
  plan_task31 decision 2), and `run_phone_drive.py` already joins on it
  (run_phone_drive.py:209, 229-232). That key is what the offline return/render
  join uses.
- **The prerequisite is on record three times.** plan_task26:98-106 ("carry
  `t4` for the previous exchange on its next ping... should be settled before
  task 33, which cannot be honest without it"), plan_task31:18-19 and :139,
  plan_task32:102.

## Decisions taken (by recommendation — not signed off by the user)

**D1. The `t4` wire field is in scope, and first.** Options: (a) do it now;
(b) ship task 33 on the one-way estimate; (c) attribute cross-device segments
only offline. Chosen: (a). The one-way bound is a delay spread that says
nothing about the delay floor (timebase.py:953-960) — measured on this pair
the floor is tens of milliseconds, the same order as the segments being
attributed, so (b) publishes numbers whose error exceeds their value; (c)
leaves the live record unable to say what the link is doing during the drive.
Three plans already call (a) the prerequisite.

**D2. Interleaved four-stamp shape: ping N+1 carries the previous exchange.**
Three new absent-tolerant fields on a *ping* (`absentable` pattern per
plan_task26:100-103; helpers exist at messages.py:238-254, Messages.kt:117-128):
`prev_exchange_id`, `t_prev_pong_wire_mono_ns` (echo of pong N's Jetson wire
stamp), `t_prev_pong_recv_mono_ns` (the phone reader's receipt of pong N).
All three or none, mirroring the pong's own all-or-none rule
(messages.py:917-925). The Jetson then forms, **with zero pending state**, the
exchange (t1 = pong N's wire stamp, local; t2 = phone receipt of pong N,
remote; t3 = ping N+1's own `t_wire_mono_ns`, remote; t4 = ping N+1's reader
receipt, local) — every ordering check in `TimebaseEstimator.add`
(timebase.py:375-385) holds, and the phone's inter-ping think time cancels in
`rtt_ns` exactly as a responder's service time does (timebase.py:177-183).
The phone holds the only state: the last pong's (exchange_id, echoed wire
stamp, receipt), kept in `Session` where `handleTimeSync` already sees the
receipt (Session.kt:544-573). The Python `TimeSyncInitiator`
(timebase.py:653) gains the same carry so `scripts/interop_jetson_peer.py`
exercises the full path without a handset.

**D3. Phone stage stamps ride the camera frame header, same-clock.**
`CapturedFrame`/`CameraPipeline` record `t_encode_start_mono_ns` and
`t_encode_done_mono_ns` around `compress` (CameraPipeline.kt:120);
`CameraFrameMessage`/`CameraFrame` gain the two fields (both codecs, spec
message table, golden frames); camera sends set `wantsWireStamp = true` so the
channel dwell (enqueue→wire, where latest-wins eviction lives) separates from
the network. Every phone-side segment — capture→encode_start (gate+pack+queue),
encode, encode_done→enqueue (buffer dwell + sender poll,
CameraFrameSender.kt:17-19), enqueue→wire — is then a difference of two stamps
on one clock: **exact, no timebase involved**, and the `SessionLog` gets them
for free because it logs the header verbatim. Rejected alternative: keeping
phone stages only in a phone-local log, which would make even the exact
segments reachable only through an offline join.

**D4. Transport and return are conversions and say so.** `transport` =
Jetson arrival (reader receipt, phone_source.py:236-241) minus the converted
phone wire stamp, published with `bound_ms`, `estimate_id` and `source`.
`TimebaseStamp` (time_sync.py:28-76) gains `source`
("round_trip" | "one_way" | "proxy"); `PhoneClockAdapter` tries the round-trip
estimator first, the one-way second, proxies last, and counts which — a
converted number must never look as measured as a same-device one
(timebase.py:262-281). `PipelineStats` keeps round-trip-converted transport
samples in a separate series from one-way-converted ones; they have different
error semantics and one distribution must not launder the other. `return` =
phone receipt of the advisory minus the advisory's Jetson wire stamp,
computed **offline** (D6) because the Jetson never learns the arrival and the
phone has no estimator; advisories therefore also ask for the wire stamp
(Python `AdvisoryMessage.to_wire` + `RESERVED_ALLOWED`, like
messages.py:876; the phone's inbound path already tolerates it,
Messages.kt:377-379).

**D5. The Jetson tick gains a `stages` object; existing keys are not
redefined.** Precedent: frames.py:56-58 — redefining a stamp "would have
changed what every existing latency figure measures without changing a test";
the dashboard reads `stage_ms['observe']` and `['policy_advisory']` by name
(ui/dashboard.py:113). So `stage_ms` keeps its exact current meanings, and a
new per-tick `stages` dict carries the canonical task-33 names:

| stage    | from → to                                   | clock  | basis      |
|----------|---------------------------------------------|--------|------------|
| capture  | the anchoring instant (analyzer callback)   | phone  | instant    |
| encode   | encode_start → encode_done                  | phone  | measured   |
| transport| camera wire stamp → Jetson reader receipt   | cross  | converted + bound, or absent + reason |
| jpeg_decode | around `_decode` in `_accept`            | jetson | measured (unnamed by the task; needed for a complete chain) |
| detect   | = stage_ms.detect                           | jetson | measured   |
| track    | = stage_ms.track_distance                   | jetson | measured   |
| fuse     | feed_fusion.own + peer merge inside `build` | jetson | measured (sub-segment of observe) |
| infer    | `actor.act`                                 | jetson | measured   |
| decode   | `advisory_decoder.decode`                   | jetson | measured (infer + decode = policy_advisory) |
| return   | advisory wire stamp → phone receipt         | cross  | offline, converted + bound |
| render   | phone receipt → first `current()` return    | phone  | measured, 250 ms quantised |

Every entry is `{ms, basis, clock}` plus `bound_ms`/`estimate_id`/`source`
when converted, or `{ms: null, basis: "absent", reason: ...}` — the
None-not-zero discipline of `RollingStats.summary` (pipeline.py:40-59) and
`link_s` (time_sync.py:50-67) applied uniformly. The phone-side segments the
Jetson can compute exactly (capture→encode_start, encode, encode_done→enqueue,
enqueue→wire) are published under `stages` as phone-clock measured entries.
`return` and `render` do **not** appear as eternal nulls in the tick record;
they are per-advisory facts in the phone log, merged by the offline report —
the deliverable is explicitly the pair of records plus the join, stated in
the run summary. `fuse` timing comes from the builder recording its own
`last_timings` (precedent: detector.py:120, 196); `infer`/`decode` from two
new sub-stamps in `pipeline.step`; `jpeg_decode` measured in
`PhoneCameraStream._accept` and carried on `Frame` (camera_stream.py:30-38
gains a field; `None` for local sources, which decode elsewhere).

**D6. Return and render are witnessed by the phone and joined offline.**
The delivery callback grows the reader's receipt stamps (`onFrame` today
receives a bare `Frame`, Session.kt:693; `Received.recvMonoNs/recvWallNs`
exist at Queues.kt:266-267) so the arrival stamp is the reader's, not the
delivery thread's — the same correction the Python side already made
(phone_source.py:236-241). `SessionLog` gains a distinct inbound line type
for advisories: `{"dir":"in","recv_mono_ns":...,"recv_wall_ns":...,
"header":{...verbatim...}}` — the outbound "verbatim, one object per line"
contract (SessionLog.kt:11-17) is untouched. `AdvisoryHolder` records the
first `current()` that returned each advisory and reports render latency in
its stats; the service writes an `advisory_shown` line via a holder callback.
Offline, `eval_run.py --phone-log <session.jsonl>` joins phone advisory lines
to Jetson ticks on `t_capture_mono_ns` (exact key, run_phone_drive.py:229),
converts the phone receipt with the offset of the nearest
`timebase_estimate` record (nearest by the NTP-locked wall clocks,
phone_source.py:23-27), and emits `return_ms` (converted + bound) and
`render_ms` (measured, quantised) in the ten-stage table. Rejected
alternative: echoing advisory receipts back up the wire for a live return
figure — more protocol surface for a number nothing consumes at runtime;
flagged below.

**D7. Timebase estimates are persisted for re-derivation.**
`ConvertedInstant.estimate_id` promises offline re-derivation
(timebase.py:262-281), but nothing persists the estimates —
`estimator.to_record` keeps only the current one (timebase.py:571-597). The
`run_demo` worker writes a `{"type":"timebase_estimate", ...}` line to
metadata.jsonl whenever the adapter's estimate_id changes (≤ the sync cadence,
~1-4 Hz worst case, a few hundred bytes each). `eval_run.load_records`
already ignores unknown types (eval_run.py:74).

**D8. The one-way estimator stays as a fallback.** A phone build without the
new ping fields would otherwise regress from "converted with a spread bound"
to "proxied forever". The adapter's preference order and the per-stamp
`source` field keep the two distinguishable everywhere they land. On a rebind
both estimators reset together (the existing reset at phone_link.py:428-429
extends to the pair — a new session is a new peer clock).

## Amended after implementation, 2026-08-31 — step 3's mechanism

Step 3 below says the advisory asks for a wire stamp by way of `AdvisoryMessage`.
It was implemented instead as a `wants_wire_stamp` parameter on `Session.send` and
`MessageRouter.send`, which is the mechanism the Kotlin transport already uses for
the same purpose.

The reason is that the literal instruction would have changed the bytes of the
frozen `message_advisory` golden vector, and a change to a frozen vector forces a
`PROTOCOL_VERSION` bump that nothing in this task discussed. Checked rather than
assumed: after the change all 20 pre-existing golden cases are byte-identical,
exactly two cases were added, and `PROTOCOL_VERSION` is untouched.

Recorded here rather than left in a commit message because this file is the fixed
target the validator audits against, and a plan that describes a mechanism the code
does not use turns every audit of that area into a false finding.

## The work

1. **Spec first**: specs/transport_protocol.md — three new optional ping
   fields on `control` (table row at :344, time-sync section ~:486-535),
   all-or-none rule, and the wire stamp now requested on `camera` and
   `advisory`; new golden-frame cases via
   scripts/generate_transport_golden_frames.py.
2. **Kotlin transport**: `TimeSyncMessage` fields; `Session` holds the last
   pong's (exchange_id, echoed `t_wire_mono_ns`, receipt) and
   `sendTimeSyncPing` carries them; camera sends ask for the wire stamp;
   delivery callback carries receipt stamps.
3. **Python transport**: `TimeSyncMessage` fields + all-or-none;
   `TimeSyncInitiator` carries the trio; `AdvisoryMessage` wire stamp.
4. **Jetson estimator wiring**: `phone_link._answer_pings` builds the
   interleaved `TimeSyncSample` (D2 orientation) and feeds a
   `TimebaseEstimator` held beside the `OneWayEstimator`;
   `PhoneClockAdapter` preference order + `TimebaseStamp.source`; both reset
   on rebind; both in `to_record` under distinct headings.
5. **Phone capture/encode**: stamps in `CameraPipeline`/`CapturedFrame`,
   fields in both camera codecs, pass-through in `CameraFrameSender`.
6. **Jetson stages**: JPEG decode timed in `PhoneCameraStream._accept` and
   carried on `Frame`; `ObservationBuilder` records `last_timings` for fuse;
   `pipeline.step` sub-stamps infer/decode; per-tick `stages` object built
   per D5; separate transport stats series per source; `timebase_estimate`
   metadata lines (D7).
7. **Phone return/render**: inbound advisory lines and `advisory_shown` lines
   in `SessionLog`; `AdvisoryHolder` first-shown stamp and render stats.
8. **Offline report**: `eval_run.py --phone-log` join producing the ten-stage
   table with basis/bound tags; runs without a phone log emit the eight
   Jetson-side stages and name the two absent ones.

Each step lands with its tests — section G's rule is that instrumentation is
written as part of the implementation, not added afterwards.

## Tests

- **Timebase**: interleaved samples recover a known synthetic offset and skew;
  the ordering and admission guards fire on out-of-order stamps; partial
  trios are refused with the null-consistency reason; a run whose pings lack
  the trio converges the one-way path only, and every stamp says
  `source: "one_way"`.
- **Adapter**: preference order round_trip → one_way → proxy under scripted
  estimator states; `proxied` and `proxy_reasons` still count; a rebind
  resets both estimators.
- **Pipeline**: `stages` present on every tick; a local-camera tick reports
  transport/jpeg_decode absent-with-reason, not zero; a proxied tick reports
  transport absent with the proxy reason and contributes **no** sample to
  either transport series; fuse ≤ observe and infer + decode = policy_advisory
  within rounding; existing `stage_ms` keys byte-identical in meaning
  (dashboard and old-run replay unaffected).
- **Kotlin**: capture ≤ encode_start ≤ encode_done ≤ enqueue on every frame
  the pipeline emits; the trio rides ping N+1 and matches pong N's stamps;
  golden/interop parity for every new field; inbound advisory line written
  with the reader's receipt stamp; `AdvisoryHolder` marks first-shown once
  per advisory and never for an expired one.
- **Offline join**: a synthetic phone log + Jetson run joins every advisory
  to exactly one tick, return_ms carries the bound of the estimate it used,
  an advisory with no matching tick is counted (unmatched), and an old run
  without `stages` still reports rather than crashing.

## What this establishes, and what it does not

It establishes, per tick, where the loop's time went — on which device, on
which clock, with what confidence — and makes a missing, proxied or
cross-clock number distinguishable from a measured one in the record itself,
which is the recurring defect class tasks 31/32 name. It gives the Jetson a
delay-free, bounded offset for the first time, and it gives the phone-side
stages exactly, with no clock conversion at all.

It does not establish that the numbers are *good* (no gates move), does not
measure shutter→callback capture bias, does not measure actual pixel-draw
time behind the 250 ms UI tick, does not produce a live return figure, and
does not validate `ASSUMED_SKEW_PPM` (timebase.py:96-110) — the wall-clock
cross-check fields the four-stamp samples carry for that purpose
(timebase.py:166-174) remain unconsumed.

## Open items flagged for the user

1. **"decode" is read as advisory decode**, from the task's own ordering
   (detect, track, fuse, infer, *decode*, return) — my first instinct was
   JPEG decode, i.e. checking the list contradicted the instinct. Both are
   instrumented (`decode`, `jpeg_decode`); only the naming needs sign-off.
2. **"fuse" is read as the feed-fusion + peer-merge sub-segment** inside
   `builder.build` (observation_builder.py:302, :473). If "fuse" meant the
   whole observation assembly, that is the existing `observe` and the new key
   is redundant naming — say which.
3. **"capture" is an instant, not a duration.** The shutter→callback bias is
   real and unmeasurable on this clock (CapturedFrame.kt:16-24). Accepting
   that as out of scope is a choice the user should see.
4. **Render is quantised by the 250 ms advisory tick**
   (MainActivity.kt:255). Tightening it means either a faster UI poll or draw
   instrumentation; neither chosen here.
5. **Return is offline-only.** A live return figure needs the phone to echo
   advisory receipts up the wire; deferred, no strong preference.
6. **How long the one-way fallback lives** once every handset build carries
   the trio — retiring it simplifies the adapter but breaks old-app drives.
7. **Wire growth on the busiest channel**: two int64 fields + one wire stamp
   per camera frame (~70 bytes of header per frame, ~0.3% at 25 kB frames).
   Judged negligible; recorded because it is per-frame forever.

## Open items carried in

- Task 27's HERE parse has never met a real response body. (No HERE work in
  this task; the key never appears in any URL or log.)
- Task 28's observation vector cannot be feed-informed without a
  simulator-side change.
- `MAX_QUERY_RADIUS_M` (10 km) unchecked against HERE v7's accepted range.
- `achieved["camera_hz"]` overstates the sustained rate when the channel
  evicts; needs `Session.send` to return the displacement it computes.
- ~~Four-stamp time sync needs the phone to carry `t4` on its next ping,
  before task 33.~~ Closed by this plan (steps 1-4).
