# Section E status

Running position on the phone app (tasks 17–25). Updated as work lands; read the
bottom section first if you only want what needs you.

Repo: `dsrc`, branch `main`. Everything below is pushed unless marked otherwise.

---

## Needs you

Nothing here blocks progress — each is a call I made and recorded rather than waited on.

| # | question | where | my call |
|---|---|---|---|
| 1 | GPS "receipt time" has **no field in the frozen wire contract**. The task asks for fix time *and* receipt time. | task 19 O1 | Logging receipt locally. Adding `t_receipt_mono_ns` to `gps` is a coordinated Python + Kotlin + golden-vector change and not this task's to make. |
| 2 | `minSdk 29` is a guess about the handset. | task 17 O1 | 29 covers everything used; the API-31 emulator tests above it. One place to change. |
| 3 | The 9 km HERE corridor has no legitimate path source. The phone has no route, and picking one is a sensing decision the spec now forbids it. | earlier HERE work | Deferred to task 21. Interim would be a heading projection, which leaves the road on any real curve. The honest fix is widening the downlink. |
| 4 | `t_capture_mono_ns` is an arrival stamp, not the shutter. | task 18 O1 | Kept on `elapsedRealtimeNanos` because it must share a clock with the header's enqueue stamp. **Its bias is unquantifiable from the two stamps available** — differencing them came out negative, which is impossible for one clock base and is itself the proof they differ. Needs task 19's enqueue stamp. |
| 5 | Non-rate settings (camera resolution, JPEG quality, GPS accuracy, HERE query shape) are Jetson-owned but `rate_cmd` carries only rates. | task 17 O2, task 18 O2 | Local config stand-in, shaped like the future wire object. |
| 6 | The GPS adapter can **never send `valid: false`**, so the Jetson cannot tell "no fix" from "the phone stopped sending". | task 19 | Left open. `LocationManager` only delivers a `Location` and a `Location` always carries a position. The platform signal exists (`onProviderDisabled`) and is unwired; whether it belongs on `gps` or `telemetry` is a task 24 question. |
| 7 | Camera frames can be dropped in **two** places and only one is visible to the receiver. | tasks 18/19 | Accepted and counted. `FrameBuffer` and the channel queue are both depth-1 latest-wins; a drop before the sequence number is assigned leaves no gap, so sequence-gap accounting undercounts camera loss. |

---

## The watcher

`../.pipeline/` holds a stall detector, because the failure mode of this run has
been stopping after a report rather than continuing to the next task. A cron job
in the Claude session runs `watch.py` every 11 minutes while the REPL is idle.

Idle is not the same as stalled, which is why it answers with one of seven
verdicts instead of a boolean. `HOLD` covers a stop that is deliberate — the
state file names something only the user can decide — and `WORKING` covers the
six minutes after any commit or uncommitted edit, so a nudge cannot land in the
middle of an implementation step. Only `CONTINUE` resumes anything.

`ESCALATE` is the part worth keeping. Four consecutive nudges that change nothing
in the repository stop the nudging and report instead: a watcher emitting
`CONTINUE` forever against a wedged run generates a stream of activity with no
progress in it, which reads like it is working.

The counters live in `pings.json` rather than `state.json`. Folding them together
makes the position file's mtime partly the watcher's own footprint, so "how long
since anything happened" ends up measuring the instrument instead of the work.

All seven verdicts were exercised before arming it, plus two paths that carry the
logic: progress resets the idle counter, and an uncommitted edit registers as
activity. The cron job dies with the session, so it covers "stopped but alive"
and not "the session died" — `state.json` is the cold-restart path for that,
naming the next action in full.

### Sign-off, precisely

There has been **no validator sign-off on any task in this section.** Each ended
with me fixing the last round's findings and moving on, so the final round's
fixes went in unaudited: task 17 after round 3, task 18 after round 2, task 19
after round 1. That is not a formality. Round 3 on task 17 found that round 2's
try/catch ran correctly and the process died anyway, and it caught a `stopSelf`
pin that had regressed to surviving. Round 2 on task 18 found three defects in
round 1's fixes. Every round so far has found something in the previous round's
fixes, which is the reason the run does not get to stop when the fixes feel
finished.

---

## Done and pushed

### Task 17 — Android project skeleton
Two Gradle modules: `:transport` (pure Kotlin/JVM, no Android, so the wire contract is
testable at laptop speed) and `:app`. **82 JVM tests + 20 instrumented, 0 failed.** APK
6.8 MiB; cold build 8 s with no daemon or cache. 15 manifest facts read back off the
merged manifest, verified to fail when wrong.

Three defects no unit test could find, all caught on the emulator:

- an ignored intent left a resident service forever, because `startService` creates one
  to deliver an intent whether or not the state machine acts on it;
- a teardown throw escaped an unguarded `react(STOPPING)` and killed the process — aimed
  squarely at task 18, since `onSensingDown()` is where the camera gets released;
- `startForegroundService()` is a promise to call `startForeground()` whose breach kills
  the process **on bring-down, 1 ms later, not on a timeout** — so no `try/catch` could
  survive it. Fixed by not making the promise: the only caller is a visible Activity.

Validation ran 3 rounds. Round 1 found a spec-drift test that could never fire (the spec
was not a declared Gradle input), a manifest gate blind to the three permissions the app
cannot start without, and permission constants that asserted against themselves. Round 2
found the foreground-promise crash **in my round-1 fix**. Two of my tests were vacuous —
an "exhaustive sweep" that pinned no row of the transition table.

Deferred with reasons in the plan: `MainActivity`'s wiring is unpinned pending an
Activity harness (scheduled into task 23), and two test seams ship in the APK.

### Task 18 — Camera capture
**178 JVM tests + 37 instrumented, 0 failed.**

| commanded | achieved | source |
|---|---|---|
| 1 Hz | 0.90 | 28.4 fps |
| 5 Hz | 5.00 | 28.5 fps |
| 15 Hz | 15.00 | 28.3 fps |

Sustained 30 s at 10 Hz: 299 accepted, 9.97 Hz, no encode failures, no stall. Slow
drain: 151 accepted, 20 drained, 131 dropped, accounting balanced. JPEG p50 **25.7 KB at
1280×720** quality 85 on a synthetic scene.

The rate gate carried the task's content and took **four attempts**, each failure found
by a test:

1. scheduling slots from *now* undershoots — a 30 Hz source into a 10 Hz target gives
   7.5 Hz, which reads as the camera underperforming rather than the gate miscounting;
2. scheduling from the previous slot is exact but pays a stall back as a burst, and on a
   throttling phone the stall and the burst arrive in that order;
3. below ~1.1e-10 Hz — *inside* the wire's legal range — the period saturates and adding
   it wrapped negative, so a command meaning "almost never" produced **full-rate**
   capture;
4. re-sending an unchanged rate silently cost a quarter of the frame rate, because
   re-anchoring converts the exact schedule back into the undershooting one.

Four counters reported failure as success. A `pack()` throw left a frame with no
outcome, so total failure read as an encoder backlog. Frames discarded at shutdown were
counted nowhere, making the balance identity false after every stop. A frame refused
because sensing had stopped was reported as rate limiting. And one identity **could not
fail at all** — `gated` was derived from the other terms, so
`seen == accepted + gated + refused` reduced to `seen == seen`.

`ResolutionSelector` silently ignores the requested size unless the aspect ratio is
stated: it defaults to 4:3, so a 16:9 request is filtered out before any resolution rule
runs. The emulator lists 1280×720 as supported and returned 640×480 anyway, and
1856×1392 under a prefer-higher rule — both looking like device limitations.

The chroma row stride was completely unpinned, **and the emulator cannot pin it**: its
virtual camera reports `rowStride == width` and planar chroma, so the padded and
semi-planar paths are inert there. That is exactly where the packer's bugs live.

---

## In progress: task 19 — GPS capture and forwarding

The largest task in the section. "Forwarding" is the word that brings the whole Kotlin
transport: framing, canonical JSON, eight typed messages, per-channel queues with
priority and overflow, the session with handshake and keepalive, the time-sync responder.
GPS is the first passenger.

The acceptance test is deliberately not "GPS arrives": it is (a) all 18 cases in
`specs/transport_golden_frames.json` byte-exact both directions, and (b) the Kotlin
client against the **live Python session** over a socket with both sides' counters
agreeing. Golden vectors pin bytes; only a real peer pins behaviour.

**Step 1 of 11 done — Python-compatible double formatting.** 1,256 reference cases
generated by running `json.dumps`, all matching byte for byte.

It refuted this plan's own reasoning. I wrote that the digits could be borrowed from
`Double.toString()` and that recomputing them risked picking a different shortest form.
The opposite is true: on JDK 17 `toString` is documented only to distinguish a value
from its neighbours, not to be minimal, and it isn't — `Double.MIN_VALUE` spells as
`4.9E-324` where the shortest form is `5e-324`. Five cases diverged on that. Computing
the shortest form left two more, both ties, because `MathContext(int)` rounds HALF_UP
where Python takes the nearer value.

That only surfaced because the reference was **generated** rather than transcribed.
Transcribing risks agreeing with the bug, and the golden vectors are no help here at
all: every float in them sits in the range where Kotlin and Python already agree. The
first real divergence would have appeared in **task 20**, on a near-zero gyro reading.

**Steps 2–4 done — canonical JSON, framing, and all 18 golden vectors byte-exact.**
Both directions: our encoding matches the recorded prefix, header bytes, length and
SHA-256 for every case, and our decoder reads the frozen bytes back to the recorded
fields. The second direction earns its keep — a codec wrong in a self-consistent way
passes the first test and fails that one. The vector file is a declared Gradle input,
verified by corrupting one recorded hash and watching the task re-run and fail.

The vectors corrected two of my design assumptions:

- **`n` is transport-owned.** The caller supplies a logical header; the framer derives
  `n` from the payload, and a caller value is honoured only if it agrees. I had it as a
  required input.
- **Extensions are top-level header keys, not a nested object.** The file nests them
  under `extensions` for readability, which is what made it look otherwise. Being
  top-level is what makes the header additive.

And the same "agree where it's safe" pattern showed up twice more, both handled:

- `JsonValue.Num` (Long) is a separate type from `Real` (Double). The token's *shape*
  decides the type, not the value, so `35.0` stays a float. An integer beyond `Long` is
  refused rather than widened, since widening loses exactly the precision the type
  exists to keep.
- Object keys sort by **code point**. Kotlin's natural `String` order compares UTF-16
  units, putting a surrogate pair at 0xD800 before BMP characters at 0xE000+; Python
  puts astral characters last. Only a non-ASCII *key* separates them, and no vector has
  one.

**Step 5 done — the channel table and outbound queues.** The table is tied to the spec
row for row (direction, priority, overflow, depth) in *both* directions, so a channel
added to the spec and not to the code fails too. Mutation-checked: changing one depth or
one priority turns the suite red.

Two spec rules drive the queues, and both look like details until they are wrong:

- **A sequence number is assigned at enqueue, before the overflow decision.** That is
  what makes a gap in received sequence numbers the peer's evidence of a drop. Assigning
  at send time would renumber the survivors and hide every drop.
- **The hello spends `control` sequence 0**, so ordinary control traffic starts at 1. A
  peer restarting control at 0 would duplicate the hello's number, and the gap rule
  cannot see it — it only fires on a sequence *greater* than expected — so the divergence
  would be silent and permanent.

**Step 6 done — the refusal vocabulary and the GPS message.** The GPS record encodes
byte-identically to both recorded headers (full fix and all-null), which is the first
message-level conformance on the Kotlin side. Four spec subtleties pinned:

- an absent nullable field is refused, not read as null — absent would conflate "the
  sensor said nothing" with "an older build that never had this field", and the two
  halves deploy separately;
- the coordinate range check is **conditional on `valid`**, which is what the table
  actually says;
- a fractional count is refused rather than truncated, since truncating hides a sender
  bug behind a plausible number;
- an integer is accepted where a float is expected, so acceptance does not depend on how
  the producer spelled a round value.

`MessageError` is deliberately not a `FramingError`: a bad message costs one message, a
bad frame ends the session, and the difference is recoverability.

**Step 7 done — the session.** Two sessions run against each other over a real loopback
socket pair, because the behaviours worth testing only appear with a genuine peer:
handshake ordering, keepalive cadence, stall detection on completed reads, and what a
framing error does to a live session. Piped streams deadlock in exactly those cases.

**Two of my own tests claimed more than the protocol promises**, and both are worth
recording:

- One asserted that 200 messages all arrive. But `gps` is depth 64 and reliable, so
  bursting past a writer that can't keep up *should* drop some — lossless delivery is the
  wrong claim for that channel. Replaced by the right properties: every message accounted
  for under exactly one heading, and a drop leaving a **gap in the sequence numbers the
  peer sees**, which is the whole reason sequences are assigned before the overflow
  decision.
- The other was **flaky — green, red, red**. The stall test gave both peers the same
  injected clock, so advancing time to trigger the timeout also fired the *peer's*
  keepalive, which arrives as read progress and resets the very timer under test. The
  peer is now a raw socket that goes silent after its hello, and the mechanism that
  caused the flake is asserted deliberately: a keepalive *is* read progress and defers a
  stall. Five consecutive runs green.

**120 transport tests, 0 failed, stable over five runs.** (Two commit messages undercount
by one; the counts here are right.)

**Step 8 done — and this is the one that matters most. Kotlin now talks to the live
Python transport.** The peer is `scripts/interop_jetson_peer.py` running the real
`deployment/jetson/transport` Session, not a mock — a mock would agree with whatever the
Kotlin side happens to do.

Measured across the wire, Kotlin sending and Python receiving:

- 40 GPS records arrive, sequence numbers 0–39, monotonic, none dropped at depth 64
- a 40,960-byte camera payload crosses
- keepalives flow in both directions
- an advisory from Python is delivered to us and decodes to the fields the spec names —
  the downlink direction nothing else exercises

The sharpest of these separates **arrival from acceptance.** Python's counters keep
`received` apart from `delivered`, so a frame that arrives intact and is *then* rejected
by the far side's message layer shows up as such. Ours are 10 of 10 delivered with zero
dropped inbound — the first evidence the Kotlin encoder produces what the Python decoder
actually wants, not merely what the framing allows.

Both the script and the Python transport package are declared Gradle inputs, so a change
to either re-runs the one test that can catch the two implementations drifting apart.

Two harness bugs, both the same shape — failure and success looking alike. The Python
drain thread died on a non-existent attribute and the summary reported `frames_received:
0` while the session's own counters showed five arrivals; a crashed drain was
indistinguishable from a quiet link. And my JSON reader returned the *string* `"null"`
for a JSON null, so every healthy run failed an `assertNull` — the tests were red while
the interop worked perfectly.

**126 transport tests, 0 failed, stable over three runs.**

**Step 9 done — the time-sync responder, and with it the whole transport half of task
19.** One message type in both directions, told apart by nulls, because the channel is
the discriminator for everything else and a second type would need a `kind` field. Both
ping and pong encode byte-identically to the recorded headers.

Three things worth keeping:

- **A partial pong is refused.** Accepting one would let a consumer compute an offset
  from a mixture of set and missing terms — the arithmetic would run and produce a
  number, which is worse than an error.
- **The pong echoes the ping's own wire stamp**, not our clock. That was task 15's
  correction to its own plan: substituting our clock replaces the initiator's t1 with a
  value from a different device, leaving the offset wrong by the whole link delay.
- **The phone runs no estimator**, deliberately. My first attempt to pin that used
  reflection over member names — needs `kotlin-reflect` at runtime and would pass for an
  estimator called something else. Replaced with the behavioural property: a hundred
  exchanges in between don't change the answer to an identical ping.

One of my tests was wrong **twice**, instructively. It asserted a drop leaves a gap
*between* received sequence numbers. Where the gap falls is timing-dependent: if the
whole burst is enqueued before the writer wakes, drop-oldest leaves the survivors as one
contiguous run and there is no interior gap at all; if the writer interleaves, the gap
lands in the middle. Both are correct. What holds either way is the **count** — the
sequence numbers the peer never saw equals what the sender dropped — and that's also the
only form a receiver can act on.

**143 transport tests, 0 failed, stable over four runs.**

**Step 10 done — GPS capture with both clocks.** Fix time and receipt time are different
instants and neither substitutes for the other: the difference is the location stack's
own latency, seconds rather than milliseconds on a cold start. Only the fix time reaches
the wire, because the frozen contract has no receipt field (needs-you item 1).

Three deliberate choices:

- **The pipeline reuses `RateGate`** rather than growing a second rate implementation.
  "At the commanded rate" has to mean the same thing on every modality, and the gate is
  the piece that took four attempts to get right.
- **No buffer in front of the transport.** `gps` is reliable at depth 64, so the
  transport's queue *is* the buffer; a second one would mean two places dropping for
  different reasons, with only one of them visible to the peer as a sequence gap.
- **An invalid fix is forwarded and counted, not discarded.** Silence and no-fix are
  different facts and only one is actionable. An out-of-order fix stamp is counted too,
  since the receiver's freshness arithmetic assumes they are monotonic.

### Validator round 1 — ten findings, eight closed

**359 JVM tests across 25 classes, 0 failed, stable over three runs.**

The worst was **a session that reported healthy forever while discarding everything.** A
header could pass the size probe and then grow past `MAX_HEADER_BYTES` at write time,
because the probe used sequence 0 and the real `seq`/`t_mono_ns`/`t_wall_ns` are
substituted later. The resulting `FramingError` escaped the writer's catch, the thread
died *without ending the session*, `isRunning` stayed true, `send()` kept returning true —
and because the stall check lived inside the writer loop, the timeout died with it.
Verified at four times the timeout with no end reason recorded.

**My first fix for the stall check missed the point.** Moving it to the top of the writer
loop covers an idle link, which is the case that doesn't matter. A wedged peer blocks the
writer *inside* `output.write()`, so the top of the loop is never reached again. Both IO
threads spend their lives blocked, so neither can be trusted to notice silence. There's a
watchdog thread now.

**The sender rule was not implemented at all.** The spec makes every refusal a sender rule
too, and Python runs its typed decoder on every outbound message; `send()` checked only
reserved keys and the channel name, leaving **six of nine refusal reasons unreachable
outbound**. Worse, a non-finite value made `send()` throw at the caller — on the phone,
into a sensor callback.

**`sendHeartbeat` destroyed application messages.** It drew its sequence number by
enqueueing and immediately polling; `enqueue` appends and `poll` takes the head, so it
returned whatever was already waiting — a control message destroyed with `dropped` still
zero, then the heartbeat written twice with a duplicate sequence number.

**The double formatter was wrong on 46 exact powers of two**, and the reason is sharper
than my docstring anticipated: the shortest round-tripping decimal is not always the
*nearest* k-digit decimal, because a double's rounding interval is asymmetric at a power
of two. 2^-24 is exactly `5.9604644775390625e-08`; at 16 digits that's a tie,
nearest-even picks `...062` which doesn't round-trip, and only `...063` does. Both
neighbours are tried now, against a second reference set of 17,568 dyadic cases generated
from CPython. Every `Float` widened to a `Double` is in that family.

Also: `t_wire_mono_ns` was reserved on the Python side and not here, so a caller could
set the field the peer's timebase reads as our departure stamp. The JSON parser accepted
seven shapes CPython rejects, one of which **corrupted silently** — `toIntOrNull(16)`
accepts a leading sign, so `\u-041` became U+FFBF. And a valid GPS fix with a null
coordinate was accepted here and refused by Python.

### Validator round 2 — seventeen findings, and round 1's fixes were unpinned

The heavier round. Three of round 1's fixes turned out to be **pinned by nothing**: the
widest-value probe, the write-time framing guard, and the wire stamp could each be reverted
with the whole suite green. That is the same failure the earlier rounds kept finding, one
level up — the fix was right and the test could not tell.

The worst finding was underneath the probe rather than in it. `t_mono_ns` and `t_wall_ns`
were read on the **writer** thread, where the spec defines them as the sender's clocks *at
enqueue*. So `t_mono_ns - t_capture_mono_ns` — which the spec names as the queueing
latency — measured capture-to-write and reported the queue's own depth as if it were the
sensor's. The comment in `writeMessage` asserted the field was an enqueue stamp two lines
above the call that made it a write stamp. It also inverted the wire stamp, which was built
before the header's clock call and so came out **deterministically earlier** than the field
it exists to postdate: 8 of 8 frames negative when measured. Both clocks now travel on the
message, which makes the wire stamp later by construction and lets the probe check the real
header instead of a guess.

This would have poisoned the O1 experiment the plan and this file both promise. It would
have produced a plausible, wrong number, which is worse than no number.

Two more of the same shape as round 1's dead writer, in the two places round 1 did not
look. The **reader** had no guard beyond `EOFException`/`FramingError`/`IOException`, so an
application router bug killed the thread with `ended` still false — `isRunning` lying,
`send()` returning true, inbound frames consumed by nobody with no counter moving, and the
session finally ending as `STALLED` with a null cause, blaming the network for an app
fault. And the **keepalive** lived in the writer's idle branch, so a phone under sustained
camera traffic sent none at all.

**The timebase finding, where the validator had the direction backwards.** It reported that
the phone cannot answer a ping. The spec says the opposite — "The phone initiates and the
Jetson only ever answers" — and Python's `TimeSyncInitiator` is documented as "The phone's
side". So the phone had the *wrong half* built, and that half was reachable from no code
path: nothing initiated, so the timebase never ran in either direction and
`t_wire_mono_ns` had no consumer at all. Worse than reported, not better.

**Still open:** nothing drives the ping cadence, and five of eight typed messages are
absent — which is why `rate_cmd` with `rates: 0` is still accepted, the spec's own worked
example for why the sender rule exists.

---

#### What round 2 came to

Thirteen findings closed and mutation-pinned; two pushed back on with counter-evidence;
two recorded as deliberate gaps with the condition for revisiting them. **212 transport,
213 app, 41 instrumented tests, 0 failed.**

The push-backs matter as much as the fixes. F15 reported a "phantom `STALLED`" when
`close()` races the watchdog, measured at 1 in 30. It cannot happen: `finish()` is a
compare-and-set, so a losing caller cannot overwrite the reason. **My own fix for it was
worse than the finding** — a flag to make `close()` win, which removing again survives 200
rounds, because the CAS had already decided. Shipping a guard nothing can observe is the
failure mode where dead code reads as load-bearing, so it went back out and the reasoning
lives in `finish()`'s docstring.

And F13's direction was inverted, which made the real finding worse rather than better. The
spec says the phone initiates; the phone had the responder built, and it was reachable from
nothing.

Two things are recorded rather than built, each with the condition that forces the issue.
**Inbound queues** — the spec describes them, Python has them, and the phone's only inbound
channels are low-rate; the moment a handler does anything slow (a disk write in task 25, a
UI hop in task 23) synchronous delivery couples that latency to the read loop, which is
what the stall timeout watches. And the **ping cadence** — `sendTimeSyncPing` exists and is
exercised, but no timer calls it, so a running app still does not perform the exchange.

One thing is worse than a gap: the interop test asserts `"dropped_inbound": 0` against the
**Python peer's** counter, so it reads as a check on our inbound accounting and is nothing
of the kind. It goes with the F9 work.

---

### One of the two open findings is now closed

**F8 is closed** (`7c4b632`). The adapter, the link and the service wiring are real:
`SensingService` owns one session that camera and GPS share, and **393 JVM tests across
29 classes pass, 0 failed**.

The adapter is `LocationManager`, not the fused provider the plan named. `num_sats` is a
required, non-nullable field and the fused provider has no satellite count — it is not on
`Location`, and `GnssStatus` is a `LocationManager` facility — so every fused fix would
have gone out as a *valid* fix claiming zero satellites, with no null in that field to
mean "unknown".

Both GPS clocks come off `elapsedRealtime`, so unlike the camera's pair the difference is
a real delivery latency rather than a latency plus an unknown offset. That is the
measurement task 18's O1 could not make.

A test found a defect that would have ended a drive. A version mismatch throws
`FramingError`, which extends `Exception` rather than `IOException` or `RuntimeException`,
so it escaped both catches and killed the link thread outright — no reconnect, every send
refused, nothing recording why. It is also the one handshake failure guaranteed to happen
in the field, after the Jetson is updated and the phone is not.

And `camera` was exempt from the sender rule and should not have been: the
highest-volume channel was the one place an unchecked field would travel thousands of
times before anyone looked.

**F9 is still open.** The interop test remains far weaker than the plan's step 8 promises
— 1,000 frames with deliberate overflow, a malformed message and a framing error, and
per-reason counters compared field by field; actual is 40 frames with none of those
provoked. The evidence stands: **removing the hello's reservation of control sequence 0
leaves all six interop tests green.** Held until validator round 2 reports, because its
findings name the mutations the interop leg has to detect.

`imu`, `here` and `telemetry` messages land with tasks 20, 21 and 24 — their producers.
The golden vectors pin framing, not message decode, so they pass without them.

---

## Not started

Tasks 20 (IMU), 21 (HERE client), 22 (sensing-config handling), 23 (advisory display,
plus task 17's deferred Activity harness), 24 (thermal), 25 (local session logging).

**Section E will not finish in one sitting.** Each task is taking a validator
conversation of several rounds, and those rounds keep finding real defects — including
several in my own fixes. The bar is worth more than the pace.

---

## Two arguments I made, and why both were wrong

Worth recording separately from the findings, because the same mistake produced both and
it is a mistake about method rather than about Android.

Twice I removed a guard on the strength of an argument that the code it protected could
not be reached, and twice a validator reached it. The first: `onSensingUp` cannot be
entered with live fields, because every route into `STARTING` has been through teardown.
The enumeration was of the routes where teardown *succeeds* — `onSensingDown` ran eleven
releases in sequence and nulled the seven fields last, so a throw part-way through
skipped every release behind it and left them all set. The second, after that was fixed:
`react(STARTING)`'s `try` encloses `handle(Started)` as well as `onSensingUp()`, and
`handle` publishes the state, so anything raised while publishing `RUNNING` is caught as
a *start* failure and offered as `Failed` while the machine is already `RUNNING` — an arm
the machine accepts, and one no teardown stood behind.

Both fixes are structural rather than trigger-specific. Each release runs under its own
guard, and teardown belongs to **entering a stopped state** rather than to the one
transition that happens to pass through `STOPPING`. The point is not that two guards were
missing; it is that "this cannot happen" was doing the work a test should do, and an
enumeration of routes is exactly the kind of argument that looks complete while missing
one.

A third instance, in the tests rather than the code: the first test for the second
finding used a throwing status listener, which is how the validator reproduced it. But
containing listener failures — a fix in its own right — closed that door, so with both
fixes in place the test passed without exercising the route at all. Mutating the teardown
away left all 51 instrumented tests green. A test can pass for the wrong reason the moment
a *different* fix lands, and nothing announces it.

## What the suites cost, and the one thing that made them lie

252 transport + 245 app JVM + 53 instrumented + 952 Python. The instrumented suite is
~2 minutes and the Python suite ~47 seconds; the JVM suites are seconds.

Instrumented runs were intermittently reporting failures in tests nobody had touched,
with no assertion message. The cause is not in the code: installing `com.dsrc.phone`
force-stops any running process of it, so two `connectedDebugAndroidTest` runs against
the one emulator kill each other, and whichever test was executing when the install
landed is named as the failure. A diagnosis pointing at innocent code is worse than a
crash. `scripts/with_device.py` takes an exclusive lock; every device-touching command
goes through it, including from a validator's mirror.

## The pin harnesses

`scripts/remutate.py` and `scripts/remutate_device.py` re-apply every mutation that some
test is supposed to catch, and report one that no longer fails. They exist because a pin
lapsed silently: a test written for the `Failed`-from-`RUNNING` route was driven by a
throwing status listener, and a later unrelated fix -- containing listener exceptions --
closed the door that trigger used. The test kept passing, and mutating the teardown away
left all 53 instrumented tests green.

Run them after landing a **batch** of fixes, not after landing one. That is when a fix
defeats another fix's pin, and it has now happened twice: the second time, removing an
inert-looking exemption from the control decoder made `handleTimeSync`'s error path
unreachable, and the harness caught it between two commits.

An anchor that no longer matches is reported as a failure rather than skipped, because
the code having moved is exactly when the question matters.

```
python3 scripts/remutate.py           # JVM and Python, ~4 min
python3 scripts/remutate_device.py    # instrumented, ~2 min per entry plus lock wait
```

## How to run any of it

```
cd dsrc/phone
JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home ./gradlew :transport:test :app:check
python3 scripts/with_device.py -- env JAVA_HOME=... ./gradlew -p phone :app:connectedDebugAndroidTest
```

`:app:check` includes lint and the merged-manifest gate; `:app:testDebugUnitTest` alone
skips the gate. The instrumented suite needs the `dsrc_test` AVD booted. Only JDK 17
works — the Gradle on `PATH` runs on JVM 26, which no current AGP supports, so always
use the wrapper.
