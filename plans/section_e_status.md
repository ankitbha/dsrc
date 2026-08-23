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

### Task 19 remaining (4 of 11 steps)

- the time-sync responder — small, since the phone runs no estimator (task 16 established
  the converting side must be the initiating side, and the Jetson converts);
- the Kotlin↔Python interop test, which is the half of the acceptance criterion the
  golden vectors cannot cover;
- GPS capture itself, with the FusedLocation adapter and both clocks;
- wiring the session into `SensingService` so camera and GPS share one connection.

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

## How to run any of it

```
cd dsrc/phone
JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home ./gradlew :transport:test :app:check
JAVA_HOME=... ANDROID_HOME=/opt/homebrew/share/android-commandlinetools ./gradlew :app:connectedDebugAndroidTest
```

`:app:check` includes lint and the merged-manifest gate; `:app:testDebugUnitTest` alone
skips the gate. The instrumented suite needs the `dsrc_test` AVD booted. Only JDK 17
works — the Gradle on `PATH` runs on JVM 26, which no current AGP supports, so always
use the wrapper.
