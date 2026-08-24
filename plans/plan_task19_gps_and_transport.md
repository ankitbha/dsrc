# Plan: Task 19 — GPS capture and forwarding

> GPS capture and forwarding, logging both fix time and receipt time.

## Short version

The word doing the work is **forwarding**. Task 18 produced frames into a buffer and
sent nothing; this is the first task that puts bytes on a wire, so it brings the whole
Kotlin half of `specs/transport_protocol.md` with it — framing, canonical JSON, the
typed messages, per-channel queues with priority and overflow, the session with its
handshake and keepalive, and the time-sync responder. GPS is the first passenger.

That makes this the largest task in section E, and the only one whose deliverable is a
*contract* rather than a feature. Two implementations of one wire format drift unless
something holds them together, so the acceptance test is not "GPS arrives" but:

1. all 18 cases in `specs/transport_golden_frames.json` encode to exactly the recorded
   bytes and decode back to the recorded fields, and
2. the Kotlin client talks to the **live Python** `Session` over a loopback socket and
   both sides' counters agree.

The golden vectors pin bytes; they cannot pin behaviour. Handshake ordering, keepalive
cadence, stall detection, priority, overflow and drop accounting only interact under a
real peer, which is why (2) exists and why it is worth the setup.

### Two traps, restated because they now become code

Recorded in task 17's plan as the evidence for a hand-written codec; this is the task
where they bite.

1. `large_ints` carries `9223372036854775807`, `-9007199254740993` (−2^53−1) and a
   `seq` of `9007199254740993`. Gson and kotlinx.serialization both parse JSON numbers
   into `Double` by default, silently returning `9007199254740992`. The frame then
   re-encodes to different bytes and the peer's sequence arithmetic is wrong.
2. Kotlin's `Double.toString()` and Python's `json.dumps` agree on **every float in the
   vectors**, which is what makes this dangerous rather than safe. They diverge outside
   the range those vectors sample: Kotlin emits `1.5E-5` where Python emits `1.5e-05`,
   and `1.0E7` where Python emits `10000000.0`. A near-zero IMU gyro reading is an
   ordinary value in the first range — so task 20 is where a miss here would surface,
   not this one.

`non_ascii_extension` adds a third, smaller one: `señal 温度 ±2°C 🌡` must go out as raw
UTF-8, not `\u` escapes, and the thermometer is outside the BMP, so a surrogate pair
has to survive.

### Scope boundary

**In:** `:transport` in full — `Frame`, canonical `Json`, Python-compatible `Doubles`,
`Channels`, `Queues`, `Stats`, the eight typed `Messages`, `Session`, the time-sync
responder; `GpsSource` with a FusedLocation adapter and a fake; both GPS clocks;
wiring the session into `SensingService` so camera *and* GPS flow; golden-vector
conformance; the Kotlin↔Python interop test.

**Out:** reacting to `rate_cmd` (task 22 — the rate is still local), the advisory UI
(23), thermal (24), on-phone logging (25). The other six message types are implemented
because the golden vectors pin them and a partial `MESSAGE_FOR_CHANNEL` would be a lie,
but only `gps`, `camera` and `control` have a producer in this task.

### Open items

- **O1. Two GPS clocks, and the task asks for both.** "Fix time and receipt time" are
  `Location.getElapsedRealtimeNanos()` (when the fix was made, on the monotonic clock)
  and our own `elapsedRealtimeNanos()` at callback. The wire has one field for each:
  `t_capture_mono_ns` takes the fix time, and the header's `t_mono_ns` is enqueue —
  which is receipt-ish but not receipt. **A dedicated receipt field does not exist in
  the frozen contract.** Recording receipt separately therefore needs either a new
  extension key on `gps` or the local session log from task 25. This plan logs it
  locally and flags the gap rather than inventing a field.
- **O2. `utc_epoch_ns` and `t_capture_mono_ns` are different clocks by design.** The
  first is `Location.getTime()`, GPS wall time, which can step; the second is
  monotonic. Both are required on the wire. They must never be differenced.
- **O3. The phone is a time-sync responder only.** Task 16 established that the
  converting side must be the initiating side and the Jetson converts. The phone echoes
  `t_peer_recv_mono_ns`, `t_peer_recv_wall_ns` and the ping's wire stamp, stamps its own
  `t_wire_mono_ns` at write time, and runs no estimator. Worth stating so nobody ports
  one.

---

## 1. Grounding

### The GPS message

From `specs/transport_golden_frames.json`:

```json
{"altitude_m":35.0,"ch":"gps","fix_quality":1,"hdop":0.9,"heading_deg":91.2,
 "lat":51.5074,"lon":-0.1278,"n":0,"num_sats":9,"seq":1,"speed_mps":13.4,
 "t_capture_mono_ns":1000000001,"t_mono_ns":1100000000,
 "t_wall_ns":1755648000000000000,"utc_epoch_ns":1755648000000000000,"valid":true}
```

and the all-null case, which every numeric field takes when there is no fix:

```json
{"altitude_m":null,"ch":"gps","fix_quality":0,"hdop":null,"heading_deg":null,
 "lat":null,"lon":null,"n":0,"num_sats":0,"seq":1,"speed_mps":null,
 "t_capture_mono_ns":1000000002,"t_mono_ns":1100000000,
 "t_wall_ns":1755648000000000000,"utc_epoch_ns":null,"valid":false}
```

Nullable: every numeric field except `valid`, `fix_quality` and `num_sats`. Payload
empty — a payload on `gps` is `unexpected_payload`. And `lat`/`lon` out of range **while
`valid`** is `out_of_range`, which means the range check is conditional on the flag, not
absolute.

### What the transport owes

Framing is `[4B payload_len][2B header_len][header][payload]`, big-endian, with both
limits checked *before* allocating. Five required header keys, `n` equal to
`payload_len`. Canonical JSON: UTF-8, keys sorted **recursively**, separators `,` and
`:`, no non-ASCII escaping, NaN and Infinity refused. `hello`, `heartbeat` and
`t_wire_mono_ns` reserved. Phone opens the connection; both sides send a hello before
either reads; hello spends `control` seq 0 so the session's own control traffic starts
at 1. Keepalive every 1.0 s, stall at 5.0 s measured on *completed reads* with a
read never larger than 8192 bytes. Priority high → normal → bulk, round-robin within a
tier, never preempting a frame mid-write. `reliable` drops the oldest, `latest_wins`
holds one; `seq` is assigned at enqueue *before* the overflow decision, so a gap is how
the peer sees the drop. A malformed message costs one message and is counted by reason
from a closed vocabulary; a framing error ends the session. A sender must refuse to emit
what its own decoder would refuse, counted separately.

---

## 2. Decisions

Taken by recommendation under `plan_dsrc_rec`. **None is user sign-off.**

| # | decision | why | runner-up |
|---|---|---|---|
| D1 | Hand-written canonical JSON, no library | The two traps. A library that parses to `Double` corrupts 2^53+1, and none formats doubles as Python does | kotlinx.serialization; fails both |
| D2 | A `JsonValue` sealed hierarchy with `JsonLong` distinct from `JsonDouble` | The integer/float distinction is load-bearing on this wire: `35.0` must not become `35`, and `9007199254740993` must not become a double | one numeric type; loses one or the other |
| D3 | Blocking sockets, dedicated reader and writer threads | Mirrors the Python session one-for-one, and the spec states stall detection in terms of completed reads — natural in blocking IO, awkward in NIO | coroutines/NIO; the two implementations stop being comparable |
| D4 | Writer stamps `t_wire_mono_ns` immediately before the bytes leave | The spec requires it, and task 15 found the enqueue stamp's queueing delay dominates a timebase estimate | stamp at enqueue; defeats the field's purpose |
| D5 | Reserve the widest `t_wire_mono_ns` at encode time | Python hit exactly this: substituting a 19-digit stamp into a header sized for a placeholder overflowed `MAX_HEADER_BYTES` and killed the writer thread with `send()` still returning true | re-encode after stamping; the length can then change under the reader |
| D6 | Golden vectors as a data-driven test over the JSON file, not transcribed cases | Transcribing invites a typo that agrees with the bug; and a regenerated file must fail loudly rather than be silently re-approved | hand-written cases; drift-prone |
| D7 | Interop test spawns the real Python session as a subprocess | The only thing that catches behavioural drift between two implementations of one protocol | mock peer; would agree with whatever Kotlin does |
| D8 | GPS receipt time goes to a local log, not a new wire field | O1 — the contract has no field for it, and inventing one is a cross-language change that is not this task's to make | add `t_receipt_mono_ns`; needs Python, Kotlin and vector changes together |
| D9 | `GpsSource` behind an interface, **`LocationManager`** adapter plus fake | Same reason as the camera: the policy is pure and the adapter is thin | call the platform directly; untestable off-device |

---

## 3. Files

```text
phone/transport/src/main/kotlin/com/dsrc/transport/
  Json.kt         # JsonValue, canonical encode, Long-preserving decode
  Doubles.kt      # Python-compatible double formatting
  Frame.kt        # layout, limits, encode/decode
  Channels.kt     # the table: direction, priority, overflow, depth
  Messages.kt     # the eight types + the closed refusal vocabulary
  Queues.kt       # per-channel queue, seq assignment, drop counting
  Session.kt      # handshake, keepalive, stall, reader/writer threads
  TimeSync.kt     # the responder
  Stats.kt        # per-channel, per-reason counters
phone/app/src/main/kotlin/com/dsrc/phone/
  sensors/GpsSource.kt      # interface + fake
  sensors/GpsLocationSource.kt   # FusedLocation adapter
  net/SessionHolder.kt      # owns the connection for SensingService
```

---

## 4. Steps

Ordered so the contract is proven before anything rides on it. Each ends green with a
local commit.

| # | step | done when |
|---|---|---|
| 1 | `Doubles` | matches `json.dumps` across `1e-7 … 1e17`, both signs, and the exponent boundaries the vectors miss |
| 2 | `Json` encode | recursive key sort, raw UTF-8 including astral plane, NaN/Inf refused |
| 3 | `Json` decode | `9007199254740993` and `Long.MAX_VALUE` survive a round trip unchanged |
| 4 | `Frame` | all 18 golden cases byte-exact both ways, including `frame_sha256`; limits checked before allocation |
| 5 | `Channels`, `Queues`, `Stats` | priority order, round-robin, both overflow policies, seq-before-overflow, per-reason counts |
| 6 | `Messages` | every golden message case round-trips; every row of the refusal table has a test; sender-side validation counted apart |
| 7 | `Session` + `TimeSync` | handshake, control seq from 1, keepalive at 1.0 s, stall at 5.0 s, framing error ends the session |
| 8 | Interop | Kotlin against the live Python session: 1000 frames, deliberate overflow, deliberate malformed message, deliberate framing error; counters agree |
| 9 | `GpsSource` + fake | fix time and receipt time both recorded and distinct; no-fix maps to all-null |
| 10 | `LocationManager` adapter | real fixes on the emulator via a mock provider |
| 11 | Wire into `SensingService` | camera and GPS both flow over one session |

---

## 5. Tests

**Conformance** — all 18 vectors byte-exact both directions; double formatting against
Python across the ranges the vectors miss; `2^53+1` and `Long.MAX_VALUE` round-trip;
astral-plane UTF-8; NaN/Inf refused on encode.

**Transport behaviour** — a saturated normal tier does not starve high, and bulk can be
starved; round-robin within a tier; `reliable` drops oldest and counts, `latest_wins`
replaces and counts; `seq` continues across a drop so the peer sees a gap; hello first
in both directions and a version mismatch closes; keepalive cadence; stall measured on
reads not frames; every framing-error row ends the session; every refusal row drops one
message and increments the right reason; reserved keys refused on data channels and
honoured on `control`.

**GPS** — a full fix maps to every field; no fix maps to all-null with `valid` false;
`lat`/`lon` out of range is refused *only* when `valid`; fix time and receipt time are
both captured and differ; the rate gate from task 18 is reused, so a commanded GPS rate
is honoured.

**Interop** — the Kotlin client against the real Python `Session`, with both sides'
per-channel and per-reason counters compared field by field.

---

## 6. Experiments

1. Golden conformance: 18/18, reported as a count.
2. Interop: frames delivered, per-channel drops, per-reason refusals, both sides side by
   side.
3. Loopback throughput and queueing latency by channel at a camera-like payload. This
   also produces the first measurement of what a `here` message costs on the wire, which
   tasks 14 and 16 never took — they measured latency on small frames only.
4. `t_mono_ns − t_capture_mono_ns` for camera and GPS.

   **This was described wrongly and the description mattered.** The plan called it "the
   second operand task 18's O1 lacked, so the first honest read on that bias". It is not.
   O1 asks how far `t_capture_mono_ns` sits from the shutter, and this subtraction *starts*
   at `t_capture`: the shutter is on the far side of it and appears in neither operand.
   What this measures is capture-stamp to enqueue — the pipeline's own cost, which is worth
   having and is newly meaningful now that the stamp is genuinely taken at enqueue. The
   shutter offset still needs a hardware timestamp the camera does not expose, so **O1
   remains unmeasured**, and a number reported against it would have been the wrong
   quantity wearing the right name.

**Not measured:** real GNSS quality, real link behaviour, anything needing the handset.

---

## Measured

Run with `./gradlew :transport:test --tests '*TransportMeasurementsTest' -Ddsrc.measure=true`.
**Loopback on the laptop, not the link** — the in-car path is `adb reverse` over USB and the
development path is Tailscale, so these bound the transport's own cost and say nothing about
either.

| channel | bytes on the wire | frames/s | MiB/s | dropped |
|---|---|---|---|---|
| `gps` (depth 64, no payload) | 265 | 69,269 | 17.5 | 0 of 2,000 |
| `camera` (depth 1, 25.7 KB JPEG) | 25,889 | 6,861 | 169.4 | 0 of 200 |
| `here` (depth 16, 64 KiB body) | 65,884 | 10,764 | 676.3 | 0 of 200 |

The `here` row is the measurement tasks 14 and 16 never took: **a 64 KiB traffic reply costs
65,884 bytes framed**, so the header is 348 bytes of it. At the planned 0.2 Hz that is about
13 KB/s, against the camera's 25.9 KB per frame at 5 Hz — so HERE is roughly a tenth of the
camera's load, not the rounding error the cadence table implies.

Queueing latency, `t_mono_ns − t_capture_mono_ns`, 400 samples each:

| channel | p50 | p95 | max | negative |
|---|---|---|---|---|
| `gps` | 2 µs | 5 µs | 451 µs | 0 |
| `camera` | <1 µs | 6 µs | 352 µs | 0 |

**Zero negatives is the result worth keeping**, because it is the check on round 2's fix: the
enqueue stamp now postdates the caller's capture stamp on every sample, where the wire stamp
used to precede the enqueue stamp on every stamped frame.

Golden conformance: **18 of 18**, counted from the vector file rather than restated.

### Two numbers I had to throw away

The first camera throughput read **1,692 MiB/s at 68,536 fps with one frame of two hundred
arriving.** It divided *accepted* by elapsed time, and `send` returns true on a
latest_wins channel even when the message it displaced is dropped — so it was a
queue-insertion rate on a channel discarding 99.5% of its traffic, and no slowness in the
link could have made it fall.

The first camera queueing p50 read **2,995 µs**, about a thousand times the corrected value.
The pacing wait sat between the capture stamp and the send, so the measurement contained
this harness's own `Thread.sleep` granularity. Moving the stamp after the wait gave 3 µs;
replacing sleep with a spin moved camera's throughput from 378 fps to 6,861, which is how
much of the original figure was the harness.

## 7. Risks

- **Size.** This is four or five tasks' worth of surface in one. The step order is
  chosen so that a partial task still leaves a proven artefact: the vectors pass before
  any session exists, and the session is proven before GPS rides on it.
- **Two implementations will drift.** The interop test is the only thing that catches
  it, and it has to be runnable on demand rather than once by hand.
- **The Python side is the reference and may itself be wrong.** Where they disagree, the
  spec decides, not whichever is older.
- **The emulator's GPS is a mock provider.** It proves the plumbing and the null
  handling; it says nothing about real fix quality, HDOP or satellite counts.

---

## D9 revised: `LocationManager`, not the fused provider

D9 originally named `FusedLocationProviderClient`. It cannot satisfy the frozen wire
contract, and that only became visible once the message table was fixed.

`num_sats` is required and non-nullable. The fused provider has no satellite count:
it is not a field on `Location`, and `GnssStatus` callbacks are a `LocationManager`
facility. Every fused fix would therefore go out as a *valid* fix carrying
`num_sats: 0`, and there is no null in that field to mean "unknown", so a receiver
reads it as a contradiction rather than as missing information. The only way to fill
the field honestly is the API that reports it.

Independently: fused positions are blended across GNSS, wifi and cell, and
interpolated. This phone exists to record what the road did, and a smoothed track is
the wrong ground truth to compare a traffic model against.

`hdop` stays null on every record. No Android API exposes it. `Location.accuracy`
is a metre radius and HDOP is dimensionless satellite geometry, with no conversion
between them, so deriving one from the other would place an invented number on the
wire next to measured ones.

The satellite count is attached from the most recent `GnssStatus`, not from one
measured at the instant of the fix -- the two callbacks are separate and do not
arrive together. It is off by at most one status interval. Counted as *used in fix*
rather than visible, following the GGA convention, because a visible-but-unused
satellite contributes nothing to the position.

## An open gap: this adapter can never send `valid: false`

`GpsRecord.noFix()` and the whole `valid: false` shape are unreachable from
`GpsLocationSource`. `LocationManager` only delivers a `Location`, and a `Location`
always carries a position, so a real phone produces valid records or produces
nothing. The no-fix record is reachable only from the fake source in tests.

The consequence is on the Jetson: it cannot distinguish "the phone has no fix" from
"the phone stopped sending", because both look like an absence of `gps` messages.
The signal exists on the platform -- `onProviderDisabled`, and provider availability
-- and is not wired to anything. Left open rather than guessed at, because whether an
explicit no-fix record or a `telemetry` field is the right carrier is a question for
task 24, where the phone's other health reporting lands.

## A cost worth naming: the camera drops in two places

`FrameBuffer` is depth-1 latest-wins and so is the `camera` channel queue, so a frame
can be dropped in either. `GpsPipeline` deliberately has no buffer for exactly this
reason -- its own docstring argues against two droppers -- and the camera has one
because task 18 needed somewhere to put a frame between the encoder and a consumer
that did not exist yet.

Both counts are visible: `FrameBuffer.Stats.dropped` and the channel's own counters.
What is *not* visible from the far side is the first kind, because a drop before the
sequence number is assigned leaves no gap for the peer to see. A receiver reading
only sequence gaps undercounts camera loss.

## Step 9 was the wrong half: the phone initiates, it does not answer

The plan called step 9 "the time-sync responder". The spec calls the direction
explicitly: **"The phone initiates and the Jetson only ever answers."** Python agrees --
`TimeSyncInitiator`'s own docstring reads "The phone's side" -- so the responder is the
Jetson's half and the phone should never answer a ping at all.

What was built was the responder, and it was reachable from no code path: the class
existed, had tests, and nothing routed a `control` frame to it. So the phone had the wrong
half of the exchange, and the half it had was inert. Nothing initiated, which means the
timebase never ran in either direction, and `wantsWireStamp` -- the field the whole
estimate depends on -- had no consumer.

The transport now owns the exchange the way it owns keepalives, and the role decides which
half runs. A phone sends pings and delivers pongs upward, because the estimate belongs
above a transport that does not know what the samples are for. A Jetson answers pings. The
wrong direction arriving is a protocol error counted as `unknown_value`, which is the
reason the spec names for it, "because the alternative is treating one as the other and
silently producing an offset with the sign inverted".

The responder stays in the Kotlin module rather than being deleted: a Kotlin session plays
the Jetson in the loopback tests, and that is now the role that reaches it.

**Still open:** nothing drives the ping cadence. `sendTimeSyncPing` exists and is
exercised, but no timer calls it, so in a running app the exchange still does not happen
on its own. The cadence and the estimator belong with whatever consumes the offset, which
is not this task.

## Inbound queues: a deliberate divergence, and the test that hides it

The spec says "Inbound queues use the same policies and depths". The Kotlin session has
none: it calls `onFrame` synchronously on the reader thread. Python has real inbound
queues with a per-channel `dropped_inbound`.

The counting half of the divergence is fixed -- inbound refusals are keyed by channel and
by reason, as the spec asks -- but the queues themselves are not built, deliberately, for
two reasons. The phone's only inbound channels are `advisory` and `rate_cmd`, both
low-rate and both handled by fast code, so the backpressure a queue would absorb does not
exist yet. And adding one changes the app's threading contract: today a handler runs on the
reader thread, and moving it introduces a second place inbound messages can be dropped,
which is the exact arrangement `GpsPipeline` argues against for the outbound side.

**What is not acceptable is the interop test.** It asserts `"dropped_inbound": 0` against
the *Python peer's* counter, so it reads as a check on our inbound accounting and is
nothing of the kind -- we have no such counter to check. A test that appears to cover the
divergence is worse than no test, and it goes with the F9 work.

The condition for building them: the moment a handler does anything slow -- a disk write in
task 25, a UI hop in task 23 -- synchronous delivery couples that latency to the read loop,
and the reader is what the stall timeout watches.

## 8. Needs sign-off

1. **O1** — whether GPS receipt time should get a wire field (`t_receipt_mono_ns` on
   `gps`), which is a coordinated Python/Kotlin/vector change, or stay in the local log.
2. Whether the interop test should run against the **sandbox Python** in-tree or a
   pinned copy, given the Python side is under active development in the same repo.
