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
| D9 | `GpsSource` behind an interface, FusedLocation adapter plus fake | Same reason as the camera: the policy is pure and the adapter is thin | call the platform directly; untestable off-device |

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
| 10 | FusedLocation adapter | real fixes on the emulator via a mock provider |
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
4. `t_mono_ns − t_capture_mono_ns` for camera and GPS. This is the second operand task
   18's O1 lacked, so it is the first honest read on that bias.

**Not measured:** real GNSS quality, real link behaviour, anything needing the handset.

---

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

## 8. Needs sign-off

1. **O1** — whether GPS receipt time should get a wire field (`t_receipt_mono_ns` on
   `gps`), which is a coordinated Python/Kotlin/vector change, or stay in the local log.
2. Whether the interop test should run against the **sandbox Python** in-tree or a
   pinned copy, given the Python side is under active development in the same repo.
