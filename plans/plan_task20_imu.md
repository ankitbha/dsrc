# Task 20 — IMU capture and forwarding

## Short version

Capture accelerometer and gyroscope, pair them into one `ImuSample`, gate to the
commanded rate, and forward on the `imu` channel. The wire message, its decoder and the
Python peer already exist from task 19; this task is the phone-side producer and its
wiring into `SensingService`.

Three things decide whether the samples are worth anything, and all three are about
time or pairing rather than about reading a sensor:

1. **`SensorEvent.timestamp` is not guaranteed to be the clock everything else uses.**
   The rest of this app stamps on `SystemClock.elapsedRealtimeNanos` — GPS fixes, the
   transport's enqueue stamp, the timebase exchange. `SensorEvent.timestamp` is
   *documented* as the same base, and on the emulator it is, but it is a well-known
   vendor bug for it to be `System.nanoTime` or even a boot-relative count with a
   different epoch. A sample whose capture stamp is on a different timebase is worse
   than a missing sample: it arrives looking valid and lands the Jetson's fusion at the
   wrong instant. So the source **measures the offset** rather than assuming, and refuses
   to convert silently.
2. **Accelerometer and gyroscope are separate streams.** `ImuSample` carries both in one
   message, so they must be paired, and the pairing is a real approximation — the two
   sensors fire independently and at rates that drift against each other. The sample's
   `t_capture_mono_ns` can only be one of them.
3. **The commanded rate is 50 Hz, and Android's sampling period is a hint.** The platform
   delivers faster or slower than asked, in bursts if the sensor is batching. The
   `RateGate` that camera and GPS already use is what makes the emitted rate the
   commanded one.

**Scope boundary.** No orientation, no fusion, no filtering, no calibration — the phone
forwards raw axes and the Jetson interprets them. `ImuSample` has no orientation field
and this task does not add one.

## Open decisions — taken, not asked

Recorded here because `plan_dsrc_rec` takes the recommended option rather than raising
it.

| Decision | Taken | Why |
|---|---|---|
| Which sensors | `TYPE_ACCELEROMETER` + `TYPE_GYROSCOPE` | The calibrated pair. Uncalibrated variants expose bias estimates the Jetson has not asked for, and the message has no field for them. |
| Which event drives a sample | The **accelerometer** | The stamp has to belong to one of the two readings, and it should be the one that produced the sample. **The original reason given here — "it is the higher-rate stream" — was wrong**: both sensors are registered at the same commanded period, and on the emulator both report 2–100 Hz. With equal rates the gyro's age is roughly uniform over one period, so `staleGyroSamples` will not be rare. |
| The sample's capture stamp | The accelerometer event's own timestamp | It is the stamp of the reading that produced the sample. Using "now" would fold the delivery delay into the capture instant, which is the mistake task 19 corrected for the pong's wire stamp. |
| Pairing skew | Recorded, not hidden | `gyroAgeNs` is a phone-side statistic on `ImuPipeline.Stats`, **not** a wire field — `ImuSample` carries `ax..gz` and `accuracy` and the contract is frozen. A sample whose gyro half is older than one commanded period is still sent, and the age is counted so a bad pairing is visible rather than inferred. A gyro stamped *after* the accelerometer is real (one handler, equal rates) and gets its own counter rather than a negative age that would cancel real error in the mean. |
| A sample with no gyro yet | Dropped, counted | At startup the accelerometer may fire before the first gyro event. There is no defensible filler: zeros are a reading, and nulls are not what the message allows. |
| Requested sampling period | `SENSOR_DELAY_GAME` equivalent, computed from the commanded rate | Asking for exactly the commanded period and gating on top is what the camera does. |
| Batching (`maxReportLatencyUs`) | Zero — no batching | Batched delivery hands over a burst of events whose timestamps are correct but whose *arrival* is late, and the rate gate would then pass one and drop the rest of the burst. Latency here is worth more than the wakeup saving. |
| `accuracy` | From `SensorEvent.accuracy`, of the accelerometer | The field is nullable and per-sample. Reporting the driving sensor's accuracy is the only reading that corresponds to the stamp. |
| Timebase mismatch | Refuse to capture, report it | See below. |

## The timebase check

On start, the source reads `SensorEvent.timestamp` from the first event and compares it
against `SystemClock.elapsedRealtimeNanos()` taken at delivery. The delivery stamp is
necessarily *later* than the capture stamp, so:

**This is not enough on its own, and the first version got it wrong.** The headline vendor
bug is `SensorEvent.timestamp` on `System.nanoTime` (CLOCK_MONOTONIC) rather than
`elapsedRealtimeNanos` (CLOCK_BOOTTIME), and those two do not differ by an *epoch* — they
differ by the device's accumulated suspend time, which starts at zero and grows. A handset
that had suspended for 1.5 s produced a difference of 1.5 s, inside any bound generous
enough not to false-alarm, and every sample after it carried a silent constant 1.5 s
error: about 21 m of road at 50 km/h.

So both candidate clocks are read at delivery and the event is attributed to whichever it
is nearer. The ambiguous case is exactly the harmless one — when the two clocks have
barely diverged it does not matter which the sensor uses, because choosing wrong costs at
most that gap — and the gap is recorded, so an *accepting* verdict still says how wrong it
could be. Two tolerances, not one: how long a delivery may plausibly take, and how far the
clocks may drift before the attribution is worth making.

If the check fails the source does not convert and does not guess an offset — it reports
`ImuTimebase.MISMATCHED` and **tears the registrations down**. An earlier version only
stopped *emitting*: both listeners stayed registered and the thread stayed up, so two
sensors were held awake at the commanded rate for the whole drive to produce nothing. A wrong offset applied silently is the
failure this exists to prevent, and an IMU stream is not worth taking the phone down for:
sensing continues with camera and GPS.

This is the one piece of task 20 the emulator cannot really exercise, because its virtual
sensors report the same clock. The check is therefore driven in tests by calling
`ImuPairing` directly with both clock readings — there is no injected event *source*; the
platform listener is thin glue and everything it decides lives in that class. What the
device does pin, through `dumpsys sensorservice`, is the glue itself: which sensors were
registered, at what rate, with what batching, and whether they were given back.

## Shape

Mirrors the GPS pair, which is the closest existing model.

- **`ImuSource`** — the platform edge. Owns a `HandlerThread` named `dsrc-imu` (so the
  thread census in `SensingServiceTest` can see it, which is how three resources were
  found unpinned in task 17), registers both listeners on it, pairs the streams, runs the
  timebase check, and hands `ImuReading`s to a sink. No rate policy, no transport.
- **`ImuPipeline`** — the rate gate and the counters, and the conversion to `ImuSample`.
  Mirrors `GpsPipeline`: `seen`, `accepted`, `gated`, `refusedStopped`, `delivered`,
  `refusedBySink`, plus `unpaired` (no gyro yet) and a `gyroAge` summary.
- **Wiring** — `SensingService.allocateAndStart` creates both, `onSensingDown` releases
  them as two more independent `release(...)` steps, and the seven-field invariant becomes
  nine — which silently broke a mutation-harness anchor that had been written against the
  seven, and left that pin inert until round 2 noticed.

## Counters, and the identity they have to satisfy

`seen == accepted + gated + refusedStopped + unpaired`, asserted per heading and not only
as a sum. Round 6 found that a sum is blind to a swap between its own terms — an
identity held perfectly while a delivery failure was filed as a success — so every
heading gets its own assertion alongside the balance.

## Tests

JVM. There is no injected event *source* — the platform listener is thin glue, and
everything it decides lives in `ImuPairing`, which a test drives directly. Round 1 is why:
with the decisions inside the `SensorEventListener`, thirteen mutations to them survived
267 tests, and four applied at once left the instrumented suite green while the channel
transmitted nothing all session. A pure verdict function was pinned and its *use* was not,
which is the distinction that matters — extracting a function only helps if the caller is
reachable too.

- Rate: a commanded 50 Hz over a simulated minute emits within tolerance of 3,000, and a
  re-commanded rate re-anchors rather than drifting.
- Pairing: a sample carries the *latest* gyro reading, not the first; the gyro's age is
  recorded; an accelerometer event before any gyro event is `unpaired` and counted.
- Stamps: the sample's capture stamp is the accelerometer event's, not the delivery
  instant. Pinned by driving the two apart — with them equal the assertion cannot fail,
  which is the trap that made a camera receipt test unfailable in an earlier round.
- Timebase: a matched clock captures; a mismatched one reports `MISMATCHED` and emits
  nothing. Both directions, because a check that only ever sees the good case is a
  premise that is never active.
- Teardown: the `dsrc-imu` thread is gone, from `WORKER_PREFIXES`. **Unregistration is not
  a JVM test** — it needs the platform's record, and lives in `ImuCaptureTest`
  (instrumented) instead. It went untested for two rounds because `quitSafely()` ends the
  thread whether or not the listeners were removed, so the thread census reported success
  on a leak.
- Accounting: the identity above, per heading, plus the balance.

Instrumented, on the device:

- The `dsrc-imu` thread starts with sensing and is gone after it stops, and after a
  failed teardown — added to `WORKER_PREFIXES`, so the existing teardown tests cover it
  without new tests.

## What was built, and the two decisions that moved

Built as planned, with two departures worth naming.

The timebase check's *verdict* is a pure function on the companion
(`ImuPairing.verdictFor`), not a private method. The plan said the policy would be pinned
rather than the platform's behaviour, and that is only true if a JVM test can reach it --
the emulator's virtual sensors report the same clock, so the mismatched branch is inert on
a device and an instrumented test of it would assert nothing.

`unpaired` is offered through its own entry point (`offerUnpaired`) rather than as a
nullable reading. The alternative was an `ImuReading` with nullable gyro axes, which would
have put a null in the one place the wire format has none and invited someone to fill it
with zeros later.

Two of the tests written for this could not fail, and both were found by mutation rather
than by reading them:

- The gyro ages in the summary test only ever rose, so "keep the maximum" and "keep the
  last" gave the same answer. Fixed by making them descend.
- The stale-gyro test used 5 ms and 25 ms against a 20 ms period, stepping over the
  boundary, so `>` and `>=` agreed. Fixed by asserting exactly one period, which is *not*
  stale because "older than one period" is strict.

That is the same shape as the two findings in section E's status note, and it keeps
happening in the tests rather than in the code.

## What this task does not settle

- Whether the pairing approximation is good enough for the Jetson's fusion. That is a
  question about the fusion, and it needs data from a real drive, not a decision here.
- The real delivered rate and jitter on the target handset. The emulator's virtual
  sensors are synthetic; the number is worth measuring on hardware and is not worth
  asserting here.
- Whether any handset we ship on actually has the timebase bug. The check is cheap and
  the failure is silent, which is enough reason to have it regardless.
