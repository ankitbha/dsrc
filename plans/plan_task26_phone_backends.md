# Task 26 — Phone backends for `CameraStream` and `GpsReader`, fed from the transport

## The short version

The backends already exist. `sensors/phone_source.py` has `PhoneCameraStream` and
`PhoneGpsReader` behind the same consumer surface the pipeline uses
(`start`/`stop`/`wait_for_fresh`/`latest`/`is_stale`), with 34 tests. What does not
exist is any way to *use* them: `run_demo.py`, `replay_demo.py`, `bench_latency.py`,
`pipeline.py` and `eval_run.py` between them mention `phone_source` zero times.
Nothing outside the test file constructs one. The module docstring's claim that those
entry points "drive a phone exactly as they drive a USB camera" is true of the
interface and false of the wiring.

So task 26 is **not** writing the backends. It is standing up the stack that feeds
them — acceptor, session, router, timebase — and letting an entry point select it.

**One thing blocks it, and it is not wiring.** The Jetson cannot convert the phone's
timestamps to its own clock, because a spec-compliant Jetson never obtains the
estimate. See "The timebase problem" below; it decides the shape of the work and it
is the only thing here I would want overruled if I have called it wrong.

## Scope boundary

In: the two backends reaching a real entry point, fed by a real session from a real
phone, with a timebase good enough for freshness.

Out, and deliberately: HERE ingestion (27), fusion (28), the sensing controller and
its rates (29), shadow/live gating (30), the tick-loop integration and the advisory
return path (31), the end-to-end run (32). Task 26 ends when the Jetson can run its
existing pipeline on phone-fed camera and GPS. Sending anything back is 31.

Also out: latency attribution. See the second open decision.

## The timebase problem

The pipeline decides freshness by comparing stamps against the Jetson's
`time.monotonic()`. The phone's monotonic clock counts from its own boot — measured
67.57 hours apart on this pair. Unconverted, `gps_age` is −243,264 s against a 2.0 s
threshold, so `gps_fresh` is False on every tick, ego speed falls back to neutral,
and the loop keeps producing advisories that look fine. Conversion is not optional.

Conversion needs an offset estimate, and here is the bind:

- `specs/transport_protocol.md:507` — **"The phone initiates and the Jetson only ever
  answers"**. A Jetson receiving a pong is a protocol error, dropped and counted as
  `unknown_value`.
- The phone enforces it. `Session.checkTimeSyncDirection` computes
  `wrongWay = if (role == ROLE_PHONE) !message.isPing else message.isPing`, so a real
  phone refuses a ping from the Jetson.
- `TimeSyncSample` needs four stamps — t1 local send, t2 remote recv, t3 remote send,
  t4 local recv. **The initiator has all four. The responder has three.** The Jetson,
  answering, learns the phone's send stamp, its own receipt and its own departure. It
  never learns t4, the phone's receipt of the pong.

So the side that must convert is structurally the side that cannot estimate.

`scripts/run_loopback_pipeline.py` sidesteps this by having the Jetson initiate, and
its own docstring says that contradicts the spec and "needs sign-off". It works there
only because that harness owns both ends and `transport/timebase.py` is role-symmetric.
Against a real phone it cannot work — the phone will refuse the ping. **The loopback
harness is not evidence that a phone-fed run works.**

### Recommended: a one-way offset on the responder side

The Jetson estimates `offset ≈ t1 − t2` from the pings it already answers — the
phone's send stamp against its own receipt. The error is the one-way delay, not
zero, and it is one-directional so it does not cancel.

That is adequate for what task 26 needs and inadequate for something task 26 does not
do, which is why it is worth being precise rather than calling it "good enough":

- **Freshness** is the only consumer here. The threshold is 2.0 s. Measured one-way
  delay on this pair is ~28 ms (a 56.9 ms handshake round trip over `adb reverse`,
  7.2 ms on emulator loopback). Three orders of magnitude of margin.
- **Frame-to-frame intervals** are deliberately not converted at all — they stay on
  the phone's clock so link jitter stays out of them. Unaffected either way.
- **Latency attribution** would inherit the full one-way delay as a systematic bias.
  At ~28 ms against a 200 ms end-to-end target that is 14%, and it is a bias rather
  than noise, so averaging does not remove it. Task 26 must not be read as having
  delivered a clock fit for that.

Rejected alternatives, with why:

- *Jetson initiates* (the loopback arrangement). Needs a spec edit and a phone change,
  since the phone rejects the ping. Two codebases and a protocol change to obtain
  something a one-way estimate already gives to within milliseconds.
- *Phone converts before sending.* Breaks the rule that a capture stamp is on its
  sender's clock, which is the invariant the whole timebase design rests on.
- *Ship the estimate in a new wire field.* More on this below — it is the right answer
  later and the wrong one now.

## Open decisions

**1. The one-way estimate is a task-26 decision, not a permanent one.** Recorded here
rather than buried in a commit. If it is wrong, it is wrong now, before the wiring is
built on top of it.

**2. Full four-stamp samples need a wire field, and section G needs them.** The clean
fix is for the phone to carry `t4` for the *previous* exchange on its next ping: the
phone already computes it, the control channel already exists, and the thermal work
just established the pattern for adding an absent-tolerant field to a shipped protocol
without a flag day (`absentableNumber` / `absentable_number`, omit-when-absent, both
decoders tolerate absence). That gives the Jetson complete samples and a delay-free
offset. **It is out of scope for 26** — it touches the phone, the Python transport and
the spec, and 26 does not need it. It should be settled before task 33 (per-stage
timestamps), which cannot be honest without it.

**3. Camera and GPS must be selected together.** They share one session and one
router; there is no arrangement where the camera comes from a phone and GPS does not,
short of two sessions to the same handset. So the selection is one flag, not two.

## The work

1. **A phone-session assembly, once, in one place.** Acceptor → session → router →
   estimator → `PhoneClockAdapter` → the two backends. This exists in
   `scripts/run_loopback_pipeline.py` against a synthetic phone; promote the assembly
   into `deployment/jetson/sensors/` so an entry point can call it, and leave the
   harness using it rather than duplicating it.
2. **Responder-side offset estimation.** A one-way sample path in `transport/timebase.py`
   fed from the pings the Jetson answers, kept distinct from `TimeSyncSample` so the
   two are not confused: a one-way estimate and a round-trip estimate have different
   error, and a reader must be able to tell which one produced a number.
3. **Selection in `run_demo.py`.** One flag standing up the phone stack for both
   camera and GPS. A run that asks for a phone and gets none is a hard failure with a
   clear message, not a silent degrade to the simulator — the whole point is to know
   whether the phone fed the run.
4. **Lifecycle.** The phone dials, so the Jetson waits. Bounded wait, clear failure.
   Reconnects are already handled inside `_PhoneSource`
   (`_on_reader_ended` / `_on_reader_restarted`); confirm rather than rebuild.
5. **Provenance.** Every run must record which clock produced its stamps and how the
   offset was obtained, one-way or round-trip. Without it a run where conversion
   worked and one where it silently did not are indistinguishable — the loopback
   harness's own stated lesson.

## Tests

- One-way offset: recovers a planted offset within the injected delay; a one-way
  estimate is labelled as such and never presented as a round-trip one.
- The Jetson never initiates: a ping from the Jetson is refused by a phone-role peer,
  pinned so the loopback arrangement cannot quietly return.
- Selection: asking for a phone and getting none fails; it does not fall back.
- Freshness end to end: with a 67.57-hour planted offset, `gps_fresh` is True and
  `gps_age` is plausible — the exact failure the conversion exists to prevent.
- Frame intervals survive conversion unchanged (affine), which the existing tests
  already assert; confirm they still hold through the new assembly.

## Needs sign-off

- The one-way offset for task 26 (decision 1). Everything else here follows from it.
- That `specs/transport_protocol.md:507` stands as written — the phone initiates, the
  Jetson answers, and the loopback harness's contrary arrangement is the thing that
  gets corrected rather than the spec.
