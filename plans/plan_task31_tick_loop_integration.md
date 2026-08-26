# Task 31 — Integration into the tick loop, advisory returned to the phone

## The short version

Sections E and F built six things the live loop has never constructed:
`SensingController`, `ModeHolder`/`command_for`, `HereFeed`, `FeedFusion`, the
`telemetry` reader, and the advisory return path. `run_demo.py --phone` today
takes camera and GPS off the handset and **sends nothing back down the link** —
no advisory, no rate command. The phone's screen stays blank for the whole drive
and its sensors run at whatever the operator last set by hand.

Task 31 closes the loop: one place per tick where the Jetson decides what to send
back, and a supervisor that survives the phone hanging up.

**Scope boundary.** In: the send path on `PhoneLink`, the controller/mode wiring
in the tick loop, the rate-command cadence, and the redial supervisor deferred
from task 26. Out: the USB path (task 32 is still network), per-stage
instrumentation (task 33), the `t4` wire field (before 33), and any change to
`SensingController.decide` or to what `shadow` means on the wire.

## The four decisions

### 1. The return path belongs to `PhoneLink`, not to `run_demo`

`PhoneLink` is the object that owns the router, the session and the clock
adapter; it is already the unit that gets selected. A caller reaching into
`phone.router` to send would be the second place that knows the assembly order,
and the first one exists because that knowledge leaking into
`run_loopback_pipeline.py` is exactly what task 26 was fixing.

So: `send_advisory(advisory, *, t_capture_mono_ns)` and `send_rate_command(cmd)`,
both returning `bool` — **False when there is no session**, not an exception. A
run whose phone has dropped must keep ticking; the supervisor below is what
reattaches it, and a raise on every tick in between would take the drive down for
the one condition the supervisor exists to survive.

### 2. The advisory carries the frame's capture stamp, on the Jetson's clock

`AdvisoryMessage` has no frame id. Its only link back to the frame that produced
it is `t_capture_mono_ns`, so that field gets `tick.t_capture_mono` — the phone's
own capture stamp already converted into Jetson time by `PhoneClockAdapter` — and
not `time.monotonic()` at the moment of sending. Emit time is not capture time,
and a log where every advisory claims to be about a frame captured at the instant
it was sent cannot be joined to anything afterwards.

The stamp is provenance, not control: `AdvisoryHolder` on the phone expires on
**local arrival** and says why — "relating the two takes the timebase exchange,
and a display that goes blank because a clock estimate wandered would be a fault
invented by its own safety check". Nothing here changes that, and nothing here
may come to depend on the phone reading the stamp.

### 3. A rate command goes down when it changes, not every tick

`rate_cmd` is `RELIABLE` at depth 16. The tick loop runs at camera rate. Sending
a command per tick would put ~30/s into a 16-deep reliable queue whose oldest is
dropped on overflow — so the channel designed never to lose a record would be
losing records continuously, and the phone would spend the drive applying
identical commands.

So the loop sends when the command differs from the last one **sent**, plus a
heartbeat every `RATE_CMD_HEARTBEAT_S` so a phone that reconnected mid-drive is
not left running whatever it had. Comparison is on the decided content — rates,
trigger, query — and never on `t_capture_mono_ns`, which differs every tick and
would make the "unchanged" test always false. It is also **not** on the `shadow`
flag alone: a mode flip must send, because the flag is the whole difference
between a recorded decision and a gated one.

`SensingController` already has its own dwell and hold. This is a transport
concern layered on top of that, not a second policy: the controller decides, this
decides whether the wire has already been told.

### 4. Shadow is the default, and the supervisor rebinds rather than ending the run

The loop comes up in `SHADOW` — a drive that gates for real because nobody passed
a flag is the wrong failure — and `--live-rates` opts in. Task 30's `ModeHolder`
is constructed once and read once per tick.

The redial supervisor is the piece task 26 deferred. Today a phone that hangs up
ends the run: `_answer_pings`, `_read_here` and `_read_telemetry` all return for
good once `session.is_closed`, the backends go quiet, and `wait_for_phone` is
never called again. The supervisor watches for the session ending, tears the
backends down, and rebinds to the next phone that dials in, **without restarting
the process**. What it must not do is pretend the gap did not happen:

- the run record counts sessions and names each end reason (`_end_reason_of`
  already spells them the way the rest of the repo does);
- ticks during a gap are ticks with no camera, and must be visibly absent rather
  than silently repeated — a frozen last frame re-advised is the failure this
  project has already paid for once;
- the timebase estimate does **not** carry across a rebind. A new session is a new
  peer clock, and `OneWayEstimator`'s samples from the old one are not comparable.
  Reattaching without resetting would let the first ticks of the second session
  convert against the first session's offset and look perfectly healthy.

## The work

1. `PhoneLink.send_advisory` / `send_rate_command`, both no-ops returning False
   with no session, both counted.
2. `PhoneLink.rebind()` and a supervisor thread: detect session end, stop the
   backends and the three readers, reset the estimator, wait for the next dial-in,
   `_begin()` again. Counters for sessions seen and gap duration.
3. A `SensingLoop` (or equivalent) holding `SensingController`, `ModeHolder` and
   the last-sent command, with one `on_tick(tick, phone) -> Decision | None` entry
   point. Its inputs come from `tick` (margin, density bin, GPS) and from
   `phone.telemetry` / `phone.here`, aged against the tick's own clock.
4. `run_demo.py --phone`: construct the above, call it per tick, add `--live-rates`
   and `--rate-heartbeat-s`, and fold the mode record and send counters into the
   run summary.

## Tests

- An advisory sent from a tick carries **that tick's** capture stamp, not now.
- With no session, both senders return False and neither raises; the loop keeps
  ticking.
- An unchanged decision sends once; a changed decision sends again; a mode flip
  sends even when the rates are identical; the heartbeat fires with nothing
  changed.
- A session that ends and a second phone that dials in: the run continues, the
  record shows two sessions, and the estimator does not carry samples across.
- Ticks during the gap are absent from the log, not repeats of the last frame.
- Every command the loop emits round-trips through `decode_message(RATE_CMD, ...)`
  and every advisory through `decode_message(ADVISORY, ...)` — the wire is the
  arbiter, as in tasks 29 and 30.

## Experiment

A scripted drive over the real loopback transport with a phone-role peer: assert
advisories arrive and match their frames, rate commands are sent on change rather
than per tick (report the ratio), a forced session drop is survived, and no frame
is refused. Then the same over the network backend — which is task 32's job, so
task 31 stops at the loopback.

## Open, carried in

- `MAX_QUERY_RADIUS_M = 10 km` has never been checked against HERE v7's accepted
  `in=circle;r=` bound. Not closable here: the API must not be called.
- Task 27's parse has never met a real HERE body.
- Task 28's vector cannot be feed-informed without a simulator-side change.
- Four-stamp time-sync samples need the phone to carry `t4` on its next ping —
  before task 33, not here.
