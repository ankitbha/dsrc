# Task 29 — The sensing controller

## The short version

Three tasks of the phone reporting; this is where the Jetson decides. Four
independent rates — `camera_hz`, `gps_hz`, `imu_hz`, `here_hz` — plus the
per-modality settings that ride with them, chosen from four inputs and sent as a
`rate_cmd`.

Nothing in `deployment/jetson/` has ever constructed a `RateCommand`. Only the tests
and two scripts do, so the whole decision path is new.

**One of the four named inputs is invisible to the Jetson today.** `PhoneLink` reads
camera, GPS, control and HERE. It does not read `telemetry`, so `thermal_status`,
`thermal_headroom` and the `skin_temp_c` added for exactly this purpose reach the
Jetson and are dropped on the floor. Thermal backoff cannot be implemented until
that channel is ingested, and adding it is part of this task.

## Scope boundary

In: the four rates, the `here` query that goes down with `here_hz`, the trigger
vocabulary, the telemetry ingestion the thermal input needs, and the decision
record.

Out: whether the decision is *applied*. `RateCommand.shadow` exists and task 30 owns
it, so this task produces decisions and marks them; it does not gate. Also out: the
tick-loop wiring and the send itself (31), and per-stage timing (33).

## The four inputs

**1. The free always-on tier as a trigger proxy.** IMU and GPS run continuously and
cost almost nothing — two ~200-byte frames a second against a camera stream. So they
are the cheap detector: acceleration and speed changes say *something is happening*
before the camera has been asked to look. This is the input that makes the whole
scheme pay, because it lets the expensive modalities idle without going blind.

**2. Advisory bin-boundary proximity.** The policy emits *bins*
(`desired_speed_bin ∈ {slow, nominal, fast}`), so there is no continuous advisory
value to sit near a boundary. The computable meaning is the **policy's own margin**:
`ActorRuntime` publishes `head_probs`, a softmax per active head, so proximity is
`top1 - top2`. A small margin means a small input change flips the advice, which is
exactly when better inputs are worth paying for. Recommended, because the
alternatives — thresholding the decoded speed, or watching the bin flip — either
invent a continuum the policy does not have or react after the flip rather than
before it.

**3. Disagreement between sources.** Task 28 established the feed reaches no
observation field, so this is not a disagreement inside the vector. It is between the
feed's derived congestion (`ObservationResult.feed`) and what the camera sees locally
(`local_density_bin`, `local_mean_speed_bin`). When a feed says jammed and the camera
sees free-flowing road, one of them is wrong about where we are, and that is worth
spending samples to resolve.

**4. Thermal backoff.** `thermal_status` is the actionable signal; `skin_temp_c` is
the one that moves early — measured on the handset, status stayed `nominal` through a
5.4 °C rise. Backoff is the only input that *lowers* rates against the others' wishes,
so it wins ties by construction.

## Decisions taken

**A closed trigger vocabulary.** `trigger` is a free string on both sides today —
`requireString`, no membership check. Task 34 is "trigger attribution: which rule
fired, for which sensor, and why", and free text makes that a text-mining problem
over a drive's logs. A closed set is defined here and validated on the way out.

**Rates cannot express "off".** The spec constrains rates to `(0, 1000]`, so zero is
refused. The controller can idle a modality low but never disable it, and anything
wanting "off" needs a protocol change rather than a rate of zero. Named because the
obvious implementation of "stop the camera" is silently invalid.

**Hysteresis, and a dwell.** Every rate change costs the phone: re-registering
sensors, rebinding the camera. A controller that re-decides every tick would thrash
hardware and produce a drive whose rates are noise. So a rate moves only when the
evidence crosses a band and has held for a dwell — and the record says which rule
fired, so a drive can be read afterwards.

**The controller proposes; nothing here applies.** Consistent with the phone's own
rule that it reports and the Jetson decides, this task's output is a `RateCommand`
plus its reasoning. Task 30 decides whether it is gated for real.

## The work

1. Ingest `telemetry` in `PhoneLink`, beside the other channels, so thermal is
   visible at all. Its reader exits on a closed session like the others.
2. A `SensingController` taking the always-on tier, the policy margin, the feed and
   camera views, and telemetry; producing four rates, a `here` query, a trigger, and
   a per-rule record.
3. The trigger vocabulary and the rate bounds, as named constants with reasons.
4. Hysteresis and dwell.

## Tests

- Every produced rate is inside `(0, 1000]`, for every input combination — including
  the ones that "want" zero.
- Thermal backoff beats every other input that would raise a rate.
- A quiet drive settles to the idle rates and stays there: no thrash without evidence.
- A trigger fires the modality it names and not the others — four independent rates
  means camera evidence must not raise `imu_hz`.
- The trigger is always a member of the vocabulary.
- Margin: a confident policy does not raise rates; a near-tie does.
- Disagreement between feed and camera raises what resolves it.
- With no telemetry ever received, the controller still produces valid rates and says
  thermal was unknown rather than assuming nominal.

## Needs sign-off

- The policy-margin definition of bin-boundary proximity.
- The closed trigger vocabulary, since task 34's attribution is built on it.
- That thermal backoff outranks every other input, including a near-tie margin that
  would otherwise raise the camera.
