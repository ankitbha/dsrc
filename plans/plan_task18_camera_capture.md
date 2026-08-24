# Plan: Task 18 — Camera capture

> Camera capture at the commanded rate, JPEG encode, per-frame monotonic
> timestamps from `elapsedRealtimeNanos`.

## Short version

Turn the camera into a stream of JPEG frames carrying honest timestamps, produced
at whatever rate the Jetson asks for. Nothing is transmitted yet — task 19 is the
first task that forwards — so the deliverable is a frame source with a queue, a
rate gate, and counters, sitting behind an interface the transport will later drain.

Three things carry the risk, and all three are why this is not just "call CameraX":

- **The rate gate must not drift or accumulate credit.** A naive "has a period
  elapsed since the last kept frame" gate silently emits bursts after any stall,
  which on a thermally-throttled phone is exactly when it must not.
- **`ImageProxy` must be closed exactly once, on every path.** CameraX hands out a
  bounded pool; leak one and the pipeline stalls permanently after `maxImages`,
  with no error. Close it twice and it throws. Both are silent in a unit test.
- **Encoding must not happen on the capture thread.** JPEG compression at 720p is
  tens of milliseconds; doing it inline stalls the analyzer and drops frames the
  drop counter never sees, which is the failure mode task 17's report already
  argued against — a loss with no record is worse than a counted one.

The `camera` channel is `bulk` / `latest_wins` / depth 1, so the design is
deliberately lossy: one frame in hand, a newer one replaces it, and the
replacement is counted. That is the spec's choice, not a shortcut.

### Scope boundary

**In:** a `CameraSource` interface with a CameraX adapter and a fake; the rate gate;
frame ids; `elapsedRealtimeNanos` capture stamps; JPEG encode on a worker; a
depth-1 latest-wins queue with drop counting; wiring into `SensingService`'s
`onSensingUp`/`onSensingDown`; JVM tests for everything pure; instrumented tests
against the emulator's virtual camera.

**Out:** transmitting anything (task 19 brings the transport), reacting to a
`rate_cmd` off the wire (task 22 — here the rate is set locally), thermal backoff
(task 24), and any image content analysis, which is the Jetson's job.

### Open items

- **O1. `elapsedRealtimeNanos` versus the CameraX timestamp.** The task names
  `elapsedRealtimeNanos`, and that is what will be used. Worth recording that
  CameraX's `ImageInfo.timestamp` is the *sensor* timestamp and is the more accurate
  shutter time; on many devices its base is `uptimeNanos`, not `elapsedRealtime`, so
  the two are not interchangeable and mixing them would put two clocks in one field.
  Sticking to one clock matters more here than the accuracy difference, and the spec
  is explicit that `t_capture_mono_ns` shares a clock with `t_mono_ns`. If the
  shutter time is later wanted, it belongs in a *new* field, not this one.
- **O2. Resolution and JPEG quality are Jetson-owned settings with no wire field.**
  Per `specs/transport_protocol.md`, the Jetson owns them; `rate_cmd` cannot carry
  them yet. They come from the local config stand-in, same as task 17's O2.

---

## 1. Grounding

The `camera` message, from `specs/transport_golden_frames.json`:

```json
{"ch":"camera","format":"jpeg","frame_id":1841,"height":720,"n":4096,
 "quality":85,"seq":1,"t_capture_mono_ns":1000000000,
 "t_mono_ns":1100000000,"t_wall_ns":1755648000000000000,"width":1280}
```

So a frame owes: `frame_id`, `width`, `height`, `format`, `quality` (the one
nullable field), `t_capture_mono_ns`, and the JPEG in the payload. Channel policy
is `bulk` / `latest_wins` / depth 1.

`queueing latency = t_mono_ns - t_capture_mono_ns` is the one subtraction the
protocol sanctions without an offset estimate, which is only true if both come from
the same clock on the same device — hence O1.

The emulator advertises `android.hardware.camera` and `camera.any`, so the CameraX
adapter is exercisable on `dsrc_test` rather than only on the handset.

---

## 2. Decisions

Taken by recommendation under `plan_dsrc_rec`. **None is user sign-off.**

| # | decision | why | runner-up |
|---|---|---|---|
| D1 | `ImageAnalysis` use case, not `ImageCapture` | Gives a continuous stream with backpressure we control; `ImageCapture` is built around discrete shutter presses and its own queueing | `ImageCapture`; wrong shape and hides the drop accounting |
| D2 | `STRATEGY_KEEP_ONLY_LATEST` backpressure | Matches the channel's `latest_wins` exactly, so the camera and the transport agree about what loss means | keep-all; would build an unbounded queue upstream of a depth-1 channel |
| D3 | Rate gate as a pure function of (target Hz, last emitted stamp, now) with **no credit accumulation** | A gate that carries a deficit emits a burst after any stall — worst behaviour precisely when the phone is struggling | token bucket; useful for smoothing, wrong when the constraint is thermal |
| D4 | `elapsedRealtimeNanos` for `t_capture_mono_ns` | The task says so, and it must share a clock with `t_mono_ns` for the sanctioned subtraction to hold | `ImageInfo.timestamp`; more accurate shutter time, different clock base — see O1 |
| D5 | JPEG encode on a single-thread executor, depth-1 latest-wins input | Keeps compression off the analyzer thread; one thread because two would reorder frames | thread pool; reordering makes `frame_id` monotonicity a lie |
| D6 | `frame_id` counts frames the gate **accepted**, not frames the sensor produced | It is the id of a frame that exists; gaps then mean "dropped after acceptance", which is the number worth seeing | count all sensor frames; the id would jump for reasons nobody can act on |
| D7 | `ImageProxy` closed in a `finally` in the analyzer, before any encode | Encoding copies out of the proxy first, so holding it during compression pins a pool slot for tens of ms for no reason | close after encode; halves the effective pool |
| D8 | Frames buffered in a `FrameBuffer` the transport will later drain | Task 19 attaches the transport with no change here; until then the buffer is the observable output and its counters are the test surface | hold frames in the service; makes the counters untestable |

---

## 3. Files

```text
phone/app/src/main/kotlin/com/dsrc/phone/
  sensors/CapturedFrame.kt      # the record: ids, dims, quality, stamp, jpeg
  sensors/RateGate.kt           # pure: should this frame be kept?
  sensors/FrameBuffer.kt        # depth-1 latest-wins, drop counting
  sensors/CameraSource.kt       # interface + fake
  sensors/CameraXSource.kt      # the adapter (Android only)
  sensors/JpegEncoder.kt        # YUV_420_888 -> JPEG, off the capture thread
  config/SensingConfig.kt       # rate, resolution, quality (the O2 stand-in)
```

---

## 4. Steps

| # | step | done when |
|---|---|---|
| 1 | `RateGate` | pure tests: steady rate, stall then resume with no burst, rate change mid-stream, zero and absurd rates refused |
| 2 | `CapturedFrame` + `FrameBuffer` | depth-1 replacement counted; drain returns the newest; counters monotonic |
| 3 | `JpegEncoder` | encodes a synthetic YUV plane; refuses a malformed one rather than emitting a truncated JPEG |
| 4 | `CameraSource` + fake | the fake drives the whole chain at a commanded rate on the JVM |
| 5 | `SensingConfig` | rate/resolution/quality read from one place, validated on the way in |
| 6 | `CameraXSource` | frames arrive on the emulator; `ImageProxy` closed exactly once |
| 7 | Wire into `SensingService` | `onSensingUp` starts capture, `onSensingDown` stops it; a teardown throw is already handled |
| 8 | Instrumented | real camera on `dsrc_test`: frames flow, the rate is honoured, no pool exhaustion over a sustained run |

---

## 5. Tests

**Pure**
- `RateGate`: at 10 Hz over 100 simulated frames the accepted count is within one of
  the expected; after a 5 s stall exactly one frame is accepted immediately and the
  next no sooner than one period later (the no-burst property); a rate change takes
  effect on the next decision; `0` and `> 1000` Hz are refused, matching the wire's
  `(0, 1000]`.
- `FrameBuffer`: replacement counts a drop; drain empties; drain on empty returns
  null rather than blocking; counters never decrease; a drop is counted exactly once.
- `JpegEncoder`: a known synthetic plane produces a decodable JPEG of the expected
  dimensions; a plane whose stride disagrees with its width is refused.
- `SensingConfig`: out-of-range rate, zero resolution and quality outside 1..100 are
  each refused with the reason named.

**Sanity**
- The full chain on the JVM with the fake: commanded 5 Hz over a simulated minute
  yields ~300 accepted frames, every `frame_id` consecutive, and dropped + delivered
  equals accepted.
- Every accepted frame's `t_capture_mono_ns` is non-decreasing.
- Stopping mid-stream leaves no frame half-encoded and the buffer drained.

**Instrumented**
- CameraX delivers frames on the emulator and the commanded rate is honoured within
  tolerance over 10 s.
- A sustained 30 s run does not stall — the `ImageProxy` leak would show as frames
  ceasing after the pool size, so this is the test that would catch D7 being wrong.
- `onSensingDown` releases the camera: a second start succeeds.

---

## 6. Experiments

1. Accepted-vs-commanded rate on the emulator at 1, 5, 15 Hz over 10 s each.
2. JPEG size and encode time distribution at 720p, quality 85.
3. Drop counts under a deliberately slow drain, to show latest-wins accounting.
4. A 30 s sustained run: frames delivered, drops, and whether the rate holds.

**Not measured:** anything about a real phone camera, real thermal behaviour, or
image quality. The emulator's virtual camera produces a synthetic scene, so encode
sizes are not representative of a road.

---

## 7. Risks

- **The emulator camera is synthetic, and narrower than it looks.** It proves the
  plumbing, the rate gate and the pool discipline. It says nothing about real exposure,
  real frame intervals or real JPEG sizes — and, more sharply, **it cannot exercise the
  packer at all**: the virtual camera reports `rowStride == width` and
  `pixelStride == 1` planar, so both stride padding and the semi-planar chroma path are
  inert on it. That is exactly where `YuvPacker`'s bugs live, which is why they are
  pinned by pure tests rather than by the device.
- **`ResolutionSelector` needs the aspect ratio stated, or the resolution is silently
  ignored.** It defaults to 4:3, and a 16:9 request is excluded from the candidate set
  *before* any resolution rule is consulted. Measured on the emulator, which lists
  1280x720 among its supported sizes: no selector gave 640x480, prefer-lower also gave
  640x480, prefer-higher gave 1856x1392 — and with the ratio supplied, all three rules
  give exactly 1280x720. So a size that looked unsupported was being filtered by ratio,
  and either fallback direction looked like a device limitation. The selector now
  derives the ratio from the configured size.
- **The configured resolution is still a request, not a guarantee.** A device need not
  offer the requested size, and the fallback prefers lower — going under costs detail,
  going over costs encode time and payload on a link that carries every frame.
  `CapturedFrame` reports the frame's own dimensions rather than the config's, so what
  goes on the wire is always what arrived.
- **`ImageProxy` accounting is silent when wrong.** A leak stalls the stream with no
  exception. `KEEP_ONLY_LATEST` makes the pool effectively depth-one, so a single leak
  stops the stream on the *first* frame and four of the five camera tests fail — a
  wider net than the sustained run alone.
- **O1 means `t_capture_mono_ns` is an arrival-ish stamp, not the shutter.** It is
  taken in the analyzer callback, so it carries the pipeline's own latency. That is a
  known bias on the same clock as everything else, and the more accurate alternative is
  a different clock.
  **Its magnitude is unquantified, and cannot be measured from these two stamps.**
  Differencing the arrival and sensor stamps returns offset *plus* latency
  inseparably — measured negative, which is impossible for two stamps sharing a base
  and is itself the evidence that they do not. Only the offset-free part is available:
  the *spread* of that difference, 7.0 ms peak-to-peak on the emulator, since a constant
  offset cancels in the variation. The absolute bias needs the transport's enqueue stamp
  as a second operand, which arrives with task 19.

---

## An unpinnable guard, kept and not claimed

`react()` reaches `STOPPED_ERROR` and `STOPPED_PERMISSION_REVOKED` through `release()`
alone, so a throw part-way through `onSensingUp` left every allocation above it live until
`onDestroy` — and cleanup then depended on `stopSelf(lastStartId)` winning, which the
startId overload exists precisely to *lose* when a start is queued behind it. A retried
Start re-entered and overwrote all seven fields, orphaning the first set: a link thread
reconnecting forever, a sender polling forever, a GNSS callback still registered.

`onSensingUp` now releases anything already published before allocating, and releases what
it allocated if the come-up throws. **Neither change is pinned by a test, and that is
stated rather than papered over.** Removing either leaves the instrumented suite green,
because `onDestroy` cleans up by another route. Observing the difference needs the service
to survive its own `stopSelf`, which needs a start queued behind it — a window of about two
milliseconds that neither the validator nor I could force.

The two tests added alongside are named for what they do verify — a failed start leaves
nothing running, and a restart does not accumulate threads — not for the guard.

## D2 (`KEEP_ONLY_LATEST`) is unpinned, and why

Swapping `STRATEGY_KEEP_ONLY_LATEST` for `STRATEGY_BLOCK_PRODUCER` survives the suite. The
two are hard to tell apart from outside: under a slow analyser both yield a reduced frame
rate at the pipeline, and the difference — whether the camera skipped frames or the producer
was throttled — lives in statistics CameraX does not expose. Distinguishing them would need
the camera's own dropped-frame counters.

It stays as `KEEP_ONLY_LATEST` on the plan's reasoning (a stale frame is worth less than a
fresh one, and blocking the producer couples the camera's rate to the encoder's), recorded
as a choice the tests do not defend rather than one they do.

## 8. Needs sign-off

1. **O1** — whether `t_capture_mono_ns` should stay `elapsedRealtimeNanos` at the
   analyzer, or become the sensor timestamp with a documented clock change.
2. **O2** — resolution and quality defaults, until the downlink can carry them.
