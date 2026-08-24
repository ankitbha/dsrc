# DSRC Project Task List

The active work is a live phone-plus-Jetson advisory system: the phone carries
four sensors and the driver display, the Jetson does all compute, and a sampling
controller on the Jetson adjusts each sensor's rate in real time.

Tasks are sequenced so that **nothing requiring the phone and Jetson to be in the
same room happens until late**. The transport is built with two backends — a
network backend over Tailscale for development, and a USB backend for
deployment — so the whole system can be developed with the phone in hand and the
Jetson wherever it lives. Tasks needing physical colocation are marked
**[COLOCATED]**.

## Architecture

```text
  phone                    transport                  jetson
  -----                    ---------                  ------
  camera --+          network (dev) / USB (car)   +-- perception (TensorRT)
  GPS -----+-- sensor frames ------------------>  +-- fusion (camera + HERE)
  IMU -----+   (each at its own commanded rate)   +-- policy inference
  HERE ----+                                      +-- advisory decode
                                                  |
  display <---- advisory --------------------------+
  rate ctl <--- per-sensor rate commands ----------+-- sampling controller
```

The phone stays dumb: it captures or queries at the commanded rate and forwards
raw data. All interpretation, association, and control live on the Jetson.
The cloud supplies observability through HERE; it is not in the control loop.

## Non-goals

Both devices are powered from the car, so **energy and battery life are not
metrics**. The binding costs for the sampling controller are HERE API quota,
thermal headroom, and Jetson compute. Thermal is a real constraint and the phone
must not overheat.

Out of scope entirely: any traffic-flow effect (one vehicle, advisory-only),
human compliance with the advisory, and anything fleet-level. Those claims come
from simulation or not at all.

## A. Status: what already exists

**Simulator foundation** — maintain, do not rebuild: simulator integration,
project interfaces, the topology ladder, demand and vehicle lifecycle, human
behavior profiles, metrics and logging, local sensing, the common
safety/etiquette/physical-control layer, the baseline ladder, model-free RL.

**Jetson prototype** — reuse rather than rewrite:

- `perception/detector.py`, `tracker.py`, `distance.py`, `observation_builder.py`
- `policy/sim_contract.py`, `actor_runtime.py`, `advisory.py`
- `pipeline.py` tick loop and rolling latency tracking
- `logio/` metadata logging, `eval_run.py` gated reporting

`camera_stream.py` and `gps_reader.py` already abstract their sources and need a
phone backend, not a rewrite. The GPS `dialout` blocker is resolved: the u-blox
enumerates and `/dev/ttyACM0` is readable by `edge`. The Jetson is reachable over
Tailscale at `ssh jetson`, so its runtime can be developed remotely.

## B. Environments

1. ~~Python environment on the Mac with `highway_env`, torch, and the project dependencies.~~ **DONE** — Python 3.12.14 venv at `.venv/`; numpy 2.5.2, torch 2.13.0, gymnasium 1.3.0, highway_env 1.12.1; 138 tests pass. Section C is unblocked.
2. ~~Android toolchain on the Mac.~~ **DONE** — command-line SDK only, no Studio (per plan). Temurin 17.0.20; cmdline-tools; platform-tools, platforms;android-35, build-tools;35.0.0, emulator, system-images;android-31;google_apis;arm64-v8a; AVD `dsrc_test` (Android 12, arm64); Gradle 8.9 wrapper on AGP 8.7.3; `adb` owned solely by the SDK. Plan: `scratchpad/plan_task_02_android_toolchain.md`.
3. ~~`adb` on the Jetson.~~ **DONE** — adb 1.0.41 (platform-tools 28.0.2-debian) at `/usr/bin/adb`, plus `51-android.rules`; 8 packages added, nothing else changed. Carries the `adb forward` TCP tunnel that is the in-car transport (D1). RSA authorization still deferred to task 41. Plan: `scratchpad/plan_task_03_jetson_adb.md`.
4. ~~Tailscale on the phone.~~ **DONE** — `moto-g-power` `100.75.142.126` under `bhardwaj.ankit275@` (same account matters: the Jetson is a *shared* node from `taila2630c`, and sharing is per-account). Phone→Jetson TCP verified with a real payload in both directions; path upgraded DERP→direct, 55 ms. Plan: `scratchpad/plan_task_04_phone_tailscale.md`.

## C. Simulation study — BLOCKED IN SUBSTANCE by the task 8 result

**Tasks 10 and 11 cannot produce meaningful numbers until the simulator can show
a control effect.** The sufficiency study degrades an observation and measures
what the controller loses; the sampling-policy evaluation credits a flow-level
benefit. Both require an effect to degrade or to credit, and task 8 established
there is none — +0.12 to +0.14 m/s where the measurement is cleanest.

**Project-level risk, recorded here deliberately:** the paper's only outcome
claim comes from simulation, and the simulator currently cannot produce one. One
instrumented vehicle can never demonstrate throughput or delay. Repairing the
simulator is therefore a prerequisite for the paper having a result at all, not a
tidying task. It is not scheduled — sections D through G are unblocked and come
first — but it must not be filed as done.

5. ~~Per-field variance audit to identify inert inputs before any ablation.~~
   **DONE** — `src/analysis/observation_audit.py` + `scripts/audit_observation_fields.py`,
   41 tests. Ran 162 conditions (6 topologies × 3 controllers × AV penetration
   {0.05, 0.10, 0.20} × 3 seeds, 120 steps), 10,569 samples.
   Plan: `scratchpad/plan_task_05_variance_audit.md`. Artifacts:
   `outputs/validation/observation_audit/`.

   **Findings that bind sections D–F:**

   - **Only 2 of 39 encoded fields are uninformative everywhere.** `is_active`
     is structurally constant at 1.0 — the observation map contains only active
     AVs, so the flag can never be false. `distance_to_next_merge` is hardcoded
     to `0.0` in `src/sensing/local.py`. Neither should be sensed, and neither
     may be ablated as if the result meant anything.
   - **8 fields are penetration-gated:** dead at 5% AV penetration, informative
     at 20% — `nearby_av_count`, `active_av_count_local`, `nearby_av_density`,
     `nearby_av_lane_distribution.{1,2}`, `downstream_congestion_estimate` and
     its `cooperation.*` twin, and `target_lane_rear_required_decel`. No field
     moves the other way. The local-aggregate cooperation block therefore only
     earns its sensing cost above roughly 10% penetration, which the sampling
     controller (task 29) should treat as a rate-allocation input rather than
     sensing unconditionally.
   - **Inertness is strongly topology-gated:** ring 18/39 constant,
     `straight_single_lane` 14/39, `merge` 8/39, `straight_multilane` 7/39,
     `inverted_tree` 4/39, `inverted_tree_bottleneck` 3/39. Any ablation must be
     read per topology, never pooled.
   - **Coverage caveat:** 60 of 162 conditions produced zero samples — 54 are
     `no_av` (correct by design) and 6 are non-ring topologies at 5% penetration
     where no AV spawns at all. The effective matrix is 102 conditions.
   - **`controlled_vehicles` is inert outside ring.** `HighwayTopologyEnv`
     clears `agent_ids` and defers to the demand spawner whenever continuous
     demand is active, i.e. on every topology except ring. AV population must be
     set through `demand.av_penetration`. This bit the first run and is recorded
     as Amendment 2 in the plan.
6. Sufficiency harness: evaluate a fixed policy under configurable observation
   degradation (field ablation, added lag, added noise, forced fallbacks).
7. Baseline sweep with the current sensing defaults, to establish the reference
   the degraded conditions are measured against.
8. ~~Exercise the topology ladder beyond ring so the study is not
   single-topology.~~ **DONE — superseded by a simulator health check**, which
   absorbed and extended it. `src/analysis/simulator_health.py` +
   `scripts/check_simulator_health.py`, 213 tests. Ran 72 cells / 648 runs
   (6 topologies × 4 demands × 3 penetrations × 3 controllers × 3 seeds, 120
   steps) in 18m37s. Plan: `scratchpad/plan_task_08_simulator_health.md`.
   Artifacts: `outputs/validation/simulator_health/`.

   **Result: 0 of 72 cells pass all four criteria. There is no usable operating
   point, and the simulator cannot currently support a flow-level claim.**

   | criterion | fails in |
   |---|---|
   | `baselines_separate` | 68/72 |
   | `episodes_complete` | 44/72 |
   | `throughput_holds` | 31/72 |
   | `congestion_reachable` | 27/72 |

   - **The controllers have no measurable effect, and crashes are not the
     reason.** Cells that complete separate in 7% of cases, cells that crash in
     5% — the hypothesis that truncation was hiding the effect is refuted. In the
     four cells that both congest *and* complete, measured on 2–3 congested
     shared seeds, the best controller moves mean speed by **+0.12 to +0.14 m/s**
     against a 1.0 m/s threshold. That is the cleanest measurement the grid
     offers and it is near-zero.
   - **Congestion is reachable but topology-structured.** `inverted_tree`,
     `inverted_tree_bottleneck` and `ring` congest in 12/12 cells,
     `straight_single_lane` in 9/12, and **`merge` and `straight_multilane` in
     0/12** — including `merge`/high, which congests at a single penetration but
     fails the per-seed rule.
   - **Only 24 of 216 (cell, controller) pairs were ever measured with a
     congested shared seed.** Most separation failures are therefore not
     evidence that a controller cannot help; the controller was evaluated where
     there was nothing to control. Without `congested_shared_seeds` the report
     would have read as 68 controller failures.
   - **40% of runs crash** (260 of 648 never reach their configured duration).
   - **The penetration axis is substantially noise.** The mechanism is correct —
     pooled realised spawn fraction is 0.200 against a nominal 0.20 — but only
     60–78 vehicles spawn per run, so the standard error is ±0.05 and single-seed
     realised penetration swings between 0.08 and 0.43 at a nominal 0.20. Per-cell
     verdicts rest on single seeds, so nominal 0.05 and 0.10 cells can realise the
     same fraction. Concurrent AV count, which is what actually acts as an
     actuator, peaks at 5–6 out of ~40 active vehicles.
   - **243 of the 648 runs are duplicates.** `burst` is bit-identical to `medium`
     on all six topologies (162 runs), and ring disables demand so its four
     demand levels collapse to one (81 runs). Config defects, out of scope here.

   **Next diagnostic when this is picked up** (not run): raise demand *and*
   episode length *and* penetration together, so concurrent AV count reaches
   double digits. Raising nominal penetration alone would leave the realised
   concurrent count in single figures for the same small-sample reason. That run
   distinguishes "too few actuators" from "these controllers do not control this
   simulator" — two diagnoses leading to completely different work.
9. Sensing model calibrated from drive measurements. *Waits on section G data;
   the harness is built now and the parameters filled in later.*
10. Sufficiency study proper: derive the sensing requirement specification.
11. Sampling policy evaluated in simulation for the flow-level benefit one
    vehicle cannot demonstrate. *Waits on the policy from section F.*

## D. Transport

Two backends behind one interface. Build the network backend first.

**Binding constraint — phone initiates, Jetson listens.** The Jetson cannot
originate ordinary IP traffic to tailnet peers: its Tegra kernel has
`CONFIG_NF_CONNTRACK_MARK` unset, so the conntrack struct has no `mark` field and
Tailscale's connmark rules cannot install. The feature is compiled out, not a
missing module, so no package or out-of-tree build fixes it short of rebuilding
NVIDIA's kernel. Inbound works, and both directions were verified on a
phone-initiated TCP connection: sensor data up, advisory and rate commands back
down the same socket. Design the transport so the phone always opens the
connection. This affects the network backend only; the in-car `adb forward` path
tunnels over USB and is unaffected.

12. ~~Transport interface: framed bidirectional messaging, backend-agnostic.~~
    **DONE** — `deployment/jetson/transport/`, stdlib only, 291 tests. Plan:
    `scratchpad/plan_task_12_transport_interface.md`; wire contract:
    `specs/transport_protocol.md`; frozen encodings:
    `specs/transport_golden_frames.json`.

    One socket, channels multiplexed by a JSON header in front of an opaque
    payload; framing, priority, overflow, sessions and counters all sit above a
    three-method `ByteConnection` seam, so tasks 13 and 40 each implement a byte
    stream rather than a transport. The loopback backend ships, so the pipeline
    runs with no phone and no Jetson.

    **Measured.** 291/291 tests pass identically on the Mac (3.12) and on the
    Jetson (aarch64, 3.10). Loopback at the planned sensor rates: every channel
    at its commanded rate, zero drops, zero gaps, transport latency p50 0.02–0.08 ms.
    Over a real Tailscale socket, Mac standing in for the phone, 60 s at
    **428 KB/s (3.4 Mbps)**: camera 10.02 Hz achieved against 10.0 commanded,
    zero outbound drops on any channel, zero sequence gaps, round trip p50
    **11.3 ms** / p95 21.0 ms / p99 70.2 ms, handshake 12.9 ms. The Jetson's own
    per-channel account closes exactly on every channel. All four failure modes
    provoked over the real link and independently confirmed in the listener's
    record: version mismatch refused, `displaced`, `framing_error`, and
    `stalled` at 5.0 s against a 5.0 s timeout — with the listener surviving
    each and serving the next connection.

    **Overflow only appears once the consumer is slower than the offered load.**
    Throttled to 2 Hz against a 10 Hz camera: camera dropped 241 of 300 keeping
    only the newest, IMU 1186 of 1501, GPS 28 of 150, while `advisory` and
    `rate_cmd` dropped nothing and held sub-0.1 ms latency — the priority and
    per-channel policy decisions working as specified.

    **Validation: four rounds, 30 findings, 26 of them defects in the previous
    round's fixes.** Worth reading before task 13 touches this code:
    a caller message carrying a reserved key was destroyed with no counter and
    no gap; one silent peer could wedge the listener for a whole drive; the
    stall timer measured completed frames, so a slow link was killed and would
    reconnect forever; a delivered frame was reported as a loss; a thread dying
    outside its expected exception types left a session claiming to be healthy
    while transmitting nothing; and `delivered` counted arrivals rather than
    collections, so it was always equal to `received` and double-counted
    anything displaced — that last one found by the experiment, not by
    validation, because no test drove inbound overflow and checked the account
    at the same time.

    **Three assumptions only a real socket can break**, documented in
    `connection.py` and untested until task 13: `recv_exact` must raise rather
    than return short or empty (returning `b""` at EOF spun the reader at
    millions of calls per second before it was guarded); `close()` must unblock
    a read already in progress from another thread, or the handshake timeout
    leaks a thread per attempt (counted in `handshake_workers_leaked`, not
    prevented); and a write failure must surface as `OSError`.
13. ~~Network backend over Tailscale, for development with devices apart.~~
    **DONE** — `transport/tcp.py` and `transport/client.py`, plus
    `scripts/run_transport_listener.py` and `scripts/run_transport_client.py`
    which replace the scratchpad tools task 12's experiment ran from. Plan:
    `scratchpad/plan_task_13_network_backend.md`. 431 transport tests.

    Both ends: the Jetson accepts, the phone dials, and a `SessionClient`
    composes dial + handshake + `Session` with reconnection that never gives up.
    Backoff 0.25 s doubling to a 5 s cap with ±25% jitter; the schedule resets
    only after a session lasting `max(cap, stall × 2)`, derived so it cannot
    coincide with the stall timeout — at the shipped 5 s it would have, and the
    escalation would then never have engaged for the commonest failure in a car.

    **Both seams now have conformance suites**, run against every backend, so
    task 40's USB work inherits them: `ByteConnection` (13 checks × 2) and
    `Acceptor` (9 × 2, added after the two implementations disagreed about what
    `accept(timeout=0)` means). An accepted connection is also run through the
    `ByteConnection` checks, because the two suites were disjoint and a USB
    acceptor handing back a non-compliant connection would have passed
    everything.

    **Measured on the Jetson** (Linux 5.15.148-tegra aarch64, Python 3.10.12):
    **431/431 pass**, identical to the Mac. That is the point of running it
    there — two requirements were only ever verified as *call order* on macOS,
    which releases a blocked `recv` on `close()` alone. On Linux:

    ```text
    blocking recv, close() only        never released (still blocked at 6 s)
    blocking recv, shutdown()+close()  released in 0.001 s
    ```

    So `shutdown()` before `close()` is essential there, not belt-and-braces:
    without it a session shutdown leaves its reader blocked forever. The
    platforms disagree the *other* way on `accept`: macOS releases a blocking
    accept but not a timed one, which ran its caller's full 5 s; Linux releases
    both at once. The internal accept poll is what macOS needs and Linux does
    not.

    **Over the real Tailscale link**, Mac standing in for the phone, 60 s at
    **428 KB/s (3.4 Mbps)**: every channel at its commanded rate, **zero drops,
    zero sequence gaps**, round trip p50 27.8 ms / p95 71.7 ms / p99 280 ms,
    and the client process flat at 27 fds and 11 threads sampled through the run
    — the two per-attempt leaks validation found would have shown as a slope.
    All four failure modes provoked and confirmed in the listener's own record:
    version mismatch refused, `displaced`, `framing_error`, and `stalled` at
    5.0 s. A **genuine half-open** — the client process frozen with SIGSTOP, so
    the socket stays open with no FIN, no RST and no application data — was
    reaped as `stalled`. That is the case the timer exists for and no unit test
    can produce.

    **The handshake-timeout question is settled: 5.0 s has roughly 50× margin.**
    Client-side handshake round trip measured 21.5–101.9 ms across runs. Note
    the *listener's* `handshake_round_trip_ns` is **not** the link round trip —
    both sides stamp send-then-read, but the dialler's read waits a full round
    trip while the listener's hello arrives after the client's is already
    queued, so it read 0.3 ms against the client's 101.9 ms on the same session.
    Renamed in the report accordingly.

    **Validation: four rounds, 32 findings**, and the transferable part is that
    the fix was right every time while *what trailed it moved outward each
    round* — round 1 the evidence trailed the code, round 2 the observability
    trailed the evidence, round 3 the consumers trailed the library, round 4 the
    report trailed the question it was built to answer. Worth reading before
    task 40: one silent peer could wedge the accept loop for a whole drive; a
    retryable accept error killed the listener permanently while claiming it was
    closed; a retry with no pause spun at 2.2 M calls/s; a nullable field
    crashed the run report in exactly the interesting case; and a salvage I
    added in round 3 created phantom sessions that inflated `connected` and
    `reconnects`, so I reverted it.

    **The seam question is settled, the other way.** I raised `ByteConnection` as
    one property short — twice this task wanted to ask a connection "are you
    still usable". Checked: nothing consumed the `is_closed` that TcpConnection
    already had, and both problems were already solved without it. So it was
    **removed** rather than added: a liveness flag is stale the moment it is
    read, and check-then-act on one is the exact shape of this task's two worst
    races. The protocol stays at four members, `connection.py` says why there is
    deliberately no liveness query, and the contract suite pins both that every
    member is present and that no undeclared surface appears — so a backend's
    extras are a deliberate act. Task 40 implements four, not five.
14. ~~Wire protocol: sensor messages upstream, advisory and rate commands
    downstream, each carrying its own timestamps.~~
    **DONE** — `transport/messages.py`, stdlib only, 639 transport tests (157 on
    this layer). Plan: `scratchpad/plan_task_14_wire_messages.md`; the contract
    Kotlin implements is the Messages section of `specs/transport_protocol.md`.
    The two measurement harnesses ship as `scripts/run_message_exercise.py`
    (loopback, header and validation audit) and `scripts/run_message_link.py`
    (both roles over a real socket), so every number below can be re-run.

    Seven types over the eight channels, and **the channel is the discriminator**
    — there is no `kind` field to consult, so a message cannot claim to be
    something its channel is not. Sensor fields ride the JSON header and only
    opaque bytes ride the payload, which is what lets a camera frame and an IMU
    sample share one codec.

    **Unavailable is `null`, present.** Never absent, never a sentinel: absent
    cannot be told from a sender that forgot the field, and a sentinel is a real
    number that a consumer will average. NaN cannot cross at all — the encoder
    refuses non-finite rather than emitting invalid JSON — so a missing field is
    the *only* way to say "no value", and the observation builder sees one shape.

    **The three nested objects are additive and `action` is strict.** Rates,
    achieved rates and drop counts must carry every known key and *ignore*
    unknown ones, because the sensor set will grow and refusing an unknown key
    breaks a rolling deploy in both directions at once. The four action heads are
    a closed set from `specs/action_schema.md`, so an extra head is refused.

    **A malformed message is not a malformed stream.** Fifteen refusal
    conditions across a nine-reason closed vocabulary, each with its own row in
    the spec so the Kotlin side reads off which to emit; the message is dropped
    and counted per channel *and per reason*, and the session stays open. That
    differs from a framing error, which ends the session, and the difference is
    recoverability: framing succeeding proves the byte stream is still aligned,
    so one bad record costs one record.

    **The same table binds the sender.** A receiver rule alone leaves a sender
    free to emit garbage and learn about it as someone else's drop counter, so
    `send` refuses anything its own decoder would reject. A zero in `rates` is
    the case that shows why: it is read as a period, so the field that should
    say "10 Hz" says "never", 12 ms away from the code that could have caught
    it. It raises `InvalidMessage`, deliberately **not** a `MessageError` — that
    type means the peer sent something bad and its whole idiom is
    drop-and-count, so a consumer wrapping its own sends in it would silently
    swallow its own bug. The two counters stay apart for the same reason.

    **Measured on the Jetson** (aarch64, Python 3.10.12): **696 pass, 1 skip**
    (the skip needs the sim repo), including the golden-vector regeneration
    check — the cross-language contract is now proved byte-identical on a second
    architecture and a second Python, which is the entire reason those vectors
    are frozen.

    Header budget, the one hard limit this layer can breach: the widest encoded
    header per channel is **110–393 B against the 8192 B cap**, so 1.3–4.8% of
    it, advisory being the widest. Per message, encode costs 10–18 µs at the
    median and the send-side validation adds **6–19 µs** — it roughly doubles
    per-message CPU, and at the planned rates that whole guarantee costs under
    0.1% of one core. An empty-queue `recv`, the call a control loop makes most
    often, costs **4 µs** at the median and 22 µs at the worst of 3000.

    **240 s over loopback at the planned sensor rates**: every channel on
    cadence (camera 10.01/10.0, IMU 50.02/50.0), 15,971 messages up and 2,872
    down, **zero decode errors, zero invalid sends, and the account closes with
    no gaps**. Capture-to-read p50 1.0–1.6 ms, p99 1.9–2.4 ms, max 4.6 ms, and
    **zero negative samples in 18,843** — reported as a distribution because a
    sign check would have passed on a broken clock.

    **180 s with every 20th message deliberately corrupted**, rotated through
    three kinds so the breakdown has to distinguish causes rather than just
    count: **598 injected, 598 counted, and every per-reason bucket matches
    exactly** (camera 90 → 60 `wrong_type` + 30 `missing_field`, IMU 450 → 300 +
    150, and so on). Both corruptions of a *type* — a field and the capture
    stamp — correctly land in one bucket. Advisory and rate_cmd kept flowing
    with zero errors throughout, which is the point of dropping a record rather
    than a session.

    **Typed messages both ways over the real Tailscale link**, Mac standing in
    for the phone, 170 s: camera 1699 frames at 9.99 Hz, IMU 8496 at 49.98 Hz,
    **zero decode errors and zero invalid sends on either side**, and the
    Jetson's own per-channel account closes against what the phone sent. Round
    trip p50 **12.2 ms** / p95 21.2 ms / p99 134 ms / max 333 ms, measured on the
    phone's clock alone — the Jetson answers each frame with a typed advisory
    carrying the frame id back, so nothing is subtracted across two machines.
    Relating those clocks is task 15's job and the protocol forbids doing it by
    hand.

    **One anomaly, unresolved and recorded as such.** The first link run offered
    ~40% of its commanded cadence on every channel at once. Nothing was lost —
    the Jetson received every message sent — so it was the sender falling behind,
    not the transport. It did not reproduce in three later runs at the same and
    shorter durations, all of which held full cadence with zero iterations behind
    schedule. It cannot be attributed now because nothing recorded which path
    the link took, and a DERP-relayed link and a LAN link are different
    experiments; the harness now records that, so task 15 will not have this
    hole.

    **Validation: three rounds, 28 findings.** The recv budget was the one that
    mattered: passing the caller's original timeout on every skipped record made
    a stream of malformed messages block for an unbounded multiple of what was
    asked (**6.9x measured** on a 50 Hz channel with one broken field, the shape
    a single bad phone build produces). The fix regressed into something worse —
    checking the deadline before the queue, so the default `timeout=0.0`, the
    poll idiom a control loop uses, returned nothing while messages sat waiting.
    That survived because all nine call sites passed an explicit timeout, so the
    default was never once exercised; the test now enumerates the whole argument
    domain.

    Round 3 closed on a different lesson: **a test that pins nothing looks
    exactly like a test that passes.** The injected-clock test used a *frozen*
    clock, which is indistinguishable from a real one because neither expires a
    budget, so it passed with the injection ignored entirely. And the refusal
    table had no test at all — both of its numeric bounds could be silently
    halved, which for a cross-language contract means the other implementation
    reads the wrong number and its messages are dropped here. Every round-3 fix
    was then mutation-tested against the test that names it (15/15), which found
    that the first version of the new spec-bound assertion was satisfied by a
    second copy of the number elsewhere in the document.
15. ~~Shared timebase with clock-offset estimation and drift tracking, so
    phone-side and Jetson-side events are comparable.~~
    **DONE** — `transport/timebase.py`, stdlib only, **829 tests on the Jetson**.
    Plan: `scratchpad/plan_task_15_shared_timebase.md`; contract: the Shared
    Timebase section of `specs/transport_protocol.md`; harness:
    `scripts/run_timebase_probe.py`.

    `clock.py` forbids comparing a phone monotonic value with a Jetson one, and
    tasks 13 and 14 both reported round trips on one clock only to honour that.
    This is the sanctioned way across, and the whole discipline is that **no
    conversion returns a bare number** — a converted instant carries its bound
    and the id of the estimate that produced it, so a cross-device timestamp
    cannot be mistaken for a same-device one. Below the gate it **raises** rather
    than answering with a widened bound.

    **One typed message on `control`, not two.** The channel is the
    discriminator everywhere else, so a ping type and a pong type would need a
    `kind` field — which the spec refuses. The null convention carries it
    instead: `t_peer_recv_mono_ns` null means ping, set means pong, and since the
    phone always initiates the receiver's role settles which. `control` had no
    typed message before this; it is why `no_typed_message` exists.

    **Three things implementation found that planning had wrong.** The pong must
    echo the ping's wire stamp, because an initiator cannot read its own — the
    writer applies it after the caller lets go — so without the echo the only t1
    available carries the queueing delay, fixing one side of a symmetric
    calculation and not the other. The skew fit must run over min-filtered
    buckets: fitting raw samples put the whole delay tail in the residuals and a
    planted **+20 ppm came back as −1.55 ppm**. And half the min round trip is a
    *guaranteed* bound, not the optimistic one the plan feared — a sample's error
    is `|up−down|/2` against a round trip of `up+down`, so it cannot escape
    `rtt/2`.

    **Measured on the Jetson** (aarch64, Python 3.10): 829 pass, 1 skip. Loopback
    null case, where one machine means the truth is zero: offset spread **12 µs
    over 145 s** and a fitted slope of −0.05 ppm, so the estimator invents no
    structure where there is none.

    **Over the real link, 330 s**: 357 exchanges, every one matched, **zero
    refused, and all nine outcome counters close against pings sent** without
    subtracting anything. `rtt_min` p50 **14.8 ms**, bound p50 **8.0 ms** (min
    6.96, max 9.65), offset spread 2.7 ms across the run. Under the full sensor
    load — 10 Hz camera at 40 KB, 50 Hz IMU — 218 exchanges with every channel on
    cadence (IMU 49.9/50.0, camera 9.99/10.0).

    **The premise the whole guarantee rests on is now measured, and the
    instrument I designed for it was invalid.** `ASSUMED_SKEW_PPM = 50` is the
    only bound on the true relative skew of the two monotonic clocks. The plan
    proposed estimating the offset twice, once on each pair of clocks, and
    differencing the slopes. That cancels the quantity of interest: with each
    device's own wall-versus-monotonic slew written `s`, the mono-pair slope is
    `σ` and the wall-pair slope is `σ + (s_remote − s_local)`, so the difference
    is `s_local − s_remote` and **σ is gone**. Confirmed rather than argued: it
    reported **+12.06 ppm** against an independently measured slew difference of
    **+12.00 ppm**.

    What works needs no network. Each device's own `wall − monotonic` slew is
    exact, needs no delay model, and — since both wall clocks are NTP-locked to
    UTC — states how far that device's monotonic clock runs from UTC:

    ```text
    phone (Mac)   +12.00 ppm over 329 s   (halves +12.00 / +12.00, 0 steps)
    jetson         +0.00 ppm over 368 s   (halves -0.00 / +0.01, 0 steps)
    => true monotonic skew  -12.00 ppm    against 50 ppm assumed, 4.2x margin
    ```

    **And that measurement is what vindicates the bound's hardest fix.** The
    estimator *fitted* −1.09 ppm while the truth was −12.00 ppm, so the error
    from applying its slope is 10.9 ppm — ten times the fitted magnitude. A
    charge of `|fit|` alone, or of the fit's standard error alone, would have
    been an order of magnitude too small. The additive form charges
    `50 + |fit| = 51.1 ppm` and covers it with 4.7x margin. The floor is doing
    the work, not the fit.

    **What the write-time stamp bought, measured**: the phone's own
    enqueue-to-departure gap on its control frames, read by the Jetson from the
    two stamps the frame carries, is p50 **0.088 ms** and max 0.271 ms — real,
    but about 1% of the bound. The ~100 ms case the design reasoned about needs a
    relayed or congested link, and every run today went direct at LAN speed. So
    the fix is right in principle and its measured benefit on *this* link is
    small; the case that justifies it remains unproduced.

    **Validation: four rounds, 34 findings**, and the two that matter most were
    both mine. A **critical** one: `_stamped_at_wire` sat one line above the `try`
    whose `except BaseException` exists precisely to stop a writer dying with the
    session reporting health — and it calls `encode`, which raises when a
    15-digit departure stamp overflows a header the one-digit placeholder fit.
    Measured: writer gone, `is_closed` False, `send()` still returning True,
    nothing transmitted again. Fixed at the root by reserving the widest possible
    stamp at enqueue, so the caller's own encode is the verdict.

    **The transferable lesson is about tests, and it cost two rounds.** After
    round 1 I had fixed fifteen findings and pinned almost none of them: **16 of
    19 reverts left the suite green**. After round 2, six more survivors. Worst
    of them, each passing for the reason it should have failed — a test planting
    *zero* true skew, so `max` and `sum` agree at the one point it measured; a
    `recv_with_receipt` test wrapping its call in a **retry loop**, so a call
    that returns nothing and leaves the good record queued is indistinguishable
    from "not here yet"; and a no-bare-number check that evaluated every
    candidate to `None`, so it was empty by construction. All three mutation
    passes now kill 18/18, 13/13 and 7/7 by the test that names each.

    **Two guards with no live path**, kept and pinned rather than deleted:
    `no_typed_message` (every channel has a type now) and the skew baseline check
    (20 buckets 10 s apart span 180 s, over the 120 s it asks for). Both reached
    by tests that remove the thing that dominates them, on the principle that a
    reason no test can reach is a reason that rots.

    **The cadence anomaly from task 14 recurred once and again did not
    reproduce.** One run offered ~12% of every commanded rate at once, uniformly,
    losing nothing; three later runs held full cadence. The link path was
    recorded this time — `direct` on both — so the relay hypothesis is dead and
    the cause is still unattributed. It is a phone-side scheduling effect, not a
    transport one.
16. ~~Loopback test with synthetic sensor frames and synthetic advisories;
    transport latency instrumented from the start.~~
    **DONE** — `sensors/phone_source.py` and `scripts/run_loopback_pipeline.py`,
    **913 tests** (895 on the Jetson; the difference is `test_sim_contract`,
    which needs the sim repo). Plan:
    `scratchpad/plan_task_16_loopback_pipeline.md`.

    This joins the two halves of the project. Tasks 12–15 built a transport that
    carries phone sensor data and returns advisories; `pipeline.py` has run
    perception → observation → actor → advisory since before any of it existed.
    They had never been connected.

    **The connection is not wiring, it is a clock.** The pipeline decides whether
    a reading is fresh by comparing it against the Jetson's `time.monotonic()`,
    and two devices count from their own boot — measured on this pair, **67.57
    hours apart**. Unconverted, `gps_age` is −243,264 s against a 2.0 s
    threshold, so `gps_fresh` is False on every tick of every drive, ego speed
    silently falls back to neutral, and the loop keeps producing advisories that
    look fine.

    **And the failure is not symmetric, which is the part the plan got wrong.**
    Which direction you get depends on which device booted first, and only one is
    safe:

    ```text
    Jetson booted later   age +243,265 s   fails the threshold -> neutral fallback
    phone booted later    age negative     a one-sided `age <= 2.0` ACCEPTS it
    ```

    The second means the policy acts on arbitrarily stale data believing it
    current. The freshness gate is now conservative on both sides: the past side
    charges the stamp's own uncertainty, the future side allows only that
    uncertainty, and a bound wider than half the window is refused outright with
    its own diagnostic rather than answered badly.

    `PhoneCameraStream` and `PhoneGpsReader` sit behind the interfaces
    `CameraStream` and `GpsReader` already expose, converting once on the way in
    — so every comparison downstream is same-clock by construction rather than by
    review, and frame-to-frame intervals survive exactly because conversion is
    affine. The bound travels attached to the reading rather than looked up
    beside it, because pairing a stamp with the wrong record is a mistake this
    project has already made.

    **The estimator was on the wrong side, and that needs sign-off.** Task 15 has
    the phone initiate, so the phone holds the offset — but the Jetson runs the
    pipeline and converts the incoming stamps, and a responder sees only t2 and
    t3, so it has no path to the offset at all. The exchange therefore runs
    Jetson-initiated, which needs no wire change and no code change because both
    roles are role-symmetric. It contradicts one sentence of
    `specs/transport_protocol.md`, **which has not been edited**. The wording to
    approve is *"exactly one side initiates, and it must be the converting
    side"*, plus an explicit prohibition on both sides initiating at once:
    `exchange_id` has no side tag, so two initiators would collide in the id
    space and match a pong to the wrong pending exchange.

    **Latency is two segments now**, because a reader cannot tell a slow link
    from a slow Jetson in one number and the deployment claim is about the
    Jetson. That moved a gate: `eval_run` asserted `e2e` p95 < 200 ms, a claim
    about this hardware that would have failed a run for the network's behaviour
    — and would have *loosened silently*, since the link segment drops out of the
    sum whenever the timebase cannot convert. It gates `jetson_ms`.

    **Measured.**

    ```text
                       ticks  jetson_ms p50   link_ms p50    fresh   converted
    Mac loopback  60s    601      1.18 ms       0.23 ms       100%     590/601
    Jetson  60s          601      3.11 ms       0.52 ms       100%     590/601
    real link 120s      1092      3.22 ms      11.96 ms       100%   1085/1092
    ```

    On the real link the one-way segment is p50 **11.96 ms** with p95 170 ms and
    max 320 ms — the same heavy tail task 14 measured (p99 134 ms, max 333 ms),
    and the reason the two segments are reported apart. The conversion bound
    tracked at p95 **8.72 ms**, against task 15's 8.0 ms. Every run: account
    reconciled, zero lost, and **100% of ticks with a fresh GPS fix — the number
    the task exists for, since unconverted it is zero.**

    **Convergence, three runs**: first conversion at 1.102, 1.108 and 1.110 s;
    11 ticks proxied; and every one of those ticks still produced an advisory.
    The proxy's reasons are recorded, not just its count — `no samples` twice,
    then the estimator's window filling one sample at a time — so a drive that
    proxies for longer than expected says why.

    **Not run**: the real-detector variant. The TRT engine is gitignored and
    absent from this tree, so `detect_ms` here is a scripted stand-in and
    `jetson_ms` above **excludes real detection**. `bench_latency.py` with the
    engine remains the shipping number; these two must not appear in one column.

    **Validation: six rounds, 27 findings.** The two worth reading: the harness's
    pass/fail gate could not fail on the defect it exists for — against a
    conversion replaced by the identity it reported `usable: true` and exit 0
    while the same report carried a **67.6-hour link segment** — and the phone
    reader thread died on any exception with every counter reading healthy, which
    is the exact failure class `transport/session.py` fixed in both its loops and
    documented against itself, reintroduced one layer up.

    **The transferable lesson is about my own fixes, and it recurred every
    round.** After round 1, sixteen of nineteen reverts left the suite green: I
    had fixed fifteen findings and pinned almost none. Then a test that planted
    *zero* skew, where the two candidate formulas agree. Then one that
    reimplemented the code it was testing. Then a guard with no live path,
    introduced in the same round I removed another for being one. Then a fix that
    *replaced* a guard instead of adding to it, losing what the old one caught —
    a healthy run reported unusable, and a run that spent 67% of itself on the
    proxy reported usable. Every one was found by mutating the fixes or by the
    validator; none by the tests passing.

    **Carried as stated limitations**, none able to produce a wrong value, a lost
    message, a hang, a crash or a false report: `pipeline_stats.*.n` is
    window-capped at 300, so quote `latency.*.n`; the rounded record's
    `e2e == jetson + link` identity is off by 0.01 ms; the GPS backend lacks
    `sim`/`start_wall`/`start_mono` (task 26); `gps_timebase_unresolved` is a
    guard for a constants change that **cannot fire today** — the widest bound
    reachable on a link the gate admits is ~106 ms against a 1 s threshold; the
    stats sampling race fails safe, reporting a gap that is not there rather than
    hiding one that is; and the rendering paths are unpinned because nothing
    asserts rendered output.

## E. Phone app — phone is in hand, no Jetson needed

17. ~~Android project skeleton: Kotlin, CameraX, permissions, foreground service.~~
    **Done.** Two Gradle modules: `:transport` (pure Kotlin/JVM, no Android, so the
    wire contract is testable at laptop speed) and `:app`. 82 JVM tests + 20
    instrumented on the `dsrc_test` AVD, 0 failed. APK 6.8 MiB; cold build 8 s with
    no daemon or cache. Manifest facts read back off the merged manifest (15 checks),
    verified to fail when wrong.
    Three defects the tests could not have found, all caught on the emulator:
    an ignored intent left a resident service forever, because `startService` creates
    one to deliver an intent whether or not the state machine acts on it; a teardown
    throw escaped an unguarded `react(STOPPING)` and killed the process, which is
    aimed squarely at task 18 since `onSensingDown()` is where the camera gets
    released; and `startForegroundService()` is a promise to call `startForeground()`
    whose breach kills the process **on bring-down, 1 ms later, not on a timeout** —
    so no `try/catch` could survive it. Fixed by not making the promise: the only
    caller is a visible Activity, so `startService` suffices.
    Validation ran 3 rounds. Round 1 found a spec-drift test that could never fire
    (the spec was not a declared Gradle input), a manifest gate blind to the three
    permissions the app cannot start without, and permission constants that asserted
    against themselves. Round 2 found the foreground-promise crash in *my* round-1
    fix. Two of my tests were vacuous — an "exhaustive sweep" that pinned no row of
    the transition table — and one instrumented race let a queued stop land inside the
    next test. Known gaps recorded in the plan: `MainActivity`'s wiring is unpinned
    pending the task-23 UI harness, and two test seams ship in the APK.
18. ~~Camera capture at the commanded rate, JPEG encode, per-frame monotonic
    timestamps from `elapsedRealtimeNanos`.~~
    **Done.** 178 JVM tests + 37 instrumented on the AVD, 0 failed. Achieved rate
    5.00 Hz at a commanded 5 and 15.00 at 15 against a ~28.4 fps source; 0.90 at a
    commanded 1, within one frame. Sustained 30 s at 10 Hz: 299 accepted, 9.97 Hz,
    no encode failures, no stall. Slow drain: 151 accepted, 20 drained, 131 dropped,
    accounting balanced. JPEG p50 25.7 KB at 1280x720 quality 85 on a synthetic scene.
    The rate gate carried the task's real content and took four attempts, each
    failure found by a test. Scheduling slots from *now* undershoots (a 30 Hz source
    into a 10 Hz target gives 7.5 Hz); scheduling from the previous slot is exact but
    pays a stall back as a burst; below ~1.1e-10 Hz -- inside the wire's legal range --
    the period saturates and adding it wrapped negative, so a command meaning "almost
    never" produced *full-rate* capture; and re-sending an unchanged rate silently cost
    a quarter of the frame rate, because re-anchoring converts the exact schedule back
    into the undershooting one.
    Three counters reported failure as success: a `pack()` throw left a frame with no
    outcome so total failure read as an encoder backlog; frames discarded at shutdown
    were counted nowhere, making the balance identity false after every stop; and a
    frame refused because sensing had stopped was reported as rate limiting. A fourth
    identity could not fail at all -- `gated` was derived from the other terms, so
    `seen == accepted + gated + refused` reduced to `seen == seen`.
    `ResolutionSelector` silently ignores the requested size unless the aspect ratio is
    stated: it defaults to 4:3, so a 16:9 request is filtered out before any resolution
    rule runs. The emulator lists 1280x720 as supported and returned 640x480 anyway,
    and 1856x1392 under a prefer-higher rule -- both looking like device limitations.
    The chroma row stride was completely unpinned, and the emulator cannot pin it: its
    virtual camera reports `rowStride == width` and planar chroma, so the padded and
    semi-planar paths are inert there. `onDestroy` leaked both worker threads on every
    platform teardown, since it never called `onSensingDown`.
19. ~~GPS capture and forwarding, logging both fix time and receipt time.~~ **DONE** — `GpsLocationSource` (`LocationManager` + `GnssStatus`, not FusedLocation, because `num_sats` is required non-nullable) and `GpsPipeline`, plus the transport this task grew to carry: framing, the eight channels with their policies and depths, inbound queues with a delivery thread, the timebase exchange, and the typed messages for every channel. Both clocks come off `elapsedRealtime`, so the fix-to-receipt difference is a real latency rather than a latency plus an unknown offset between two clock bases. Eight validation rounds; every finding closed and mutation-pinned, and the ones worth remembering are in `plans/section_e_status.md`. Two of them were arguments of mine that a branch could not be reached, both refuted. `Session.stats()` returned self-contradictory snapshots until the per-channel maps were read before the session totals -- 18,530 in 49.8M samples, and inverting the *writes* had fixed only one field pair. `scripts/refusal_reasons.py` reconciles the Kotlin and Python refusal tables case by case; the one divergence left is check order on multi-fault records, recorded with the precedence rule to adopt in `specs/transport_protocol.md`.
20. ~~IMU capture and forwarding.~~ **DONE** — `ImuSource` (accelerometer + gyroscope on a `dsrc-imu` thread), `ImuPairing` (the decisions, with no Android in them) and `ImuPipeline` (rate gate and accounting). The accelerometer drives and the gyroscope is paired from its latest reading, with the pairing skew carried as a statistic. `SensorEvent.timestamp` is only *documented* to share `elapsedRealtime`, so the offset is measured on the first event of either stream and a mismatch stops the modality rather than emitting a stream of confidently wrong timestamps. Four validation rounds; the behavioural defects were samples counted but never sent, the two sensor streams transposable without any suite noticing, four of six axes free to swap, and a gyroscope on a different clock still pairing. All are pinned by `ImuWireTest`, which reads the frames from a peer really listening on the port the phone dials — the physics does the work, since a stationary accelerometer reads one g and a gyroscope reads nothing.
21. ~~HERE client: query the road ahead at the commanded rate and in the commanded
    query shape, forward the raw response with request and response timestamps,
    no interpretation on the phone.~~ **DONE** — `HttpHereClient` behind an
    interface (no test touches the network; the key is shared with Nash
    production) and `HerePipeline` on a `dsrc-here` thread. The phone makes no
    call until told what to query: a default shape would be the phone
    originating a sensing decision. `rate_cmd` gained an **optional** `here`
    object to carry the shape, receiver-tolerant from the start on both sides,
    with twelve reconciled rows and a golden vector. A failed call is forwarded
    rather than swallowed — status 0 means no response at all, and the one
    validation finding was that a reply which stalled mid-body was reported that
    way too, losing a status the phone had already seen. The key never reaches
    `request_url`, which goes on the wire and into every artifact.
22. ~~Sensing-configuration handling across all four modalities -- rates and
    per-modality settings alike -- applied without restarting capture.~~ **DONE**
    — `ConfigApplier` routes a decoded `rate_cmd` to all four pipelines *and* to
    the IMU and GPS sources, because a rate gate can only ever lower a rate:
    commanding 200 Hz gave 50 on the wire while the phone reported 200, with
    nothing on either side recording the difference. `shadow` changes nothing at
    all, per the spec's definition. The link now starts **last** in come-up: it
    used to start sixty lines before the applier existed, so a command arriving
    in that 3–6 ms window was dropped while the transport counted it delivered —
    and since HERE makes no call until a query arrives, losing that one command
    meant no HERE traffic for the drive with every counter healthy. The only
    non-rate setting the downlink carries is the HERE query; camera geometry and
    JPEG quality stay compile-time.
23. ~~Advisory display: the driver-facing UI.~~ **DONE** — `AdvisoryHolder` plus a panel on `MainActivity` showing the Jetson's own strings: `rec_speed_display` and `current_speed_display` arrive already converted into `units`, and the phone formats nothing, because a phone that rounded 30.4 to 30 while the Jetson meant 30.4 would be showing a recommendation nobody made. The task's correctness content is staleness: the transport keeps only the newest, but that governs the *queue*, and once nothing arrives there is nothing to displace what is on screen. The holder expires three seconds after **arrival** (not `t_capture_mono_ns`, which is the Jetson's clock — a panel going blank because a clock estimate wandered would be a fault invented by its own safety check), and the panel redraws on a tick, because an event-driven redraw can only react to an event that happened. Validation found two more ways a stale advisory reached the driver: returning to the app repainted the old one for ~29 ms, and an advisory could land after Stop because teardown joins no delivery thread. Also closes task 17's deferred Activity harness, which is what the first of those needed.
24. ~~Thermal monitoring reported upstream, plus throttle-safe capture that
    degrades rather than failing.~~ **DONE** — `ThermalReader` and
    `TelemetryReporter`, plus the telemetry deferred from task 22. "Degrades
    rather than fails" means the phone **reports** and the Jetson decides:
    nothing here lowers a rate on its own, because a phone that quietly halved
    its camera rate when warm would leave the Jetson comparing a model against
    inputs it never asked for and cannot see it did not get. `achieved` beside
    `rates` *is* the shortfall. `thermal_headroom` is nullable on the wire
    because `getThermalHeadroom` returns NaN when it has no estimate, and
    canonical JSON refuses a NaN on both sides — so a NaN would fail the whole
    frame and take the thermal status with it, silencing the phone about being
    hot at the moment it was hottest. Validation found that same call is **API
    30 against a minSdk of 29**, unguarded, inside a lambda whose caller
    swallowed the `NoSuchMethodError`: on an Android 10 handset the entire
    telemetry stream produced nothing for a whole drive with no log line. Lint
    had the answer and nothing ran lint; `scripts/check.sh` does now.
25. Local session logging on the phone, for ground truth and post-hoc analysis.

## F. Jetson runtime — developed over SSH

26. Phone backends for `CameraStream` and `GpsReader`, fed from the transport.
27. HERE response ingestion: link association from GPS, caching, staleness
    tracking, explicit failure semantics.
28. Fusion / estimator: per-field source ownership between the wide-lagging feed
    and the narrow-current camera, with a staleness aging term. The sources
    observe different parts of the state and are not substitutable.
29. Sensing controller producing four independent rates and the per-modality
    settings that go down with them. Inputs: the free always-on IMU/GPS tier as
    a trigger proxy, advisory bin-boundary proximity, disagreement between
    sources, and thermal backoff from the phone.
30. Shadow / live mode flag. In shadow mode the controller emits the decisions it
    would make without gating; in live mode it gates for real. Both paths
    implemented, flag flippable at runtime.
31. Integration into the existing tick loop, advisory returned to the phone.
32. End-to-end run over the network backend, phone and Jetson apart, exercising
    the whole loop before any USB work.

## G. Instrumentation

Written as part of the implementation, not added afterwards.

33. Per-stage timestamps across the loop: capture, encode, transport, detect,
    track, fuse, infer, decode, return, render.
34. Trigger attribution in the controller: which rule fired, for which sensor,
    and why.
35. Shadow-mode decision log emitted alongside the full-rate reference, so every
    candidate policy can be scored against identical traffic from one drive.
36. Per-tick field provenance and missingness.
37. Thermal and throttle-event log for both devices.
38. Failure event log: GPS dropout, HERE failure or quota exhaustion, dropped
    frames, transport stalls, with recovery outcome.
39. Session summary generator: latency percentiles, achieved versus commanded
    rates, API calls made, trigger counts, failure counts.

## H. Colocation and integration — **[COLOCATED]**

Everything above is done before the devices meet.

40. **[COLOCATED]** USB transport backend behind the same interface, swapped in
    for the network backend.
41. **[COLOCATED]** `adb` first connection: accept the RSA authorization on the
    phone screen, tick "always allow".
42. **[COLOCATED]** Bench loopback over USB with both devices on a desk;
    end-to-end latency measured against the 200 ms target.
43. **[COLOCATED]** Shadow-mode correctness: logged shadow decisions match what
    live gating produces on the same input.
44. **[COLOCATED]** Live-mode verification: flip the flag and confirm gating
    genuinely changes sampling rates, the loop still closes, and the advisory
    remains sane. Shadow mode is not evidence that live mode works.
45. **[COLOCATED]** Thermal soak: sustained maximum-rate run to steady state;
    confirm the phone stays within limits and the controller backs off.
46. **[COLOCATED]** Failure injection: revoke GPS, kill HERE, unplug the link,
    and confirm each degraded mode behaves as specified.
47. **[COLOCATED]** Sim-contract parity: the observation vector produced live
    matches the simulator's sensing model field for field.
48. **[COLOCATED]** In-car install: 12 V power for both devices, mounts, cable
    routing.

## I. Measurement drives — **[COLOCATED]**

49. Shakedown drive: short and local, purely to confirm the system records
    readable, aligned data.
50. Drive set 1, shadow mode at maximum rate: the full-rate reference plus every
    candidate policy's decisions against identical traffic.
51. Drive set 2, live mode: the controller gating for real, verifying the
    shadow-mode predictions held.
52. Repeat across congested and free-flow conditions on at least three separate
    days, on a corridor known to congest.

Measured on the drives: end-to-end and per-stage latency; achieved versus
commanded rates and trigger attribution; HERE-reported speed against experienced
speed, and feed lag; camera-derived local speed variance against the feed's
scalar; how often the camera changes the advisory; provenance and missingness on
real roads; advisory bin distribution and churn; safety-layer intervention
counts; thermal behavior; and failure and recovery events.

## J. Reproducibility

53. One-command dry runs for the simulation matrices.
54. Smoke validation small enough for routine regression testing.
55. Artifact manifest: where checkpoints, session recordings, metrics, plots, and
    validation summaries live.

## Blockers

- **HERE API key** — blocks task 21 only. Everything else in E proceeds.
- **Physical colocation** — blocks section H onward. Nothing before it.
- Repository layout: phone app in `dsrc/android/`, Jetson runtime extends
  `deployment/jetson/`.
