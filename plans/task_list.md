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

**The phone is tethered, not on its own SIM.** It reaches the internet through a
personal phone's WiFi hotspot, because the alternative was buying a SIM and a plan
for one experiment. The system does not depend on which it is: the sensor link to
the Jetson is USB in the car and loopback over `adb reverse` on the bench, and the
only phone traffic that leaves the handset by IP is the HERE query. Two measured
quantities do depend on it, and both are reported as properties of this rig rather
than of the system: HERE feed lag includes the tether hop and the tethering phone's
radio, and the phone's own radio load during the thermal soak is WiFi rather than
cellular. Every telemetry report names the network it was built on
(`network_transport`, with `network_transport_absent` when there is none), so which
network a drive ran on is recorded rather than assumed — a tether can drop and be
replaced mid-drive by a fallback or by nothing.

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
25. ~~Local session logging on the phone, for ground truth and post-hoc analysis.~~ **DONE** — `SessionLog` writes the frame headers **verbatim**: the canonical JSON the transport encoded, one object per line, so the log cannot disagree with what the Jetson received because it is the same object. Payloads are not written (`n` already distinguishes sizes). Offers go to a bounded queue and drop rather than block, and the file stops at its cap rather than rotating — the start of a drive holds the setup and the timebase exchange, and a file that stops early and says so never claims to be complete. `Stats.complete` is the single place that says whether the artifact is whole, and validation found three ways it could say yes to a short file. **Open:** nothing prunes `filesDir/sessions`; the cap is per file and the storage is per device, and bounding it needs a retention decision that is a research question rather than an engineering one.

## F. Jetson runtime — developed over SSH

26. ~~Phone backends for `CameraStream` and `GpsReader`, fed from the transport.~~ **DONE** — the backends already existed; nothing could use them. `run_demo.py`, `replay_demo.py`, `bench_latency.py`, `pipeline.py` and `eval_run.py` mentioned `phone_source` zero times between them. `PhoneLink` is the assembly from socket to the two backends, and `--phone` selects it — both sensors or neither, since they are channels of one session, and a hard failure if no phone dials in rather than a silent fall back to local sources. The clock was the real problem: the spec makes the phone initiate time sync and the Jetson only answer, and a responder never learns t4, so **the side that must convert is structurally the side that cannot measure**. `OneWayEstimator` forms the offset from arrivals alone — `t1 - t2` sits below the truth by exactly the one-way delay, so the largest gap in a window is the fastest crossing. Its bound is a delay *spread*, not half a round trip: a constant 80 ms delay reports a spread of zero while every stamp is 80 ms wrong, which is why it is fit for a 2 s freshness threshold and unfit for latency attribution. **The experiment is what found the defect that mattered**: `Session.sendTimeSyncPing` was called only from tests, so no drive ever sent one, the Jetson saw `samples_accepted: 0`, and every frame proxied while the run looked perfectly healthy. `TimeSyncDriver` sends them at the spec's cadence; after it, 270 of 274 ticks convert and the offset reproduces across runs to 247 µs. **Open:** a redial still ends the run rather than rebinding — the supervisor for that belongs to task 31. Full four-stamp samples need a new wire field (the phone carrying `t4` on its next ping) and should be settled before task 33, which cannot attribute latency honestly without them.
27. ~~HERE response ingestion: link association from GPS, caching, staleness
    tracking, explicit failure semantics.~~ **DONE** — nothing on this side had
    ever opened a HERE body; `downstream_congestion_estimate` came from V2V peers
    as a hardcoded `0.0` or a neutral fallback, so a field the advisory partly
    rests on had never been informed by traffic data. `HereFeed` parses,
    associates against the vehicle's fix, caches links (not answers — between
    responses the query is re-answered from geometry against a fresh position),
    and ages. **No failure returns a congestion number**: eight named outcomes,
    because `0.0` there does not read as "unknown", it reads as "clear road
    ahead". **Two ages and only one knowable** — response age is measured, and
    HERE v7 flow carries no per-result observation time, so the feed's own lag is
    recorded as null with a note rather than summed into a figure that would look
    measured. Validation found seven defects in round 1, one of which killed the
    reader for the whole drive on a single malformed body (`OverflowError` out of
    `float()` on an oversized JSON integer). Round 2 found that my round-1 fix had
    turned a fail-safe race into a fail-dangerous one — one wrongly-fresh reading
    in 396,380 queries, reporting a 1.0 s age for links 46 s old — now one frozen
    snapshot published in a single store. Round 3 found the distance and lateral
    offset I had just added were computed from two different points, inverting the
    very judgement they exist to support. **Open:** the parse is written from the
    v7 documentation and has never met a real body — the experiment used synthetic
    responses over the real transport, which proves the wiring and not the schema.
    One captured response committed as a fixture is the only thing that would.
28. ~~Fusion / estimator: per-field source ownership between the wide-lagging feed
    and the narrow-current camera, with a staleness aging term. The sources
    observe different parts of the state and are not substitutable.~~ **DONE** —
    and it concluded that **the feed owns no observation field**, which is the
    opposite of what the plan set out to do. Fusion here is ownership, not
    averaging: the camera cannot see 2 km ahead and the feed cannot see the car in
    front, so nothing blends them. Two candidates were taken and both retracted
    under validation, for the same reason one field apart. `segment_target_speed`
    passed a units check — HERE's `freeFlow` is a free-flow speed in m/s, exactly
    the simulator's quantity — and the check was too shallow: the simulator fills
    it *and* `nearby_av_mean_speed` from one constant, so they are perfectly
    correlated in every training sample, and the sim's `min(target_speed,
    free_flow)` safety clamp is a no-op only *because* they are equal. Decoupling
    them advised **268 mph** on a parser-legal 120 m/s link.
    `downstream_congestion_estimate` went the same way: `src/sensing/local.py:203`
    is a **block** gate, pinning congestion, merge pressure and target speed
    together when no AVs are near — **0 of 1,095** rollout samples have congestion
    above zero without AVs, and a lone instrumented car never has equipped
    neighbours, so writing it would put the policy in that empty cell on every
    tick. The reading is published beside the vector instead
    (`ObservationResult.feed`, `diagnostics["feed"]`) for the task-29 controller.
    Experiment: 20 ticks owning a reading over the real transport, congestion
    0.067–0.9, **0 observation differences** against the identical no-feed run.
    The check that found both, which a units question cannot ask: *does this field
    move alone in a block the simulator moves together?* **Open:** making the
    vector legitimately feed-informed needs the **simulator's** sensing model to
    produce congestion without AVs — a training-side change, outside section F,
    and the real blocker behind both retractions.
29. ~~Sensing controller producing four independent rates and the per-modality
    settings that go down with them. Inputs: the free always-on IMU/GPS tier as
    a trigger proxy, advisory bin-boundary proximity, disagreement between
    sources, and thermal backoff from the phone.~~ **DONE** — nothing in
    `deployment/jetson/` had ever constructed a `RateCommand`, and one of the four
    named inputs was invisible: `PhoneLink` never read the `telemetry` channel, so
    thermal status, headroom and skin temperature arrived and were dropped. Bin
    proximity is the policy's own `top1 - top2` margin, since the policy emits
    discrete bins and has no continuous value to sit near a boundary. Thermal is a
    trailing multiplier so it wins by construction, skin temperature backs off
    before the status moves (measured: 5.4 °C of warming with the status
    `nominal` throughout), the free tier is never scaled because it is what
    notices the next event, and silence is not nominal. Rates cannot express
    "off" — the wire refuses zero — so every combination is swept against its
    bounds. **Validation found nine defects in three rounds, four of them in code
    written to fix the previous round's.** The pattern worth keeping: *more
    evidence produced lower rates* (a fresh event mid-hold took the camera 5 Hz →
    1 Hz, and a straddling signal caused a rebind per tick — the exact thrash the
    dwell existed to prevent); the *backoff cancelled itself* (call rate ÷6.7,
    query area ×45, so cellular bytes and heat were unchanged); a *GPS dropout
    stopped every rate command* (this codebase spells "no position" NaN, not
    None, and `MessageRouter.send` raises rather than drops, so one NaN fix took
    all four rates with it); and the round-2 bridge *reinstated the round-2
    defect* at `critical`, because it re-derived "was active" from a thermally
    scaled rate. Experiment: a 220-tick scripted drive, **0 commands refused by
    the wire**, no query through the tunnel, radius flat under backoff. **Open:**
    `MAX_QUERY_RADIUS_M = 10 km` has never been checked against what HERE v7
    accepts for `in=circle;r=` — nothing in the repo documents a bound and the API
    must not be called, so if their ceiling is lower the largest radii are refused
    by HERE rather than by the codec, and neither side would catch it.
30. ~~Shadow / live mode flag. In shadow mode the controller emits the decisions it
    would make without gating; in live mode it gates for real. Both paths
    implemented, flag flippable at runtime.~~ **DONE** — half of it already existed
    and was right: `ConfigApplier` on the phone treats a shadow command as changing
    "nothing at all", and counts what it shadowed. What was missing was the Jetson
    deciding which it is sending. The property the whole thing rests on is that
    **the mode never reaches the decision** — task 43 checks that logged shadow
    decisions match what live gating produces on the same input, and a `decide()`
    that could see the mode would make that check compare a function against itself
    and pass whatever the code did. So the mode is applied strictly on the way out,
    selecting one boolean, and `decide`'s signature is asserted structurally rather
    than left to review. **All three validation rounds found the same defect one
    step further in: the record read a live drive as a pure-shadow one.** Round 1:
    a pure shadow drive has no traffic feed at all, because `ConfigApplier.apply`
    returns on the shadow branch *before* `setHereQuery`, so the phone never calls
    HERE, `feed_congestion` is None on every tick, and `Trigger.DISAGREEMENT` — one
    of three raise rules — cannot fire; task 35 would have credited every candidate
    policy equally for a rule none of them had the chance to use. Round 2: the fix
    keyed on `f.was == LIVE`, and **a flip records the mode it came from**, so it
    asked "has this drive *left* live?" — a drive promoted and left there, the
    normal shape, still claimed the reference rates *and* still named the feed
    absent. Not monotone in live exposure either: False on leaving live, never on
    entering. Round 3: keying on "ever live" then collapsed a **mid-drive
    promotion** into a born-live drive, reporting nothing absent for a log whose
    leading segment had no feed — the unsafe direction, for the same reason. It now
    keys on being live from the first tick and publishes `feed_possible_from_mono`
    beside it, a lower bound because the query goes down at the flip and the
    response arrives later. **Four tests pinned nothing they were named for.** The
    unreachability test read `trigger`, which is `raises[0]`, so a co-firing event
    hid the rule; the absent list was compared against the constant it came from,
    so declaring `camera_density_bin` absent passed; `command_for` could drop the
    HERE query or the capture stamp with the suite green, because
    `shadowed.here == live.here` is two references to one object; and a concurrency
    test **executed its loop body zero times** — `Thread.start()` releases the GIL,
    so 400 flips finished before the main thread was rescheduled, leaving four dead
    assertions and one whose failure message read "the flipper did not finish, so
    the reader raced nothing" while passing on a run where the reader raced
    nothing. Replaced by forcing the interleave: `flip_to` calls the injected clock
    inside its critical section, so the clock *is* the middle of the flip.
    Experiment: a 120-tick drive replayed in both modes — **120/120 decisions
    identical, 120/120 commands differing by exactly the flag, 0 refused by the
    wire**, and four mode histories each recording itself. **The method lesson cost
    more than any single defect.** A throwaway mutation harness scored on
    `returncode != 0` reported nine of nine CAUGHT having run no tests, because
    `pytest-timeout` is absent and `--timeout=60` exits 4 on a usage error;
    rescored, three had survived. `scripts/remutate.py` already scores on *which*
    test failed for exactly that reason — the reason I did not reach for it is that
    it rebuilt both Gradle suites per mutation, so it now takes a kind filter. It
    then caught a bad pin of mine by naming a module instead of a test: deleting a
    two-line block orphaned the `if` beneath it, so the "catch" was the Python
    parser. Collection errors are refused as verdicts now. 17 pins, 0 survived.
    **Open:** making the feed available in shadow would mean letting a shadow
    command carry a query into effect, which changes what `shadow` means on the
    wire — a protocol decision, raised rather than taken.
31. ~~Integration into the existing tick loop, advisory returned to the phone.~~
    **DONE** — `run_demo.py --phone` took camera and GPS off the handset and sent
    **nothing back**: no advisory, no rate command. Everything sections E and F
    built — `SensingController`, `ModeHolder`, `HereFeed`, `FeedFusion`, the
    telemetry reader — had never been constructed by a live path, and the traffic
    feed reached no consumer at all: `pipeline.step` had no `feed` parameter, so the
    whole HERE ingestion path terminated in a log record and `Trigger.DISAGREEMENT`
    could not fire on any drive. **Two cadences, because the two channels fail
    differently**: `advisory` is latest_wins at depth one and goes every tick;
    `rate_cmd` is reliable at depth 16 and the loop runs at camera rate, so a
    command per tick would make the channel designed never to lose a record lose
    records continuously. Commands go on changed content, on the query going stale
    **in space** (it is centred on the vehicle and formatted to ~1 m, so it differs
    almost every tick and cannot be part of the changed test), or on a heartbeat.
    Shadow is the default; `--live-rates` opts in. **Sixteen defects across three
    validation rounds, and each round found its defect inside the previous round's
    fix.** Round 1: `build_components` binds `camera = phone.camera` once and the
    worker closes over it, so a rebind that built new backends reconnected the link
    to objects nobody read — and the run was already dead, because the camera's
    `end_of_stream` fires within one 5 ms poll and the worker breaks on exactly
    that. Round 2: the identity fix for that carried the previous phone's **frame-id
    high-water mark**, so a run surviving a redial refused every frame the second
    phone sent — silently, with `reader_alive` True, `end_of_stream` False, and even
    the drop counter flat, because it fires on the one condition that is false here.
    Round 3: the camera's fix was **absent from the GPS reader**, which served the
    previous handset's position — valid, fresh, stamped `measured` — to every tick
    until the new phone spoke; and the V2V beacon gates on `fix.valid` with no age
    test at all. The estimator reset was justified as "a new session is a new peer
    clock"; the same argument covers thermal state, the traffic feed and the
    position, and each round found one more field it had not been applied to.
    **Experiment** (`scripts/run_phone_drive.py`): 586 ticks over 20.3 s at 30 Hz
    through a real 0.73 s outage — the run did not end, 14 ticks passed with no
    camera, **280 of 280** advisories the phone saw matched a frame this side
    actually processed (0 unmatched), **13 rate commands against 586 ticks** — a
    ratio of 0.022, 45× fewer than per-tick — and **0 refused by the wire**. Pacing
    was not cosmetic: run flat out the same code reports a cadence that is an
    artefact of the harness. **The method lesson:** a round-2 pin mutated two lines
    as one anchor, so deleting just `_latest = None` restored the defect in full and
    the pin still read CAUGHT — *a pin whose granularity is coarser than the defect
    is not a pin*. Nineteen tests also bound a real fixed TCP port and flaked 6 in
    40; I had reverted that fix once for lack of justification, and the measured
    rate was the justification. **Open:** the loop does not attempt a send during an
    outage — the camera yields no frame, so `on_tick` is never reached — so the
    no-session path is covered by unit tests and not by the drive.
32. ~~End-to-end run over the network backend, phone and Jetson apart, exercising
    the whole loop before any USB work.~~
    The address the phone dials comes from a `link.json` pushed to the app's
    external files directory and read once at service start, with the source
    (`file` or `default`) carried into the record. `SensingService` had built two
    `LinkConfig` instances, one for the link and one for the status line, which
    agreed only because both were defaults. A port at or above 2^32 was truncated
    into range and accepted — `Long.toInt()` keeps the low 32 bits, so 4295015107
    became 47811 and 4294967297 became 1 — and no test distinguished the truncating
    implementation from the correct one. **Three validation rounds, and each round
    found its defect inside the previous round's fix.** Round 1 (seven findings):
    `summary["network"]` recorded how this machine *would* reach each online peer,
    which is the same whether or not a session used that route, so a run carried
    over `adb reverse` — phone dials 127.0.0.1, data crosses USB — wrote the same
    network record as one that crossed the tailnet. The fact that settles it was
    already computed and already dropped: the accepted socket's own remote address,
    which `_wire_record` iterated past. A second line looked the phone up by
    `Settings.Secure.ANDROID_ID` in a dict keyed on Tailscale's `HostName` —
    different namespaces, never equal — and replacing that lookup with `started =
    {}` survived the whole suite, which is the finding. Round 2 (six findings, four
    inside round 1's fix): `path_for_address` returned a relay region for a direct
    connection, because Tailscale sets `Relay` on a direct connection too, charging
    a relay hop of tens of milliseconds to a connection that made none; and a
    tailnet address whose peer had gone offline — the case the redial timeout exists
    for — was recorded exactly as a USB run. Round 3 (three findings, all inside
    round 2's fix): membership was then tested against `100.64.0.0/10`, which is the
    shared CGNAT block rather than a Tailscale allocation, so it also matched
    `100.100.100.100`, Tailscale's own resolver; and `to_record` read `self.session`
    four times, so one record could carry two `session_id` values and two handsets'
    channel counters, with nothing in it to say which was which.
    **Experiment** (`run_demo.py --phone`, phone dialling 100.90.108.88:47811 over
    the tailnet, `adb reverse` empty so no USB is in the data path): **786 ticks
    over 180 s across one forced redial** — the link down 9.3 s on `peer_closed` and
    back on the same handset, 2 sessions accepted, 0 displaced, 0 refused. **786
    advisories sent, 0 refused by the wire**; 34 rate commands, one per 23.1 ticks.
    The path was **direct, not relayed**, taken from the accepted socket's remote
    address 100.75.142.126:37458 rather than from reachability. **Link segment mean
    106.1 ms, p50 87.7 ms, p95 223.4 ms; Jetson segment mean 31.9 ms, p50 31.7 ms,
    p95 32.9 ms** — end-to-end is the sum of the two by construction, so the split
    is the information and not the total. Two prior fixes were confirmed against
    live data rather than fixtures: the hostname disambiguation fired on a real
    tailnet, where two peers share the hostname `device-of-shared-to-user` and one
    carries `Relay: nyc` on a direct path; and from the first tick after the redial
    the position read `valid: false, lat: null, num_sats: 0` rather than the
    previous session's fix, which is task 31 round 3's defect.
    **The finding the run itself produced, from reading the record and not from a
    test.** On the control channel the first run reported **241 frames received
    against 121 delivered**, with `dropped_inbound` and `abandoned_inbound` both
    zero, having lost nothing. The transport generates keepalives and also consumes
    them — `_record_inbound` counts one in `received` and returns before the inbound
    queue — and the record published every term of the inbound account except
    `heartbeats_received`, which `SessionStats` had carried all along, while the
    comment beside those fields asserted an identity the control channel does not
    satisfy. A reader applying it would have read 120 consumed keepalives as 120
    lost messages, indistinguishable from a run that really lost 120. Every channel
    balances in the second run.
    **The method lesson:** two entries in the mutation table pinned nothing, and the
    harness reported both in a form that reads like progress. One printed SKIP for
    several rounds because its anchor named code that had since been rewritten — a
    skipped entry is not a pin, and the code it named was edited three times
    underneath it. The other had become an equivalent mutant, which was settled by
    running seven colliding peers through both versions (7 of 7 kept either way)
    rather than by arguing from the code.
    **Open:** the phone evicted **61 of 655 camera frames** on its outbound queue
    (9.3 per cent) and the Jetson's record cannot see that, because the eviction
    happens on the sender; the policy bundle is random-init, so advisory values are
    placeholders and the run establishes wiring rather than decision quality; and
    the two machines sat on one desk, so the link segment is a tailnet path within
    one building and says nothing about a cellular link or a moving vehicle.
    **A test that measured the transport and reported it as the command.**
    `aCommandRaisingTheRateIsHonouredOnTheWire` failed on handset ZY227VV4XC at
    53.0/s and 53.3/s against a 56.8/s floor, on production code that had not
    changed since the gate was last green (`8b4811a`). The raise did reach the
    source: that session recorded `rateHz=200.0`, `seen=765`, `delivered=756`,
    `refusedBySink=0`, with mean gyro age falling from 13.9 ms to 8.3 ms. What did
    not reach the wire was the sample —
    `imu=ChannelCounters(enqueued=756, dropped=0, sent=525, abandoned=0)`, so 231
    samples were queued and the socket drained about 50 frames per second, while
    the 50 Hz baseline already sat at that ceiling. **The wire rate is bounded by
    the smaller of the source rate and the socket's drain rate, so it failed under
    two different conditions — a command that never reached the source, and a socket
    that could not carry the result — with nothing to say which.** Both sensors
    advertise a 500 Hz maximum, so the hardware was not the cap.
    The assertion now reads `ImuPipeline.seen`, incremented in the sensor callback,
    so the quantity is what the platform delivered to the process; the windows,
    the noise-derived floor and the command are unchanged, and the method is
    `aCommandRaisingTheRateReachesTheSource`. Two quantities were rejected, for
    opposite reasons: `stats.rateHz` is `gate.hz`, the stored command, and is the
    number that was lying in the original defect; the wire rate is the one that
    cannot see a raise on this handset. **Measured on both sides of the change:**
    with the source re-requesting its period the source rate went 50.3/s to
    114.3/s, a factor of 2.27, while the wire went 50.3/s to 51.7/s, a factor of
    1.03; with `ImuSource.setRate` returning without re-registering — the pre-fix
    behaviour — the source rate went 50.7/s to 50.3/s and the test failed. So the
    source rate separates the defect from the fix by a factor of 2.27 and the wire
    rate by 2.6 per cent, which is inside its own noise. The wire rate is still
    measured and logged, because the transport's ceiling is worth seeing.
    Coverage given up, named rather than implied: this test no longer covers the
    segment from pipeline to peer. Half of that is bought back by requiring
    `refusedBySink` not to increase across the raise. The other half is not — a
    channel eviction is not a sink refusal, since `Session.enqueue` drops the oldest
    and returns true, and the phone-side channel counters are not reachable from an
    instrumented test.

## G. Instrumentation

Written as part of the implementation, not added afterwards.

33. ~~Per-stage timestamps across the loop: capture, encode, transport, detect,
    track, fuse, infer, decode, return, render.~~
    Four of the ten already existed on the Jetson; six did not. Capture, encode,
    return and render happen on the phone, transport was a single capture-to-arrival
    lump converted through a one-way clock estimate whose own docstring calls it
    "unfit for attributing latency", and fuse, infer and decode were buried inside
    two undivided segments. The work: land the `t4`-on-next-ping field so the Jetson
    forms four-stamp samples and feeds the round-trip estimator it already shipped
    but never used; carry the phone's capture and encode instants on the camera
    header as same-clock stamps; split the Jetson tick; and publish a per-tick
    `stages` object in which every entry says whether it was **measured** on one
    clock, **converted** across two with a stated bound, or **absent with a named
    reason** — never a zero.
    **Three validation rounds, and each round's defects were inside the previous
    round's fix.** Round 1 (13 findings): `run_demo` persisted a timebase estimate
    without consulting the estimator's usability gate and `eval_run` converted
    against it, so with one time-sync sample the live adapter recorded `proxy=True,
    reason="only 1 samples in the offset window"` while the offline join reported
    `converted, ms 80.0` against a truth of 40.0 ms with a stated bound of 0.007 ms —
    an error some 5,700 times its own bound, reachable on every drive. Round 2 (9):
    round 1's own `capture_stamp_ns` unification had missed a third call site, so
    `run_phone_drive`'s advisory-match metric disagreed with itself on ~1.5 per cent
    of ticks; the old-log compatibility fallback reopened the RTT-ceiling clause it
    had just closed; `superseded = received - shown - expired` went to −1 because
    `shown` and `expired` are not a partition; and the test pinning the session id
    asserted source text with `inspect.getsource`, passing for every behavioural
    defect on the field. Round 3: round 2's new pin was itself unsound, failing 3
    runs in 8 — the callback assigned on every CONTROL frame into one variable named
    for the first, so the second pong overwrote it and the assertion read
    `expected:<57000> but was:<32000>`, **naming the session's correct value as the
    wrong one**.
    **Experiment**, 900 ticks over 180 s with the phone dialling the Jetson over a
    relayed tailnet path and `adb reverse` empty: **900 advisories logged by the
    phone, 900 matched to a tick, 0 unmatched** — the capture-stamp unification
    confirmed in the field, where the pre-fix rate would have left about 13
    unmatched. Per-stage p50s: capture-to-encode 4.3 ms, encode 7.6 ms,
    encode-to-enqueue 10.4 ms, enqueue-to-wire 12.6 ms, transport 26.8 ms, JPEG
    decode 10.2 ms, detect 17.8 ms, infer 0.5 ms, return 11.1 ms, render 93.0 ms;
    Jetson segment mean 30.7 ms, zero dropped frames. **`transport` converted on 898
    ticks and absent on 2** ("only 3 samples in the offset window"), **`render`
    measured on 665 and absent on 235** ("no advisory_shown line for this capture
    stamp") — both absences named rather than zeroed, on real data, which is the
    property the task exists for.
    **The experiment found a gap no test could.** All 900 ticks recorded the `stages`
    block and the report printed none of it: every stage name appeared zero times in
    `report.md`. The measurement existed and the surface meant to carry it did not.
    Building that surface then produced two more instances of the same rule — an
    instant carrying `ms: 0.0` was averaged as a duration, printing `capture | n 900
    | mean 0.0`, a stage that took no time; and the aggregation read tick records
    rather than joined rows, so the ten-stage table had twelve rows and lacked the
    two only the phone witnesses.
    **The method lesson: four test defects, three of one shape** — a test inferring
    another thread's state instead of waiting on an observable, then naming the
    production code when the inference failed. Also four separate cases of a harness
    reporting something that reads like a result while having measured nothing: a
    mutation table entry printing SKIP because its anchor had drifted, a mutation run
    reporting SURVIVED with the module's test count at 816 against a 1091 baseline, a
    no-op mutation whose two independent guards meant removing either changed
    nothing, and three instrumented runs scored as failures that installed nothing
    and ran zero tests. The rule that survives all four: score on the count matching
    the baseline, not on failures being zero.
    **Open:** nothing confirms CameraX's frame timestamp is on `elapsedRealtimeNanos`
    on this handset, so `capture_to_encode_start` could be a subtraction across two
    clocks reported as `measured`, in the one stage no test reaches; `fuse`'s absent
    branch is unreachable, so its protection is argued rather than demonstrated; one
    test still uses `Thread.sleep(200)` where an observable exists (never seen to
    fail); and runs recorded before this task carry the old pair of capture-stamp
    spellings, so their `--phone-log` joins will show about 1.5 per cent unmatched.
34. ~~Trigger attribution in the controller: which rule fired, for which sensor,
    and why.~~
    The controller decides one global level bit, then scales two named keys for
    thermal, then clamps — a composition chain, not a rule per sensor — so the record
    follows the chain the code actually walks: per rule a closed three-state status
    (**fired** / **quiet** / **not evaluable with the missing inputs named**), a
    `gates` block separating "fired but blocked by the dwell" from "idle and quiet",
    and a `per_sensor` block carrying base rate, level sensitivity, thermal scale or
    exemption, clamp, previous and changed, with a reconstruction identity
    `_clamp(base × scale) == rates[k]`. Nothing the controller decides changed: rates,
    the trigger word, `rules_fired` and the `reasons` texts are byte-identical.
    **Three validation rounds found no defect in the implementation.** Round 1 drove
    `decide()` over 12,000 randomized calls asserting the record against the decision
    on every one — 0 inconsistencies. Round 2 replayed 15,000 decisions through the
    pre-task and post-task trees and got byte-identical records, same md5, on a corpus
    that produced all 6 trigger words, all 16 reachable `rules_fired` combinations and
    14 distinct `reasons` templates; that closed the no-behaviour-change contract by
    measurement rather than by reading the diff. **Every finding across all three
    rounds was a guard that would not have noticed if the code became wrong.** The
    blocker: deleting the whole attribution from the emitted record left all 1543
    tests passing, because every test read the dataclass and none read the tick log —
    the task's own defect class turned on the task's own output. Its mechanism is
    worth keeping: `gates` and `per_sensor` are the *same objects* in the record as on
    the dataclass, so object assertions pin them transitively, while `rules` is
    transformed and `first_decision` copied — exactly the two fields no test asserted
    values for. Round 3 then found that filtering the emitted `rules` to fired-only
    also survived, restoring absence-as-ambiguity in the log, and that the event rule
    was the one whose `not_evaluable` branch nothing asserted. That last is unreachable
    today because the observation builder substitutes a neutral float — and becomes the
    live path the moment task 36 lands provenance, which is why it is pinned now.
    **Experiment**, 899 ticks over 180 s, phone dialling the Jetson over a direct
    tailnet path with no USB in the data path: every tick carried an attribution block,
    and **all three states appear on real data** — `advisory_margin_narrow` fired on
    899 (an untrained random-init bundle), `event_from_free_tier` and `thermal_backoff`
    quiet on 899, and `source_disagreement` **not evaluable on 899** with `missing:
    ["feed_congestion"]`, because HERE is unconfigured. That row is the deliverable
    proving itself: before this task the rule's absence was indistinguishable from a
    calm road. `feed_declined` carried `"feed_outcome"` on every tick, so the field
    delivers and open item 1 resolves toward keeping it. The thermal cause was null
    throughout, so the telemetry thread did not die as the code's comment warns it has
    on Android 10, and `gapped` was false on all 899 ticks.
    **Two estimates corrected by measurement.** Record growth was projected at 0.8-1.0
    KB of attribution on a ~1.5 KB record, about +60%; measured 1354 B on a mean record
    of 7847 B — a third larger in absolute terms, and 17% rather than 60% in relative
    terms, because the base record is some five times the assumed size. And with HERE
    unconfigured, `Trigger.DISAGREEMENT` is unreachable in practice as well as on
    shadow drives.
    **Open:** `local_density_bin` carries the same substitution blind spot as the
    accelerometer but in the opposite direction — a camera seeing nothing gives a bin
    index of 0, so a blind camera beside a congested feed **fires** the disagreement
    rule and raises the camera rate; the accelerometer case under-reports and changes
    nothing, this one over-reports and moves a rate, and open item 2 now names both.
    Crossing two quiet entries in the emitted record still survives, accepted as a
    limit. `RuleCheck.to_record()` does not reject non-finite floats, judged
    unreachable because the transport refuses them at framing.
35. ~~Shadow-mode decision log emitted alongside the full-rate reference, so every
    candidate policy can be scored against identical traffic from one drive.~~
    **The task's wording promises more than the system can deliver, and the plan says
    so.** `shadow_mode.py` already records that the traffic feed is structurally absent
    from a pure shadow drive — the phone makes no HERE query, so the feed stays silent
    and the disagreement rule cannot fire — and task 34's drive confirmed it from the
    other side at 899 of 899 ticks not evaluable. So "identical traffic" is defined as
    identical *recorded inputs to the decision function*: never the HERE feed, never the
    trajectory, and only the full-rate reference until the first live tick. The log
    carries the exact 13-field `Inputs` unrounded (the attribution's evidence rounds to
    four places, so scavenging it would diverge silently), the controller's own clock
    read, and a per-tick reference witnessed from the phone's `achieved`/`dropped` —
    decoded since forever and read by nothing until now, so the reference stopped being
    an assumption about the phone's defaults. `score_shadow.py` replays the log and must
    reproduce the incumbent byte for byte before it scores anyone; a log that fails that
    identity scores nobody.
    **Three validation rounds, and the defect that mattered most was mine.** Round 1
    (ten findings): the witness could not tell a live phone from one that reported once
    and died — two 300-tick drives produced a byte-identical block claiming 300 ticks
    with achieved, while the controller itself had recorded `stale_telemetry` on 250 of
    them. Round 2: **the counter I specified to fix that was wrong across a whole
    regime**, not a corner — `1 + count(age decreasing)` fails whenever the tick
    interval is at least the telemetry interval, and idle camera rate is 1.0 Hz against
    1 Hz telemetry, so the ordinary drive sits exactly on that boundary. An idle drive
    with 120 genuine reports counted 1, indistinguishable from the dead phone the fix
    existed to catch. Replaced with distinct arrival instants. Round 2 also found the
    witness's staleness predicate disagreeing with the controller's on NaN, where the
    tick landed in neither partition and a bare `NaN` reached the JSON. Round 3 found a
    truncated log scoring clean: `summary.json` was already loaded and the run's own
    tick count never compared against it, a check `eval_run` has pinned and this tool
    did not do.
    **Experiment**, two drives because mode is fixed at construction so one drive is
    either all-reference or all-contaminated: **599 ticks shadow and 353 live-rates,
    `replay_identity` 0 mismatched on both** — the first logs this tool ever read that
    it had not itself written, with floats through `MetadataLogger`'s buffered writer.
    Every tick of both carries `decision_inputs`, `decided_at_mono` and `reference`.
    The additions cost 595 and 605 B against a mean record of 8511 and 8552 B, 7.0 and
    7.1 per cent, against an estimate of ~560 B and ~7 per cent — an estimate that held,
    unlike task 34's.
    **The method lesson: a candidate only differs where the drive exercises the quantity
    it keys on.** Three candidates were needed to produce one difference. Lowering the
    acceleration threshold did nothing on a stationary handset; halving the margin
    threshold did nothing because an untrained bundle on a static scene held the margin
    constant at 0.0123 for all 599 ticks; only a threshold below the margin itself
    differed, at 596 of 599 with `first_differ_tick_id` 3, the first three agreeing
    because the dwell had not yet promoted the incumbent.
    **Open:** the staleness reconciliation agrees at zero on both drives, which shows
    only that the witness raises no false positives — neither drive had stale telemetry,
    so non-zero agreement is untested on hardware. `first_differ_tick_id` is exercised
    but not pinned; the candidate that would pin it must move rates across the dwell.
    `source_disagreement` was not evaluable on every tick of both drives, so the only
    claim available about it is that the refusal is correctly named. Bare `NaN` appears
    in the metadata log from NaN GPS before a fix — 70 lines and 1 — which is
    pre-existing and does not affect the identity gate.
36. ~~Per-tick field provenance and missingness.~~
    Every value the encoder reads now says where it came from: a closed eleven-member
    vocabulary, a `field_sources` entry for all 39 encoder slots where the informal map
    covered 33, the classes carried into the controller's `Inputs` and into
    `score_shadow`, and every log recorded before this task refused **by name** rather
    than scored against defaults. The distinction the task exists for is between a zero
    that was measured and a zero that stands in for an absence.
    **Three validation rounds and two fix rounds, and the rounds kept finding defects
    inside the previous round's fix.** The critical one: both GPS readers return the
    held fix with no age invalidation, so a receiver that goes silent presents as the
    same `valid=True` fix growing older -- not as `valid=False`, which was the only
    shape the task's own dropout tests used. The staleness guard measured from the last
    appended sample rather than from the fix's own capture, so it fired a full
    `gps_stale_after_s` late: for 2.0 s the speed read `fallback_neutral` while the
    acceleration read 0.0 labelled `derived`, which is a calm road reported from a dead
    sensor. **The plan's own scope claim was false.** It said the guard could only ever
    remove a rate raise; `ego_acceleration` is an encoder slot, so the guard also
    changes the actor's input and moves `policy_margin`, which is itself a raise rule.
    The mechanism was measured -- 11 of 12 random-init bundles moved, by up to 0.0027
    against a 0.15 threshold -- and no crossing was exhibited, because there is no
    trained bundle to exhibit one with. **Four of the task's own new behaviours had no
    catching test**, each surviving the full suite; the sharpest read the wrong
    observation key, which would have recorded `measured` where the truth was
    `derived_empty`. A further defect neither guard caught: on the first fresh tick
    after an outage the window holds samples from both sides of the gap, so a slope was
    fitted **across an interval containing no data**, labelled `derived`, and cleared
    the event threshold -- 20 m/s, a 5 s dropout, GPS returning at 0 m/s and staying
    there produced -3.6 m/s^2 and latched the camera at 5 Hz for the full hold. It is
    identical at the parent commit, so it was pre-existing rather than introduced.
    Fixed by clearing the sample window on any non-fresh tick, which gives one
    stateable invariant: the window holds only samples from an unbroken run of fresh
    fixes. The last round also found `eval_run` still deciding encoder coverage by
    **counting** keys -- the defect the previous round had fixed in the builder and left
    standing in the surface an operator reads -- and a test that fix round had added
    passing without ever reaching the clause it was named for, because its fixture's
    order made an earlier assertion decide the outcome.
    **Experiment**, two drives, the second with a deliberately induced 46 s GPS outage
    because the first drive's only gap was cold-start acquisition. 600 and 500 ticks,
    the laptop's tailnet path direct but **the phone-to-Jetson session relayed via nyc
    on both** -- a different pair of hosts from the leg that looks healthy. The census
    over 23,400 field-ticks: `fallback_neutral` 67.09%, `derived_empty` 7.69%,
    `static_config` 7.69%, `derived` 7.47%, `sim_parity` 5.13%, `approximated` 2.56%,
    `measured_converted` 2.36%, with every tick carrying exactly 39 entries.
    `score_shadow` scored the new log (600 of 600, `replay_identity` 0 mismatched) and
    **refused both task-35 drives by name**, printing the four missing keys and the
    first tick id. **The synchrony fix holds on hardware**: across six recoveries on two
    drives there is no tick where `ego_speed` is substituted and `ego_acceleration` is
    not, and two independent code paths agree exactly on the count, 51 = 51 and
    320 = 320. Acceleration stays substituted for two to three ticks after speed
    recovers while the cleared window refills to its 0.3 s minimum span, which is
    visible as a third missingness value one tick wide either side of every recovery.
    **Two numbers came out worse than estimated.** The added bytes measured 852 per
    record, 8.9% of it, against an estimate of 630 and 7% -- 26% below, with one field
    omitted from the estimate entirely; task 35's estimate had held. And the missingness
    figure rose 6.02 percentage points on identical footage, from 0.611 to 0.671, purely
    because the denominator went from 33 slots to 39 and all six added slots are
    substituted on a lone instrumented car.
    **The experiment found what 1,694 tests could not**, in the shape task 33 found
    first: `report.md` printed the missingness mean and no spread, and on the second
    drive that printed mean of 69.9% **occurred on zero ticks** -- the drive is bimodal
    at 66.7% and 71.8%. The percentiles were already in `report.json` and the renderer
    ignored them.
    **The method lesson: run the old code before calling something a repair.** A
    per-tick-bucketing hazard was written up as a defect this task's design had
    introduced by composing two decisions that were never considered together. Extracting
    the pre-task function verbatim and running it against real records showed it never
    emitted that key at all -- the hazard was created and pre-empted inside the same
    change, and the write-up had attributed to the old code a behaviour it never had.
    The same drive showed the corrected summariser has still never run on real data,
    because the branch needs a continuous value and the only candidate was null on every
    tick of both drives.
    **Open:** while GPS is fresh, one held fix is re-appended every tick, so a real slope
    decays toward 0.0 and is still labelled `derived`; fixing that changes acceleration
    values upward and would make the event rule fire more often, so it is its own task.
    The camera was occluded on both drives -- 0 detections across 1,100 ticks -- so
    `derived_empty` on 100% of ticks came from a covered lens rather than an empty road,
    which is exactly the distinction that class names and cannot itself resolve. Four of
    the second drive's five recorded gaps were single ticks where the fix aged 1 to 30 ms
    past the freshness bound; the class alone does not separate those from the 46 s
    outage. The event rule fired three times on a handset lying on a desk, because the
    fused speed reported 5.3 m/s while stationary -- the provenance record is correct and
    the value is wrong, which is the boundary of what this task claims: it records where
    a number came from, never whether the number is right. Two defensive branches in the
    numeric summariser survive mutation and were left unpinned rather than pinned with a
    manufactured state, since no evidence key a rule can carry on every non-evaluable
    tick is boolean or mixed-type. `summarise({})` reports 0.0 missingness for an empty
    map, unreachable from the builder and unchanged from before.
37. ~~Thermal and throttle-event log for both devices.~~
    **The two devices were in opposite states, and finding that out was the plan's
    first job.** The phone's thermal input was already live: it reads the platform
    status and its own zones once a second, sends both on the telemetry channel, and
    the controller maps them to the multiplier applied to the camera and HERE rates.
    Task 34's drive had already confirmed the whole chain, since its thermal rule was
    quiet on 899 of 899 ticks and quiet requires a scale of exactly 1.0. The Jetson,
    by contrast, had **no thermal reading at all**, and the one sampler adjacent to it
    degraded to a silent no-op when its optional import failed, wrote records nothing
    collected, and had no test. So the task was a record on one device and a
    measurement on the other. A temperature sample is task 33's stage timing
    (**measured**, **stale** with an age, or **absent with a named reason**) and a
    throttle event is task 34's rule attribution (**fired**, **quiet**, or **not
    evaluable with its missing inputs named**) -- `count` is 0 on both quiet and not
    evaluable, so the status word carries the distinction and the count never does.
    Behaviour changed in exactly one place, a status-change listener on the phone; no
    commanded rate moves on either device.
    **Three validation rounds and three fix rounds.** A phone redial erased the drive's
    throttle count and wrote a phantom event, because the sampler copied the phone's
    counter instead of accumulating and a redial restarts it at zero -- a drive on which
    the phone reached `severe` printed `quiet -- 0 status transitions` with three event
    lines in the log, violating the plan's own rule that the count equals the number of
    event lines, in both directions at once. A Jetson whose cooling devices were readable
    **once** in 180 passes produced a record byte-identical to one readable on all 180,
    because the flag was set and never cleared. The report printed a `stale` count that
    could never be non-zero while omitting the count that carried the signal, so a
    sampler that died 10 s into a 180 s drive rendered identically to a healthy
    10-second drive. The phone half of the module was never executed by the Python suite
    at all. **The second fix round existed mainly to undo the first round's regression:**
    refusing to say `quiet` on partially observed cooling devices was right, but the same
    change zeroed a real, already-logged throttle count and discarded `fired`, so a drive
    that observed and recorded throttling reported that it said nothing about whether the
    Jetson throttled. The last defect found by reading a fix rather than running it: the
    phone's count and its last transition were read under two separate locks with a
    binder call between them, so a transition could be counted with its description
    missing -- and on the first transition of any run that needs no race at all, because
    the description is still null when the count reaches one.
    **The experiment proved the feature was inert on the one Jetson it was written for.**
    150 s on the real Orin at the deployed commit: 1,200 ticks over five drives told the
    story, but the first drive alone settled it -- **751 ticks, 0 sample records, 0 event
    records**, every tick reading `absent`, reason `sampler_stopped`. Three of that
    machine's nine thermal zones answer `EAGAIN`, which surfaces through the buffered
    text layer as a `TypeError` that the reader's `except OSError` did not catch, so the
    sampler thread died on its first pass. **The fixtures could not have caught it**:
    they make a zone unreadable by deleting the file or denying permission, and both
    raise `OSError`. The blast radius was worse than the crash -- one sysfs quirk took
    the *phone's* thermal record down with the Jetson's, on a drive where the phone was
    connected and delivering telemetry throughout. **What the vocabulary did right is the
    reason to keep it**: nothing read `quiet`, nothing read as a zero, and the report
    said outright that the drive answered nothing. The failure was recorded as a failure.
    **The census settled the plan's largest unknown and broke its estimate.** Nine zones,
    six usable, and **13 cooling devices where the estimate assumed 3** -- the entire 39%
    overrun in sample-record size, against per-item figures that were right to a tenth of
    a byte (22.7 per zone, 23.5 per device). The wire cost was exact at +68 B/s. Zone
    selection used the preferred-name arm and picked `tj-thermal`; the hottest-zone
    fallback, which exists precisely because nobody knew these names, never ran. Thermal
    headroom became a counted fact rather than a source comment's assertion: **not a
    number on 456 of 456 and 297 of 297 reports**, on a handset whose thermal HAL is
    connected and answering. The independence property held under a real 54.98 s tick
    stall -- 55 samples at 1.005 to 1.009 s spacing, with the phone's age climbing 1.6 to
    45.9 s on the face of each record.
    **After the repo fix, a confirmation drive on the same Orin**: 1,200 ticks, 241
    samples, `measured 241, absent 0`, zero ticks reading `sampler_stopped`, and the
    sampler's mean interval corrected from 1.0091 s to 1.0003 s. The `quiet` line now
    carries the evidence for its own claim -- `241 of 241 passes fully readable` -- where
    before it asserted "readable throughout" and printed the counters only on the branch
    where the claim was *not* being made.
    **The method lesson: a fixture's failure mode has to be the field's.** A deleted file
    and a denied permission both raise one exception; the real device raised a different
    one, and every test passed while the feature did nothing. The second lesson is about
    instruments: **five distinct false readings** were produced by measurement harnesses
    on this task alone -- stale bytecode, a shell modifier that mangled a build task name
    so an empty results directory read as no failures, two partial tree copies that
    collected fewer tests than the baseline, and a stall analysis that compared two
    different clocks and reported a clean-looking zero. Every one was caught by the same
    rule: reproduce the baseline count and kill a known-bad control before quoting any
    verdict. A sixth near-miss came from `sort` collation differing between two machines,
    which made identical trees look 60 lines apart.
    **Open:** the Jetson's `fired` arm never ran in the field -- all 13 cooling devices
    held constant across every drive, so the event-writing path and its byte estimate are
    confirmed by tests only. `stale` was never produced on any of 8,718 ticks, and
    `read_error` is unreachable through the `EAGAIN` zones because the absence is
    absorbed per node before it can raise. The guard that stops a Jetson read failure
    nulling the phone's record was never exercised in the field, because no Jetson read
    failed. Two transitions inside one telemetry period stayed unexercised: the
    instrument sets the floor, since each injection round trip costs about a second. And
    no real thermal excursion was reached on either device -- 1,560 s of sustained
    streaming took the handset to 43.9 C skin with its status still nominal, the phone's
    transitions were injected rather than provoked, and the platform's severity
    thresholds are internal to its thermal service, so how much heat would have been
    needed cannot be stated from this device.
38. ~~Failure event log: GPS dropout, HERE failure or quota exhaustion, dropped
    frames, transport stalls, with recovery outcome.~~
    **The plan's first section was an inventory, and it is what the task turned on: 186 failures
    are already detected across the two devices, and almost none of them can be read.** Four of
    the 186 record when the failure happened. None records an episode -- a second endpoint and an
    outcome. Several have no reader at all: the metadata logger's own write-failure and
    dropped-record counters are set in five places and read nowhere outside that module, the only
    pre-existing failure record type is written by one line and read by nothing, and the
    sequence-gap counters -- the system's sole cross-device loss evidence -- were omitted by hand
    from the record that carries them off the device. Two failures were detected nowhere: the tick
    loop's no-frame branch counted nothing, so a drive blind for 110 of 120 seconds wrote the
    artefact of one that was never blind, and the worker was `try/finally` with no `except`, so an
    exception still ran teardown and wrote a summary that read like a clean short run. The task
    adds one stream as a **projection of counters that already exist**, not a second detector:
    where the log and a counter disagree the counter is right and the log names the disagreement.
    Its vocabulary is imported rather than invented -- task 34's three rule words, task 33's basis
    words -- and exactly one closed set is new, whose third member is the point: an episode is
    `recovered`, `open_at_end`, or **`unobservable`**, because a source that stopped being readable
    must never report a recovery nobody witnessed.
    **Three validation rounds, a fix round, a re-audit, and a second fix round.** The first round
    found the feature able to report `quiet` on a drive where the camera went blind 40 times, with
    "blind ticks: 40" printed three lines away and nothing reconciling them; a single continuous
    outage counted as 30 discarded episodes, because the cap check returned before constructing the
    episode so nothing was ever marked open; and **five of six wiring points unpinned**, so the
    entire feature could be absent with every test passing. **The re-audit then found the fix round
    had introduced a critical regression**: the backwards-counter record was changed from one entry
    to a list so repeated occurrences would accumulate, and the consumer was not changed, so
    rendering raised and **no `report.md` was written at all**. Worse in combination -- the same
    round's other fix was **inert**. It declared the camera's dropped-frame counter session-scoped
    so a redial would stop being read as a backwards jump, but the gate also requires a session id
    and that accessor never supplied one, so a redial still recorded the false step, and that step
    now crashed the report. One redial, no report. The re-audit also defeated the round's own
    structural pins: wrapping either failure-recording call in a dead branch left all 1,849 tests
    passing, silencing the log entirely -- this task's defect class, in the code that reports it.
    **Experiment**, 300 s and 1,073 ticks on the real Jetson with an induced 82-second link outage,
    the phone's first session relayed via nyc and the second direct after the redial. **The drive found
    seven defects that 1,858 Python and 443 Kotlin tests did not**, and the sharpest is the section's own
    defect class arriving through a door no test opened: a camera that delivered no frame for 82 seconds
    was recorded as **21 separate recoveries**, opening on an exact 4.001 s period, each closing
    `recovered` after about 2.15 s and together covering 45.2 s of the outage. The source is fed by direct
    notification, so its scan accessor reports "nothing moved" by construction, and the generic path read
    that as quiet and closed the episode every three passes; its own three-second timer never fired. Three
    further defects made the record misstate what happened: every `not_evaluable` row whose source
    recovered printed `-- missing ;` and named nothing, because the reason is cleared on the readable
    path; the report printed the *readable* count inside a sentence about unreadable passes, overstating
    one unobserved window 150-fold; and two phone-side sources reported themselves readable through the
    entire outage from a pre-outage snapshot, because the link clears telemetry on rebind rather than on
    session loss. A `by_reason` breakdown also dropped one reason entirely while its total stayed correct.
    **What the drive confirmed carries equal weight.** The accounting invariant held on **30 of 30**
    sources against real counters, with the one non-zero third term appearing exactly where it was added
    for. The redial produced no false backwards step, and `report.md` rendered in full -- the crash that a
    backwards step caused two days earlier did not recur, though a backwards step also did not occur, so
    that fix remains untested in the field. The wire cost was **exactly 0 bytes**, verified against the
    diff rather than asserted. And the per-tick block was blind to the whole outage while the 1 Hz scan
    stream carried it, which is the plan's own sampling decision confirmed on real data. Byte costs: the
    two per-record figures the plan named as the ones to check were exact at 131 B and 197 B; the summary
    missed by **77.8%** (7,899 estimated against 14,045 measured), a third of it from the 28-versus-30
    source count and the rest from five fields the plan's sample row did not carry. The drive could not
    exercise `open_at_end` or `unobservable` -- no episode was open at teardown, and none was open when
    its source went unreadable -- nor the worker-exception path, nor any HERE source, the build having no
    key by design.
    **The method lesson: check which file the instrument reads.** Nine false readings across this
    task and the last came from measurement harnesses rather than from the code under test -- stale
    bytecode, two partial tree copies collecting fewer tests than the baseline, a shell modifier
    that mangled a build task name so an empty results directory read as "no failures", a quoting
    error that read as a compile failure, a units error comparing two clocks, and a listing that
    reported no differences because two machines sorted with different collation. The last was
    mine: I refuted an agent's claim that a platform class could not be extended by running `javap`
    against the mock jar used at test *runtime*, when the compiler resolves sources against the
    `compileSdk` jar, where the class is `ACC_FINAL`. The agent was right and my evidence came from
    the wrong file. The rule that survives all nine: reproduce the baseline count and kill a
    known-bad control before quoting a verdict, and confirm which artefact the tool actually reads.
    **Open:** four registry pins outside this section resolve but catch nothing -- each mutation
    leaves the suite green -- and a fifth has never compiled, its mutation text naming a parameter
    of a different function; all five are recorded rather than fixed, by decision, since closing
    them means writing tests in four unrelated subsystems. Four of the eight phone failure kinds
    are untested: `imu.timebase_mismatched` for two independently verified reasons (the sensor class
    is final on the compile classpath, and the mock jar's accessor is compiled to return a constant
    with no field or constructor to change it, so the source's dispatch cannot match in any JVM
    test), and three more reachable only through the Android service lifecycle. Closing any of them
    needs a mocking library or Robolectric, which is a dependency decision taken deliberately and
    declined. **The plan itself is wrong in eight places**, all recorded: it says "twenty-eight
    sources" throughout while its own table and the code have 30, and 28 is exactly the number of
    rows whose device is the Jetson -- one device's rows written as the total, so every derived
    figure carries the error; its cross-check test is impossible as written, since the ten phone log kinds and the
    two phone-device registry rows are disjoint sets; two open items describe mechanisms that do not
    exist, one of which is the direct cause of the inert fix above; the summary byte estimate does
    not reproduce; a stated reading rule is not checkable from the records as emitted; a
    reconciliation test it specifies was never written; and its record specification still lists two
    scan-block aggregates that were deliberately removed for being constant by construction, one of
    them a literal nothing could ever increment.
39. ~~Session summary generator: latency percentiles, achieved versus commanded
    rates, API calls made, trigger counts, failure counts.~~
    rates, trigger counts, API calls made, failure counts.~~
    **The plan's first job was to find the honest gap, and two of the five items were already
    rendered.** Latency percentiles and failure counts had surfaces. The other three did not, in
    the pattern this section kept finding: trigger counts were written into `summary.json` and had
    **zero non-test readers**; achieved and commanded rates both existed on every tick and were
    never put side by side; and the phone's own count of API calls placed crossed the wire on every
    telemetry frame and **reached no reader at all**. So the task is one section in the report a
    human reads, not a seventh artefact beside the six that already existed.
    **The design question was how to aggregate five three-state vocabularies without collapsing the
    third state.** The answer: each of seven axes reports `answered of attempted` as two
    independently counted integers, plus the census of that axis's own reason words. No percentage
    anywhere, no scalar health field in the JSON -- because a scalar is what a dashboard plots on
    its own, and once plotted the enumeration is gone. `0 of 0` is never "answered". A word outside
    a declared vocabulary is counted verbatim and flagged rather than absorbed. Ten reconciliations
    compare numbers that are supposed to agree.
    **Three validation rounds, and the critical finding was the section's own defect class inside
    the mechanism built to detect it.** Seven of the nine reconciliations reported `held` having
    compared nothing: the predicates read with defaulting accessors, so an absent field equalled an
    absent field, and an empty population produced an empty failure list, which read as success. On
    a real drive the summary printed `9 reconciliations, 9 held, 0 failed` while one of them
    compared a field present on **0 of 1073 ticks** against a zero. The second critical was the same
    shape one level up: `## Sensing` rendered `0 HERE calls` with a causal explanation attached, on
    a log that never carried the field. And two axes describing the same 1,073 records disagreed by
    a factor of 1,073 -- one read the field that says whether telemetry arrived, the other counted a
    missing field and called it the same thing.
    **The finding worth carrying forward is about pinning, not about summaries.** The rule that a
    count must be measured rather than derived was pinned on the one axis that already obeyed it,
    and unpinned on the three that violated it. The pin was placed where it would pass. It resolved,
    it caught its mutation, and it read as coverage -- while the behaviour it named went unguarded
    three feet away. The consequence was concrete: a count derived by subtraction cannot disagree
    with its own census, so the census being wrong was invisible from inside the axis.
    **Experiment: three drives, one of them deliberately degraded, because a clean drive proves
    nothing for a task whose only claim is that a bad drive says so at the top.** A live 300-second
    run with an induced link outage, a run with no phone and the thermal sampler disabled, and the
    previous task's log evaluated on the backward-compatible path. **A reader of the degraded
    drive's summary alone would not know it went badly, and the reason is structural rather than a
    defect.** Ticks are produced by phone frames, so the 54.58-second outage destroyed `attempted`
    and `answered` together: at the drive's own 5.01 ticks per second it cost about 273 of the
    roughly 1,502 ticks that would have existed, and the summary reports `attempted = 1229` and
    calls every axis fully answered. **18.2 per cent of every denominator was missing and no axis
    could say so, because each axis is a ratio whose denominator the same event removed.** Its whole
    account of a dead 54-second stretch and a real redial is `no_telemetry 1` on two axes, above
    `10 reconciliations, 10 held`; searching the section for the words that would name the event
    matches only its heading. An axis of the form `answered of attempted` is blind by construction
    to an event that stops the attempting -- the vocabulary this section built protects the
    numerator, and nothing was watching the denominator. The drive also found eight places where
    the summary and the detail it summarises disagree, the sharpest being a `failures` axis reading
    `749 of 749 ticks answered` on a run where 20 of 30 sources were unreadable on every pass and
    the section below it said so 20 times; four `See ##` references to sections the same document
    did not render; and a zero attributed to shadow mode when the real cause, recorded two sections
    away, was a missing API key. **And it settled a disagreement in both directions.** A claim of
    mine that the provenance axis was wrong on the earlier log was itself wrong -- that log carries
    three non-substituted classes -- but the concern behind it was right: `ego_headway_s`,
    `target_lane_front_gap` and `uncongested_low_speed_flag` carry a non-excluded class on 100 per
    cent of ticks of all three drives, so the axis can never report zero in the field, because
    `derived` is assigned even when every input to the derivation is substituted. On the no-phone
    drive every input to those three was a fallback and the axis still read fully answered. The
    guard passes its unit test and cannot fire on this hardware.
    **Two corrections went the other way, and both were mine.** I reported the provenance axis as
    wrong to call a drive fully measured; the implementer asked for the evidence, and the drive in
    fact carried three non-substituted classes -- I had generalised from one field of thirty-nine.
    And in building a control I called the reconciliation entry point with its arguments reversed,
    producing nine uniform verdicts that looked exactly like a finding; the control I had built to
    fail, and which did not, is what caught it. That is the same hazard the plan's own decision to
    replace a six-tuple with a dataclass exists to prevent, met from the caller side where no type
    checker was watching.
    **Also worth keeping: an agent declined to add two pins it had been asked for**, on the ground
    that for an exact two-way partition the derived and counted forms are algebraically
    indistinguishable, so no input could tell them apart. Refusing to write a pin that would pass
    regardless is the exact opposite of the defect above, and the right call.
    **Open:** four §5.3 record fields remain specified and absent, one of which an open item's
    stated bound rests on. The plan was wrong in nine places, all recorded -- among them an identity
    that counts only episode opens where the code counts opens and closes, a cited symbol that does
    not exist, a worked example that cannot distinguish the mean it demonstrates from a plain
    average, and a §13 statement that the exit code does not change where its own D13 specifies
    a new one.

### What section G's validation loop could not establish

Seven audit rounds against an independent validator found nineteen defects that change a reported
number or produce a plausible-but-wrong result, each reproduced before being acted on. Twelve were
in the original instrumentation; the rest were introduced by fixes and caught by auditing the fixes
as new code. Every one of the twelve original findings has the same shape: **a record that cannot
distinguish a failure from a success.** A completeness check measured against a quantity the outage
destroys on both sides. A dead writer reported as healthy. A recovery inferred from silence. An
84-second camera blackout recorded as 21 recoveries.

The items below are not open defects. They are claims the loop could not test, and each bounds
something the instrumentation now asserts. Further audit rounds do not shrink them.

- **Every drive in evidence is a shadow drive** (`ever_live: False` on all five), so the tick cadence
  is constant for the whole run. The coverage and time-weighting fixes are specifically about a
  cadence that changes — `IDLE_RATES` 1.0 Hz against `ACTIVE_RATES` 5.0, and 0.15 Hz under worst-case
  thermal backoff, a 33.3× spread nothing has produced. Those fixes were found and verified against
  hand-built live-mode fixtures and have never met a real adaptive cadence. **Task 44 is the input
  that closes this, and it is worth more than another audit round.**
- **The perception path has never processed a vehicle**: zero vehicle sightings and exactly one
  detection across all 4,151 ticks of the five drives. Every perception-derived field is the fallback
  path on every tick, which is why the `target_lane_front_gap` misclassification was uniform across
  the corpus rather than intermittent. `perception/distance.py` and `sim_contract`'s encoder sit
  downstream of a path that has never produced an input; audit them **before** the first drive with
  traffic, not after.
- **`_score_candidate` and `vs_incumbent` are entirely unexercised** — `candidates: {}` on every
  drive, and the code runs only when `--candidate` is passed, which nothing does. It cannot report a
  wrong number because it reports none. Audit it before the first candidate policy is scored.
- **Per-notification timestamps are not in the artifact.** A source recording N episodes gives no
  direct way to see the stream behind them. On `run_20260902_143427` the true structure was
  recoverable only because it was perfectly regular; both the validator and I misread that drive from
  the summary alone before the event records settled it.
- **A2's zone census cannot be confirmed against any existing drive**, and neither can the
  `missingness` reclassification: `eval_run` reads `obs_diagnostics.missingness` from the tick
  records, so both take effect on the next capture rather than on re-analysis of an old log.

### Section G validation loop — resumed and closed out 2026-09-03

**The blocking unknown: no mutation verdict produced in this loop is trustworthy yet.**

`scripts/remutate.py`'s `run(kind)` calls `subprocess.run(...)` on the Python arm and **discards the
result** — only the Gradle arm assigns it — and `failing_tests` swallows `ElementTree.ParseError`
with a bare `continue`. So an empty name list means both "no test failed" and "no test ran", and the
caller prints `*** SURVIVED ***` for both. Confirmed by direct reproduction: **no XML at all**
(pytest never started — bad flag, crash, OOM, conftest import failure), **XML truncated mid-write**
(killed run, full disk), and **zero tests collected** (the partial-tree-copy shape) each yield `[]`
and print SURVIVED, indistinguishable from a genuine survival.

This is section G's own recurring defect — a record that cannot distinguish a failure from a
success — living inside the instrument that certifies section G. `failing_tests`' own docstring
states the principle it violates: *"A harness that reports a false SURVIVED is the same failure as
one that reports a false CAUGHT."* `run()`'s docstring says naming the failing test settles the
ambiguity; it settles a false CAUGHT, and the false-SURVIVED asymmetry was never closed.

**How to settle it.** Three changes, then a re-run: capture the returncode on the Python arm (pytest
0 and 1 are the only valid inputs to a verdict — 2/3/4/5 are a third state, print `INCONCLUSIVE`,
and count it in `survived` so the exit code still fails); treat a missing or unparseable XML as
inconclusive rather than zero failures; cross-check the collected testcase count against the
expected baseline, which catches the partial-tree case independently.

**STATUS: settled and landed** in `e6fed93`. Re-running the loop's fifteen affected pins against it
gives **all fifteen CAUGHT, `survived: 0`**, so no verdict this loop relied on was a false clean.

**It proved itself on first use, which is the part worth keeping.** The first re-run returned
**13 of 15 `INCONCLUSIVE`** rather than CAUGHT. That was not a harness bug: the collected-count check
correctly detected that the tree under test no longer matched its baseline, because a test had been
added to `deployment/jetson/tests/` concurrently and moved the real count from 2059 to 2060 mid-run.
Under the old harness that same condition would have printed `SURVIVED` — a false clean, silently.
The check caught, on its first outing, an instance of the exact failure it was written for.

**One maintenance burden it introduces:** `EXPECTED_PYTHON_TESTCASES` is a hardcoded 2060 and goes
stale the moment a test is added, turning every pin `INCONCLUSIVE` until someone updates it. That
failure is loud and self-explaining, which is the right direction, but it should probably derive the
baseline from a clean run rather than a literal.

**Where the work stopped.** Branch `main`, HEAD `902ee08` (this commit), **27 commits ahead of origin
and nothing pushed**; baseline for the loop was `c8ef736`. Suite **2059 passed**; registry **372
anchors, all resolving exactly once**.

**A20 and the `feed_derived` reversal landed in `f307c72`.** `segment_target_speed`,
`nearby_av_mean_speed` and `nearby_av_density` are all `SOURCE_DERIVED` and agree with each other;
`nearby_av_count`, a direct count of receptions, stays `measured` and carries what evidence a peers
tick has. A test asserts no field in `field_sources` ever carries a `feed`-family class. `SOURCE_FEED`
stays reserved for the traffic feed, which `feed_fusion.py:26` and `observation_builder.py:350` both
document as owning no observation field.

**Two operational hazards this loop hit repeatedly, worth knowing before touching this code again.**
`remutate.py` edits files in place, so a `git status` showing one modified file during a run is a
live mutant, not lost work — check for the `.remutate-restore` sidecar, whose first line names the
file, before reaching for anything. Never `git checkout --`, `git stash` or `git add -A` while it
runs: the first two destroy uncommitted work, which has happened in this project three times in one
session, and the third commits the mutant. And any stray `.py` under the repo root becomes a test
case via `test_no_undefined_names.py`'s `REPO.rglob("*.py")`, changing the suite count and failing
the run if it has an undefined name.

**Still outstanding:** one comment-trimming pass over the comments this loop added. Seven rounds of
fixes left narration that argues with a reviewer who was never there — finding identifiers like
`A20:`, sentences beginning "this test then briefly asserted", and the history of classes a field
used to carry. The code is correct; the comments describe how it got that way rather than what it
does.

**Traps already mapped, do not re-derive.** `remutate.py` edits files in place: a `git status` showing
one modified file during a run is a live mutant, not lost work — check for the `.remutate-restore`
sidecar before touching anything, and never `git add -A` while it runs. Any stray `.py` under the
repo root becomes a test case via `test_no_undefined_names.py`'s `REPO.rglob("*.py")` and fails the
suite if it has an undefined name. `eval_run.py` reads `obs_diagnostics.missingness` from the tick
records, so provenance fixes cannot be verified by re-analysing an old drive — verify against the
builder. Five real drives and every fixture built during the loop are in the session scratchpad,
which is disposable; they are re-fetchable from `jetson:/home/edge/dsrc_logs/`.

## H. Colocation and integration — **[COLOCATED]**

Everything above is done before the devices meet.

40. ~~**[COLOCATED]** USB transport backend behind the same interface, swapped in
    for the network backend.~~ **DONE 2026-09-05** — `transport/usb.py`. `UsbAcceptor`
    composes a loopback `TcpAcceptor` with an `adb reverse` lifecycle, so no new
    `ByteConnection` was needed: `adb reverse` leaves ordinary TCP at both ends.
    Registered as the `usb` backend and acceptor in both conformance suites, gated on
    an attached serial.

    **Note the spec is wrong and was not edited.** `specs/transport_protocol.md:21`
    and section D's preamble both say the USB path is `adb forward`. It is `adb
    reverse` — `forward` inverts the direction the same paragraph mandates, and the
    only production socket construction in `phone/` is a dial. The spec is frozen, so
    the discrepancy is recorded rather than fixed.

    Two defects only real hardware produced: `adb reverse --list`'s first column is
    not the serial, so the original filter never matched and every timeout falsely
    re-established the mapping; and `verify()` compared only the device port,
    discarding the local one, so a mapping pointing at the wrong local port read as
    healthy. Across three 180 s drives: `reverses_reestablished` 0,
    `reverse_reestablish_failures` 0, `reverses_swept` 0, and `adb reverse --list`
    empty before and after every run.
41. ~~**[COLOCATED]** `adb` first connection: accept the RSA authorization on the
    phone screen, tick "always allow".~~ **DONE 2026-09-04** — `ZY227VV4XC` on
    `usb:1-2.2`, reporting `device`; `adb shell getprop` returns `moto g power`
    running Android 11, so the link carries commands rather than merely
    enumerating. Survives an `adb kill-server` / `start-server` cycle. The
    Jetson's identity key is `/home/edge/.android/adbkey`; no `.pub` file exists,
    which is normal — adb derives it on demand.

    **The port was the whole difficulty, and it is worth recording because the
    symptom is silent.** The phone spent some time plugged into the Orin's USB-C
    port, where it never appeared on the bus at all: no error, no partial
    enumeration, nothing in `lsusb`. That port is registered as a USB *device*
    controller (`/sys/class/udc/3550000.usb`, the `tegra-xudc` driver backing the
    `l4tbr0` gadget interface), so the Jetson presents itself as a device on it
    and can never act as host. Two devices both waiting to be enumerated produce
    no diagnostic. **The phone must go in a USB-A port**, or a hub hanging off
    one — the u-blox GPS and the Bluetooth radio are on the same USB 2.0 hub.

    **"Always allow from this computer" was ticked** (Ankit, at the phone). It is
    not independently checkable from this side: the device has no root
    (`su: inaccessible`), so `/data/misc/adb/adb_keys` cannot be read. The
    empirical confirmation, if it is ever wanted, is a replug — the USB device
    number increments and the grant should hold with no new prompt. Worth knowing
    because a session-scoped grant would only reveal itself at the next reboot,
    and the next reboot is likely to be in the car.
42. ~~**[COLOCATED]** Bench loopback over USB with both devices on a desk;
    end-to-end latency measured against the 200 ms target.~~ **DONE 2026-09-05** —
    **target met in all three runs.** `e2e_ms` p95 **90.72 / 139.52 / 111.52 ms**
    against 200 ms, pooled 116.19 over 2,684 ticks; converted-only differs by at most
    0.12 ms within a run. The tailnet baseline **missed** the target at 215.63 ms, so
    USB removed roughly 99 ms of p95. Recomputed from the raw logs independently of
    the campaign's own report.

    **Two bounds on that claim.** `link_ms` p50 36.83 ms against a conversion bound of
    1.97 ms is resolved by ~18×, but `transport` p50 sits *at* its bound — exceeding
    it by 0.02 and 0.30 ms in two runs and falling 0.03 ms below it in the third. The
    USB wire hop is of order 2 ms and the instrument cannot place it more precisely;
    on the tailnet the same stage cleared its bound by 17.64 ms. So the reduction is
    established and the wire hop's magnitude is not.

    The three runs disagree by 48.80 ms at p95, and 46.40 ms of that (95.1%) is
    `enqueue_to_wire`, the phone-side send-queue interval measured on the phone's
    clock at both ends; no other stage moves more than 1.97 ms between any pair, and
    `phone.dropped` tracks it 0/3/17. **The spread is phone-side queueing, not the
    link.** `link_ms` is now partitioned by `timebase.source` and never pooled, which
    corrected the baseline's own figure from 185.38 to 185.78 ms.
43. ~~**[COLOCATED]** Shadow-mode correctness: logged shadow decisions match what
    live gating produces on the same input.~~ **DONE 2026-09-05** —
    `check_shadow_commands.py` replays the incumbent and compares shadow against live
    commands **decoded through the real `rate_cmd` wire codec**, not through the
    in-process objects. Three drives, 899/900/885 ticks: command-replay mismatches 0,
    logged `sensing.shadow` correct, and the phone's own applier counters
    `applied == 0` with `shadowed == commands_sent` (37/38/38).

    The check itself carried this section's signature defect and it was caught in
    validation: when the phone-side half *could not run*, it printed
    `phone_applier ok=False` and exited **0**. Three real causes reach that branch.
    Fixed, and the fix is verified in both directions — a check that cannot run now
    exits 2, one that ran and passed exits 0, one never requested exits 0.

    Its logcat scoping was also wrong twice over: unscoped, so after three runs the
    last match won and run 3's counters were compared against run 1's `rate_cmd.sent`;
    then scoped to a window built on the *Jetson's* clock and matched against the
    *phone's*, which runs ~0.93 s ahead. The offset is now measured at handshake,
    recorded per session, and the window refuses when sessions disagree by more than
    the margin.
44. **[COLOCATED]** Live-mode verification: flip the flag and confirm gating
    genuinely changes sampling rates, the loop still closes, and the advisory
    remains sane. Shadow mode is not evidence that live mode works.

    **PARTIAL 2026-09-05, bench.** `run_20260905_142351`, 900 s with the phone on
    USB and `--live-rates`: `sensing.shadow` is `False` on all 3,486 ticks, the
    first live-mode run in the project.

    **The actuation link is closed.** Every rate command the controller issued was
    applied. Eleven segments and ten transitions, delivered frame rate measured on
    the Jetson against the rate commanded on the same tick:

    | commanded | segments | delivered |
    |---|---|---|
    | 5.0 Hz | 385.4 s, 3.6 s, 5.8 s, 0.6 s, 0.8 s | 4.989, 4.990, 5.003, 4.950, 5.031 Hz |
    | 3.0 Hz | 89.0 s, 4.7 s, 12.0 s, 0.6 s, 393.0 s | 3.001, 2.989, 2.993, 3.086, 3.000 Hz |

    `sends_by_reason` records `changed` exactly 10 times against 171 heartbeats;
    the phone's own log encoded 3,487 frames and the Jetson received 3,486; zero
    frames dropped on the Jetson and four on the phone's send queue. Before this
    run every drive reported `applied == 0`.

    The stimulus was the phone's own temperature, not an injected value. Skin
    (`xo_therm`) rose from 28.857 C to a maximum of 42.513 C with p50 40.201, and
    the platform `thermal_status` stayed `nominal` on all 900 reports, so the
    `SKIN_WARM_C` path is what carried it. Jetson latency mean 31.6 ms, p95
    33.0 ms; USB counters all zero and no leaked reverse mapping.

    **What this run could not do, by construction.** The level was `active` on
    3,485 of 3,486 ticks, because `advisory_margin_narrow` fires on every tick when
    the scene is empty. The dwell, hold and bridge transitions are therefore still
    unexercised, and so is the whole of the advisory-sanity half: a stationary
    camera indoors gives the policy nothing to be sane about. Those need traffic
    and belong with tasks 50 and 51. `SKIN_HOT_C` at 45.0 C was not reached
    (max 42.513), and GPS and HERE both ran at 0.0 Hz, HERE because the installed
    APK reports `here.unconfigured` -- no key in `local.properties`.

    See task 57 for the defect this run found.
45. **[COLOCATED]** Thermal soak: sustained maximum-rate run to steady state;
    confirm the phone stays within limits and the controller backs off.
46. **[COLOCATED]** Failure injection: revoke GPS, kill HERE, unplug the link,
    drop the tether, and confirm each degraded mode behaves as specified.

    The tether is the likeliest network failure in the car and the newest, because
    it depends on a second handset's battery, thermal state, hotspot idle timeout
    and incoming calls. It is worth injecting separately from "kill HERE" because
    the phone cannot tell them apart: a HERE call with no route reports
    `status = 0`, the same value a DNS failure and an unreachable HERE produce.
    What distinguishes them in the record is `network_transport` moving to
    `no_active_network` or to a fallback, so the injection is also the check that
    that field reports what it is supposed to.
47. ~~**[COLOCATED]** Sim-contract parity: the observation vector produced live
    matches the simulator's sensing model field for field.~~ **DONE 2026-09-05**, and
    **the task as worded is false** — which is the result. "Matches field for field"
    does not hold for 31 of the 39 slots, and cannot: six have no rear sensor on the
    device at all. `src/analysis/observation_parity.py` is the first module importing
    both sensing models; before it, `test_sim_contract.py` fed one hand-written dict
    to two *encoders* and compared their output.

    The ledger over 7 real scenes: **8 identical, 18 approximated, 7 substituted, 6
    structurally absent.** Every substituted or absent slot carries a provenance class
    inside the `SUBSTITUTED` partition on every scene. Checked against all three
    campaign drives: 0 re-encode mismatches, 0 constant mismatches over 2,684 ticks.

    So the deliverable is a **parity ledger with a per-slot claim**, not a
    field-for-field equality: it says which slots agree, which are approximations and
    by what, and which the device cannot produce.
48. **[COLOCATED]** In-car install: 12 V power for both devices, mounts, cable
    routing.

## I. Measurement drives — **[COLOCATED]**

49. Shakedown drive: short and local, purely to confirm the system records
    readable, aligned data.
50. Drive set 1, shadow mode at maximum rate: the full-rate reference plus every
    candidate policy's decisions against identical traffic.
51. Drive set 2, live mode: the controller gating for real, verifying the
    shadow-mode predictions held.
52. ~~Repeat across congested and free-flow conditions on at least three separate
    days, on a corridor known to congest.~~ **DROPPED 2026-09-05** — the drives will
    happen once. This is not a deferral: there is no later occasion on which the
    repeat could be run. It is struck rather than deleted because it bounds what
    section I's numbers can support.

    **What a single drive cannot separate.** Condition, day and run are confounded
    in one sample. Of the quantities listed below, end-to-end latency and thermal
    behavior depend mainly on load and are the least affected. Advisory bin
    distribution and churn, how often the camera changes the advisory, safety-layer
    intervention counts, and HERE-reported speed against experienced speed all
    depend on the traffic condition, and each gets exactly one observation. A
    statement of the form "the advisory changed N times per hour" then describes
    that drive; it is not an estimate for the corridor and it carries no interval.

    **The run-to-run term is not small enough to ignore.** The only repeat-measure
    evidence in the project is task 42's three bench runs of an identical 180 s
    configuration: the largest and smallest of their three `e2e_ms` p95 values
    differ by 48.80 ms, and 95.1% of that difference is the phone-side send-queue
    stage. That is a stationary bench with no traffic and no thermal excursion, so
    it is a lower bound on how far two drives would differ, and there will be no
    second drive against which to measure it.

    **The one contrast still available is inside the drive, not across days.**
    Choose the corridor and the departure time so the route passes through a
    congested segment and a free-flow segment, and record the segment boundaries so
    the two can be separated afterwards. That holds day, device and thermal state
    constant, which the three-day design did not, and it confounds condition with
    location, which the three-day design did not either. It is weaker than the
    dropped task and it is not nothing. Tasks 50 and 51 now carry it.

Connectivity on every drive is the tethered configuration described under
Architecture. `network_transport` in each telemetry report, and the run tally in
`summary.json`, record which network each drive actually ran on, a mid-drive change
included; a drive whose reports name no network is not assumed to have had one.

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
56. The suite does not pass on the Jetson, and one assertion is why.
    `test_score_shadow.py:660` compares a computed float mean for exact dict
    equality: `rw["achieved_mean"] == {"camera_hz": 4.97, ..., "imu_hz": 49.8, ...}`.
    `sum([49.8]*8)/8` is exactly `49.8` on the Mac (arm64, CPython 3.12) and
    `49.800000000000004` on the Jetson (aarch64, CPython 3.10), so the assertion
    fails there and nowhere else.

    **The architecture is not the defect.** Comparing a float mean for exact
    equality is wrong on every machine; the Mac passing is luck about summation
    order, not evidence. `pytest.approx` is the fix.

    It is filed here rather than as a test-quality nit because of what it costs:
    the suite cannot go green on the box where the runs happen, so nothing routine
    runs it there, and every future audit re-establishes "1 failure, pre-existing"
    by hand — a cost already paid twice. Verified pre-existing: zero commits in the
    tasks 40/42/43/47 loop touched `test_score_shadow.py` or `score_shadow.py`, and
    the failure reproduces at that loop's baseline `a61458d` on the Jetson.

## Blockers

None outstanding. All three items below are struck; they are kept so a reader can
see what was cleared and on what evidence.

- ~~**HERE API key** — blocks task 21 only. Everything else in E proceeds.~~
  **CLEARED** — the key is shared with Nash production, and task 21 is done. It
  never reaches `request_url`, so it is on no wire and in no artifact.
- ~~**Physical colocation** — blocks section H onward. Nothing before it.~~
  **CLEARED 2026-09-05** — the phone is USB-attached to the Jetson and tasks 40, 41,
  42, 43 and 47 all ran on the attached pair. What tasks 44 to 48 still need is not
  this blocker: 44 needs the live flag flipped, 45 a sustained maximum-rate run, 46 a
  hand present for one injection, and 48 vehicle installation hardware, meaning a
  12 V supply and mounts, which no item on this list has ever named.
- ~~Repository layout: phone app in `dsrc/android/`, Jetson runtime extends
  `deployment/jetson/`.~~ Never a blocker, and half of it is wrong: there is no
  `android/` directory at any level of this repository. The phone app is `phone/`, a
  Gradle project whose sources are rooted at
  `phone/app/src/main/kotlin/com/dsrc/phone`. The Jetson half is correct.

## K. Found in passing

57. **Thermal backoff has no hysteresis, and a phone on the threshold rebinds its
    camera repeatedly.** A section F runtime defect, numbered here so nothing above
    is renumbered.

    `_thermal_scale` compares `skin_temp_c >= SKIN_WARM_C` directly, with no dead
    band, no dwell and no hold. Measured on `run_20260905_142351`: between t=476.7 s
    and t=506.7 s the skin reading sat on 40.0 C and the commanded camera rate
    changed **eight times in 30 seconds** -- 5.0, 3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 3.0 --
    on readings of 39.991, 40.001, 39.994, 40.027, 39.998, 40.005 and 39.991 C. Two
    of those segments lasted 0.6 s. Each change is a real camera rebind on the
    handset, because in live mode the phone applies what it is told.

    **The evidence path already solves this and the thermal path was never given the
    same guard.** `RAISE_DWELL_S` and `HOLD_S` exist for exactly this, and
    `decide`'s own comment states the failure they prevent: "With a signal straddling
    the threshold it produced a camera rebind per tick, which is the exact thrash the
    dwell and the hold exist to prevent." The thermal multiplier bypasses both.

    **It was invisible in shadow mode by construction.** The command changed, nothing
    applied it, and the cost of a rebind falls on the phone. Every earlier drive
    recorded the same chatter as a sequence of decisions and could not have shown it
    as a cost. This is the defect that justified running task 44 on a bench rather
    than first discovering it on a drive.

    Not yet fixed. A dead band on the skin thresholds is the obvious remedy, but the
    choice between a dead band, a dwell, and reusing `HOLD_S` is a design decision
    that should be taken deliberately: backing off late and recovering late are not
    symmetric costs when the reason for backing off is heat.
