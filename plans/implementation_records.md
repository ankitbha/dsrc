# Implementation records

What was built, in the order it was built, with what each step found. Renamed from
`task_list.md` on 2026-09-12 and split: the work the paper rests on stayed here, and the
project's superseded formulation moved to `detours.md` without being deleted.

**If you are writing the paper, `paper_deploying_self_regulating_cars.md` is the argument
and tells you which file to open for each section. This is the evidence underneath it.**

What moved out, and why, so nothing reads as missing:

* **Section C**, the MAPPO replication on `inverted_tree`. Superseded by task 141; the
  flow-level half is now the SRC port on Mainz, in `mainz_src_port.md`.
* **Section K items 77 to 140**, that leg's findings. Items 87 and 141 stayed, because
  the paper cites them: 87 is why the simulated gate and the deployed gate are not the
  same rule, and 141 is the supersession itself.
* **The ordering correction and items 72 to 76**, which sequenced that leg's training.
* **The paused state of 2026-09-09**, which is where that leg stopped.

One caveat for a reader following a citation. The per-task plans in this directory cite
this file by line number -- `implementation_records.md:1143-1144` and so on -- and those
numbers were taken before the split. The passages still exist, here or in `detours.md`,
but not at those offsets. Search for the text rather than jumping to the line.


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

## The paper

**The contribution is the deployed system: the phone-plus-Jetson advisory rig, the
sampling controller, and the safety and etiquette filters.** Those are what this
project built and what the paper argues for. `src/safety/` holds the filters
(`safety_layer.py`, `etiquette.py`, `constraints.py`); sections D through I hold the
system and the evidence it runs on a road.

**The paper's three sections are design, experience and experiments**, and the venue is
ICRA. `plans/paper_deploying_self_regulating_cars.md` carries the argument and the
placement of every result.

**The road-network-level half CHANGED on 2026-09-11 and section C no longer describes
it.** It was the MAPPO replication on `inverted_tree`; it is now the SRC controller
ported to SUMO on Mainz, with the observation restricted to what a traffic API returns.
Held-out seeds 16-30, paired on seed: the traffic-API arm gains 230 +/- 44 veh/h over no
control and SRC's original six features gain 224 +/- 61, and the two differ by 0.2%,
inside both bars. The plan is `plans/mainz_src_port.md` and the supersession is recorded
as task 141.

**One instrumented vehicle can never demonstrate throughput or delay.** That is why
the flow-level half comes from simulation and the drives are never asked to support
it. The drives support the deployment claims and calibrate the sensing model.

## Critical path

**Six items, in order. Everything else in this list is either done or deliberately
off the path.** Nothing here needs hardware, another drive, or a decision from
outside the project.

1. ~~**Task 67** — fix the 40% episode-truncation rate.~~ **DONE 2026-09-09.**
   The cause was not the controllers: `highway_env`'s `IDMVehicle` follows only
   within its own lane, and every node here funnels lanes, so human traffic drove
   through itself. Measured on 81 **distinct** runs against a same-run plain-IDM
   control: completion 43/81 to 57/81, +17pp; collisions 155 to 126, −19%.
   `no_av` could never have shown the problem, because `terminated` checks AV
   crashes and a `no_av` run has none.
2. ~~**Task 63** — fix the 90-degree frame rotation.~~ **DONE 2026-09-09.**
   Rotation applied at the decode boundary; intrinsics moved with it
   (`cx_px` 640→360, `cy_px` 360→640); `horizon_y_px` measured at 717, not the
   old landscape centre of 360; `hood_line_y_px` set to 1022. The offline replay
   derives its own rotation from each run's recorded config.
3. ~~**Task 9** — fill the five sensing-model parameters.~~ **OFF THE PATH 2026-09-12.**
   The parameters define `LocalObservationBuilder`'s input distribution for the
   local-sensing formulation, and that formulation left the paper when the flow-level
   half changed. Kept because the measurements behind it are real: `latency_s` and
   `queue_speed_mps` were settled from the drives against `latency_s: 0.0` in every
   config, a 96.7 ms median. Nothing in the paper now reads them -- the SRC port queries
   HERE at a 60 s decision interval and models no observation latency.
4. ~~**Task 68** — train MAPPO on `inverted_tree`.~~ **SUPERSEDED 2026-09-11**, task 141.
5. ~~**Task 69** — evaluate trained MAPPO against `no_av` for throughput.~~
   **SUPERSEDED 2026-09-11**, task 141. The flow-level result is the Mainz SRC port.
6. **Score the six shadow runs.** `deployment/jetson/score_shadow.py` replays the logged
   per-tick inputs and gates on the incumbent reproducing byte-for-byte. Until it runs,
   whether the shadow-mode predictions held in live mode is unanswered, and that is the
   comparison the two live drives were collected for. No new driving, no new decisions.
7. ~~**Run the HERE-observation policy on the rig.**~~ **RESTATED 2026-09-12 as task 145.**
   Its old wording -- "what remains is loading that policy" -- was wrong: `SrcQNetwork`
   reads the whole network's state and its weights are Mainz's, so there is no Mainz
   observation in the drive corpus and no policy for the roads that were driven.
8. **Measure the gate's firing rate**: how often local safety clamps an advisory and what
   it costs. Folded into task 144, which has to make the gate run before its rate can be
   measured.
9. **Section L, tasks 142 to 145.** Four claims the paper plan makes that the repository
   contradicts, found by reading the two against each other on 2026-09-12. 142 is the only
   one that touches a number the paper quotes.
10. Write the paper.

**Tasks 67 and 63 are independent of each other**, so their relative order is free.

**Scope boundary.** The drives are finished and will not be repeated; the corpus is
recorded in section I. The simulation half is one network at one demand deliberately:
SRC was already simulated on this topology, at a different demand, in a different
environment, and across the two papers the coverage is PTV Vissim against SUMO,
Wiedemann-99 against EIDM with a one-second reaction, and the published 18,000 veh/h
surge against a sustained 4,500. A sweep here would re-establish what the SRC paper
did.

**Open decisions.** Whether the rotation fix belongs on the phone or the Jetson
(task 63) — the camera intrinsics assume landscape, so it is not purely cosmetic
either way. Everything else on the path has a defensible default.

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

    **ADVANCED 2026-09-08 on the road, and still open.** Two live drives, `run_20260908_161422` (2,555
    ticks, 44.4 min span, 26.6 min moving) and `run_20260908_170849` (2,426 ticks,
    27.1 min, 20.9 min moving): 4,981 live ticks over 40.7 to 42.4 km. The loop
    closed throughout and the advisory stayed populated at highway speed.

    **What is still not verified is the clause the task is named for.** That the
    loop closes and the advisory is sane are both now shown on the road. That live
    gating *genuinely changes sampling rates* was not measured: it needs the
    achieved-rate series compared between a live and a shadow run under comparable
    conditions, which the recorded data supports and nobody has run.

    **One defect found, and it is not fixed.** Selecting live mode on a redial is a
    silent no-op: `run_demo.py` reads the mode once at startup, so a phone
    reconnecting to a running `run_demo` rejoins in the mode the process already
    has. On 2026-09-08 a live-mode tap was absorbed into a running shadow session
    and the drive continued in shadow. The operational workaround is to restart
    `dsrc-drive` between modes; the code fix is unwritten.

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
    (max 42.513), and GPS and HERE both ran at 0.0 Hz.

    **The missing key was only one of two reasons HERE was inert, and correcting
    this is the point of the paragraph.** The key is now configured, the rebuilt APK
    is installed and `HttpHereClient` constructs without complaint -- and the Jetson
    still sends `here=false` on every rate command, because `_here_query` returns
    `None` without a usable position: "a query centred on a position we do not have
    describes nowhere, and the phone would spend a cellular call on it." GPS itself
    is not a defect. `ACCESS_FINE_LOCATION` is granted, `location_mode` is 3, and
    `dumpsys location` shows `com.dsrc.phone` holding a GPS request at a maximum
    interval of 1 s, last active during the run; the app asked correctly and the
    receiver returned no fix, which is what a desk indoors produces.

    So **HERE and GPS are one blocker and it is physical**: the handset needs sky.
    No amount of configuration exercises the HERE path at this desk, and a reader who
    adds a key and expects HERE frames will be misled by the sentence this replaced.

    See task 57 for the defect this run found.
45. **[COLOCATED]** Thermal soak: sustained maximum-rate run to steady state;
    confirm the phone stays within limits and the controller backs off.

    **PARTIAL 2026-09-05. Open, and the soak is deliberately not scheduled** --
    recorded here so the next reader does not repeat the analysis to find out which
    of the four clauses already have answers.

    **"The controller backs off" is settled, on hardware.** Two 900 s live runs:
    the thermal rule fired on 1,504 of 3,486 ticks and 1,561 of 3,451, cause
    `skin_warm`, and in live mode the phone applied the result -- delivered frame
    rate 3.000 and 2.998 Hz against a commanded 3.0. The decision function reads
    only device temperature, never ambient, so this clause does not depend on the
    environment and a car run would add nothing to it.

    **"To steady state" is answered in the negative, and measured.** Both runs ended
    because their clock expired, not because the handset equilibrated. The skin
    slope over the last two minutes was **+0.368 and +0.367 C/min**, and it stops
    falling after about minute five: +0.330 then +0.363 C/min over minutes 5-10 and
    10-15 in the first run, +0.452 then +0.398 in the second. It is climbing close to
    linearly. Extrapolated, `SKIN_HOT_C` at 45 C is about six minutes past where both
    runs stopped.

    **"Sustained maximum rate" was not run, in two ways.** The camera held 5.0 Hz for
    roughly the first 380 s and the loop then cut it to 3.0 Hz, so maximum rate and
    an engaged backoff cannot both hold -- the task's own wording is in tension here.
    And `gps_hz` and `here_hz` were 0.0 throughout both runs, so two of the four
    modalities carried no load at all, which are exactly the two `THERMAL_SCALED_KEYS`
    scales.

    **"Stays within limits" holds only as a statement about a desk.** The platform
    `thermal_status` was `nominal` on every one of roughly 2,700 telemetry reports
    across every run to date, at a maximum skin of 43.087 C, and no thermal shutdown
    occurred. The `skin_hot`/`severe` tier and every non-`nominal` status tier are
    unit-tested and have never executed on hardware.

    **Superseded in the field on 2026-09-08, and the answer is worse than a lower
    rate.** On the first drive the handset reached `Thermal Status: 3` (`severe`)
    with `xo-therm` at **51.6 C**, `modem-skin` 50.0 C and the GPU 50.8 C, mounted
    on a windscreen in sunlight. The backoff behaved correctly -- scale 0.3, camera
    commanded 1.5 Hz -- but that is not what ended the drive. **Android's own
    throttling of the app stalled the pipeline and closed the session**: the final
    tick recorded `link_ms` 135,359 against a p50 of 52 ms, and the app hung up.

    So the thermal work protects the commanded rates, and the thing that actually
    ends a daylight drive is the platform throttling the process, which nothing in
    this system detects or reports. No gate looks at `link_ms`. That bears directly
    on whether tasks 50 and 51 are runnable in daylight without shading or active
    cooling.

    One measurement worth keeping: it was a single tick, not a decline. Every tick to
    1311 had a normal link time, so the session died sharply rather than degrading --
    the drive's 279 s and 6.24 km are usable, and only the last tick is not.

    **One thing the desk actively hides, noted and accepted.** No policy module reads
    the Jetson's own temperature: `sensors/thermal.py` and the metadata logger record
    it, nothing consumes it, and there is no Jetson-side backoff. On both runs it sat
    at 53.7-55.6 C, a 1.4 C spread over fifteen minutes, which is equilibrium -- it
    has a heatsink and airflow. A closed car has neither, and no code path would
    respond. Raised and set aside deliberately.

    **What would close it**, if it is ever wanted: one two-hour run from ambient with
    the handset at a window so GPS and HERE carry their load, the pass condition
    written down first, and steady state judged in analysis as a trailing five-minute
    slope under 0.05 C/min rather than by terminating the run, so that "did not
    settle" stays a reportable outcome.
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

    **Expectations, written and committed before the injections were run** -- the
    commit order is the point. "Behaves as specified" has no spec for three of the
    four: `specs/` carries no failure or degradation document, and the nearest thing
    is the 30-source closed vocabulary in `logio/failure_log.py`. Without a written
    expectation every injection passes, because something always appears in the log
    and nobody can say it was the wrong thing.

    | injection | how | expected | must NOT change |
    |---|---|---|---|
    | link drop | `adb reverse --remove tcp:47811`, restored 30 s later | `link.down` opens with reason `no_session`, plus `link.session_end` and `link.sends_lost`; the phone redials and fails; on restore a second session is accepted | the phone keeps capturing, so `phone.dropped` grows on `camera` |
    | network drop | `adb shell svc wifi disable`, restored 30 s later | `network_transport` leaves `wifi+vpn`; with no data SIM in this handset, `network_transport_absent` should read `no_active_network`, one of the five values `specs/transport_protocol.md:434` allows | **the USB link**, because `adb reverse` runs over USB and must survive wifi going down -- this separation is the whole discriminator |
    | GPS revoke | `pm revoke ACCESS_FINE_LOCATION`, granted back afterwards | one of three, and which one is the result: the platform kills the app, so the link dies; or `requestLocationUpdates` raises `SecurityException`; or it degrades with a record distinguishable from baseline | -- |
    | kill HERE | not injectable | **skipped, and not for want of trying.** HERE is never called without a position, so `here.refused`, `here.reader_failures` and `phone.here_errors` cannot fire. Nothing to kill | -- |

    **Two predictions worth stating before the fact.**

    The link-drop expectation may be wrong in an interesting way: removing an `adb
    reverse` mapping closes the device's listening socket, and an already-established
    TCP session may simply continue. If it does, that is the finding -- the
    `reverses_reestablished` recovery path guards against something other than "the
    link went away", and it has read 0 on every run ever done, along with
    `reverse_reestablish_failures` and `reverses_swept`.

    The GPS revoke should raise. `GpsLocationSource.request()` calls
    `requestLocationUpdates` under `@SuppressLint("MissingPermission")` with no
    try/catch, and there is no catch on the path `ConfigApplier.apply()` ->
    `setGpsRate` -> `locations.setRate` -> `request`, up to the bare
    `applier.apply(command)` at `SensingService.kt:683`. **That path became reachable
    only when live mode was turned on**: in shadow mode `apply()` returns before
    touching any target. Predicted from reading, not reproduced, which is what this
    injection is for.

    Also expected to be indistinguishable, and recorded so the null is legible:
    `gps.not_fresh` with reason `absent` already opens at tick 0 of every indoor run,
    so the revoked state and the baseline no-fix state may produce the same record.

    **PARTIAL 2026-09-05.** Three of four injected in one 660 s live run,
    `run_20260905_181058`, each stamped against the same unix clock the tick stream
    carries. Two of the three expectations above were wrong, which is the argument
    for having written them down.

    **Link drop: the expectation was wrong and the flagged alternative was right.**
    Removing the mapping at t=120.0 s and restoring it at t=150.1 s produced **no
    `link.down`, no `link.session_end`, no `link.sends_lost`, and zero tick gaps
    over 2 s** across the whole run. `adb reverse` governs new connections; an
    established TCP session is untouched by removing it. So this injection does not
    simulate unplugging the cable, and the real unplug -- which also takes away adb
    and charging -- remains untested and needs a hand.

    **It did, however, fire task 40's recovery path for the first time.**
    `reverses_reestablished` read **1** against `reverse_reestablish_failures` 0,
    where every previous run in the project read 0 and 0. The manual restore used a
    raw `adb reverse` add, which does not touch that counter, so the increment is
    the runtime's own `verify` loop noticing the mapping gone and putting it back
    inside the 30 s window. It cannot be timed more precisely than that.

    **Network drop: the field reported the change, at a different value than
    expected.** `network_transport` moved off `wifi+vpn` for **56 of 384 telemetry
    reports**, to `vpn` rather than to the predicted `no_active_network` -- the VPN
    interface stayed up as an active network with wifi down, so the absent path was
    never reached and `network_transport_absent_counts` is empty. That still meets
    the clause above, which asks for `no_active_network` **or a fallback**. The USB
    link survived, as required, which was the discriminator.

    **One thing that weakens the attribution, recorded rather than smoothed over.**
    `wire.seq_gaps` opened at t=245.6 s, twenty-five seconds *before* the wifi
    injection, so seq gaps occur unprompted on this bench. The gap at t=271.6 s is
    consistent with the injection and is not exclusively attributable to it. A
    second pair at t=328.6 s, twenty-eight seconds after wifi was restored, matches
    the transport counts, which show `vpn` persisting past the restore.

    **GPS revoke: the platform kills the app, and the predicted defect is
    unreachable this way.** Ticks stopped 0.7 s after the revoke, `link.down` opened
    at t=421.6 s with reason `no_session`, `camera.blind_ticks` followed at
    t=431.5 s, `pidof com.dsrc.phone` returned nothing and Camera2 logged the client
    closed. The Jetson behaved correctly: `end_reason` `peer_closed`, then
    "waiting up to 120s for a redial" that could not come.

    **No `SecurityException` appears anywhere in the captured logcat.** Android
    terminates the process on runtime permission revocation before any rate command
    can reach `requestLocationUpdates`, so the missing try/catch predicted above is
    neither confirmed nor cleared -- this injection cannot reach it, and a different
    trigger would be needed. The prediction stands unproven and is not filed as a
    defect.

    Whether a revoked GPS is distinguishable from an indoor no-fix also remains
    unanswered, for the same reason: the app died instead of running without the
    permission.

    **A defect in the injection harness itself, worth more than the injection.**
    `run_device_session.py` never exited after the app was killed, so the wrapper sat
    in `wait` and its restore trap did not fire. **The location permission stayed
    revoked for about five and a half minutes** until it was granted back by hand.
    Device restoration must not depend on the wrapper reaching its last line; it
    belongs on a timer inside the injection step, or in a separate process.

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
48. ~~**[COLOCATED]** In-car install: 12 V power for both devices, mounts, cable
    routing.~~ **DONE 2026-09-08.** Both devices ran from the car for 152.8 minutes
    across eight runs with no power interruption. The phone is USB-attached to the
    Jetson, which is both the sensor link and the phone's charge source; the Jetson
    reaches the network through the phone (task 61).

    **Two install faults worth recording, because both cost drive time.** Cable
    type is not interchangeable: the Moto's cable failed in the car and the app was
    moved to the OnePlus Nord N10 (`a1411577`) mid-session. And direct sunlight on
    the dashboard heated the phone enough that the tethering handset's hotspot shut
    off, which ended a drive; the mount needs shade, not just a clamp.

## I. Measurement drives — **[COLOCATED]**

49. ~~Shakedown drive: short and local, purely to confirm the system records
    readable, aligned data.~~ **DONE 2026-09-08.** Two drives in Westfield, NJ, on
    the OnePlus Nord N10 (`a1411577`) rather than the Moto -- the Moto's cable
    failed and the app was installed on the Nord in the car, with the swap recorded
    automatically in `installed_apk.json`.

    **`run_20260908_142253`: 1,632 ticks, 279.5 s, 6.24 km**, mean 22.9 m/s and max
    28.0 m/s. GPS valid on 1,299 of 1,313 ticks at the point measured; 4.69 Hz mean
    tick rate, 203 ms median spacing, two gaps over 2 s totalling 5 s. Distance
    agrees to 0.01 km between integrating reported speed and the great-circle path
    through the fixes, which is two independent routes to one number.

    **First working GPS and first working HERE in the project.** Every bench run had
    `gps_hz 0.0` and `here=false`; this drive had a real fix at highway speed and
    `here=true`, so the HERE path has now been asked to run for the first time since
    task 21 built it.

    **It did what a shakedown is for: it broke.** Three defects, each filed
    separately -- the 90-degree frame rotation that explains every zero-detection
    drive in the project (task 63), frames being discarded so the drive could not
    explain itself (task 64), and the `severe` thermal tier ending a session rather
    than merely lowering rates (recorded under task 45).

    **What it did not establish is the "aligned" half of its own wording.** Video
    position could not be tied to ticks at the time -- that is what task 64 added
    afterwards, and the second drive `run_20260908_155910` is the first with
    `video_index.jsonl` beside `video.avi`.
50. ~~Drive set 1, shadow mode at maximum rate: the full-rate reference plus every
    candidate policy's decisions against identical traffic.~~ **DONE 2026-09-08,
    with one qualification.** Six shadow runs, 17,948 ticks, 45.9 to 47.7 km:
    `run_20260908_142253`, `_144642`, `_155910`, `_173809`, `_183538`, `_190548`.

    **The qualification is that HERE did not run at maximum rate.** It was set to
    one query per minute in shadow mode on 2026-09-08 so that shadow drives collect
    HERE at all, a deliberate change made mid-session. The camera sustained 5.0 Hz.
    A reader must not treat these as a full-rate HERE reference.

    **The "every candidate policy" half is still available offline and was not
    run.** `deployment/jetson/score_shadow.py` replays the logged decisions and
    scores candidate sensing controllers against the same recorded per-tick inputs.
    It first requires the incumbent to replay byte-for-byte and refuses otherwise,
    so the first step is running that gate against these six runs.
51. ~~Drive set 2, live mode: the controller gating for real, verifying the
    shadow-mode predictions held.~~ **DONE 2026-09-08.** Recorded under task 44:
    `run_20260908_161422` and `run_20260908_170849`, 4,981 live ticks over 40.7 to
    42.4 km. Whether the shadow-mode predictions held is the offline comparison
    named in task 50 and has not been run.
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

**Four of those depend on seeing other vehicles and were not obtained**, because of
task 63: camera-derived local speed variance, how often the camera changes the
advisory, and any quantity built on detection range or count. They are recoverable
offline from the stored frames once the rotation is fixed. The rest were recorded.

### Data collected 2026-09-08

One day, eight runs, one vehicle. This is the whole corpus; there will not be
another collection day.

| | |
|---|---|
| runs | 8 — six shadow (17,948 ticks), two live (4,981 ticks) |
| recorded span | 152.8 min, of which 114.3 min above walking pace |
| distance | between 86.55 km and 90.06 km |
| ticks | 22,929, with a valid GPS fix on 22,734 (99.1%) |
| video | 1.676 GB, 19,000 indexed frames across six runs |
| HERE | 123 response bodies, 3.7 MB, every one HTTP 200 |
| phone-side | 658,862 raw IMU samples in 783,519 records |
| archive | 187 files, 2,180,405,759 bytes, SHA-256 verified against the Jetson |

**Distance is a bracket, not a number.** 86.55 km sums great-circle steps between
fixes less than 3 s apart, which omits distance covered during gaps; 90.06 km
integrates GPS speed, which counts gaps but trusts the speed field. Quoting either
alone overstates the precision.

**Two runs are not fully usable.** `run_20260908_142253` recorded no video at all
(`video.avi` is 0 bytes). `run_20260908_144642` holds 180.7 MB of video with no
index, because the frame index (task 64) was added mid-day — the imagery exists but
cannot be tied to ticks.

**One run lost its tail.** `run_20260908_183538` has no `summary.json` and its
`metadata.jsonl` ends in 1,246 NUL bytes: the filesystem extended the file but the
buffered tail never flushed, because the process was killed by a service restart
rather than shut down. 386 ticks survive against 478 frames indexed. The lesson is
recorded because it is general: a run must be closed by stopping the service, which
flushes, and never by killing it — the final teardown of `run_20260908_190548` wrote
283,845 bytes of buffered metadata.

## J. Reproducibility

53. One-command dry runs for the simulation matrices.
54. Smoke validation small enough for routine regression testing.
55. Artifact manifest: where checkpoints, session recordings, metrics, plots, and
    validation summaries live.
56. ~~The suite does not pass on the Jetson, and one assertion is why.~~ **DONE 2026-09-09** — `pytest.approx`. Verified against the Jetson's actual value: exact equality rejects 49.800000000000004, `approx` accepts it, and a genuinely wrong 51.0 is still rejected, so the assertion can still fail.
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

**Two items gate the critical path as of 2026-09-08.** Neither needs hardware, a
drive, or a decision from anyone outside the project.

- **Task 63, the 90-degree frame rotation, gates `range_m` in task 9.** Until it is
  fixed and the horizon re-established for the real mount, the offline replay cannot
  produce a distance worth calibrating against. It does not gate anything else: the
  frames are stored, so nothing is being lost while it waits.
- **Task 67, the 40% truncation rate, gates task 68.** Training against episodes that
  do not complete cannot distinguish a bad policy from a broken run.

The three items below are struck; they are kept so a reader can see what was cleared
and on what evidence.

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

Two items are here. The rest of this section, numbered 77 to 140, is the simulation leg's
findings and is in `detours.md`.


87. **Three of the four action heads are inert on SUMO, and the safety layer never
    runs.** Found by the independent audit of the SUMO migration, 2026-09-09.
    Belongs with the task 86 decision, since both concern the action contract.

    `_apply_actions` reads `desired_speed_bin` and stores `desired_headway_bin`;
    `lane_preference` and `merge_mode` are discarded. `mappo_sumo.yaml` sets
    `action_profile: full`, which activates all four heads. `desired_headway_bin`
    only feeds back into the agent's own observation — SUMO's `tau` is never set —
    so it changes what the agent sees and not what the traffic does.

    **So of the four heads, one acts on the world, one acts on the observation and
    two act on nothing.** With 5x4x3x3 action combinations and only the 5 speed bins
    changing the simulation, most of the gradient signal is on inert dimensions.
    That is a second cause of the task 86 null, independent of the speed bins being
    outside the effective range.

    **The safety layer is not invoked at all.** `apply_safety_layer` is never called
    on this path, so `info["safety"]["penalties"]` was a literal empty dict and
    `safety_penalty_for_agent` returned 0 for every agent — the per-agent reward
    equalled the team reward exactly. That is defensible in principle, because
    SUMO's car-following is the safety guarantee the layer existed to provide, but
    it was not a decision that was taken; it was an omission. The etiquette filters
    are likewise unreached.

    **Two parts fixed now**, because they are omissions rather than design choices:
    `SafetyConstraints` now comes from the topology's own `safety:` block rather
    than library defaults, and the empty penalties dict carries `layer_ran: False`
    so it cannot be read as "no penalties were incurred".

    **What remains is a decision.** `lane_preference` maps onto
    `traci.vehicle.changeLane` and `desired_headway_bin` onto `setTau`, so both are
    implementable. Whether to wire them, and whether the safety and etiquette layers
    should run on a simulator that cannot crash, is the same question as the speed
    bin rescale: it changes what the deployed actor's heads mean.

141. **The flow-level half is now the SRC controller on Mainz, and `inverted_tree` is
     out of scope.** 2026-09-11 and 2026-09-12. Supersedes tasks 68 and 69 and the
     scope of section C. Full account in `plans/mainz_src_port.md`; the paper's use of
     it in `plans/paper_deploying_self_regulating_cars.md`.

     **Why `inverted_tree` went.** Two measured reasons, either sufficient. Its
     super-segments are **one link each**, 300 to 600 m, against Mainz's 12 segments of
     2 to 33 edges and a median 3.78 km and the SRC paper's 2 to 3 km -- so there is no
     aggregation in it for a super-segment observation to summarise, which is also why
     it read null. And its recorded 24% capacity drop does not reproduce: on five seeds
     served flow is 966 to 1,114 veh/h across a 9x density range with two-standard-error
     bars of 61 to 152, and reproducing the recorded three-seed configuration exactly
     gives 864 to 1,096 with bars up to 213. The 1,298 and 992 in
     `configs/demand/sumo_saturating.yaml` were the maximum and minimum of a noisy flat
     series measured without a spread.

     **What had to change before any network could show a gain, in order.**

     1. **A link's `q = k v` hill is not a capacity drop.** Krauss produces one too, at
        15% on a straight road, and Krauss has no capacity drop. The quantity that
        decides whether a controller has anything to recover is SERVED FLOW falling as
        offered demand rises. `scripts/measure_mainz_fundamental_diagram.py` is now that
        gate and refuses a fall inside the seed spread.
     2. **The paper's W99 calibration has no capacity drop on SUMO.** Its queues
        discharge **faster** than free-flowing traffic at capacity, 1,980 veh/h against
        1,899, because a discharging W99 platoon is regular at the model's tightest
        headway where free flow at capacity has spread. Real queue discharge is 5 to 20%
        BELOW capacity. Neither `startupDelay` at 0, 1 or 2 s, nor reaction time, nor a
        lane drop, nor a metered exit changed it.
     3. **A bottleneck sets the LEVEL of throughput, not its dependence on density.**
        `configs/human_models/eidm_reaction.yaml` -- EIDM with a one-second
        `actionStepLength` -- has a 21% drop at an isolated lane drop, resolved on five
        seeds. On Mainz it needed a merge at the exit as well, because link 218 ends into
        free outflow; `--exit-lanes 2` gives a three-into-two drop and an 11.3% fall,
        437 veh/h against a combined bar of 106. That is the first gate pass in the leg.

     **The result.** Seeds 1-10 train, 11-15 select, 16-30 evaluate, 80 episodes, 100%
     penetration, sustained 4,500 veh/h, paired on seed: the traffic-API observation
     gains **+230 +/- 44 veh/h** (+6.5%) and SRC's original six features **+224 +/- 61**
     (+6.3%), both resolved, and the two differ by 0.2%. Both arms run SLOWER than no
     control, 40.8 against 42.7 km/h, while serving more vehicles.

     **Five seeds nearly published a phantom.** On seeds 16-20 alone the tree read
     +14.2% and +11.0%; on fifteen it reads +3.2% and +0.3%, unresolved. Mainz survived
     the same extension and tightened, +207 +/- 92 to +230 +/- 44. Checkpoint selection
     was unchanged in both; only the evaluation set was enlarged.

     **`DENSITY_CRITICAL` is re-anchored to 0.178** in occupancy units -- EIDM capacity
     1,877 veh/h/lane at 39.6 veh/km/lane, jam 128.5 -- which is 0.308 of the measured
     jam density, within 3% of the 0.3 SRC publishes. Under W99 the same comparison gave
     0.134. The published threshold was not wrong; the fleet was.

     **The simulation is deliberately not a reproduction**, and that is the design: the
     mechanism now holds in two simulators under two driver models at two demands, which
     is stronger than matching one number in one of them. No figure here is comparable to
     the published table; every claim is against this port's own no-control baseline.

71. **Route-aware leader gap on the Jetson, from the HERE link shape.**
    **DECIDED BY THE USER 2026-09-09. Not started — recorded only.**

    **What.** Use the already-matched HERE link to decide whether a
    camera-detected vehicle is on the ego's own road, and measure the gap along
    that link, so the live system produces the same quantity the simulator gets by
    walking its road graph.

    **Why it is needed.** The simulator's `leader_gap` has, since task 67, meant
    "nearest vehicle ahead on the ego's road-graph successor chain, at a distance
    accumulated over lane lengths". The Jetson has no road graph at all — `grep`
    for `RoadNetwork`, `lanes_dict`, `next_lane` or `road_network` under
    `deployment/jetson` returns nothing, and its entire road model is
    `lane_width_m = 3.7` plus `round(lateral / lane_width)` to bin detections into
    ego, left and right. So the actor now trains on information the deployed
    vehicle cannot produce. The same objection was used, correctly, to refuse a
    real `distance_to_next_merge`; it applies here and was applied inconsistently.

    **It is smaller than it first appears, because the primitives exist.**
    `FlowLink.points` already keeps the link's shape in order, `FlowLink` already
    has a "how far this link's shape passes from a point" method, and
    `FlowReading.link` is already a single link selected by shape distance and a
    fixed heading cone. Rudimentary map matching is therefore already running. The
    missing piece is projecting camera detections onto that link and measuring
    along it.

    **The data is already collected and needs no new API call.** The 123 HERE
    bodies stored on 2026-09-08 hold 915 road segments across 68 named roads and
    58,453 lat/lng polyline points — 2 to 27 segments per response, up to 126 links
    and 255 points per segment, with names like `US-1/Brunswick Pike` and
    `RT-27/Nassau St`. That is enough to develop and validate offline against real
    drives, without calling HERE.

    **What it does NOT unlock.** `distance_to_next_merge` stays unavailable. This is
    the traffic *flow* API: it gives named segments with geometry, not junction
    topology or lane-level connectivity. A merge is a junction between roads and
    needs routing or map-tile data, which would be a new dependency rather than new
    processing. Adjacent segments with differing `description` values hint at a road
    change, but that is inference, not topology.

    **What it does unlock.** Task 47's `leader_gap` slot could be reclassified from
    `identical` to `approximated` against a real live implementation rather than
    scoped in a docstring, which is the outstanding item recorded there. And the
    actor would stop training on a field the vehicle cannot measure.

    **Open before starting:** whether the gap is measured along the link polyline
    or along the ego's heading ray, and how a detection is assigned to a link when
    two links pass within the camera's lateral error. Both are design questions,
    not implementation details.

70. **Selecting live mode on a redial is a silent no-op, and a live drive was
    recorded as shadow because of it.** Open.

    `run_demo.py` resolves the drive mode once at process start. A phone that
    reconnects to an already-running `run_demo` rejoins in whatever mode the process
    already holds, and the handshake's `requested_mode` is ignored. On 2026-09-08 a
    live-mode tap was absorbed into a running shadow session; the drive continued in
    shadow and nothing said so. The user's own report is what caught it: "It did not
    restart by itself, it was a live run getting added to a shadow run."

    **The failure is silent, which is the part that matters.** The phone shows the
    mode it asked for, the Jetson logs the mode it started in, and neither compares
    them. A reader of the logs alone cannot tell an intended shadow run from a live
    request that was dropped.

    Workaround in use: restart `dsrc-drive` between modes. Two candidate fixes —
    re-resolve the mode on each handshake, or refuse a handshake whose
    `requested_mode` differs from the running mode and say so on the phone. The
    second is smaller and fails loudly, which suits a rig operated from a car.

66. ~~**HERE response bodies were discarded, and a HERE response cannot be fetched
    again.**~~ **DONE 2026-09-08.** `_read_here` received every body, handed it to
    the feed, which kept `downstream_congestion` and `free_flow_mps`, and dropped the
    rest. The 2026-09-08 live drive made **143 successful calls of about 55 KB each**
    and retained two floats per call. Nobody can ask what the congestion on that road
    was at that minute again, so a parse bug found later would cost the drive rather
    than an afternoon of reprocessing -- and this project has had parse bugs of
    exactly that shape.

    `logio/here_logger.py` stores each body as `here/NNNNNN.json` with a line in
    `here_index.jsonl` carrying status, size, query position and radius, the request
    URL and the arrival stamp. Same shape as the video index deliberately: payloads
    on disk, a sidecar naming them, usable without any code from this repository.

    Three properties the tests hold. The index line is written **after** the body and
    by the same call, so a line exists only for a body really on disk. Bodies are
    written whole and renamed, so a truncated file is never mistaken for a malformed
    response. And a **non-200 body is stored too** -- a refusal from HERE is evidence,
    and dropping error bodies would leave a run unable to say why it had no feed.

    The sink is called **after** the feed, not before: the feed is what the drive
    needs to run and storage is what a later reader needs, so a body that fails to
    store must not cost the tick the feed would have served. `write` never raises by
    contract, because its caller is the reader thread whose death stops the feed for
    the rest of the drive -- the same lesson `_read_here` already learned from an
    `OverflowError` out of `float()`.

    `here_log` is in the run summary beside `video_log`, reporting written, bytes,
    failures and `complete`. Written that way because `VideoLogger.dropped_frames`
    was incremented and read by nothing, leaving "none lost" and "loss not reported"
    indistinguishable.

    About 11 MB/hour at the observed call rate. Suite 2305 passed, 372 pins.

    **What this does not recover.** Today's 143 bodies are gone. What was rescued
    from the phone is the *record* of each call -- query position, radius, request
    URL, content type and payload size -- in `~/dsrc_logs/phone_sessions_20260908/`,
    which is enough to know what was asked and how big the answer was, and not enough
    to re-derive anything from it.

65. **A drive started before NTP syncs will have `t_wall` step mid-run.** Open.

    Seen on 2026-09-08: `dsrc-drive` reported `ActiveEnterTimestamp` of 14:30:44 on a
    box that had booted at 15:55:09. The service was three minutes old. The Jetson
    boots with an unsynced clock, systemd stamps against it, NTP then corrects the
    clock forward -- 88 minutes, in that boot -- and the old stamp stays behind.

    The hazard is not the stamp, it is `t_wall` in the tick records. A run that starts
    inside the pre-sync window carries wall-clock times that jump when the correction
    lands, and the last drive's duration and distance were both computed from
    `t_wall`. `t_mono` is immune, which is most of the analysis, and the completed
    drive is clean -- 279.5 s against 1,313 ticks is mutually consistent at 4.69 Hz,
    so no step happened during it.

    **The window is exactly the first minute or two of a cold boot in a car**, since
    the Jetson only syncs once the phone tether gives it internet. The pre-flight
    check is one line -- `timedatectl show -p NTPSynchronized --value` must read
    `yes` -- and the proper fix is for `run_demo` to record the sync state at start
    and flag or refuse an unsynced clock, which is the same shape as every other
    "record the environment at run time" item here.

63. **The camera frames arrive rotated 90 degrees, and that is why no drive has ever
    seen a vehicle.** Open: the fix is not yet written.

    **This is now on the critical path.** It gates `range_m` in task 9, which is one
    of the five sensing-model parameters the replication needs. Nothing else waits on
    it, and no data is being lost while it waits, because raw frames are stored.

    **Full-day measurement, 2026-09-08:** 16 detection-bearing ticks out of 22,929,
    or 0.070%, never more than one vehicle in a tick. Worse than the count suggests,
    the `vehicles` array is **empty even on those 16 ticks** — a detection was counted
    but no tracked vehicle reached the observation stage — so **zero distance
    estimates were recorded on the entire day**. The three detection-dependent
    sensing parameters have zero observations, not few.

    Found on the first real drive, 2026-09-08. Same frame, same detector, one
    rotation:

    | frame | as delivered | rotated 90 clockwise |
    |---|---|---|
    | mid-drive | **0** | **2 vehicles, conf 0.90** (car, truck) |
    | later | **0** | **5 vehicles, conf 0.87** |

    Only clockwise; anticlockwise and 180 both give zero. The frame shows a GMC
    Savana van filling much of the picture with a legible plate, so this is not a
    marginal detection.

    **This is one bug, not the several structural limits it was recorded as.** Zero
    vehicle sightings across 4,151 ticks on the Moto and 1,632 on the Nord -- two
    handsets, two cities, same zero. `leader_gap` `inf`/`fallback_neutral` on every
    tick ever recorded, `unique_tracks` and `track_lifetime_s` empty everywhere,
    `local_density_bin` always `0`/`derived_empty`: all downstream of this.

    The detector was never at fault. Given an upright road image it returns cars at
    conf 0.80, 0.67, 0.62.

    **Where the fix belongs is a real decision, not a patch.** The intrinsics
    (`fx_px 800`, `cx_px 640`, `horizon_y_px 360`) assume a landscape frame, and
    rotating swaps which axis is which -- so correcting on the phone and correcting
    on the Jetson are not equivalent for `DistanceEstimator`. To be settled at the
    desk, not in a car.

    **Two retractions of my own, recorded because both were confidently wrong.** I
    first said rotation was the fault, then said a test had disproved it: that test
    sampled frames 0/30/60/90, which is the first three seconds of a run on an empty
    residential street, because `CAP_PROP_FRAME_COUNT` returns 0 on a file still
    being written and seeking by fraction silently sampled the start. A null from
    frames containing no cars says nothing about detecting cars. I also said the
    frames "did not contain recognisable cars"; they contained perfectly
    recognisable cars in an orientation the model cannot read.

64. ~~**A drive's frames were discarded, and video position could not be trusted as a
    tick index.**~~ **DONE 2026-09-08.**

    `logio.video` was `false`, so 1,632 ticks of frames were decoded, detected on and
    thrown away. The drive's central question -- why no detections -- was
    unanswerable from its own record. Now `true`, with the reason in the config.

    **Position is no longer assumed to be the tick index.** The docstring claimed
    "video frame index == tick index == metadata record index" while the drop path
    five lines below could break it: a bounded queue, `put_nowait`, discard on
    `Full`. After one drop every later frame is off by one, silently, and every
    offline re-analysis keyed on position is wrong from there. Each written frame now
    appends `{pos, frame_id}` to `video_index.jsonl`, written by the thread that
    wrote the frame and after the write, so a line exists only for a frame really in
    the file and a drop shows as a gap in `frame_id` to a reader who never sees the
    code.

    `VideoLogger.dropped_frames` was incremented and **read by nothing** -- "no
    drops" and "drops not reported" were indistinguishable. `to_record()` now
    surfaces written, dropped, *which* ids were dropped, and
    `position_is_tick_index`, and `run_demo` puts it in the summary beside the
    camera's own separate counter.

    **The test found a worse defect than the one it was written for.** `close()` did
    a blocking `put` on a bounded queue, so with the drainer dead it never returned:
    a run would hang at teardown and never write `summary.json`, losing the whole
    drive's record over a video problem. `close` is now bounded by
    `CLOSE_TIMEOUT_S`, reports `close_blocked`, and the drainer no longer dies on a
    write error -- it records `writer_error` and keeps draining, because a dead
    drainer is what turns dropped frames into a lost summary.

    First test file this class has ever had. Suite 2270 passed, 24 skipped, 372 pins.

    **One claim in that work is wrong and the code comment says it.** I wrote that a
    dropped frame "shows as a gap in `frame_id`". It does not. Measured on
    `run_20260908_155910`: 30 non-consecutive steps across 242 positions, a ratio of
    1.57 frame_ids per written frame, and no drops -- because the camera delivers
    faster than the pipeline consumes and `wait_for_fresh` is latest-wins, so skipped
    ids were never offered to the logger at all. Gaps are normal operation. The only
    sound discriminator is `dropped_frames`/`dropped_frame_ids`. The comment needs
    correcting, because a note that teaches the next reader to misread the artifact
    is worse than no note.

    Also still missing: the drop count reaches a reader only in the summary at run
    end, so mid-drive nobody can tell whether alignment is intact. A smaller version
    of the gap this task closed, and it belongs in any health readout.

62. ~~**A dead USB tether holds the default route against working wifi, and nothing
    notices.**~~ **DONE 2026-09-06**, verified by a packet rather than by a state.

    Reproduced live on 2026-09-06 by switching the hotspot off, which left the Moto
    with no upstream while the cable stayed plugged in:

    | probe | result |
    |---|---|
    | `ping -I usb2` | **FAILS** |
    | `ping -I wlP1p1s0` | WORKS |
    | `ping` with no interface, the path everything uses | **FAILS** |

    So the Jetson had a working network and would not use it.

    **NetworkManager is not at fault and that is the point.** From its side nothing
    is wrong: the cable is in, the link is up, the DHCP lease from the phone is
    valid. The phone is still a router, just one with nowhere to route. `usb2` at
    metric 100 beats wifi at 600, and no connectivity check ever runs, so a black
    hole outranks a working path indefinitely.

    **This is section G's shape in the network stack:** `connected` and `working` are
    different claims and nothing in the state distinguishes them. `usb2` reported
    `connected` for the entire outage.

    **It therefore constrains the phone-side health readout** more than it
    constrains the routing. A status panel that reports "the Jetson has a network"
    from an interface state would have shown green throughout. That field has to
    mean a packet went out and came back, or it is worse than absent.

    The fix sets ethernet below wifi in two places: the profile that exists, and a
    `conf.d` default covering ethernet connections that do not exist yet. The second
    matters because `Wired connection 2` was auto-created by NetworkManager for this
    USB device, so fixing only the profile would let the next replug generate a fresh
    one at metric 100 and bring the bug back silently.

    **In the car this changes nothing**, because there is no wifi to win: if the
    hotspot drops, Jetson access is gone until it returns. What it fixes is the desk,
    where the Jetson otherwise spends the personal phone's mobile data while campus
    wifi sits idle beside it.

    **After the fix**, with the hotspot still off so `usb2` is still a black hole:
    wifi carries the default route at 600, `ping` over the default path WORKS, and
    `ip route get` names `wlP1p1s0`. The same three probes that failed before now
    separate correctly -- wifi WORKS, `usb2` FAILS, default WORKS -- which is the
    behaviour that was missing.

    One number came out other than as written: `ipv4.route-metric 700` produced a
    kernel route at **20700**, so NetworkManager adds an offset somewhere rather than
    using the value verbatim. It is recorded because it looks like a mistake and is
    not: the ordering is what matters, 20700 loses to wifi's 600 by a wider margin
    than intended, and in the car `usb2` is the only default route, where a metric
    decides nothing at all.

61. ~~**No way to reach the Jetson in the car.**~~ **DONE 2026-09-06.** The wifi
    route is a dead end and the phone is the answer: with USB tethering the handset
    is the Jetson's network adapter over the cable already in place.

    **Why wifi could not work, measured rather than assumed.** The saved hotspot
    profile was correct -- ssid, `hidden yes`, autoconnect, priority 5 under `nyu`'s
    10, `psk-flags 0` with the keyfile present -- and activation still failed:
    `secrets exist. No new secrets needed` then `Activation: failed`. The password
    was never in question. The Jetson simply cannot hear the AP: BSSID
    `5e:9c:01:50:f0:ed` appears nowhere in a full scan, while the Moto, cabled to the
    Jetson and so inches away, sits on it at RSSI -36. A single radio associated on
    5 GHz `nyu` picks up mains-powered campus APs on 2.4 GHz but not a phone beacon on
    channel 6. **A profile saved the ordinary way would have failed silently in the
    car**, since autoconnect also waits to hear a beacon.

    **What works.** `svc usb setFunctions rndis` gives `rndis,none,adb`, so adb
    survives. The Jetson enumerates `usb2`, NetworkManager DHCPs it with no root
    (`Wired connection 2`, autoconnect yes), gateway and DNS are the phone, ping out
    is 54-143 ms, and Tailscale comes up over it.

    **Two independent mechanisms hold it up**, because the setting does not persist
    across a replug or a phone reboot. `dsrc-usb-net.service` on the Jetson re-applies
    it, checking the current config rather than setting blindly -- a blind re-set
    every 20 s would bounce the USB bus continuously and take the drive's own link
    down with it. And `svc usb setScreenUnlockedFunctions rndis` on the phone. Note
    `persist.sys.usb.config` still reads `adb` when the latter is set, so that
    property is not how to check it.

    **The unit was verified against a genuinely broken state, not merely installed.**
    The first attempt proved nothing: the knock-out failed because the phone-side
    backstop silently restored rndis, and the unit's log was empty. Clearing that
    first produced a real failure -- config `adb`, `usb2` gone -- and the unit
    restored both within 10 s, with `config is 'adb'; enabling rndis` in its log.

    **The phone can also reach the Jetson directly**, `ssh edge@192.168.16.70`,
    confirmed by reading an `SSH-2.0-OpenSSH_8.9p1` banner from the handset against a
    `Connection refused` control. Two of my probes before that returned nothing and
    were instrument errors, not results: `nc` with stdin at `/dev/null` hangs up
    before the banner arrives, and `nc -z` is not supported by toybox.

    **One cost, needing root to fix:** `usb2` takes the default route at metric 100
    against wifi's 600, so at the desk every byte the Jetson sends -- `rsync` and
    `scp` of this repository included -- goes over mobile data. The remedy is in
    `systemd/README.md`.

60. ~~**The app had no icon, so the drawer showed a system placeholder.**~~
    **DONE 2026-09-06.** The manifest declared no `android:icon` and `res/` held only
    `values`, which the handset confirmed as `icon=0x0`. It was findable by name and
    by nothing else -- a poor arrangement for someone looking for it one-handed in a
    parked car.

    An adaptive icon: a stop-and-go wave damping into smooth flow, which is the
    project's own claim drawn as one white stroke on a dark blue ground. Adaptive
    only, since `minSdk` is 29 and every launcher that reaches this app understands
    `mipmap-anydpi-v26`; no legacy PNG densities are needed. The device now reports
    `icon=0x7f080001` on the resolved launcher entry.

    **Drawn against screenshots from the handset, not guessed at, and it took three
    passes.** At 6dp stroke with half the amplitude it read as a squiggle. At 8dp
    with four oscillations the lobes sat closer together than the stroke was wide and
    merged into a single blob. Two oscillations spaced about twice the stroke width
    apart is what survives being 48dp across.

    `verifyMergedManifest` gains the icon, matching how this project checks every
    other manifest fact: read it back out of the merged artifact rather than restate
    the intent. Confirmed the gate fails without the attribute before relying on it.

    **Two stale-artifact traps hit on the way.** An XML comment cannot contain `--`,
    so one build failed while the `adb install` chained after it reported `Success` --
    installing the previous APK. Comparing the device-side `sha256sum` against the
    local one is what caught it, and is the only check that can.

59. ~~**Choosing a drive's kind needs a shell, so it needs a laptop in the car.**~~
    **DONE 2026-09-06.** `MainActivity` now has **Start (shadow)** and
    **Start (live)**. Stop the session and start another to change kind; there is no
    mid-session switching and that is deliberate.

    **A request, not a setting.** The phone carries `requested_mode` in its
    handshake; the Jetson decides, applies it before the first tick, and records both
    the mode and who asked. `plan_deployment.md`'s rule -- the Jetson owns every
    sensing decision, and a setting chosen on the phone is a decision that never
    reaches the log -- therefore still holds, because this decision does reach it:
    `sensing.mode.mode_origin` reads `phone_request` or `command_line`.

    **Precedence:** the phone wins when it asks, the command line decides when it
    does not. A handset older than the field expresses no opinion, so an existing
    service configuration behaves exactly as before. `resolve_drive_mode` is a
    module-level function rather than an `if` inside `run_live`, because it decides
    what a whole drive is and was otherwise only reachable by starting a run with a
    handset attached.

    **Verified on the device, against a service whose own arguments select shadow:**

    | button pressed | recorded mode | origin | flips |
    |---|---|---|---|
    | Start (live) | `live` | `phone_request` | 0 |
    | Start (shadow) | `shadow` | `phone_request` | 0 |

    So one service configuration served both kinds, and the button decided.

    **Three design points that are not obvious.** A hello with no opinion omits the
    key rather than sending `null`, so it is byte-identical to one from before the
    field and `transport_golden_frames.json` stays valid. A value outside the two is
    refused at the door on both sides rather than defaulted, because running shadow
    silently when the driver pressed live is a wasted drive that reads as a good one.
    And `handshake.py` restates the two modes instead of importing
    `policy.shadow_mode`, because its own spec says the transport is opaque -- with a
    test asserting the two sets stay equal, since restating them costs drift.

    Two buttons rather than a toggle plus one Start: a toggle has state that outlives
    the session that set it, so the next drive would inherit the last one's kind from
    a control nobody looked at.

    **Mid-session switching is left undone on purpose.** `ModeHolder` already
    supports it -- `flip_to` exists with 22 tests and no production caller -- but
    `eval_run` treats `ever_live` as a property of a whole drive, so a promoted
    drive's shadow segment loses its caveat at `eval_run.py:1639` and its unapplied
    commands read as a rate shortfall. The record can express a flip; the scoring
    cannot.

    `specs/transport_protocol.md` gains the field, since a wire change absent from
    the contract is worse than an edit to it. Suites: 2265 Python, 458 app, 286
    transport, 372 pins.

    **A defect in the bench, not the feature.** The acceptance script printed
    `VERDICT: the phone's request decided the drive` while its own Python was raising
    `FileNotFoundError` -- the summary is not written until the run ends, and
    `check ... | tee` takes its exit status from `tee`. The verdict was reported by a
    check that never ran, which is this section's recurring shape. The result above
    was re-derived from the summaries afterwards.

58. ~~**Nothing starts the Jetson runtime in the car.**~~ **DONE 2026-09-05.**
    Until this, `run_demo.py` could only be started from a shell, which meant `ssh`
    over Tailscale, which meant the Jetson needed internet and the car needed a
    laptop. Checked all three autostart routes on the box first: no systemd unit,
    no cron entry, no `rc.local`. The phone half was never the problem --
    `MainActivity` has a Start button, and `SessionHolder` retries a refused dial
    indefinitely, so tap order does not matter.

    `deployment/jetson/systemd/` now carries a **user** unit, its environment file
    and a README. A user unit because installing to `/etc` needs a password an
    unattended deploy does not have, and `loginctl enable-linger` does not. The
    procedure in the car is: 12 V on, plug the phone in, tap **Start sensing**.

    Two settings carry the weight. `--rebind-timeout-s 0` is new and means never
    give up on a handset that went away; the old fixed 120 s **ended the run**, so a
    call, an app restart or a thermal shutdown longer than two minutes would have
    cost a drive silently with nothing there to restart it. `--duration-s 0` was
    already unlimited.

    **Verified end to end on the bench**, not just installed: service enabled and
    started, the phone tapped, ticks flowing, then the runtime killed by its
    systemd `MainPID` -- a **new** run directory appeared with the phone
    reconnected and no human action, `NRestarts` 1, and a clean stop left
    `inactive/success`.

    **Three defects in this work, found by running it rather than by writing it.**
    The environment file did not parse as shell (`--usb: command not found`) even
    though systemd would have read it, so it is quoted now and valid under both.
    `journalctl --user -u dsrc-drive` reported **"No journal files were found"** on
    this box, so the README's own instruction showed nothing -- output goes to
    `~/dsrc_logs/dsrc-drive.log` with `PYTHONUNBUFFERED=1`, because Python
    block-buffers to a file and an empty log reads the same as a dead service. And a
    normal stop left the unit `failed` until `SuccessExitStatus=143` was added; a
    unit that reports failure after being told to stop teaches the reader to ignore
    its state.

    `--rebind-timeout-s` also broke five tests at once, because they build an
    `argparse.Namespace` by hand and a new option leaves it short a field. The
    parser is now `build_parser()` and a test compares the fixture against it, so
    the next option fails once, naming the missing field, instead of raising an
    `AttributeError` deep inside `run_live`.

    **What it does not solve:** the service restarts the runtime, not the phone. If
    the app dies -- which task 46 showed `pm revoke` makes it do -- somebody must
    tap Start again. There is no channel from the Jetson to the app: the USB link is
    adb with the Jetson as host.

57. ~~**Thermal backoff has no hysteresis, and a phone on the threshold rebinds its
    camera repeatedly.**~~ **DONE 2026-09-05.** A section F runtime defect, numbered
    here so nothing above is renumbered.

    **Fixed and verified on the handset.** `run_20260905_153043`, 900 s in live mode
    on commit `d7b224d`, with `source_tree_check` and `apk_last_update_time_check`
    both `matched`:

    | | before | after |
    |---|---|---|
    | commanded rate transitions | 10 | **2** |
    | changes while sitting on 40 C | 8 in 30 s | **none** |
    | delivered rate at commanded 5.0 / 3.0 Hz | 4.989 / 3.000 | 5.000 / 2.998 |

    One crossing at t=377.6 s on a reading of 40.009 C, then held for the remaining
    520.4 s to a maximum of 43.087 C. `sends_by_reason` records `changed` twice.

    **The dead band was exercised, not merely unopposed.** After the crossing the
    skin fell below 40.0 C on **237 of the 1,561 remaining ticks**, as deep as
    39.48 C. Under the previous bare comparison every one of those would have
    released the backoff and re-engaged it. The latch held on all 1,561.

    That deepest dip is 0.520 C, which settles a judgement made before it could be
    measured: the 1.0 C band was chosen over 0.25 C on the grounds that 0.25 C sat
    below the largest single-sample step then observed. **0.25 C would have failed on
    this run**, and 0.5 C would have been marginal.

    The platform `thermal_status` was `nominal` on all 900 reports again, at a
    maximum of 43.087 C, so the skin path remains the only one that moves on this
    handset.

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

    **A dead band and not a dwell**, because the two directions are not symmetric:
    a dwell would delay backing off, which is the direction where lateness costs
    heat, while a dead band delays only the recovery, at a temperature already below
    the trip point. Entering still takes the bare threshold.

    `SKIN_HYSTERESIS_C` is 1.0 C, sized from the sensor rather than chosen: over the
    900 samples of the run that found the defect, the 1 Hz sample-to-sample change
    had p50 0.023 C, p95 0.087 C and a maximum of 0.369 C.

    The latch is deliberately **not** cleared when telemetry goes absent, stale or
    unstamped. A gap in the reporting is not evidence the handset cooled, which is
    the same reading of silence the `unknown` tier already takes.

## L. Claims the repository does not support

Four defects found on 2026-09-12 by reading `paper_deploying_self_regulating_cars.md`
against the files it points at. Each is a statement the paper plan makes that the code or
the artifacts contradict. None was found by a test, because in every case the thing that
would have failed is absent rather than wrong.

142. **The paper's headline simulation result has no generator and no artifact.**
     **RESOLVED 2026-09-12.** The figures were never wrong; nothing could stand behind them.
     `scripts/evaluate_mainz_checkpoints.py` now produces them and
     `results/evaluation/mainz_paired_seeds.{json,jsonl}` records the run (commit `79a4a84`).
     Measured over seeds 16-30, 60 episodes in 403 s: here **3,764.89** (+229.93 +/- 44.36,
     resolved at 5.18x its bar), src **3,759.26** (+224.30 +/- 60.95, 3.68x), no control
     **3,534.96**, and a `zeroed_head` falsification arm at **2,846.67** -- 15.74 times no
     control's own bar below it, with a paired gain negative and resolved at 15.11x. Every
     asserted value reproduces to the rounding at which it was published: means within
     0.26 veh/h, bars within 1%.

     Two things the work settled beyond the number. The fifteen-seed figures were produced
     under HEAD's `SPEED_ACTION_FRACTIONS` mapping, so `eb86e72`'s 0.0200% action-set shift
     touches only the five-seed `test` blocks in `results/checkpoints/*_result.json` and not
     these. And the run configuration, which nothing had ever recorded, was recovered by
     arithmetic -- `mean_vehicles` 858.0370370370371 is exactly `115835/(5*27)`, pinning 27
     accumulation samples and therefore a window opened at 900 s -- then confirmed by
     reproducing the stored no-control block at 0.00e+00 on all seven metrics.

     **One defect remains, found while the experiment ran and after two validation rounds
     had closed.** The artefact's provenance reports `dirty: true`, caused by the run's own
     output: `results/evaluation/` is tracked and not gitignored, so the evaluator dirties
     the tree by writing and then samples that as its own provenance. `git status
     --porcelain --untracked-files=no` returns nothing; the code was `578d689` unmodified.
     The flag exists to say whether the code differed from the commit beside it, the answer
     is no, and the artefact reads as though it were yes. Fix: exclude the evaluator's own
     output path from the check, or record *what* is dirty rather than only that it is.
     Two validation rounds could not see it because it appears only when the tool runs
     against a real tree.

     `3,765 / 3,759 / 3,535 veh/h` with paired gains of `+230 +/- 44` and `+224 +/- 61`
     over seeds 16-30 is asserted in five places: this file, `mainz_src_port.md`,
     `results/README.md` and the paper plan. Nothing produces it.

     `scripts/train_mainz_src.py` reads `TEST_SEEDS = tuple(range(16, 21))` -- five seeds.
     The committed `results/checkpoints/mainz_{here,src}_result.json` carry that five-seed
     read and nothing else: **3,774.7 / 3,756.4 / 3,568.0**, which is +5.8% and +5.3%. No
     script in the tree evaluates seeds 16-30, and none ever did -- `git log -S "range(16,
     31)" --all` returns no commit. The per-seed values the `+/- 44` is computed from exist
     nowhere.

     The checkpoints are committed, so the number is re-derivable rather than lost. What
     the repository currently does is assert one figure while storing another.

     **The task:** an evaluator that loads a committed checkpoint, runs seeds 16-30 against
     the same no-control baseline, pairs on seed, and writes the per-seed table with its own
     two-standard-error bar. Then run it and either reproduce the quoted figures or correct
     every document carrying them. `mainz_src_port.md` also heads its result table
     "Held-out seeds 16-20" nine lines above "Fifteen evaluation seeds, 16 to 30"; whichever
     way the numbers land, that heading is wrong today.

143. **The vendored contract's equality check has no reference left to check against.**
     Implemented 2026-09-12 (implementer-143) against
     `plans/plan_task143_contract_golden_vectors.md`.

     `policy/sim_contract.py` vendors the encoder, the scales and the action heads from sim
     commit `d477dba`, and the paper plan calls this "what makes 'the device runs what was
     trained' checkable rather than asserted". The check was
     `deployment/jetson/tests/test_sim_contract.py`, which opened
     `pytest.importorskip("src.rl.encoders")`.

     `src/rl/` holds only `src_q.py`. `encoders.py`, `actions.py` and `models.py` were
     deleted in `6b538f2`. Measured before this task: the file collected one test and
     reported `1 skipped`.

     Restoring the deleted modules would have undone a deliberate cleanup to serve a test.
     Built instead: the idiom this repository already uses across two languages --
     `specs/transport_golden_frames.json` freezes the wire format and both implementations
     test against the file rather than against each other.

     **What was built.** `specs/sim_contract_golden_vectors.json` (122.7 KiB): 14 encoded
     observations (the 12 `test_sim_contract.py` already used, plus two cases giving the
     `cooperation` and `nearby_av_lane_distribution` blocks three distinct per-slot values
     each, so a reorder inside either block changes a number and not only a name), the
     action heads/values/forced defaults/profiles/default indices, both bin decoders,
     `bin_index` over a grid, the neutral cooperation fallbacks, the actor state-dict layout,
     and `contract_fingerprint()`. `scripts/generate_sim_contract_golden_vectors.py` derives
     every one of those twice -- once from sim commit `d477dba` (`git archive`d into a
     temporary directory, inserted at `sys.path[0]`, never the repository root, which would
     let the working tree's own trimmed `src` package win) and once from
     `policy/sim_contract.py` -- and refuses to write when the two disagree, in `--check` and
     in `--write --force` alike. Proved directly: with `leader_gap`'s scale mutated to 120.0
     (150.0 in the reference), both modes exit 1 and name `leader_gap`, and the file's
     SHA-256 is unchanged before and after (`b77b24e5a8ef...`). `test_sim_contract.py` is
     rewritten to read the frozen file and import no simulation module: 74 tests, 0 skipped,
     against the file's own count of 1 skipped before.

     **The mutation gate (plan section 7).** Seven entries added to `scripts/remutate.py`'s
     `MUTATIONS`, prefixed `sim contract:`. Run from a `git worktree add --detach` mirror at
     this task's commit, never the live tree, because two other agents were committing to
     `deployment/jetson/` at the same time:

     | Mutation | Caught by |
     |---|---|
     | M1 field order (`leader_gap`/`leader_relative_speed` swapped) | `test_slot_names_match_encoded_slot_names` |
     | M2 a `FIELD_SCALES` value (`leader_gap` 150.0 -> 120.0) | `test_field_scales_match_in_both_directions` |
     | M3 a `HEADWAY_BIN_S` bin edge (2.2 -> 2.3) | `test_decode_headway_bin_matches_the_recorded_grid` |
     | M4 `bin_index`'s boundary direction (`>=` -> `>`) | `test_bin_index_matches_the_recorded_grid` |
     | M5 the inf-clamp constant (200.0 -> 100.0) | `test_encoding_matches_the_recorded_vector[full_obs]` |
     | M6 an `ACTION_VALUES` order within a head reversed | `test_action_values_and_their_order_match` |
     | M7 `COOPERATION_FIELDS` order changed | `test_slot_names_match_encoded_slot_names` |

     All seven CAUGHT, 0 SURVIVED, 0 INCONCLUSIVE, 0 "did not build/import". None of
     M3/M5/M7 -- the three the fingerprint does not hash -- was caught only by a fingerprint
     test; each was caught by a golden-vector test, satisfying plan section 5.4's split.

     **`EXPECTED_PYTHON_TESTCASES` is not a literal.** The plan's own step 9 called for
     setting it to a measured count; that count moved five times in one day before this task
     even started (root suite 69 -> 88 -> 116 -> 130 -> 146, four increases;
     `deployment/jetson` 2,243 -> 2,390, one increase) as three tasks landed tests
     concurrently, and it moved twice more during this task's own mutation-gate runs (2,493,
     then 2,499). Any literal typed during this task would have gone stale within the hour --
     which is exactly the failure being fixed (the constant was 2060 against a suite that had
     already reached 2272, silently turning every Python mutation entry
     `INCONCLUSIVE`). Replaced with `_baseline_python_testcases()`: a full pytest run against
     the clean, unmutated tree, measured once at the start of each `remutate.py` invocation,
     before the first mutation is applied. It must run before, not after, any mutation: a
     mutation's own collection failure would otherwise lower both sides of the comparison
     together and hide the exact failure this check exists to catch.

     **`__pycache__` purge and `PYTHONDONTWRITEBYTECODE=1`.** Neither `run()` nor
     `_baseline_python_testcases()` cleared bytecode caches before invoking pytest. A
     same-second edit of equal byte length can reuse a stale `.pyc` (Python's default
     invalidation is mtime+size), silently re-scoring the previous mutant -- a failure mode
     that has produced a wrong verdict on this project twice before. `_purge_pycache()` now
     runs before every pytest subprocess in this file, and both subprocess calls set
     `PYTHONDONTWRITEBYTECODE=1`. Stated plainly rather than left implied, and generalised
     by validator round 2 (2026-09-12) rather than left as one function's footnote: **no
     test in this repository exercises `scripts/remutate.py` at all**, so every guard in
     it -- not only `_purge_pycache()`, but `_refuse_if_tree_is_dirty()`,
     `_COLLECTION_AFFECTING_NAMES`, and the baseline-failure refusal added in the two
     validator rounds below -- can be removed or emptied with the suite green. Verified
     directly, not inferred: the validator set `_COLLECTION_AFFECTING_NAMES =
     frozenset()` and the full Jetson suite reported 2,527 passed / 28 skipped, the
     unmutated count -- SURVIVED. `scripts/remutate.py` carries no unit tests of its own;
     every fix made to it in this task is verified operationally -- by the two controls
     below, by every mutation gate run in this task completing without a stale-cache
     false verdict, and by the reproductions recorded in the validator-round sections
     below -- not by an automated assertion, because none exists. This is the harness
     every mutation verdict on this project rests on, and it is the one thing here with
     no test behind it; closing that is outside this task's scope.

     **Two controls, because a gate that has only been seen to pass has not been seen.**
     First: `decode_speed_bin`'s `min_contextual_speed_mps` default (12.0) looked unread by
     anything -- the only direct test call site (`test_sim_contract.py`) always passes it
     explicitly from the recorded grid, and the three production call sites (`advisory.py`,
     `safety_gate.py` x2) always pass their own context value. It was CAUGHT anyway, by
     `test_regeneration_leaves_every_pre_existing_case_byte_identical`: this generator's own
     `speed_bin_mps` derivation calls `decode_speed_bin` on both sides without pinning that
     argument either, so Account A (the reference, unmutated) and Account B (the vendored
     copy, mutated) disagreed inside the generator itself, and the regeneration test's
     subprocess call returned non-zero. Real coverage, but indirect and fragile: it depends
     on that one test's `torch` and `git` dependencies being satisfied. Second, registered to
     settle the question directly: `active_heads()`'s except-branch message. Every caller in
     the tree -- `export_policy.VendoredActor`'s default, `actor_runtime.py`, this generator,
     and every test -- passes only profile `"full"`, which returns before the except branch
     is ever reached. This one SURVIVED (`*** SURVIVED ***`, exit code 1, named in the
     `survived` list) -- direct proof the harness's SURVIVED branch fires rather than the
     seven-mutation gate's clean sweep being an unexercised code path. Removed once observed;
     the gate was then re-run and returned to 7 CAUGHT / 0 SURVIVED, exit code 0, with the
     same seven catching tests as the table above.

     **Suite counts.** Mirror at `1a61b1b` (this task's parent commit, via `git archive`,
     never the live tree): `deployment/jetson/tests/` 2,391 passed / 28 skipped (24
     USB-device-gated, 3 pre-existing and unrelated in `test_run_demo_loop.py`, 1 this task's
     own `test_sim_contract` module-level skip), 2,419 JUnit testcases. Mirror at `2692c3b`
     (this task's first commit): 2,465 passed / 28 skipped, 2,493 testcases -- the 28 splits
     as the same 24 USB + 3 unrelated + 1, but that 1 is
     `test_regeneration_leaves_every_pre_existing_case_byte_identical` skipping because a
     `git archive` mirror carries no `.git` directory at all, so the generator's own `git
     archive d477dba` call fails structurally and the test detects that and skips with a
     named reason -- a property of the mirror, not of the development machine. On a live
     checkout with `.git` present (this repository, before committing): `test_sim_contract.py`
     74 passed / 0 skipped; the full suite 2,470 passed / 24 skipped, all 24 USB-gated,
     matching the plan's sign-off items S1 and S2 exactly. Cross-checked against the
     coordinator's independent measurement at `d976dcc` (a later commit on the same branch):
     `deployment/jetson/tests/` 2,465 passed / 28 skipped -- identical to this task's own
     `2692c3b` mirror. The final mutation-gate run (after the control's removal) measured its
     own fresh baseline at 2,499 testcases, higher again, from concurrent work landing on the
     branch between that run and the earlier ones; this is the dynamic count doing what it
     was built to do rather than a discrepancy.

     **Not done, and outside this task's ownership list while two other agents were live in
     the repository:** `ARCHITECTURE.md` section 6 still names a machine where the sim
     imports, which does not exist, and the maintenance-procedure rewrite the plan's section
     8 specifies was not made. Left for whichever task next holds that file.

     **Commits** (branch `mainz-src-port`): `2692c3b` (the generator, the golden file, the
     rewritten test file, the seven `MUTATIONS` entries, `EXPECTED_PYTHON_TESTCASES` made
     dynamic), `972ded9` (`__pycache__`/bytecode fix, first control registered), `939bee2`
     (control replaced), `8a321fd` (control removed). Each by explicit pathspec, never `git
     add -A`, never a bare `git commit`.

     **Validator round 1 (2026-09-12): two blocking findings, both confirmed by
     independent reproduction before this round started, both fixed here.**

     **B1.** Six tests looped over a recorded array with no length check:
     `test_decode_headway_bin_matches_the_recorded_grid`,
     `test_decode_speed_bin_matches_the_recorded_grid`,
     `test_bin_index_matches_the_recorded_grid`,
     `test_neutral_cooperation_matches_the_recorded_values`,
     `test_action_values_and_their_order_match`, `test_action_profiles_match`. `derive()`
     accumulates mismatches inside `for` loops; a loop that iterates zero times
     contributes zero mismatches and an empty list, so emptying `decoders.headway_bin_s`
     (3 entries), `decoders.speed_bin_mps` (15), `decoders.bin_index` (8) and
     `neutral_cooperation` (2) in the committed golden file left the suite at 73 passed /
     1 skipped, byte-identical to clean -- reproduced independently before this round.
     The write side had the matching hole: setting `SPEED_BIN_FREE_FLOWS`,
     `BIN_INDEX_VALUES` and `NEUTRAL_FREE_FLOWS` to `()` in the generator made it print
     "the reference and the vendored contract agree on every recorded quantity" and exit
     0, and mutation M4 (`bin_index`'s `>=` flipped to `>`) survived the full suite with
     the collected count unchanged.

     Fixed by adding the population check each test was missing: the four grid tests
     now assert their recorded length explicitly (`headway_bin_s` 3, `speed_bin_mps` 15,
     `bin_index` 8, `neutral_cooperation` 2), and the two action tests assert
     set-equality of the recorded head/profile names against
     `sim_contract.ACTION_HEADS`/`ACTION_PROFILES`. The generator now refuses to write
     when a grid constant (`SPEED_BIN_FREE_FLOWS`, `BIN_INDEX_VALUES`,
     `NEUTRAL_FREE_FLOWS`, or `sim_contract`'s own `HEADWAY_BIN_S`/
     `SPEED_BIN_OFFSETS_MPS`) is empty, or when an emitted section's length disagrees
     with what its source implies.

     Considered and not used: converting the six tests to `@pytest.mark.parametrize`
     over the recorded arrays, the form the finding suggested first. Checked directly
     (a throwaway `@pytest.mark.parametrize("x", [])` test, this pytest version, 9.1.1):
     an empty parametrized list produces one *skipped* test, not a failure -- the run's
     exit code stays 0. That does not "confirm the suite fails," the validation step the
     finding itself asks for, so the explicit length/set-equality assertion (the
     finding's own named fallback for where parametrize is awkward) was used for all six
     instead, uniformly, rather than mixing the two.

     Proved, not merely argued: re-emptying the four sections in the golden file now
     fails 5 of 75 tests (the four population assertions, plus
     `test_regeneration_leaves_every_pre_existing_case_byte_identical` -- round-1 S3's
     fix below reaching further than round-1 S3 alone asked), where before this round
     it was 73 passed / 1 skipped. Zeroing the three generator constants now prints
     `REFUSING: 3 disagreement(s)` naming all three, where before it wrote the file
     and exited 0. Both re-verified clean afterward (`git diff` empty on both
     `specs/sim_contract_golden_vectors.json` and
     `scripts/generate_sim_contract_golden_vectors.py`).

     (Disambiguation, added after the fact: "S3" names two different things in this
     record. Round-1 S3, just above and in the paragraph below, is the validator's
     should-fix finding about the regeneration test's 4-of-12-keys comparison. The
     plan's own sign-off S3 (`plans/plan_task143_contract_golden_vectors.md` Section
     12: "`EXPECTED_PYTHON_TESTCASES` equals the measured JUnit count") is a
     different item entirely, addressed in the sign-off checklist section below,
     round-1 S3 having nothing to do with it. Read "S3" bare in this record as
     round-1 S3 unless a sentence says "sign-off S3.")

     **B2.** `_baseline_python_testcases()` accepted pytest's returncode 1 (a suite with
     a failing test) but discarded the clean run's failing names (`_, total, usable =
     failing_tests_in(baseline_dir)`), keeping only the count. `run()` then returned the
     mutated run's failing names with no subtraction against that baseline, so any
     pre-existing failure in the clean tree scored every mutation CAUGHT by it and
     `survived` stayed empty -- confirmed with a planted `assert 1 == 2` test
     (`test_zz_preexisting_failure`, added and removed for the check, never committed):
     mutation M4 reported `CAUGHT (1) ... by test_zz_preexisting_failure`, `survived:
     0`, exit 0.

     Fixed: `_baseline_python_testcases()` now reads the failing names (still accepting
     returncode 1 long enough to do so) and refuses, naming them, the moment there are
     any -- never subtracting. The CAUGHT print used to show only `failed[0]`; it now
     shows every failing name, since with several tests failing the first name alone
     cannot say whether a mutation was caught by the test built for it or only by an
     unrelated one also failing.

     Proved: with the planted test present, `python3 scripts/remutate.py python
     --name="bin_index's boundary"` printed `refusing: the clean Python tree already has
     1 failing test(s) ... ['test_sim_contract.test_zz_preexisting_failure']` and
     exited 1, instead of scoring CAUGHT. With the plant removed, the same command
     reported `CAUGHT (2) ... by test_sim_contract.test_bin_index_matches_the_recorded_
     grid, test_sim_contract.test_regeneration_leaves_every_pre_existing_case_byte_
     identical` and exited 0 -- both names now visible, where the unfixed harness would
     have printed only the first.

     **A second route to the same failure, found while proving the first fix, not in
     the original finding.** This branch had two other agents committing to it
     throughout this round (144's fixer holding several files, including
     `safety_gate.py` and `test_safety_gate.py`, mid-edit). Running the fixed gate
     against the *live* worktree measured a momentarily clean baseline (no failing test
     at that instant, 2530 testcases) and began applying mutations against it. A `git
     status --porcelain` check at that same moment would have shown two modified
     tracked files; nothing in the gate checked it. Confirmed concretely, not
     hypothetically: a plain `pytest -q deployment/jetson/tests/` run against the live
     tree minutes later, with that same edit still in progress, reported 2 failed /
     2517 passed / 24 skipped in `test_safety_gate.py` -- exactly the shape of failure
     B2 exists to refuse against, reached by a second door. The live-tree gate run in
     progress at the time was interrupted by hand (SIGINT) rather than trusted;
     confirmed afterward that its `finally` restored `sim_contract.py` (`git diff`
     empty) and left no `.remutate-restore` sidecar. None of that run's printed numbers
     are quoted anywhere in this record.

     Fixed: `_refuse_if_tree_is_dirty()` now runs before any mutation is applied and
     refuses whenever `git status --porcelain` shows a modified tracked file (untracked
     files excepted), naming what is dirty. This gate must be run from its own `git
     worktree add --detach`, never a tree another agent is also committing to -- stated
     here because nothing in the tooling enforces it; the guard only refuses a live,
     already-dirty tree, it does not create an isolated one.

     **The gate's real verdict, re-run from a dedicated worktree** (`git worktree add
     --detach <scratchpad>/gate_worktree_143_round1 2936427`, `.venv` symlinked in, torn
     down after): baseline 2529 testcases, all 9 `sim contract:` mutations CAUGHT, 0
     survived, 0 inconclusive, 0 did-not-build, exit 0. `M6` (an `ACTION_VALUES` order
     reversed) and the two S4 additions below are now caught by
     `test_the_second_pinned_digest_covers_what_the_fingerprint_does_not` in addition to
     their original catchers -- S2 closing real ground, not only the coordinated-edit
     case it was written for. `M3` (the `HEADWAY_BIN_S` edge S2 was written against)
     is caught by that same new test plus its original two. Full table:

     | mutation | caught by (count) |
     |---|---|
     | M1 field order | 17 |
     | M2 a `FIELD_SCALES` value | 17 |
     | M3 a `HEADWAY_BIN_S` bin edge | 3 (incl. the new S2 test) |
     | M4 `bin_index`'s boundary | 2 |
     | M5 the inf-clamp constant | 14 |
     | M6 an `ACTION_VALUES` order reversed | 4 (incl. the new S2 test) |
     | M7 `COOPERATION_FIELDS` order | 16 |
     | S4: `SPEED_BIN_OFFSETS_MPS["slow"]` narrowed | 15 (spans `test_safety_gate.py`, `test_safety_gate_pipeline.py`, `test_score_safety.py` -- this value reaches real decisions, not only the golden file) |
     | S4: `LANE_DISTRIBUTION_LANES` reordered | 6 (incl. the D13 case added for exactly this) |

     **S2.** Added `PINNED_SECOND_FINGERPRINT` (`6ac1218184e2c3fa`), a digest over
     `HEADWAY_BIN_S`, `SPEED_BIN_OFFSETS_MPS`, `COOPERATION_FIELDS`,
     `LANE_DISTRIBUTION_LANES`, `ACTION_VALUES` and the `_number` inf-clamp's `200.0`
     (typed by hand; it has no module-level name to read) -- same idiom as
     `PINNED_FINGERPRINT`. Closes the coordinated-edit hole: `contract_fingerprint()`
     hashes only `LOCAL_OBS_FIELDS` and `FIELD_SCALES`, so changing e.g.
     `HEADWAY_BIN_S["larger"]` from 2.2 to 2.3 in both `sim_contract.py` and the golden
     file at once moved neither it nor any golden-vector comparison test -- reproduced
     independently before this round. The mutation table above shows this test also
     catches ordinary, single-file mutations it was not written for (M3, M6, both S4
     entries), not only the coordinated-edit case.

     **Round-1 S3** (not the plan's sign-off S3 -- see the disambiguation note
     above; the sign-off item is answered in the sign-off checklist section below).
     `test_regeneration_leaves_every_pre_existing_case_byte_identical` compared
     4 of the golden document's 12 top-level keys. Now asserts `fresh == DOCUMENT` after
     that subset check. Reaches further than this finding alone asked: re-emptying the
     golden sections for B1's proof above also fails this test now, since the freshly
     regenerated document no longer matches the tampered one wholesale.

     **S4.** Two mutations registered: `SPEED_BIN_OFFSETS_MPS["slow"]` narrowed −10.0 →
     −8.0 (caught by 15 tests, reaching well beyond the golden file into
     `test_safety_gate.py`/`test_safety_gate_pipeline.py`/`test_score_safety.py`) and
     `LANE_DISTRIBUTION_LANES` reordered (caught by 6 tests, including the D13 case
     added for exactly this -- without this entry nothing in the gate ever showed that
     case catches anything). Not registered: removing `_number`'s `if
     isinstance(value, bool): return float(value)` branch. It survives, but it is
     inert: both bool-valued fields (`is_active`, `uncongested_low_speed_flag`) have
     `FIELD_SCALES` of 1.0, so the branch removed and the `plain` branch it falls
     through to compute the same thing (`float(True)/1.0 == float(True)`). It would
     become separable only if a bool field were ever given a non-1.0 scale.
     Registering a mutation that can never be caught for the wrong reason is worse than
     not registering it.

     **Recorded, not fixed (validator round 1's should-record items).**

     `min_contextual_mps` is a hand-typed literal at
     `scripts/generate_sim_contract_golden_vectors.py:442` that
     `test_decode_speed_bin_matches_the_recorded_grid` feeds back in as an input rather
     than deriving independently; changing `decode_speed_bin`'s own default from 12.0
     to 8.0 leaves the suite unchanged, since every call site -- this test, the
     generator's own derivation, and all three production call sites (`advisory.py`,
     `safety_gate.py` x2) -- passes the value explicitly. It changes no device number
     today. `neutral_cooperation`'s expected values are likewise hand-typed in the
     generator, defensibly, since their source is `specs/observation_schema.md`, a spec
     document rather than executable code. The generator's own docstring ("every
     recorded quantity is derived twice") overstates the case by these two.

     A residual risk the validator found and this task is not the place to close:
     `local_density_bin` and `local_mean_speed_bin` come from `bin_index(value, edges)`
     where the edges are configuration on both sides -- `config.yaml:74-75`
     (`density_bin_edges_veh_per_km: [12.0, 30.0]`, `mean_speed_bin_edges_mps: [8.0,
     18.0]`) on the device, `SensingConfig` plus the experiment YAML in the simulation.
     They agree today and nothing checks it. Retrain under different edges and
     `LOCAL_OBS_FIELDS`, `FIELD_SCALES`, `sim_contract.py` and `SIM_COMMIT` are all
     untouched, every golden test still passes, the bundle still loads -- and
     `local_density_bin = 1` means a different density than it did in training.
     `sensing.range_m` is the same shape of risk. This belongs to the bundle manifest
     task 145 owns, not to this task's contract. Also worth recording:
     `actor_runtime.py:95` (`exported = self.manifest.get("contract_fingerprint")`)
     accepts a manifest carrying no `contract_fingerprint` key at all without complaint
     (`exported is None` short-circuits the check), so the device-side half of this
     pair's coverage is conditional on the loaded bundle actually carrying one.

     19 of the 74 tests measured before this round cannot be reached by any mutation of
     `sim_contract.py`: the 14 `test_vector_sha256_matches_the_recorded_values`
     parametrizations compare the golden file against itself; four more
     (`test_the_file_declares_itself_frozen`, `test_case_names_are_unique`,
     `test_the_note_carries_the_freeze_rule`,
     `test_the_raw_json_contains_no_bare_nan_or_infinity_token`) check only properties
     of the file itself; and `test_the_fingerprint_is_stable_across_calls` compares
     `contract_fingerprint()` to itself, a tautology. All 19 are legitimate integrity
     checks and stay; "74 passed" (75 after this round's own new test) overstates this
     file's coverage of the vendored contract by 19, not by an error.

     **L3, closed this round.** `ARCHITECTURE.md` was held by task 144's fixer for most
     of this round; released mid-task (144's fixer's own edits to the file had landed at
     `da764ca`) and addressed immediately. Section 6 no longer tells a reader to run the
     test "on a machine where the sim imports" (no such machine exists) or names
     `test_actor_state_dict_layout_matches_sim` (`2692c3b` deleted it, before
     `test_actor_state_dict_layout_matches_the_recorded_layout` existed). Rewritten to
     say what is true now: the golden file is the reference, `test_sim_contract.py`
     imports no simulation module and runs on every machine, and regeneration against
     `d477dba` is the separate step needing `git` and `torch` -- and skips where either
     is absent, which includes the device (S2's own limit as recorded above, named so
     this section cannot be read as claiming the live comparison also runs on the
     Jetson). Committed alone at `5788aa1`.

     **Suite counts.** Isolated worktree at this round's commit (`2936427`, via `git
     worktree add --detach`, not the live tree): baseline 2529 testcases for the
     mutation gate (above). Live tree, read after task 144's fixer's concurrent commit
     landed (`dedd3c1`): `deployment/jetson/tests/` 2522 passed / 24 skipped / 0 failed
     in 81.3s; `deployment/jetson/tests/test_sim_contract.py` alone, 75 passed / 0
     skipped (74 before this round, +1 from S2's new test). The 24 skips are the same
     USB-gated set as every prior measurement in this file; `test_sim_contract.py` has 0
     skips on this machine (both `git` and `torch` are present).

     **Commits** (branch `mainz-src-port`): `5788aa1` (`ARCHITECTURE.md`, alone, L3),
     `2936427` (B1/B2/S2/S3/S4 fixes across `test_sim_contract.py`,
     `generate_sim_contract_golden_vectors.py`, `remutate.py`, including the dirty-tree
     guard). Each by explicit pathspec, never `git add -A`, never a bare `git commit`.
     `deployment/jetson/policy/safety_gate.py` and
     `deployment/jetson/tests/test_safety_gate.py` were left exactly as task 144's
     fixer's own work throughout -- confirmed untouched by `git diff` immediately before
     each commit above, and clean (0 diff) once that fixer's own `dedd3c1` landed.

     **Validator round 2 (2026-09-12): one blocking finding, prototyped by the
     validator; four should-fix/accept items; two rulings confirmed rather than
     changed.**

     **R2-1 (blocking).** Round-1 S2's digest hand-listed five constants
     (`HEADWAY_BIN_S`, `SPEED_BIN_OFFSETS_MPS`, `COOPERATION_FIELDS`,
     `LANE_DISTRIBUTION_LANES`, `ACTION_VALUES`); `sim_contract.py` has eleven
     uppercase module-level constants. `ACTION_HEADS`, `ACTION_PROFILES` and
     `FORCED_ACTIONS` were in no digest at all -- shipped short the day round-1 S2
     was written, not a later drift. Reproduced independently before this round, on
     a tree without `.git`: changing `FORCED_ACTIONS["merge_mode"]` from `"normal"`
     to `"hold_lane"` in both `sim_contract.py` and the golden file left
     `test_sim_contract.py` at 74 passed / 1 skipped, identical to clean, while
     `indices_to_action({})["merge_mode"]` -- the fallback the runtime applies to
     any head the policy does not emit -- silently became `"hold_lane"`.

     **R2-2 (the fix).** Adopted a self-maintaining construction:
     `_second_pinned_digest()` now derives its payload from every uppercase,
     non-underscore name in `vars(sim_contract)`, excluding only
     `HASHED_BY_CONTRACT_FINGERPRINT` (`{"LOCAL_OBS_FIELDS", "FIELD_SCALES"}`,
     what `contract_fingerprint()` itself hashes) and `PINNED_SEPARATELY`
     (`{"SIM_COMMIT"}`, pinned by `test_the_file_is_for_this_sim_commit` instead,
     deliberately -- the sim moves for reasons that do not touch this contract).
     A constant added to `sim_contract.py` now moves this digest and fails the
     pinned-literal test below it, by design -- documented in the test's own
     docstring as a decision point the adder must resolve (put it in one of the
     two exclusion sets, or accept that it is now covered here), not as noise to
     silence.

     Added `test_hashed_by_contract_fingerprint_matches_what_it_actually_reads`,
     which checks `HASHED_BY_CONTRACT_FINGERPRINT` against
     `contract_fingerprint()`'s own source text (`inspect.getsource`) rather than
     trusting it as a hand-maintained restatement, bidirectionally: a name
     missing from the claim that the source references fails it, and so does a
     name in the claim the source no longer references. This closes R2-1 by a
     third route (a hand-maintained exclusion list drifting from what the
     fingerprint actually hashes, silently exempting a constant from both
     digests) -- confirmed independently: the validator added `FORCED_ACTIONS` to
     `HASHED_BY_CONTRACT_FINGERPRINT` and this test failed. One documented
     brittleness, found by the validator and recorded in the test's own
     docstring rather than fixed: the check is textual, so it cannot distinguish
     a constant the function hashes from one merely *named* in its docstring.
     `SIM_COMMIT` currently passes only because it happens to also be in
     `PINNED_SEPARATELY` -- it appears in `contract_fingerprint()`'s own
     docstring, in the sentence explaining why `SIM_COMMIT` is deliberately not
     hashed. Adding a sentence naming any other constant there, for any reason,
     would fail this test spuriously -- loud, not silent, so left as a known
     limit rather than "fixed" into something that greps for code and skips
     comments.

     `PINNED_SECOND_FINGERPRINT` recomputed fresh from this tree at this commit
     (`98c2e16b56ed6927`) -- explicitly not copied from the validator's own
     reproduction (`265b73ae2c792da0` -> `3eef713f1e8300fc`) or from the
     coordinator's independent check (`98c2e16b56ed6927` clean ->
     `5f6426f33b7c11e0` under the `FORCED_ACTIONS` edit), even though the clean
     value coincides with the coordinator's: both were derived by running the
     identical construction against the identical clean source, not copied.
     Both pinned-literal assertions (`PINNED_FINGERPRINT` and
     `PINNED_SECOND_FINGERPRINT`) now say explicitly, in the failure message and
     in a comment giving the exact command, to regenerate ON A CLEAN TREE and
     never from a prototype, a colleague's run, or memory.

     Re-ran the `FORCED_ACTIONS` reproduction against the fix (mutating both
     `sim_contract.py` and the golden file, restoring both in a `finally`,
     confirmed byte-identical afterward): 2 failed / 74 passed, where before the
     fix it was 74 passed / 1 skipped, identical to clean.
     `indices_to_action({})["merge_mode"]` still reads `"hold_lane"` under the
     mutation -- the device-behaviour change is real and the test suite now
     says so.

     **R2-3 (should fix).** `bin_index`'s boundary direction had no semantic
     pin: replacing the golden grid's 8 values with 8 non-boundary values
     (count preserved, so round-1's `assert len(entries) == 8` still passed)
     and then applying the `>=` -> `>` mutation left the suite unchanged -- the
     three values that would have discriminated (0.5, 1.0, 2.0) were simply
     absent from the grid. Added a 3-line pin in test source
     (`test_bin_index_counts_edges_at_or_below_the_value`) that calls
     `sim_contract.bin_index` with hardcoded edges and hardcoded expectations
     and reads no recorded file, so no regeneration and no file edit can rewrite
     it. Verified it fails against the mutation directly, applied by hand and
     reverted.

     **R2-4 (should fix).** Four of the round-1 generator checks compared two
     quantities that move together by construction -- e.g.
     `len(bin_index_grid) != len(BIN_INDEX_VALUES)`, when `bin_index_grid` is
     built by appending exactly once per element of that same `BIN_INDEX_VALUES`
     -- so no edit could ever make them disagree; shortening `BIN_INDEX_VALUES`
     from 8 to 5 wrote the file and exited 0. Deleted the four (headway_bin_s,
     speed_bin_mps, bin_index, neutral_cooperation), keeping only the five
     emptiness checks, which do fire. Chose deletion over pinning the expected
     sizes as literals -- a guard that can never fire reads as protection while
     providing none, which is worse than no guard -- because the real
     protection against a shortened grid is downstream and independent:
     `test_sim_contract.py`'s `assert len(entries) == 8` (etc.) compares against
     a literal that does not move with the generator's own constants. Verified
     both halves directly: shrinking `BIN_INDEX_VALUES` to 5 still writes and
     exits 0 at the generator, and the resulting 5-entry file still fails the
     test-side length assertion.

     **R2-5 (accept, cheap fix).** An untracked `pytest.ini` (or `conftest.py`,
     `setup.cfg`, `tox.ini`, `pyproject.toml`, an untracked `test_*.py`) changes
     what pytest collects without `git status --porcelain` ever calling the
     tree dirty in the sense round-1's B2-second-route check covers, and
     contaminates `_baseline_python_testcases()`'s self-measured baseline and
     every mutated run identically -- a false SURVIVED, never a false CAUGHT,
     because the two never disagree with each other. Reproduced: an untracked
     `pytest.ini` with `addopts = --ignore=...test_sim_contract.py` measured a
     baseline of 2470 testcases against a true 2545, and a real mutation inside
     the ignored module then reported `*** SURVIVED ***`.
     `_refuse_if_tree_is_dirty()` now also refuses on an untracked file matching
     `_COLLECTION_AFFECTING_NAMES`, naming it. Verified directly: the
     reproduction above now exits 1, naming `pytest.ini` and the reason, instead
     of writing a false SURVIVED.

     **The trade this makes explicit, stated because it was previously only
     implied.** Round 1 replaced a hand-typed `EXPECTED_PYTHON_TESTCASES`
     literal with a per-run, self-measured baseline to fix staleness (a
     literal drifting behind a growing suite had silently made every Python
     mutation report INCONCLUSIVE). That fix traded away a different piece of
     detection: an absolute expectation notices a suite that has already
     shrunk before either run starts; a self-measured baseline, by
     construction, measures whatever the tree currently collects as ground
     truth and cannot tell a legitimately smaller clean tree apart from one an
     untracked file has already shrunk. `_COLLECTION_AFFECTING_NAMES` recovers
     part of that detection, for the named files and the `test_*.py` shape
     only -- not the general case the old literal covered by being a number
     independent of the tree entirely. This is recorded as a real, partial
     trade, not as the gap being fully closed.

     **R2-6.** Moved `_refuse_if_tree_is_dirty()`'s call below the
     `ARGV`/`WANTED` parse and scoped it to `kinds` (skipping entirely when
     `"python"` is not among the selected kinds). A `git archive` mirror with no
     `.git` at all previously made `git status` itself exit 128, refusing every
     kind including Gradle-only runs that touch no Python and depend on neither
     check the guard performs -- exactly the friction that makes someone
     disable a guard rather than live with it. Verified three ways in a
     dedicated worktree: (a) clean tree, kind `python` -- proceeds; (b)
     untracked `pytest.ini` present, kind `python` -- refuses, naming it; (c)
     the same untracked `pytest.ini`, kind `transport` -- proceeds (the
     check is skipped entirely). Verified a fourth way against a true `git
     archive` mirror with no `.git`: kind `transport` proceeds (`survived: 0
     (none)`, exit 0); kind `python` still refuses, naming the `git status
     exited 128` failure. Recorded as a real behaviour change, not undone: a
     validator can no longer exercise this harness's Python guards inside an
     archive mirror at all -- it can only run the Gradle kinds there now.

     **Two rulings the validator confirmed rather than changed, both worth
     recording as settled rather than re-litigated next round.** Refusing
     (not subtracting) on a red baseline: subtraction would define the verdict
     relative to a tree already agreed not to understand, with two failure
     modes and nowhere to put either (a mutation caught only by an
     already-failing test subtracts to a false SURVIVED; a mutation that
     *fixes* a pre-existing failure produces fewer failures than baseline,
     which subtraction cannot represent as a count at all). And this task's own
     gate baseline of 2,529 (round 1's isolated-worktree measurement) reproduced
     independently, in the validator's own mirror.

     **The harness itself has no test coverage, stated as the general fact
     rather than left as one function's footnote (validator round 2).** Item
     143's original text above says "no test in this repository would fail if
     `_purge_pycache()`'s call sites were removed" -- true, but naming only one
     function invites the reader to assume the others (`_refuse_if_tree_is_
     dirty()`, `_COLLECTION_AFFECTING_NAMES`, the baseline-failure refusal
     added across both validator rounds) are covered elsewhere. They are not:
     **no test in this repository exercises `scripts/remutate.py` at all**, so
     every guard in it can be removed or emptied with the full suite green.
     Confirmed directly, not inferred: the validator set
     `_COLLECTION_AFFECTING_NAMES = frozenset()` and the full Jetson suite
     reported 2,527 passed / 28 skipped -- the unmutated count -- SURVIVED.
     `scripts/remutate.py` is the harness every mutation verdict on this
     project rests on, and every fix made to it across both validator rounds is
     verified operationally (by the reproductions recorded in this entry) and
     not by an automated assertion, because none exists. Closing that gap is
     outside this task's scope and is not attempted here; the sentence in the
     original entry above is corrected to say so generally rather than naming
     only `_purge_pycache()`.

     **Sign-off checklist** (`plans/plan_task143_contract_golden_vectors.md`,
     Section 12), the four items the validator identified as measurable and not
     yet answered with a number or a name in this record.

     - **S1** (`test_sim_contract.py` reports 0 skipped): `77 passed in 3.30s`
       -- 0 skipped (both `git` and `torch` present on this machine).
     - **S2** (the full Jetson suite passes, 24 skips, no non-hardware skip):
       `2532 passed, 24 skipped, 95 warnings in 81.65s`. JUnit skip-reason
       breakdown: 24x `"no USB device attached: no device attached (adb
       devices -l lists none in state 'device')"`, 0 of any other reason.
     - **Sign-off S3** (not round-1 S3 -- see the disambiguation note earlier
       in this entry): **superseded, not answered as the item asks, with the
       reason recorded here rather than ticked anyway.** `EXPECTED_PYTHON_
       TESTCASES` no longer exists in `scripts/remutate.py` (`grep` finds
       nothing); round 1 replaced it with `_BASELINE_PYTHON_TESTCASES`,
       measured per invocation by `_baseline_python_testcases()`, precisely
       because the hand-typed literal had gone stale (2060 against a suite
       already at 2272, turning every Python mutation INCONCLUSIVE silently --
       recorded earlier in this entry). The sign-off item asked for evidence
       that a hand-maintained constant matched reality; the task concluded
       that constant was the wrong mechanism and removed it, which answers the
       underlying question -- does the harness know how big the suite really
       is -- rather than the literal item. What plays its role now: the
       per-run baseline, for a mutation that changes what gets collected
       mid-run, and `_COLLECTION_AFFECTING_NAMES` (R2-5, above) for the one
       detection the old literal had that a self-measured baseline alone
       cannot recover -- a suite that had already silently shrunk before the
       baseline is even measured, and only for the named files and the
       `test_*.py` shape, not the fully general case an absolute literal
       covered.
     - **S4** (M1 CAUGHT): `test_slot_names_match_encoded_slot_names` (named in
       this entry's original mutation table above; re-confirmed as a catcher
       in both this task's and the validator's round-2 gate re-runs).
     - **S5** (M2 CAUGHT): `test_field_scales_match_in_both_directions` (same).

     **The gate's real verdict after both validator rounds**, from a second
     dedicated worktree (`git worktree add --detach` at `112a29a`, `.venv`
     symlinked in, torn down after): baseline 2555 testcases, all 9 `sim
     contract:` mutations CAUGHT, 0 survived, 0 inconclusive, exit 0. `M3` and
     `M4` now show a fourth/third catcher each from this round's new tests
     (`test_the_second_pinned_digest_covers_what_the_fingerprint_does_not` for
     `M3`; `test_bin_index_counts_edges_at_or_below_the_value` for `M4`); `M6`,
     `M7` and both S4-registered mutations are likewise now also caught by the
     new digest test. Full table:

     | mutation | caught by (count) |
     |---|---|
     | M1 field order | 17 |
     | M2 a `FIELD_SCALES` value | 17 |
     | M3 a `HEADWAY_BIN_S` bin edge | 3 (now incl. the R2-2 digest test) |
     | M4 `bin_index`'s boundary | 3 (now incl. the R2-3 semantic pin) |
     | M5 the inf-clamp constant | 14 |
     | M6 an `ACTION_VALUES` order reversed | 4 (now incl. the R2-2 digest test) |
     | M7 `COOPERATION_FIELDS` order | 16 (now incl. the R2-2 digest test) |
     | S4: `SPEED_BIN_OFFSETS_MPS["slow"]` narrowed | 17 (grew from 15 as task 144's fixer landed more safety_gate tests concurrently) |
     | S4: `LANE_DISTRIBUTION_LANES` reordered | 6 (now incl. the R2-2 digest test) |

     **Commits** (branch `mainz-src-port`): `112a29a` (R2-1/R2-2, R2-3, R2-4,
     R2-5, R2-6 across `test_sim_contract.py`,
     `generate_sim_contract_golden_vectors.py`, `remutate.py`), `c8bfc61`
     (documented the textual-match brittleness in the R2-2 introspection test's
     own docstring, alone). Each by explicit pathspec, never `git add -A`,
     never a bare `git commit`. `deployment/jetson/policy/safety_gate.py` and
     `deployment/jetson/tests/test_safety_gate.py` were left exactly as task
     144's fixer's own work throughout this round too -- confirmed untouched by
     `git diff` before each commit above.

144. **The safety and etiquette layer runs nowhere.** Implemented 2026-09-12
     (implementer-144) against `plans/plan_task144_safety_layer_on_device.md`. The plan's
     sign-off checklist (its own last section) is unaddressed, by the plan's own statement,
     and this record does not close it.

     The paper plan's DESIGN section names `src/safety/safety_layer.py`, `etiquette.py` and
     `constraints.py` and says "the advisory is bounded before it reaches the driver". It
     also records, correctly, that the layer is unexercised in simulation, and concludes
     "the gate is real on the device and implicit in the simulation".

     `apply_safety_layer` has exactly one caller in the repository:
     `tests/test_safety_layer.py`. Nothing under `deployment/` imports it -- the two matches
     in that tree are a comment saying one observation field mirrors `etiquette.py`, and two
     docstrings citing `plan_deployment.md`. The advisory the driver is shown comes from
     `AdvisoryDecoder.decode`, whose only bound is `max(12.0, base_speed + offset)` at
     `policy/advisory.py:87`. Task 87 established the same absence in simulation. The layer
     is unexercised in both.

     It also cannot run on the Jetson as written: `safety_layer.py` imports
     `src.envs.base_ctde_env` and `src.envs.wrappers`, which is the simulation stack
     `policy/sim_contract.py` exists to keep off the device.

     **The task:** vendor the filter the way the encoder is vendored, call it on the advisory
     path, and measure how often it clamps and by how much. That measurement is also the
     paper plan's third open item, which asks for the gate's firing rate.

     **The measurement the paper's third open item asks for cannot be obtained from what is
     in the repository, and the plan said so before any code was written.** Measured (and
     re-measured here, independently, with `deployment/jetson/score_safety.py`) over the four
     recorded runs in `outputs/task42_usb/` (3,913 tick records): zero of the layer's twelve
     rules had evidence for their inputs on any of the 3,913 ticks. Eleven rules read a field
     whose provenance is in `provenance.SUBSTITUTED` on every tick. The twelfth,
     `target_lane_front_gap`, is evaluable on 1,229 of the 3,913 (the whole 2026-09-02 run,
     the only one where that slot was tagged `derived` rather than substituted) and fires on
     0 of those 1,229 -- the recorded value is `inf` on all of them. So the deliverable is a
     per-rule **evaluability census**, not a clamp rate, and `score_safety.py` refuses to
     print a rate for a rule with zero evaluable ticks rather than reporting "0%".

     **What a firing rate would have said, if one had been reported**, also reproduced by
     `score_safety.py` over the same 3,913 ticks: the recorded action (`fast` on every tick)
     and the same action with `desired_speed_bin` forced to `nominal` both clamp 0 of 3,913;
     forcing it to `slow` clamps 3,913 of 3,913, all attributed to
     `etiquette_blocked_action: low_speed_uncongested`, and RAISES the recommended speed
     20.0 -> 22.0 m/s. The reason is algebraic: with the configured `free_flow_speed_mps` of
     30.0, `decode_speed_bin` gives `{slow: 20.0, nominal: 27.0, fast: 30.0}` and the
     etiquette threshold is `30.0 - 8.0 = 22.0`; the predicate `target < free_flow - 8`
     reduces to `desired_speed_bin == "slow"`. `lane_withheld` is `not_evaluable` on all
     3,913 ticks in every arm: the policy's recorded action is `prefer_left_if_safe`, and with
     no rear camera and no lane-change detector the lane advisory is withheld on every tick
     of every recorded run, permanently, regardless of what the policy proposes.

     **What was built.** `specs/safety_contract_golden.json` freezes `SafetyConstraints` (16
     fields) and `SafetyContext` (23 fields, in order, with defaults) from `src/safety/`, with
     a hash over both recomputed and checked against itself by both sides'
     tests -- `deployment/jetson/tests/test_safety_contract.py` (the vendored copy,
     unconditional, no `importorskip`) and `tests/test_safety_contract_matches_golden.py`
     (`src/safety/` itself), replacing the comparison-against-a-deletable-original idiom
     task 143 found goes vacuous. `deployment/jetson/policy/safety_gate.py` vendors
     `SafetyConstraints`/`SafetyContext`/`SafetyState`/`SafetyDecision`/`apply_safety_layer`/
     `physical_control_command`/`safety_penalty_terms` verbatim, calling
     `sim_contract.decode_speed_bin`/`decode_headway_bin` rather than carrying a second copy
     of either decoder (`lane_preference_to_action` is inlined instead: a 3-entry literal used
     nowhere else on the device, so a second copy of it carries none of the two-decoder risk).
     It adds `SafetyInputs` (decision 3's per-field input-class partition -- configured /
     evidence-required / structurally-absent, built from an `ObservationResult`) and
     `evaluate_rules`/`run_safety_gate`, which have no `src/safety/` counterpart at all: all
     twelve rules evaluated as independent total predicates for the record (never
     short-circuited by chain position the way `apply_safety_layer`'s own lane decision is),
     plus the withhold-the-lane-action-when-a-guard-is-not_evaluable behavior decision 3 adds
     on top of the unchanged elif chain. `pipeline.step` runs the gate between
     `advisory_decoder.decode` and `set_target_headway`, unconditionally, on every tick;
     `advisory.recommended_speed_mps`/`recommended_speed_display`/`lane_text` are overwritten
     to the bounded values (`headway_target_s` is deliberately left at the raw decoded value,
     since `set_target_headway` must keep feeding back what the policy was trained against,
     not what the gate bounded it to). `Tick` gained a required `safety_gate` field (no
     default, same reasoning as `jetson_ms`) and a `"safety"` block in `to_record()` beside
     `"advisory"`; `PipelineStats`/`stages` gained `gate_ms`/`"gate"`, always
     `StageTiming.measured` -- the gate runs the full rule evaluation on every tick with no
     early-return path, so there is no refusal to mistake for an inference the way an earlier
     version of `dsrc_infer` did (found and fixed by the concurrent task-145 fixer in this
     same file; flagged here because the hazard is the same shape). `config.yaml` gained
     `safety.withhold_lane_when_not_evaluable` (default `true`), threaded through
     `run_demo.py`'s `_build_rest` as a keyword argument -- the existing six-positional-argument
     `PerceptionPolicyPipeline(...)` call and the missing `dsrc_runtime`/`dsrc_segment_builder`/
     `dsrc_advisory_decoder`/`here_feed_source` wiring there are task 145's own separate gap,
     left untouched. `deployment/jetson/score_safety.py` follows `score_shadow.py`'s
     refuse-before-misleading idiom: the per-rule census before any rate; an incumbent
     `safety` block (once one exists) must reproduce exactly from its own recorded inputs
     before anything is reported; a log with no `safety` block at all (every run recorded
     before this task) has every number labelled `counterfactual`.

     **The golden-file gate check (plan step 4), performed before trusting it.** Mutated
     `lane_change_dwell_s` 15.0 -> 99.0 in the vendored copy, ran
     `test_safety_contract.py`, watched `test_vendored_safety_constraints_matches_golden`
     fail, reverted, watched all four tests in that file pass again.

     **The step-9 invariant, and a defect in its own first draft, found and corrected before
     trusting it.** The pipeline-level test asserting `advisory.recommended_speed_mps ==
     safety_gate.bounded_speed_mps` initially compared the two values on an unforced
     random-actor rollout, and passed even after `pipeline.step`'s `replace()` call was
     deliberately broken (fed the raw, ungated speed back to itself) -- because the random
     actor rarely proposes a speed the gate actually changes, so both sides were silently
     reading the same wrong value. Corrected by monkeypatching the actor to force
     `desired_speed_bin="slow"`, which the gate is known to clamp; re-verified against the
     same deliberate break (caught: 20.0 != 22.0), reverted, passes clean. A second,
     independent mutation (offsetting `safety_gate.run_safety_gate`'s `bounded_speed_mps` by a
     constant) was caught by `tests/test_safety_gate.py`'s existing pinned-value unit tests,
     not by the pipeline-level one -- the two layers catch different classes of defect, and
     only the corrected pipeline-level test catches a wiring regression specifically.

     **The safety-block size estimate (E6 in the plan) was wrong, and the plan said to check.**
     Measured over 300 replayed ticks on this laptop (a mix of ticks with and without a
     tracked leader, so both the always-not_evaluable rules and the one evaluable one are
     represented): the `"safety"` block is 2,330-2,696 B (mean 2,576 B, p50/p95 2,696 B)
     against a 9,837 B mean full tick record -- **+26.2% on the mean record, not the +15.0%
     the plan estimated from a hand-drafted example.** The plan's own acceptance band was
     1,300-1,800 B; the measured figure is outside it by roughly 1.5-1.8x. The extra weight is
     the twelve-rule `missing` list plus a `<field>_source` entry per missing field, which the
     plan's own drafted example under-counted relative to a real not-evaluable-on-everything
     tick; the plan's stated reason for choosing this fuller form over a cheaper 1,103 B
     variant (dropping each substituted field's provenance class "is the one thing a reader of
     this block needs") is unaffected by the estimate being wrong, so the fuller form is kept.
     `gate_ms` itself: mean 0.035 ms, p50 0.034 ms, p95 0.038 ms on this laptop -- three orders
     of magnitude under any latency budget this rig has, as expected for pure Python dict and
     dataclass work with no I/O.

     **One place this record's own rule-to-field mapping differs from the plan's draft E4
     table, found while implementing decision 3 rather than the earlier evidence section.**
     Decision 3 classifies `free_flow_speed_mps` as class (A) -- "accepted as given, never
     makes a rule not_evaluable" -- because the gate must decode the speed bin against the
     same base `AdvisoryDecoder` used, and that base is `obs.cooperation.segment_target_speed`,
     regardless of its own provenance tag. The plan's E4 table nonetheless names
     `cooperation.segment_target_speed = fallback_neutral` as the input that blocks
     `low_speed_uncongested`. Implemented per decision 3 (the later, authoritative section):
     `low_speed_uncongested`'s actual blocker in this implementation is
     `local_density_veh_per_km`, via the `derived_empty` + `last_detection_age_s` carve-out
     decision 3 also specifies (the corpus's `last_detection_age_s` is `None` on all 3,913
     ticks, so density is never evidence either way). The bottom-line count this record and
     `score_safety.py` both report -- 0 of 3,913 evaluable -- is unchanged either way; only the
     named blocking field differs from the plan's draft table, and decision 3 is what this
     implementation follows since it is the section the plan itself calls authoritative over
     the evidence that preceded it.

     **Open item 3 (what the driver sees on `emergency_override`) implemented as the plan's
     own stated recommendation, flagged for confirmation because it is untestable against any
     recorded tick.** `Advisory` gained `speed_display_withheld: bool` (default `False`,
     additive), set from the gate's `emergency_override`. `advisory.recommended_speed_mps`
     itself is left as a plain, always-defined float, per decision 2's own requirement that it
     keep meaning "the number shown to the driver" unconditionally (four call sites read it as
     exactly that); a display is free to withhold the number when the new flag is set rather
     than the data model carrying an optional field four existing readers would have to be
     taught to handle. `emergency_override` fires on 0 of 3,913 ticks in every arm measured, so
     this path has not been exercised by anything on this rig.

     **Test counts.** Baseline mirror at `b1758a7` (this task's parent commit, established
     fresh rather than trusted from an earlier record): `deployment/jetson/tests/` 2,328
     passed / 28 skipped; `tests/test_safety_layer.py` 18 passed. Mirror at `b1dfd09` (this
     task's last commit, on a branch three agents committed to concurrently):
     `deployment/jetson/tests/` 2,390 passed / 28 skipped, 0 failed;
     `tests/test_safety_layer.py` + `tests/test_safety_contract_matches_golden.py` 21 passed.
     This task's own new tests: 53, across
     `deployment/jetson/tests/test_safety_contract.py` (4),
     `deployment/jetson/tests/test_safety_gate.py` (25),
     `deployment/jetson/tests/test_score_safety.py` (7),
     `deployment/jetson/tests/test_safety_gate_pipeline.py` (7),
     `deployment/jetson/tests/test_eval_run_safety.py` (7), and
     `tests/test_safety_contract_matches_golden.py` (3). The remainder of the 62-test increase
     in `deployment/jetson/tests/` is concurrent work by other agents on tasks 142 and 145,
     landed on the same branch during this task's own commits; no test was removed or skipped
     to reach these numbers, and the skip count (28) is unchanged from baseline.

     **What was not done.** No drive with this gate running exists or was made -- the plan's
     own scope excludes it ("anything that needs a new drive"), and the paper's third open
     item stays open for the same reason the plan states: no run in the repository has a
     leader track (predates task 63's camera-rotation fix), so no corpus exists on which a
     real firing rate could ever be measured, gate or no gate. The plan's own sign-off
     checklist -- confirming decision 3's fail-closed lane default, open items 1-3, the
     zero-clamp-rate deliverable, decision 7 (Mainz is a separate task), and findings F2/F3
     staying unfixed -- is for the user.

     **Validator round 1 (2026-09-12): two blocking findings, confirmed and fixed, plus eight
     further fixes.**

     **F1 (blocking), fixed.** `run_safety_gate` fed `apply_safety_layer` the SAME context
     `evaluate_rules`'s census read, built from the observation's own values including its
     substituted defaults for not-evidence fields. Reproduced before the fix:
     `local_density_veh_per_km` substituted at `0.0` (below the 12.0 veh/km uncongested
     threshold) raised `bounded_speed_mps` from the proposed `20.0` to `22.0`, while
     `low_speed_uncongested`'s own census entry read `not_evaluable` and all twelve rules read
     `not_evaluable`. Fixed by `SafetyInputs.inert_context()`: a second `SafetyContext`, read
     only by `apply_safety_layer`, where every field lacking evidence is replaced by a value
     declared in `INERT_CONTEXT_VALUES` (beside `RULE_READS`), each derived from the one
     comparison the field's rule makes, not guessed. After the fix, the same reproduction gives
     `bounded_speed_mps == proposed_speed_mps == 20.0`, `delta_speed_mps == 0.0`,
     `low_speed_uncongested` still `not_evaluable`. Checked: no field is read by two rules with
     opposite senses (the front/rear gap-and-relative-speed pairs are each read by two rules,
     both with the SAME sense -- larger gap, non-closing relative speed, is inert for both).

     **Fix 2, the invariant that proves F1.** Asserts: every reason in `apply_safety_layer`'s
     raw diagnostics names a rule the independent census calls `RULE_FIRED` -- checked on
     genuine firings for the speed path, the lane path, and `emergency_override`, plus the
     contrapositive (all twelve rules `not_evaluable`: `bounded_speed_mps ==
     proposed_speed_mps`, the lane action is `None` by withholding rather than by masking,
     `emergency_override is False`), plus the headway carve-out (`create_gap`'s merge bonus is
     the only unconditional transformation of a value the gate REPORTS -- see the R2-2
     correction below; the first draft of this claim omitted that qualifier and was wrong).
     Every new assertion was checked by neutering the mechanism it pins (`inert_context`
     monkeypatched back to the raw context) and confirming the reproduction's own assertion
     then failed.

     **`forward_ttc` and `lane_changes_per_km`: found not evaluable while genuinely fired,
     fixed, corrected twice more, and the corrections are the more important record than the
     first fix.** Round 1 first reported this as a exception left for the plan owner
     (`forward_ttc`'s `RULE_READS` requires the always-class-(C) merge-conflict pair, so its
     census read `not_evaluable` even while a real leader correctly drove
     `emergency_override`). The coordinator overruled that: it is derivable, not a preference,
     and shipping it was task 144's own headline defect on one of its twelve rules. Three
     rulings followed, each correcting the previous one after the coordinator reproduced a
     counter-example against the fix just made:

     1. First ruling: evaluability is the pair (gap + relative speed together) that
        `_forward_hazard`'s `min()` selects, mirroring `INERT_CONTEXT_VALUES`. Reproduced as
        broken by the coordinator: `leader_gap` measured (3.0 m) with `leader_relative_speed`
        substituted still fires `emergency_override` via the GAP alone
        (`hazard_gap < min_front_gap_m`, one of `physical_control_command`'s two disjuncts),
        so requiring the pair as a unit suppresses a real emergency brake on a genuinely
        measured close gap -- and gap-measured/relative-substituted is the NORMAL state while
        a track is newly acquired (`observation_builder` tags gap `MEASURED` as soon as a
        leader exists, relative speed only once `rel_speed_valid`).
     2. Corrected principle: **evaluability must be computed over the inputs the predicate
        actually consulted on this tick, not the static list it might consult.** For
        `forward_ttc`, per disjunct: the gap disjunct consults only the operative gap; the ttc
        disjunct consults the gap and its relative speed, and only needs to be asked when the
        gap disjunct itself reads false on real evidence. `_forward_ttc_missing`
        (`policy/safety_gate.py`) implements exactly this.
     3. The same principle, applied to the state pair the coordinator found in parallel (see
        below), needed its own correction too: an inert value derived correctly (`0`/`()` for
        `lane_changes_last_km`/`lane_change_distances_m`) is not the same claim as "both
        fields are needed together" -- `_lane_change_count_exceeded` consults them as
        ALTERNATIVES (whichever branch `lane_change_distances_m`, after substitution, selects),
        never both.

     **The state pair: `inert_state()`, found missing entirely, not narrowed.** The coordinator
     derived, from the transitive closure of every function `apply_safety_layer` calls, that
     `SafetyContext` and `SafetyState` are the complete set of containers the decision reads
     sensed data from (the only module-level data touched anywhere in that closure is
     `sim_contract`'s two decoders, contract constants rather than sensed values).
     `inert_context()` neutralised the first; `run_safety_gate` was passing the second to
     `apply_safety_layer` RAW. Reproduced: a `SafetyState` with `last_lane_change_time_s=995.0`
     (`time_s=1000.0`, 5 s dwell against the 15 s default) masks a lane action while
     `lane_change_dwell`'s own census reads `not_evaluable` regardless of the real state's
     value; the same shape for `lane_changes_last_km`. Latent only because nothing on this rig
     ever advances these past their class defaults (`None`/`0`/`()`), which happen to already
     be inert -- the exact shape F1 itself had before a policy emitted `slow`, and it would not
     have stayed latent: `pipeline.py`'s own comment holds the `SafetyState` rather than
     rebuilding it "so a future detector has somewhere to write." Fixed:
     `SafetyInputs.inert_state(state)`, called by `run_safety_gate` alongside
     `inert_context()`.

     **`absolute_distance_m`: a fourth field, read by one rule, in no `RULE_READS` entry before
     this fix.** `_lane_change_count_exceeded`'s window branch reads it alongside
     `lane_change_distances_m`
     (`window_start = max(0.0, absolute_distance_m - 1000.0)`), and it can never itself be
     evidence -- no odometry sensor exists on this rig at all. Consequence, found by applying
     the per-tick-consulted-inputs principle rather than assumed: whenever a tick's
     `lane_change_distances_m` selects the window branch, `lane_changes_per_km` is
     `not_evaluable` regardless of `lane_change_distances_m`'s own evidence -- a second sensor
     this rule needs that this rig will never have, the same shape as `forward_ttc`'s
     permanently-absent merge-conflict pair. `SafetyState` has six fields; three were already
     covered (`last_lane_change_time_s`, `lane_changes_last_km`, `lane_change_distances_m`);
     `absolute_distance_m` is now a fourth. The remaining two
     (`distance_since_window_start_m`, `last_lane_index`) are read by no function
     `apply_safety_layer` reaches at all and need no evidence gating.

     **One uniform neutralisation policy cannot serve all twelve rules, and the record should
     say why rather than let a later reader re-derive it or get it wrong the way this task
     did, twice.** Rule-granularity (does the WHOLE rule have evidence) is provably sufficient
     for ten of the twelve. It is provably insufficient for `forward_ttc` (per-disjunct
     evidence; the pair-granularity attempt suppressed a real emergency brake) and for
     `lane_changes_per_km` (per-branch evidence; requiring both fields together would report
     `not_evaluable` on a tick whose `lane_changes_last_km` is real evidence and decisive, only
     because the unconsulted `lane_change_distances_m` also lacked it). The
     per-tick-consulted-inputs principle is what reconciles all twelve without a rule-by-rule
     special case for its own sake.

     **`SafetyInputs.state()` deleted.** Had zero callers (`run_safety_gate` always used its
     own `state` argument); left beside the new `inert_state()` it was exactly the trap a later
     reader would have wired to by mistake.

     **The grid test needed a third axis.** Twelve rules times evidence/no-evidence cannot
     express `leader_gap` measured with `leader_relative_speed` substituted -- the case that
     defeated the first `forward_ttc` ruling. `tests/test_safety_gate.py`'s grid now has three
     axes: default vs non-default `SafetyState`, full evidence per rule, and PARTIAL evidence
     within one rule's own reads (the axis `forward_ttc` needed). Fifteen cases across the
     twelve rules (`forward_ttc` and `lane_change_dwell`/`lane_changes_per_km` each get more
     than one, since a single case cannot exercise both a rule's correctness-when-evidenced and
     its behaviour under the specific non-default/partial-evidence shape that broke it). Run
     against a version with both follow-up fixes neutered (`inert_state` reverted to a raw
     passthrough, `_forward_ttc_missing` reverted to the generic rule): finds exactly the four
     cases the corrections above named (`forward_ttc` x2, the two state-based
     `lane_change_dwell`/`lane_changes_per_km` cases) and no others -- the nine cases
     unaffected by either follow-up fix still hold under that neutering, confirming the grid
     discriminates rather than failing wholesale.

     **`emergency_override` re-confirmed 0 of 3,913 on the real corpus after all of the
     above.** Expected: no tick in `outputs/task42_usb/` has a measured leader (`leader_gap` is
     `fallback_neutral` on all 3,913 per the F5 provenance finding), so neither the corrected
     `forward_ttc` nor `inert_state()` had anything real to act on here. Measured directly,
     matching the expectation exactly.

     **R2-2: the merge bonus is NOT the only unconditional transformation `apply_safety_layer`
     makes, and the test that claimed it was has been corrected.** Reproduced by the
     coordinator: on a `SafetyContext` at its defaults (every rule `not_evaluable`), varying
     only `ego_speed_mps` moves `physical_control_command`'s `acceleration_mps2` --
     `ego_speed_mps 0.0` gives one acceleration, `ego_speed_mps 25.0` gives another (clamped at
     `max_decel_mps2`; the specific numbers are not the claim under test and are not pinned by
     either measurement taken of them). `ego_speed_mps` is deliberately absent from
     `INERT_CONTEXT_VALUES` because it gates none of the twelve rules, so this is a second
     unconditional transformation, of an unevidenced field, that the first version of this
     test's docstring said did not exist. It is harmless today for a narrower reason than "no
     other unconditional transformation exists": `SafetyGateResult` has no `acceleration_mps2`
     field at all -- `run_safety_gate` discards `SafetyDecision.acceleration_mps2` outright.
     That is the actual reason, and it is now the one the test states and asserts (checking
     `SafetyGateResult`'s own dataclass fields), because it is what tells a later agent
     plumbing an acceleration advisory into the record that there is something else to gate,
     where the old text said there was nothing left. Renamed to
     `test_the_merge_bonus_is_the_only_unconditional_transformation_of_a_value_the_gate_reports`;
     the headway assertions themselves are unchanged.

     **F2 (blocking), fixed.** `withhold_lane_when_not_evaluable` never controlled the speed
     or headway; `config.yaml`, `ARCHITECTURE.md` and the plan's decision 3 all documented it
     as the whole gate's rollback. Reproduced before the fix: both
     `withhold_lane_when_not_evaluable=True` and `=False` gave `bounded_speed_mps == 22.0` on
     the F1 fixture, unaffected by the flag either way. Fixed by adding `safety.enabled`
     (default `true`) as the actual rollback: `run_safety_gate(..., enabled=False)` skips
     `apply_safety_layer` entirely -- `bounded_* == proposed_*` exactly, including headway (the
     `create_gap` bonus is `apply_safety_layer`'s own first step and does not apply when it
     does not run) -- while the full per-rule census still runs and is still recorded in a new
     `safety.config` block (also carrying `withhold_lane_when_not_evaluable`, `time_s`,
     `min_contextual_speed_mps`, `density_max_age_s` -- F3/F4 below).
     `withhold_lane_when_not_evaluable`'s own wording is corrected in `config.yaml` and
     `ARCHITECTURE.md` to state it covers the lane/merge action only. The plan's own sentence
     ("Setting it false restores exactly the current display") is left alone, per the brief,
     for the user to correct.

     **F3/F4, fixed.** `score_safety.py` always replayed with its own module defaults, never a
     tick's own recorded configuration -- it could not read a run recorded with
     `withhold_lane_when_not_evaluable=false` (F3), and reconstructed `time_s` as the tick's
     epoch `t_wall` rather than `pipeline.step`'s `time.monotonic()` (F4), a ~1.79e9 s
     discrepancy invisible only because it reaches the record solely through
     `lane_change_dwell`, `not_evaluable` on every tick in this corpus. Every `safety` block
     now carries the `config` sub-key named above; `score_safety.py` replays from it, refusing
     -- and naming the missing key -- for a `safety` block recorded after task 144 but before
     this fix.

     **F5, fixed, and widened by one finding.** Two additions to the per-rule census: (a) a
     count of non-finite compared values among a rule's evaluable ticks
     (`target_lane_front_gap`'s reproduction: the compared gap is `inf` on all 1,229
     "evaluable" ticks, so its own 0.0% fired rate is not a rate -- the threshold could never
     physically be crossed); (b) a provenance-consistency check, from a finding the coordinator
     measured directly against `outputs/task42_usb/`, independent of and stronger than the
     finding above: `target_lane_front_gap`'s value and `field_sources` entry are both aliased
     verbatim from `leader_gap` unconditionally (`perception/observation_builder.py`, one dict
     literal each, no other writer of either key under `perception/`), so under today's
     contract the two `field_sources` entries can never disagree -- and
     `baseline_run_20260902_183446`'s 1,229 ticks (recorded 2026-09-02; `n_detections == 0` on
     every one) do disagree (`target_lane_front_gap` tagged `derived`, `leader_gap` tagged
     `fallback_neutral`, both non-finite), meaning they were written under a superseded builder
     and are not evidence under today's rule regardless of their own recorded tag. Both tools
     still count such ticks evaluable (that is what today's rule says of the recorded source)
     and flag them as `stale_provenance_ticks`, printed rather than silently trusted or
     silently dropped. Consequence for the plan's own prose: "none of its twelve rules has
     evaluable inputs on any of 3,913 ticks" is now true without the qualifier the 1,229 used
     to force.

     **Corpus re-measurement, all four runs, 3,913 ticks, via `score_safety.py`** (before: a
     `git worktree add --detach` mirror at `972ded9`, this round's own parent commit; after:
     the current commit; same corpus files both times). Forced `desired_speed_bin="slow"`:
     before, clamped 3,913 of 3,913, mean delta +2.00 m/s (F1's bug, reproduced on the real
     corpus, matching this record's own earlier E5 measurement); after, raised on 0 of 3,913,
     lowered on 0 of 3,913. `emergency_override`: 0 of 3,913 both before and after, in every
     arm. This confirms the coordinator's own reading of the class partition: `forward_ttc`
     needs the merge-conflict pair (always class (C)) and this corpus's `leader_gap` is
     `fallback_neutral` on every tick, so F1's neutralisation and the pre-fix substituted
     default happen to coincide on THIS corpus -- F1 changes the recorded numbers only for
     `low_speed_uncongested`, the one field whose substituted default (0.0) was not already
     inert for its own rule.

     **F6, fixed.** `score_safety.py`'s arms and `eval_run.py`'s `## Safety` section both
     pooled raised and lowered speed changes into one signed mean and called the total
     "clamped", which reads as a reduction regardless of which direction moved. Both now report
     `raised_ticks`/`lowered_ticks` and each direction's own distribution (mean in
     `score_safety.py`; median and mean in `eval_run.py`, matching its existing `pctl`
     convention), rendered in words: "recommended speed raised on N of M ticks ...; lowered on
     N of M ticks ...".

     **F7: the dashboard honours the flag; the phone frame does not, and does not yet.**
     `Advisory.speed_display_withheld` was read by nothing before this round.
     `ui/dashboard.py`'s recommended-speed line (pulled into `_recommended_speed_line`,
     unit-testable without `cv2`) now shows "WITHHELD (override)" in red instead of a number.
     Stated plainly, because "F7 fixed" must not be misread as "the driver cannot see the
     withheld number": **on an override, the on-device dashboard withholds the speed; the
     phone frame still carries and shows it.** `speed_display_withheld` appears in
     `pipeline.py`, `ui/dashboard.py`, `policy/advisory.py` and `tests/test_dashboard.py`, and
     nowhere under `transport/`.

     `advisory_message_from_advisory`/`AdvisoryMessage` deliberately NOT extended with this
     field, and the reason is narrower than "the function was off limits": **changing a wire
     field's VALUE does not move `AdvisoryMessage`'s frame layout; ADDING a field does, and
     only the second invalidates `specs/transport_golden_frames.json`'s pinned bytes for the
     `message_advisory` case.** `advisory_message_from_advisory` WAS edited in this same round
     -- F8/Fix 9 changed which `Advisory` attribute populates the existing `headway_target_s`
     wire field (`headway_display_s` instead of `headway_target_s`), and that edit is safe for
     exactly this reason: the field set and its encoding are unchanged, so the golden bytes for
     any message that does not exercise the specific value that changed are unaffected, and the
     golden test itself never calls this function at all (it constructs `AdvisoryMessage`
     directly with hand-written field values). Adding `speed_display_withheld` as a NEW field
     is the different, "protocol change" case that function's own docstring means (per
     `tests/test_transport_golden.py`): it would change the encoded byte set for every
     `message_advisory` case, and regenerating the golden file runs
     `scripts/generate_transport_golden_frames.py`, in another agent's active directory this
     round (`scripts/`). If adding the field turns out to be cheap once that script is free,
     it is a follow-up, not something to do now with two agents in the transport specs at
     once.

     **F9/Fix 8, fixed.** `merge_text` is decoded from the policy's raw `merge_mode` and is not
     one of the twelve rules, so it survived lane withholding untouched -- "Creating merge gap"
     could show beside a `lane_text` already reading "Keep lane" for the identical
     `not_evaluable` reason. `pipeline.step` now resets `merge_text` to its normal value
     whenever `gate_result.lane_withheld is not None`.

     **F8/Fix 9, fixed.** The gate already bounds what the driver is shown for speed and lane;
     headway was not -- `advisory.headway_target_s` displayed the raw, unbounded decode while
     `safety.bounded.headway_s` (with `create_gap`'s bonus applied) reached no surface.
     `Advisory` gains `headway_display_s` (defaults to `headway_target_s` via `__post_init__`,
     so every existing construction site is unaffected until `pipeline.step` overwrites it);
     `headway_target_s` itself is untouched and `set_target_headway` keeps feeding it back raw
     (decision 2). `ui/dashboard.py`, `Advisory.one_line()`, and the wire field
     `headway_target_s` (via `advisory_message_from_advisory` -- a value change, not a schema
     change; confirmed the golden test never calls that function) all now show the bounded
     value.

     **F10, the code half.** `target_lane_front_gap`/`target_lane_front_ttc`'s recorded
     evidence now carries `gap_source_slot: "leader_gap"`, naming where the compared gap
     actually comes from (the CURRENT lane, not the one being changed into), so the persisted
     record cannot be misread as a genuine target-lane measurement. This is the same fact F5's
     provenance-consistency check surfaces from the other direction: F10 is why the field can
     carry evidence at all; F5's check is why 1,229 of those evaluable ticks should not be
     trusted as evidence today.

     **Two numbers added to the size discussion (E6/decision 4) above.** At the corpus's 5.0
     ticks/s, the `safety` block's measured mean (2,576 B) costs **47.7 MB/hour** of
     uncompressed JSONL, on top of **~182 MB/hour** for the rest of the tick record. The
     plan's own comparison against a cheaper form (1,103 B, dropping each substituted field's
     own provenance class) was the wrong comparison: dropping only the twelve `missing` lists
     -- reconstructible from `RULE_READS` given the rest of the block, since `RULE_READS` names
     exactly which fields a `not_evaluable` rule's `missing` would list -- gives **1,707 B**,
     inside the plan's 1,300-1,800 B band, with every provenance class still present; the cost
     is that the log is no longer self-describing (a reader needs `RULE_READS` alongside it to
     reconstruct `missing`). Not implemented, recorded for the user's decision.

     **Test counts.** Baseline stated for this round: `deployment/jetson/tests/` 2,390 passed /
     28 skipped / 0 failed at `58cb284`. Current, at this round's last commit: 2,505 passed /
     24 skipped / 0 failed. The skip count moved from 28 to 24 entirely via the 12 commits
     already on the branch between `58cb284` and this round's own starting commit (`972ded9`)
     -- concurrent tasks 142/143/145, not this round: task 143's own `2692c3b` rewrote
     `test_sim_contract.py` to 0 skipped (was 1, per that task's own record above), and the
     same span of commits also removed the 3 "pre-existing and unrelated"
     `test_run_demo_loop.py` skips this record's own earlier paragraph named. Neither file was
     touched this round. This round's own new tests: 30, across `test_safety_gate.py` (11),
     `test_score_safety.py` (9), `test_safety_gate_pipeline.py` (5), `test_dashboard.py` (3, a
     new file -- `ui/dashboard.py` had none before), `test_eval_run_safety.py` (2).

     **Which of the twelve rules can ever be evaluable on this rig is structural, not
     empirical, and is now stated as such rather than left as "3,913 ticks happened to show
     none of them."** Measured by the coordinator: built the single most favourable
     observation `safety_inputs_from_observation` can produce (every source `measured`, a
     fresh `last_detection_age_s`) and checked which rules still have no field able to carry
     evidence at all.

     | rule | can ever carry evidence on this rig | reason if not |
     |---|---|---|
     | `low_speed_uncongested` | yes | -- |
     | `target_lane_front_gap` | yes | -- |
     | `passing_lane_slow_hold` | partly (`local_mean_speed_mps` yes, `in_passing_lane` no) | no lane detection |
     | `target_lane_front_ttc` | partly (gap yes, relative speed no) | -- |
     | `forward_ttc` | partly (leader pair yes, merge-conflict pair no) | no merge-conflict sensor |
     | `all_lane_low_speed_occupancy` | no | no cooperating peers (V2V-only fields) |
     | `lane_change_dwell` | no | no lane-change detector |
     | `lane_changes_per_km` | no | no lane-change detector |
     | `target_lane_missing` | no | no lane detection |
     | `target_lane_rear_gap` | no | no rear sensor |
     | `target_lane_rear_ttc` | no | no rear sensor |
     | `target_lane_rear_braking` | no | no rear sensor |

     Seven of twelve are unreachable through the builder BY CONSTRUCTION -- the fields they
     read are class (C) unconditionally, regardless of what any observation records -- not by
     chance of what four drives happened to capture. Named in a comment beside `RULE_READS`
     (`policy/safety_gate.py`) so this does not have to be rediscovered. The lane-withholding
     consequence is sharper for the same reason: the lane advisory is withheld on every tick
     not because the drives were unlucky, but because three of its eight guards
     (`target_lane_rear_gap`, `target_lane_rear_ttc`, `target_lane_rear_braking`) read fields
     the rig has no instrument for at all, and a fourth (`target_lane_missing`) needs lane
     detection this rig also does not have.

     **A mutation run against `dedd3c1` found one of round 2's own fixes unpinned.** Reverting
     `_lane_changes_per_km_missing` to the generic "every `RULE_READS` field must be evidence"
     rule left all 52 tests in `tests/test_safety_gate.py` passing -- 0 failed. Cause, per the
     table above: `lane_changes_per_km`'s three fields are ALL class (C), so no observation the
     builder can produce distinguishes the branch-aware rule from the generic one; every test
     built through `safety_inputs_from_observation` gets "all three missing" from both. Not
     the same defect as F1's degenerate corpus (a real drive that happened not to exercise a
     rule) -- this is a rule NO drive on this hardware can ever exercise, one level further in.
     `_forward_ttc_missing` did not have this problem on its reachable (leader-pair) axis,
     because `leader_gap_m` is class (B); a mutation of that axis was caught. Its
     merge-conflict axis has the identical unreachable shape as `lane_changes_per_km`, and
     was, on inspection, equally unpinned until this fix.

     Fixed by constructing `SafetyInputs` directly (bypassing the builder) for the exact
     mixed-evidence states the rig cannot produce: `lane_changes_last_km` evidenced with
     `lane_change_distances_m` not (and the reverse), and, for `forward_ttc`, the
     merge-conflict pair evidenced and operative with the leader pair explicitly built
     WITHOUT evidence (needed so the fixture actually discriminates -- an earlier draft left
     the leader pair evidenced via the blanket-evidence test helper, which the generic rule
     would also have accepted, for the wrong reason). Each pinning test is paired with an
     explicit "disagrees with the generic rule" check, and the manual-neuter re-run (reverting
     each function to the generic rule inline) now fails on all six of these tests where
     before it silently passed one of them. `absolute_distance_m`'s inert value of `+inf`
     (unchanged from round 2) is correct for a reason worth restating because it is not
     obvious: `window_start = max(0.0, +inf - 1000.0)` is `+inf`, and no finite recorded
     distance is `>= +inf`, so the recorded-change count is zero regardless of what the
     distances list contains.

     **Round 3: `evaluate_rules` computed `missing` from EFFECTIVE (inert-or-real) values
     while running the predicate itself on the RAW context/state -- the two could disagree
     about which disjunct/branch applied, and the census could assert `fired` on substituted
     data.** Reproduced by the coordinator (their case 42): leader pair NOT evidence (raw gap
     6.0 m, raw relative speed -8.0 m/s -- real numbers, just unevidenced), merge pair evidence
     (gap 6.0 m). `_forward_ttc_missing`'s own comparison correctly picks the merge pair (its
     real 6.0 m beats the leader's inert +inf) and reports `missing=()`. The predicate then ran
     on the raw context, where `merge_conflict_gap_m(6.0) < leader_gap_m(6.0)` is False (a tie,
     not a strict `<`), so it silently kept the LEADER pair instead and reported `fired`, with
     `ttc_s 0.75` computed from the leader's own un-evidenced gap and relative speed --
     `emergency_override False` and empty diagnostics, so nothing acted on it, but the record
     positively claimed a rule was evaluated and fired, on substituted values. Worse than the
     `not_evaluable` problem F1 fixed: that at least said "unknown".

     Fixed exactly as specified: `evaluate_rules` now builds `context = inputs.inert_context()`
     and `state = inputs.inert_state(state)` before evaluating each rule, so the predicate runs
     over the SAME effective values its own `missing` computation reasoned about. Verified
     independently (not merely trusted from the report): re-ran case 42 after the fix --
     `quiet`, `missing=()`, `ttc_s 12.0` (the merge pair's own value, `6.0 / 0.5`) -- and pinned
     it as `test_evaluate_rules_agrees_with_its_own_missing_about_which_pair_is_operative`, with
     a paired test that reverts `evaluate_rules` to the raw-context mechanism inline and
     confirms it then reproduces the coordinator's exact numbers (`fired`, `ttc_s 0.75`).

     This changes nothing for the OTHER ten rules: none has a branch-aware `missing`
     computation, so whenever a simple rule is evaluable (every `RULE_READS` field has
     evidence, by the generic rule), the inert-substituted and raw values are identical for
     every field that rule reads -- substitution only ever replaces a field THAT rule needs
     when that field lacks evidence, which evaluable-by-the-generic-rule already rules out. The
     corpus census is confirmed unchanged (below).

     **`_lane_changes_per_km_missing` kept, not deleted, on the coordinator's ruling against
     the validator's proposal.** The validator argued deletion is defensible -- all three of
     its fields are class (C), so the refinement is a no-op through the builder today. Ruled
     otherwise: the first lane-change detector makes it live, a silent no-op today becomes a
     wrong census the day the hardware changes, and it is no longer invisible -- it is pinned
     by hand-built fixtures (the previous commit). `_forward_ttc_missing` was never in
     question: it changes reachable behaviour today (`forward_ttc` is one of the three rules
     ever evaluable through the builder -- next paragraph).

     **Decision 3's own wording is corrected, here and wherever it is cited.** It said the
     census reads the unmodified context; after this fix it does not, for
     `forward_ttc`/`lane_changes_per_km`'s branch-aware evaluability. The reason is one
     sentence: the census's job is to describe truthfully what the decision did, and a census
     computed over values the decision never saw describes a hypothetical, not this tick.
     Nothing is lost -- `Tick.to_record()`'s `obs`/`field_sources` still carry the actually
     observed values and their provenance regardless of what the per-rule `evidence` dict
     reports. (`ARCHITECTURE.md` sec 6.1 makes the same claim and needs the same correction;
     left alone, per this round's own standing constraint that file is task 143's while it is
     active there.)

     **The reachable-rule count is THREE, not the five an earlier sweep in this record
     estimated.** The validator's 20,000-draw sweep, through `safety_inputs_from_observation`
     only: `low_speed_uncongested` and `target_lane_front_gap` (each gated by one class-(B)
     field), and `forward_ttc` (reachable only because of its own per-disjunct refinement --
     under the generic rule its permanently-class-(C) merge-conflict pair would put it with the
     nine below). Nine of twelve can never be evaluable: seven read only class-(C) fields, and
     two more -- `passing_lane_slow_hold` (`local_mean_speed_mps` can vary), `target_lane_front_
     ttc` (its gap can vary) -- are still always blocked by the generic all-fields rule because
     their OTHER read is permanently class (C). Having one flexible field is not the same claim
     as being reachable; corrected in the code comment beside `RULE_READS` and above.

     **Corpus census independently re-confirmed unchanged after this fix.** Re-ran
     `score_safety.py` over all four archived runs (3,913 ticks) and diffed the rendered report
     against `results/safety/gate_census_corpus.json`'s own `after` section: byte-identical.
     Expected, and now measured twice (once by the coordinator, once independently here): every
     field `_forward_ttc_missing`/`_lane_changes_per_km_missing` branch on lacks evidence on
     every tick in this corpus, so the fix is forward-looking and moves nothing measurable
     today.

     **Suite:** 2,530 passed / 24 skipped / 0 failed (this round added 2 new tests to round 2's
     count; the skip-count note earlier in this record explains the 24 vs. the plan's original
     28).

     **`2ecff72`'s `lane_changes_per_km` pin still did not hold -- a third incomplete attempt
     on the same refinement, and worth stating plainly rather than folding quietly into the
     next fix.** The coordinator mutated both refinements' CALL SITES inside
     `_evaluate_one_rule` (not the functions themselves) back to the generic rule: `forward_ttc`
     caught it (4 tests failed by name); `lane_changes_per_km` did not (0 of the 3 tests written
     for it failed). Cause: those three tests called `_lane_changes_per_km_missing` directly and
     asserted only on its return value -- pinning the function's behaviour, never whether
     anything still calls it. This project has hit exactly this shape before (an AST wiring test
     that passed with its own call site wrapped in `if False:`), and it is the textbook form: **a
     test that pins a function is not a test that pins its use.** `forward_ttc`'s tests did not
     have this gap because each one also ran its fixture through `run_safety_gate` and asserted
     on `result.rules["forward_ttc"].status` -- the census, not the private function.

     Fixed by giving all three `lane_changes_per_km` tests the same second half: the hand-built
     `SafetyInputs` already constructed is exactly what `run_safety_gate` needs, so each test now
     also asserts `result.rules["lane_changes_per_km"].status == RULE_FIRED`. Verified the same
     way as every other fix in this file, not assumed: temporarily renamed the `elif name ==
     "lane_changes_per_km":` branch inside `_evaluate_one_rule` (falling through to the generic
     `else` rule, exactly the coordinator's mutation) and confirmed all three tests now fail by
     name; restored, confirmed all three pass again.

     **Commits** (branch `mainz-src-port`, each by explicit pathspec, never `git add -A`,
     never a bare `git commit`): `346fb9d` (F1 + Fix 2 + F2/Fix-3 infra, in
     `policy/safety_gate.py`), `2c660a8` (F3/F4/F5/F6, in `score_safety.py`), `e12e62c` (F5/F6,
     in `eval_run.py`), `89a1048` (F7, in `ui/dashboard.py`), `ba3ac2a` (F2's pipeline wiring,
     F9/Fix-8, F8/Fix-9 -- `pipeline.py`, `policy/advisory.py`, `run_demo.py`,
     `transport/messages.py`, `config.yaml`, `ARCHITECTURE.md`), `f788735` (F10 code half, in
     `policy/safety_gate.py`), `dedd3c1` (round 2: `forward_ttc`/`lane_changes_per_km`
     evaluability corrected twice, `inert_state()`, R2-2, R2-1 record fixes), `2ecff72` (pinned
     both refinements against the surviving mutation with hand-built `SafetyInputs`, the
     seven-unreachable-rules comment -- incompletely, see above), `fce9651` (round 3:
     `evaluate_rules` reasons about the same effective values `missing` did), and the
     `lane_changes_per_km` pin strengthening above (`tests/test_safety_gate.py`, this record),
     each committed separately so the deltas stay legible.

145. **The rig cannot run the controller the paper is about.** Implemented 2026-09-12
     (implementer-145) against `plans/plan_task145_dsrc_policy_runtime.md`. The plan's
     sign-off checklist (its section 11) is unaddressed: none of it carries user sign-off,
     by the plan's own statement, and this record does not close it.

     The deployed actor is a 39 -> 128 -> 128 -> 4x3 MLP over the local-sensing contract:
     `ObservationBuilder.build` produces the vector, `ActorRuntime.act` runs it,
     `AdvisoryDecoder.decode` turns it into the advisory. The paper's controller is
     `SrcQNetwork`, which takes the whole network's state as 12 super-segments x 5 features.

     The paper plan says of the 39-field contract, "It is not the policy's input". That is
     true of the simulation and false of the device, and it contradicts the plan's own first
     open item, which says the HERE-observation policy has not been run on the rig.

     **The harder half, which no document states.** `SrcQNetwork`'s input is the entire
     network and its weights are trained for Mainz, so the policy is network-specific. The
     123 HERE bodies collected on 2026-09-08 are New Jersey roads, so the gap cannot be
     closed by replaying the drives: there is no Mainz observation in the corpus and no
     Westfield policy. "What remains is loading that policy" understates what is left.

     **The task:** build the device-side runtime that executes a DSRC checkpoint from
     super-segment features and produces an advisory, demonstrate it on the rig against
     replayed segment state, measure its latency, and require it to reproduce the
     simulator's action for the same input. Then state plainly what still stands between
     that and a policy a vehicle could run on a road it can drive to.

     **What was built.** `policy/dsrc_contract.py` (vendored `SrcQNetwork`, action
     fractions, decision interval, feature names, `network_fingerprint`);
     `scripts/export_dsrc_network.py` and the committed `specs/dsrc_network_mainz.json`;
     `perception/segment_state.py` (`SegmentStateBuilder`, coverage classes
     measured/substituted/absent); `sensors.here_feed.HereFeed.snapshot_links`;
     `policy/dsrc_runtime.py` (`DsrcRuntime`, refusing a bundle whose network identity does
     not match the device's own, no grandfather clause); `policy/export_dsrc_policy.py` and
     the resulting `models/dsrc_policy.ts`/`.json` bundle (gitignored, regenerated from
     `results/checkpoints/mainz_here_best.pt`); `scripts/export_dsrc_golden.py` and the
     committed `specs/dsrc_golden_actions.json`; `policy.advisory.SegmentAdvisory` and
     `SegmentAdvisoryDecoder`; `pipeline.py`'s wiring of all of it behind two new optional
     `stages` entries, `segment_assemble` and `dsrc_infer`, off by default and backward
     compatible with every existing call site.

     **The demonstration.** `DsrcRuntime.act` reproduces `src.rl.src_q.greedy_actions`
     exactly on both populations `specs/dsrc_golden_actions.json` carries, checked two
     different ways. On the 185 recorded decisions across `mainz_here_best.pt`'s five
     held-out test seeds (16-20), stored and compared directly, value by value: 0 action
     mismatches, Q-values bit-for-bit identical between the TorchScript bundle and the
     eager reference on this machine (the "same machine that exported" row of the plan's
     acceptance table), and agreeing to within 1e-5 (the "any machine" row). On 20,000
     states drawn from a fixed seed over the plausible feature box, the file stores only
     the seed, the feature-box bounds and a sha256 over the reference's own
     actions-then-Q-values bytes -- storing the arrays put the file at 10.8 MB, three times
     the next-largest tracked file in the repository, for a deterministic generator's
     output; the coordinating session flagged this and it was changed before this record was
     written. The test regenerates the 20,000 states from the seed, runs them through
     `DsrcRuntime`, hashes the same way, and compares: a match is 0 mismatches on that
     population too, but as a same-machine bit-exact pin rather than a per-state count, and
     it does not itself carry the "any machine, 1e-5" claim -- that claim is checked with
     real numbers only on the 185 recorded states. Three deliberately broken runtimes were
     each seen to fail before this pass was trusted: a transposed weight fails at
     `load_state_dict` (no layer in this network is square, so a transpose is never
     shape-compatible); two segments swapped in the network definition are refused at the
     `network_fingerprint` check, before any action is computed; and `jam_factor` drawn
     independently of the recomputed formula, over its full 0-10 range, changed the action
     on 143 of 185 recorded decisions (350 of 2,220 individual per-segment actions) --
     the measurement open item 2 asked for.

     **Latency, on this development machine, not the Orin.** `dsrc_infer` p50 0.029 ms /
     p95 0.048 ms over 200 replayed decisions, matching the order of magnitude
     `greedy_actions` measured in the plan's own section 1.7. `segment_assemble` p50
     25.9 ms / p95 26.2 ms: pure-Python point-to-polyline distance checks across every
     matched segment's vertices, dominated by segments with many edges (segment 2 alone has
     33). This runs once per 60 s decision, not once per tick, so even this number sits
     more than three orders of magnitude inside the interval it has to complete within; it
     is reported because the plan's own prior was stated for `dsrc_infer` alone and did not
     anticipate `segment_assemble` being the larger of the two. The Orin number itself is
     step 12's own deliverable and was not measured -- no device access this session; the
     harness (`deployment/jetson/bench_dsrc_latency.py`) is built and runs, so measuring it
     is a rerun rather than new code.

     **experiment_dsrc, 2026-09-12: the latency figures are now a committed artefact rather
     than prose.** `results/latency/dsrc_stages_devmachine.json` and its `.log`, committed at
     `f3d27ae`. Four repeats of `bench_dsrc_latency.py --ticks 300`, every one at full
     coverage (300 of 300 ticks matched), run in a detached worktree at `d976dcc` rather
     than the live tree, whose HEAD was a different commit carrying another agent's
     uncommitted work. `segment_assemble` p50 29.91 to 30.16 ms across the four; `dsrc_infer`
     p50 0.0274 ms on the one idle repeat and 0.0417 to 0.0539 ms on the three that ran while
     task 143's mutation gate drove the full Jetson suite in another worktree.

     That contention is recorded rather than averaged away, because it reconciles the three
     different `dsrc_infer` figures this record now carries -- 0.029 ms above, the
     validator-round-2 note's 0.027 ms, and 0.042 to 0.054 ms here -- which would otherwise
     read as a regression. It moved one stage and not the other: `segment_assemble`'s p50
     spread across all four repeats is 0.25 ms on a roughly 30 ms quantity, while
     `dsrc_infer`'s contended p50 is about 1.5 to 2 times its idle one. Neither changes a
     conclusion, since both stages sit more than three orders of magnitude inside the 60 s
     decision interval.

     Three things the artefact does not establish, stated in the file itself: the Orin
     number, which still needs the device; behaviour on a recorded drive, since `--here-log`
     was not given and the synthetic input is the favourable case -- every segment matches,
     so both stages do their full work every tick; and anything at all about whether the
     advisory is correct.

     **Two things the plan got wrong, found while implementing it, corrected rather than
     followed literally.**

     1. **Mainz carries no real-world geo-reference at all.** `data/mainz/mainz.net.xml`'s
        `<location>` tag is `projParameter="!"`, SUMO's own marker for "no projection",
        so `sumolib.net.Net.convertXY2LonLat` refuses the network outright. The plan's
        section 4.1 asks for "the WGS84 polyline of each edge" as though it were an
        extraction; it is a placement.  `scripts/export_dsrc_network.py` anchors Mainz's
        bounding-box centre at a fixed New Jersey-area point already used by this
        deployment's own GPS test fixtures (`lat=40.0, lon=-74.0`) and documents this as
        fabricated in the file's own `geo_reference` block, not as a measurement.
     2. **The 123-body New Jersey HERE corpus plan step 3 names is not in this checkout.**
        It was collected on 2026-09-08 (item 71 above) but written to a run directory this
        repository's `.gitignore` excludes and never committed.

        **CORRECTION, 2026-09-12: the sentence that followed said "it is not reachable from
        this machine", and that is false.** The corpus is on this machine at
        `~/Desktop/ankit_summer_2026/dsrc_drive_data_20260908/run_*/here/` --- counted
        directly: 93 bodies in `run_20260908_170849`, 28 in `run_20260908_190548`, 2 in
        `run_20260908_183538`, exactly the 123 the record claims. "Not in the checkout" and
        "not reachable" are different statements and only the first is true.

        What that does and does not change. It does not rescue the Mainz correspondence:
        point 1 above stands, Mainz carries no projection, so there is still no real pairing
        between any HERE link and any Mainz super-segment and the tolerance sweep still has
        no ground truth to calibrate against. What it does change is that the sweep's *link*
        side need not be synthetic --- real segment geometry, 915 segments across 68 named
        roads, is available offline and was described as unavailable. Combined with point 1, no real correspondence exists
        between any HERE link and any Mainz super-segment to calibrate a match tolerance
        against. `perception.segment_state.SEGMENT_MATCH_TOLERANCE_M` reuses
        `sensors.here_feed.ASSOCIATION_RADIUS_M` (60.0 m) on that constant's own physical
        reasoning rather than a number read off a sweep over data this module had to
        fabricate; `scripts/measure_dsrc_match_tolerance.py` runs that sweep on synthetic
        data and documents why its curve is provisional.

     Separately, `perception/segment_state.py` sources the `free_flow` feature from the
     network definition's static speed limit rather than from HERE's `freeFlow` field:
     `src.sumo.mainz.MainzEnv._per_edge` sets `free_flow_kmh` from the map on every step and
     never from a live reading, and its own `HERE_FEATURES` docstring says so directly. The
     plan's section 4.2 lists `free_flow` as one of the three fields read live from HERE;
     following that literally would have fed the policy a feature that moves when the
     training data it was fit against never did, with no test positioned to catch it (the
     golden-action tests bypass this module entirely, feeding recorded states straight to
     `DsrcRuntime.act`). This is a correction to the plan's own text, not a design choice
     left open for sign-off, and is called out because the plan is supposed to be the
     validator's reference.

     **Test counts.** Baseline at `975b7a2`: `deployment/jetson/tests/` 2,247 passed / 25
     skipped; repo-root `tests/` 69 passed. Measured on this branch after this task's
     commits: `deployment/jetson/tests/` 2,332 passed / 25 skipped; repo-root `tests/` 116
     passed. Task 142 committed its own paired-seed evaluator to this same branch
     concurrently with this work (`9e154e7`), so part of both increases is its tests, not
     this task's; this task's own additions are the ones named above by file
     (`test_dsrc_contract.py`, `test_here_feed.py`, `test_segment_state.py`,
     `test_dsrc_runtime.py`, `test_advisory.py`, `test_pipeline_smoke.py`,
     `test_export_dsrc_network.py`). No test was removed or skipped to reach these numbers,
     and the skip count is unchanged from baseline.

     **What was not done.** Step 12's Orin measurement (device access). The plan's own
     section 9 (a drivable-network definition, a network-shaped HERE query, a training run
     on that network) was named out of scope by the plan itself and stays out of scope
     here. Section 11's sign-off checklist, including whether points 1 and 2 above change
     any recommendation, is for the user.

     **Three documented limits, from validator round 1 (2026-09-12), not fixed because
     each is a fact about the rig or the sweep rather than a defect in this task's code.**

     (a) The DSRC path is unreachable from `run_demo.py`: `_build_rest` there constructs
     `PerceptionPolicyPipeline` with six positional arguments and never passes
     `dsrc_runtime`, `dsrc_segment_builder`, `dsrc_advisory_decoder` or `here_feed_source`,
     and `config.yaml` carries no `policy.dsrc_bundle` key at all, so on any real run both
     `segment_assemble` and `dsrc_infer` are permanently absent and step 12 can currently
     only be exercised through `bench_dsrc_latency.py`; "off by default" above is true but
     does not say there is presently no way to turn it on.

     (b) `scripts/measure_dsrc_match_tolerance.py`'s claim to "show the matcher's own
     sensitivity to the tolerance parameter" is not supported by its own output: the curve
     is flat (15 m -> 6/12 segments matched; 30 m through 250 m -> 8/12, unchanged across a
     220 m range) -- a statement about the synthetic corpus's fixed offsets, not about the
     matcher. (Re-measured after fix 3 below changed the matcher from "any segment within
     tolerance" to "the single nearest": the pre-fix-3 curve reached 9/12 at 150-250 m, one
     segment credited only because a link within tolerance of two segments' polylines was
     counted for both; post-fix it stays flat at 8/12 through 250 m, which is the honest
     number now that a link is credited to one segment.)

     (c) The golden random population's reproducibility rests on `numpy.random.Generator`
     producing the same output stream for a given seed across environments, which NEP 19
     explicitly does not guarantee across numpy versions; a stored sha256 of the
     regenerated *states* themselves, not only of the reference actions/Q-values computed
     from them, would separate that cause from an actual `DsrcRuntime` divergence the next
     time `test_random_states_hash_to_the_frozen_digest` fails.

     **Validator round 2 (2026-09-12): the latency paragraph above quotes pre-fix-3
     numbers.** Fix 3 (validator round 1) removed the `any(...)` short-circuit in
     `_distance_to_polyline`/`SegmentStateBuilder.build`'s matching loop that let a
     distance check stop at the first vertex within tolerance; every vertex of every
     matched segment is now visited on every call. Re-measured on this machine with the
     same synthetic feed (`bench_dsrc_latency.py --ticks 300`, no `--here-log`): at
     `4ba6b7f` (before fix 3) `segment_assemble` was p50 26.1 ms / p95 26.6 ms, close to
     the 25.9/26.2 the paragraph above states; at the current commit it is p50 29.6 ms /
     p95 30.1 ms, about 14% higher -- the paragraph above was written before fix 3 landed
     and was never re-measured against it. Separately, `dsrc_infer` p50 0.029 ms above was
     wall-clock time measured around `decide()`; round-1 fix 1 changed the instrument to
     `DsrcActResult.latency_ms`, timed inside `act()` around the network call alone. A
     fresh run gives p50 0.027 ms / p95 0.057 ms, within 0.002 ms of the number above but
     produced by the changed instrument, not the one the paragraph describes.

     **`segment_assemble` cost is linear in link count and can reach the per-tick budget
     -- recorded here as a real number without a real input yet, not fixed.**
     `eval_run.py` budgets 200 ms p95 per tick on the Jetson, and
     `PerceptionPolicyPipeline._dsrc_step` runs `segment_assemble` and `dsrc_infer`
     synchronously inside `step()`, so whichever tick lands on the 60 s decision boundary
     pays the full assembly cost against that single tick's budget, not against the 60 s
     interval the reasoning above is about. Measured cost is linear at about 2.5 ms per
     link: 12 links (Mainz's own network, the only one deployed against this runtime)
     costs 30.2 ms; 192 links costs 476.3 ms. On this machine the 200 ms budget is crossed
     at roughly 90 links; the Orin, being slower, crosses it sooner. Fix 3 added 11-14% to
     this cost; the cost itself predates fix 3 and is a property of the per-vertex
     polyline-distance search run for every matched segment on every link, not of that
     fix. Mainz has 12 segments, so this does not bite today; if a larger network is ever
     deployed against this runtime, the cheap remedy is a per-segment bounding-box check
     before the vertex loop, tried before replacing the vertex loop itself.

     **`bench_dsrc_latency.py --ticks 0` raises `IndexError`, recorded and not fixed.**
     Confirmed by running it: `pctl(assemble_ms)` calls `np.percentile` on an empty list
     unconditionally, which raises inside numpy rather than returning a sensible "no
     ticks ran" result; the `dsrc_infer_ms` path a few lines below is correctly guarded
     with `if infer_ms:` and prints a named message instead. This is user error (`--ticks
     0` asks the harness to measure zero decisions) producing a crash rather than the
     harness reporting a wrong number, so it is recorded rather than fixed.

---

