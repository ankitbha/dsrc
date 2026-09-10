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

## The paper

**The contribution is the deployed system: the phone-plus-Jetson advisory rig, the
sampling controller, and the safety and etiquette filters.** Those are what this
project built and what the paper argues for. `src/safety/` holds the filters
(`safety_layer.py`, `etiquette.py`, `constraints.py`); sections D through I hold the
system and the evidence it runs on a road.

**The road-network-level result is a replication, not a claim.** Decentralized MAPPO
control improving throughput is established in the literature (Vinitsky et al., and
the Flow benchmarks). The paper reproduces it once, in this simulator, on one
topology, to show the setting behaves as published. It is not a study, and it is
scoped in section C accordingly.

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
3. **Task 9** — fill the five sensing-model parameters. **PARTIAL 2026-09-09.**
   `configs/training/mappo_deploysense.yaml`. `latency_s` and `queue_speed_mps`
   settled from the drives, `range_m` from optics. Two remain open and are marked
   in the file: the noise terms need a citation, and `range_m` still wants the
   replay measurement, which needs the Jetson.
4. **Task 68** — train MAPPO on `inverted_tree`. The single missing artifact.
5. **Task 69** — evaluate trained MAPPO against `no_av` for throughput.
6. Write the paper: the deployed system and the safety and etiquette filters, with
   step 5 as the replicated flow-level result.

**Task 9 must precede task 68, and the reason is not bookkeeping.**
`src/envs/topology_env.py` builds every agent observation through
`LocalObservationBuilder(SensingConfig.from_config(...))`, so the five sensing
parameters define the actor's entire input distribution. Training first means the
policy learns against an observation model the paper then describes as wrong. The
largest single mismatch is `latency_s: 0.0` in every current config against the
96.7 ms median measured on the drives.

**Tasks 67 and 63 are independent of each other**, so their relative order is free.
Doing 63 first lets the task 9 replay pass, which reads 1.676 GB of video, run while
task 67 is worked.

**Scope boundary.** One topology (`inverted_tree`). One algorithm (MAPPO). One
throughput comparison. The drives are finished and will not be repeated; the corpus
is recorded in section I.

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

## C. Simulation: replicate the MAPPO throughput result on `inverted_tree`

**Replication, not discovery.** The task is to reproduce a published result in this
simulator with this project's settings, on one topology. No topology ladder, no
penetration sweep, no large-scale verification of the setting.

**`inverted_tree` is the chosen topology and the only one to be run.** Task 8
measured it congesting in 12 of 12 cells, which is the property a replication needs.
The other five are out of scope, `merge` and `straight_multilane` especially: they
congested in 0 of 12.

### Correction 2026-09-08: the task 8 result does not apply to MAPPO

This section read "BLOCKED IN SUBSTANCE by the task 8 result" on the strength of
"+0.12 to +0.14 m/s". **That number describes hand-written baselines.**
`src/analysis/simulator_health.py` calls `make_baseline(controller)` against
`REFERENCE_CONTROLLER = "no_av"`, and `src/baselines/registry.py` holds `no_av`,
`random_av`, `selfish_av`, `density_lookup`, `dynamic_speed_limit`,
`av_mediated_speed_harmonization`, `backpressure` and `cooperative_smoothing`. **None
of them is learned.**

**MAPPO is implemented and has never been trained.** `src/rl/trainers.py` defines
`MAPPOTrainer` with `critic_scope = "global"` and a critic over
`physical_global_state_dim() + local_obs_dim()` — a centralized critic with
decentralized actors, which is the algorithm the replication needs — distinct from
the `IPPOTrainer` and `SharedPPOTrainer` beside it. `scripts/train_policy.py` is the
entry point and supports `--resume-from` and `--resume-latest`. `outputs/` holds no
checkpoint.

So task 8 is evidence about controllers this project does not intend to ship, and it
is not evidence about the method the paper replicates. **What task 8 does still bind
are the conditions a training or evaluation run must meet**, and tasks 67 to 69 carry
those forward: the truncation rate, the concurrent AV count, and the run-to-run
term.

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
   **OFF THE CRITICAL PATH 2026-09-08.** It exists to derive a sensing requirement
   specification, which the paper described above does not claim. Build it only if
   the replication lands with time to spare.
7. Baseline sweep with the current sensing defaults, to establish the reference
   the degraded conditions are measured against. **OFF THE CRITICAL PATH
   2026-09-08** — the only reference the paper needs is `no_av` against trained
   MAPPO on one topology, which task 69 measures directly.
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
9. Sensing model calibrated from drive measurements. **SCOPED 2026-09-08 to the
   five parameters that exist.** `src/sensing/local.py` has exactly five, and each
   now has a named source:

   | parameter | default | source |
   |---|---|---|
   | `latency_s` | 0.0 | Measured on the drives: p50 96.7 ms, p95 172.9 ms, p99 276.4 ms over 22,929 ticks |
   | `queue_speed_mps` | 5.0 | Ego GPS speed distribution, 22,734 valid fixes, p50 11.14 m/s |
   | `range_m` | 150.0 | Offline `replay_demo.py` over stored video — **needs task 63 first** |
   | `position_noise_std` | 0.0 | Published characterisations of monocular detectors |
   | `speed_noise_std` | 0.0 | Published characterisations of monocular detectors |

   **The two noise terms come from the literature by decision, not by omission.**
   Calibrating them requires an independent measurement of other vehicles' true
   position and speed, and the vehicle carried no radar, no lidar and no second
   instrumented car. Reprocessing frames yields the estimator's frame-to-frame
   self-consistency, which is a different quantity from error against truth. The
   paper must attribute these two to published characterisations and not to this
   vehicle.

   **`range_m` needs no further driving.** `deployment/jetson/replay_demo.py`
   re-runs the whole perception, observation and policy pipeline over a run's raw
   video and logged GPS. It applies to the six runs holding `video_index.jsonl`;
   `run_20260908_142253` recorded no video and `run_20260908_144642` has video with
   no index. Any distance it produces is only as good as the mount geometry, so task
   63 must land first and the horizon must be re-established for the real mount.
10. Sufficiency study proper: derive the sensing requirement specification.
    **OFF THE CRITICAL PATH 2026-09-08**, with task 6.
11. Sampling policy evaluated in simulation for the flow-level benefit one
    vehicle cannot demonstrate. **OFF THE CRITICAL PATH 2026-09-08.** The
    flow-level benefit the paper reports is the replication in task 69; crediting
    the *sampling* policy at flow level is a second claim the paper does not make.
67. **Fix the truncation rate before any training run.** Task 8 measured 260 of 648
    runs never reaching their configured duration, a 40% rate. Training against that
    corrupts the return signal: a bad policy and a broken episode become
    indistinguishable. This precedes task 68 rather than running beside it.

    Two config defects task 8 found belong here, because both shrink the usable
    grid: `burst` is bit-identical to `medium` on all six topologies (162 of 648
    runs), and ring disables demand so its four demand levels collapse to one (81
    runs). Only the `inverted_tree` half of either matters now.
68. **Train MAPPO on `inverted_tree`.** `scripts/train_policy.py --training mappo
    --topology inverted_tree`. This is the single missing artifact — no part of the
    simulation claim can be evaluated until a checkpoint exists.

    **Blocked on tasks 67 and 9.** Not by convention: `src/envs/topology_env.py:109`
    constructs `LocalObservationBuilder(SensingConfig.from_config(self.config))` and
    line 281 builds every agent observation through it, so the `sensing:` block in
    the training YAML is the actor's input distribution. Training under the current
    defaults and calibrating afterwards produces a policy trained on an observation
    model the paper does not claim. `configs/training/shared_ppo_deploysense.yaml`
    already carries such a block and is the file task 9 fills.

    **Record the training config beside the checkpoint.** Task 69 has to evaluate
    under the same block, and a checkpoint whose sensing block is unknown cannot be
    evaluated at all.

    **Run a short pilot first**, to shake out the loop and to measure how long one
    update takes. That measurement sets the watcher's stall threshold, which is a
    guess until something measures it. A pilot's result is never reported as a
    result.

    Three traps, each already established elsewhere in this list:

    - **`--controlled-vehicles` is inert outside ring.** `HighwayTopologyEnv` clears
      `agent_ids` and defers to the demand spawner whenever continuous demand is
      active, so AV population must be set through `demand.av_penetration`. This bit
      the first run of task 5 and is recorded there as Amendment 2.
    - **The defaults are too small.** `--duration-steps` defaults to 120, and task 8
      measured concurrent AV count peaking at 5–6 out of ~40 active vehicles. Raise
      demand, episode length and penetration **together** until concurrent AV count
      reaches double digits; raising penetration alone leaves the realised count in
      single figures for the small-sample reason task 8 records.
    - **Resume, do not restart.** This is the project's first long run. Use
      `--resume-from` / `--resume-latest`, and prove resumption works by killing a
      run before depending on it.
69. **Evaluate trained MAPPO against `no_av` on `inverted_tree` for throughput.**
    The replication claim, and the only flow-level number the paper makes.

    **Gate: the evaluation must use the same `sensing:` block task 68 trained
    under.** If the two differ, the measured difference is a train/test mismatch
    rather than the controller, and nothing in the output would say so. Re-use the
    recorded config verbatim rather than rebuilding it.

    Seeds are the axis to spend on, now that topology and demand are fixed. Task 8
    measured single-seed realised penetration swinging between 0.08 and 0.43 at a
    nominal 0.20, and only 24 of 216 (cell, controller) pairs were ever measured
    with a congested shared seed. A throughput difference must clear that
    run-to-run term before it is reported.

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

111. **RESULT of the pre-registered run, and it is a null on all three criteria.**
     Read 2026-09-10 by `scripts/read_training_gate.py` on the completed 25 updates
     of seed 7.

     | criterion | bar | measured | verdict |
     |---|---|---|---|
     | 1: summed entropy falls | below 1.9775, which is 90% of 2 ln 3 | lowest 2.1754, first 2.1847, range 0.0147 | **fail** |
     | 2: the score trends up | move exceeds the step-to-step standard deviation | move -0.1091, step sd 0.2186 | **fail** |
     | 3: the action distribution leaves uniform | "a clear margin" -- no threshold, see task 109 | joint modal share 0.1502 against 0.1111 uniform and 0.1378 at initialisation | **fail** |

     Final head probabilities over 300 decisions: `desired_speed_bin` 0.393 / 0.263
     / 0.344 at entropy 1.0854, `desired_headway_bin` 0.313 / 0.305 / 0.382 at
     1.0934, both against a 1.0986 maximum. **Most of the departure from uniform is
     the network's initialisation, which already reads 0.1378.**

     The gate required all three and none is met. The five changes of task 103 are
     reported as a null, and tasks 105 to 110 say why none of them could have
     worked.

110. **A DEFECT IN MY OWN INSTRUMENT, found by Ankit: the floor had no error bar.**
     Every ratio in tasks 105 to 107 divided the measured gradient norm by the norm
     from ONE random permutation of the advantages. One permutation is a single draw
     from the floor's distribution, not the floor. Ankit's question was direct:
     several arms read 10% or more above 1.0, so why are they called floor readings?

     **Measured with 60 permutations per rollout, three seeds, reporting where the
     measured value sits in the floor's own distribution in standard deviations:**

     | arm | floor mean | floor sd | mean z |
     |---|---|---|---|
     | `sumo_capacity_drop`, penetration 0.25 | 0.054 | 0.016 | -0.10 |
     | `sumo_saturating`, penetration 0.25 | 0.088 | 0.025 | -0.01 |
     | `sumo_saturating`, penetration 1.00 | 0.042 | 0.011 | **+0.82** |
     | `sumo_capacity_drop`, gamma 0.999 | 0.052 | 0.013 | +0.33 |

     **The floor's standard deviation is 20 to 30% of its mean.** So the ratios of
     1.12, 1.27 and 1.40 that looked like improvements are inside one standard
     deviation of the floor's sampling noise, and the per-seed z values swing from
     -0.30 to +1.45 on the same arm. The conclusion of tasks 105 to 107 survives,
     but it was not properly supported until now: the correct statistic is the
     z-score against a permutation distribution, not a ratio against one draw.

     **The largest reading is penetration 1.00 at +0.82**, which is under one
     standard deviation on three seeds and is nothing on its own. It is the one arm
     worth more seeds if the question is ever reopened.

     **A NAMING ERROR OF MINE, also from Ankit's question.** "Oracle" was used for
     two unrelated things and they read as contradictory. The **metering oracle** is
     a CONTROLLER: perfect state, hand-written, it drives real vehicles and serves
     -0.8 and +0.8 more of them than no control. The "oracle" in the gradient probes
     drives nothing -- it is a synthetic advantage vector, +1 where the policy chose
     one value and -1 otherwise, injected into the gradient formula to check the
     statistic can move at all. One is about traffic, the other calibrates a
     measuring instrument. It is renamed **ceiling** throughout the scripts.

109. **A defect in my own pre-registration, recorded before the final numbers.**
     Task 103's criterion 3 reads "the modal action's share exceeds 1/9 by a clear
     margin". **"A clear margin" is not a threshold**, so unlike criteria 1 and 2 it
     cannot be evaluated mechanically and could be resolved either way once the
     number was in front of me. That is the failure a pre-registration exists to
     prevent, and I wrote it.

     It is recorded now, with the number already visible at update 23 of 25 and
     before the run finished, so the record shows what the criterion was worth
     rather than how it was applied. The reading at update 23, over 13,531
     observations:

     | head | mean probability | entropy | maximum |
     |---|---|---|---|
     | `desired_speed_bin` | 0.423 / 0.257 / 0.320 | 1.0778 | 1.0986 |
     | `desired_headway_bin` | 0.351 / 0.307 / 0.342 | 1.0970 | 1.0986 |

     Joint modal share 0.1486 against a uniform 0.1111, which is 34% above uniform
     in relative terms and 0.037 in absolute terms.

     **How it should have been written**: a numeric bar, for instance a modal share
     above 0.2, or a summed entropy below the criterion 1 threshold, which would
     have made criterion 3 redundant and revealed that at the time of writing.

     **It does not change this run's verdict**, because criteria 1 and 2 are
     numeric and both fail, and the gate required all three.

     **One observation worth keeping, with its own control.** The head that moved
     is `desired_speed_bin`, toward `slow`, and the head that did not is
     `desired_headway_bin`, which sits at 1.0970 of a 1.0986 maximum. That is the
     opposite of what task 105 would predict: the speed head is the one measured to
     be equivalent across its values on 97% of decisions, and the headway head sets
     `tau`, which affects car following at any speed.

     The control is the same measurement taken at update 1, when the actor had had
     one gradient step and was effectively at its initialisation:

     | update | speed head probabilities | speed head entropy | joint modal share |
     |---|---|---|---|
     | 1 | 0.362 / 0.326 / 0.312 | 1.0966 | 0.1378 |
     | 23 | 0.423 / 0.257 / 0.320 | 1.0778 | 0.1486 |

     **A randomly initialised network is already 0.1378 rather than 0.1111**, so
     most of the departure from uniform at update 23 is initialisation and not
     learning. What moved over 23 updates is the speed head's modal share, from
     0.362 to 0.423, and its entropy by 0.0188 of a 1.0986 range -- 1.7%. That is
     movement in the direction speed metering would need, at a rate consistent with
     the roughly 1% action-advantage correlation of task 107, and it is far too
     small to clear any of the three criteria.

108. **The control on tasks 105 to 107: a trained actor reads the same as a fresh
     one.** Measured 2026-09-10 by `scripts/measure_trained_actor_signal.py`.

     Every gradient reading in tasks 105 to 107 was taken at a randomly initialised
     actor. If the correlation between the advantage and the action rises as the
     policy trains, those readings describe a starting condition rather than the
     problem, and the conclusion drawn from them is wrong. The seed-7 run had
     reached 23 updates, so its checkpoint answers this directly. Same environment,
     same seeds, same reward, same floor and ceiling:

     | actor | measured over floor | per seed | oracle over floor | implied correlation |
     |---|---|---|---|---|
     | fresh initialisation | 0.848 | 0.608, 1.330, 0.607 | 26.2 | -0.0060 |
     | 23 updates of training | 0.812 | 0.465, 0.640, 1.332 | 34.0 | -0.0057 |

     **Indistinguishable.** The measurement is about the environment and the reward,
     not about where the policy happens to start, so tasks 105 to 107 stand.

     **A defect in this script, caught by the failure rather than by the result.**
     The first version read the checkpoint path from `sys.argv` AFTER replacing
     `sys.argv` for the config loader, so the path was None. It raised rather than
     loading nothing silently, which is the only reason it did not report a fresh
     actor twice under two labels and call them the same.

107. **Penetration does not produce a learning signal either, up to 100%.**
     Measured 2026-09-10 by `scripts/measure_penetration_signal.py`, closing the
     cheapest branch of task 106's option 2. Two demands, three penetrations, three
     seeds each, shipped bins, with the same floor and ceiling controls. The last
     column is the implied correlation between the advantage and the action,
     `(measured/floor - 1) / (oracle/floor - 1)`:

     | demand | penetration | AVs present | measured over floor | oracle over floor | implied correlation |
     |---|---|---|---|---|---|
     | `sumo_saturating` | 0.25 | 12.4 | 1.055 | 19.1 | +0.0030 |
     | `sumo_saturating` | 0.50 | 24.7 | 0.869 | 27.0 | -0.0050 |
     | `sumo_saturating` | 1.00 | 50.3 | 1.403 | 45.1 | +0.0091 |
     | `sumo_capacity_drop` | 0.25 | 42.4 | 0.848 | 26.2 | -0.0060 |
     | `sumo_capacity_drop` | 0.50 | 86.5 | 0.589 | 35.1 | -0.0121 |
     | `sumo_capacity_drop` | 1.00 | 168.2 | 0.960 | 80.2 | -0.0005 |

     **Every value is within 0.012 of zero and the sign is random.** At 100%
     penetration the policy commands the entire fleet -- 168 vehicles at 2400 veh/h,
     50 at 1200 -- and one agent's action still does not correlate with its own
     advantage. That is multi-agent credit assignment in its pure form: the other
     167 agents are exploring at the same time, and their contribution to the return
     swamps the one being credited.

     **The complete list of what has now been measured against the floor.** Every
     row uses the shuffled advantage as its floor and an action-correlated advantage
     as its ceiling, three seeds each:

     | varied | range | best measured over floor |
     |---|---|---|
     | reward decomposition | team, neighbourhood, own-vehicle, both | 0.850 |
     | action hold length | 1 s, 5 s, 20 s | 0.945 |
     | discount horizon | 10 to 1000 decisions | within 1.1x |
     | critic input | with and without privileged neighbourhood | within 1.05x |
     | speed bin scaling | four schemes, binding share 6% to 29% | 1.020 |
     | operating point | 900, 1200, 2400 veh/h | 1.269 |
     | AV penetration | 0.25, 0.50, 1.00 | 1.403 |

     Nothing clears the floor. The oracle reads 14 to 80 times it throughout, so the
     instrument was capable of a positive reading in every one of those arms.

     **WHAT IS AND IS NOT ESTABLISHED, stated precisely.** The metering oracle of
     task 102 is a specific heuristic given perfect state, so it is a lower bound on
     what the best controller could do and NOT an upper bound: "no controller can
     gain here" is not proven and cannot be proven this way. What is established is
     narrower and still decisive for the plan: **the mechanism this project posits --
     in-stream AV speed modulation -- neither gains when given perfect information
     nor presents a learnable gradient, on this road, at every operating point and
     penetration tried.**

     **The recommendation, now on evidence.** Report the simulation leg as a null,
     which is task 106's option 1. The reasoning is the order of the two problems:
     a counterfactual advantage estimator (option 3, COMA-style) is precisely
     targeted at the measurement above and would probably raise the correlation, but
     it fixes the learning of a mechanism that gains nothing when handed perfect
     information. Spending a research change on a better estimator is worth it only
     after an oracle shows headroom for some controller to reach, and the cheap way
     to look for that headroom is more oracles, not more training. The one positive
     figure on record remains +7.6 +/- 15.2 arrivals at congestion onset, which is
     one standard error.

106. **RETRACTION OF MY OWN RECOMMENDATION: rescaling the speed bins does not
     help, and neither does the operating point.** Measured 2026-09-10, minutes
     after task 105 recommended the rescale. The recommendation was an argument from
     a mechanism; this is the measurement, and it contradicts it.

     **The prediction that failed.** Task 105 established that the three speed values
     are equivalent on 97% of decisions and inferred that a rescaling which binds
     would produce a gradient. Four arms, shipped bins against three rescalings,
     three seeds each, with the same floor and ceiling controls:

     | speed bins | measured over floor | decisions the command binds on |
     |---|---|---|
     | shipped: 20 / 27 / 30 m/s | 0.848 | 6.0% |
     | prevailing segment speed as the context, no 12 m/s floor | **0.727** | 19.1% |
     | 0.60 / 0.85 / 1.00 of the vehicle's own speed | **1.020** | 29.2% |
     | 0.50 / 0.75 / 1.00 of the vehicle's own speed | **0.415** | 20.3% |

     **The binding share rose fivefold and the gradient stayed at its floor.** The
     recommended option 1 is the worst of the three rescalings. So inertness was a
     true description of the action and not the cause of the missing gradient.

     **The operating point does not do it either.** Shipped bins, demand varied:

     | demand | veh/h | AVs present | mean speed | measured over floor | oracle over floor |
     |---|---|---|---|---|---|
     | `sumo_burst` | 900 | 5.6 | 21.04 | 1.018 | 14.0 |
     | `sumo_saturating` | 1200 | 9.9 | 13.29 | 1.269 | 20.8 |
     | `sumo_capacity_drop` | 2400 | 42.4 | 4.70 | 0.848 | 26.2 |

     At 900 veh/h the traffic holds 21.0 m/s, so the shipped bins DO bind, and there
     are 5.6 agents rather than 42, so one agent is eight times as large a share of
     the fleet. The ratio is 1.018. The per-seed spread is 0.80 to 1.68 and nothing
     clears the floor.

     **What the ratio implies about the correlation.** With the advantage written as
     a correlation `rho` with the action plus independent noise, the measured ratio
     is about `1 + rho * (oracle_ratio - 1)`. At an oracle ratio of 14 to 39, a
     measured ratio of 1.02 to 1.27 puts **rho at or below about 0.01**, and the
     instrument would see rho of 0.05.

     **The convergence that matters.** Two instruments built for different purposes
     now agree. The perfect-information metering oracle measured -0.8 and +0.8
     arrivals against no control on this road (task 102), which is indistinguishable
     from doing nothing. The policy-gradient signal-to-noise puts the correlation
     between one AV's speed choice and its own advantage at about 1%. **A single
     AV's speed choice on this network at these penetrations changes almost
     nothing, and that is why no policy learns: there is very little to find.** The
     oracle says it from perfect information and no learning; the gradient says it
     from the learning signal and no oracle.

     **The revised decision for the user, replacing the four options in task 105.**
     The speed-bin rescale is off the table -- measured, not argued. What is left:

     1. **Report the simulation leg as a null**, with two independent lines of
        evidence rather than one, and the deployment carrying the feasibility claim
        as it already does.
     2. **Change what the AVs can do**, not how their reward is priced: raise
        penetration well above 25%, coordinate them as platoons rather than
        independently, or give the junction an explicit meter rather than relying on
        in-stream vehicles. Penetration is the cheapest of these and is being
        measured now with the same instrument, at 0.25, 0.5 and 1.0 on two demands.
     3. **Change the advantage estimator** to a counterfactual one -- a COMA-style
        baseline that marginalises the agent's own action -- which is the standard
        answer to exactly this measurement and is a research change rather than a
        configuration one.

     No recommendation yet: the penetration measurement is in flight and it bears
     directly on option 2. Recording the retraction now rather than after it, so the
     failed recommendation is on the record in its own right.

105. **THE ADVANTAGE CARRIES NO ACTION SIGNAL, and the speed bins are why.**
     Measured 2026-09-10 with `scripts/measure_gradient_signal.py`. This is the
     answer to why tasks 96 and 101 produced no learning, and it retracts four
     comparisons made earlier the same day.

     **The instrument, and the two controls it needed.** With advantages normalised
     to unit standard deviation, the norm of `d(policy_loss)/d(actor)` measures how
     much the advantage CORRELATES with the action: a term uncorrelated with the
     action cancels across the batch and the norm falls as 1/sqrt(N), while an
     aligned one adds. `policy_loss` itself says nothing -- at a probability ratio
     of 1 it is minus the mean normalised advantage, which is zero by construction
     however informative the advantage is, which is why task 96 could not settle
     this from the loss.

     The floor is the same advantages permuted across the batch: identical
     distribution, no correlation with the action. The ceiling is an advantage built
     to correlate with the action, +1 where the policy chose `slow` and -1
     otherwise. Three seeds, 17,296 decisions on the shipped configuration:

     | advantage | gradient norm | over the floor |
     |---|---|---|
     | as measured | 0.0499 | **0.784** |
     | shuffled (the floor) | 0.0637 | 1.000 |
     | action-correlated (the ceiling) | 1.5971 | **25.1** |

     **The advantage sits at its own noise floor and the instrument can read
     twenty-five times it.** PPO therefore has nothing to ascend, which is exactly
     what a policy frozen at 99% of maximum entropy looks like.

     **Every arm tried is at the floor.** Same instrument, same controls:

     | arm | measured over floor |
     |---|---|
     | team reward alone | 0.784 |
     | per-agent neighbourhood reward, blended 0.5 | 0.850 |
     | per-agent own-vehicle reward | 0.750 |
     | both together | 0.696 |
     | action held 1 s / 5 s / 20 s | 0.784 / 0.927 / 0.945 |
     | discount horizon 10 to 1000 decisions | within a factor of 1.1 |

     **FOUR RESULTS RETRACTED, all from the same session.** Before the floor existed
     I reported that the per-agent reward lowered the policy gradient (a paired ratio
     of 0.962), that the discount horizon does not matter, that privileged critic
     features do not help the gradient, and that a longer action hold does not help.
     Each was a comparison between two floors and none of them carried information.
     The paired construction was sound -- identical trajectories, verified by
     transition count and action sums -- and it was measuring a quantity that could
     not move.

     **What still stands from that work.** The critic regression, which is a
     different statistic: giving the centralized critic the agent's own
     neighbourhood raises out-of-sample R2 on the per-agent return from 0.808 to
     0.886, and leaves it at 0.854 against 0.858 under the team reward, which is the
     control that says the improvement is about the per-agent term. And the reward
     counterfactual of task 104, which is about the reward and not the advantage:
     one agent's action moves its own neighbourhood reward 17.8 times what its
     neighbours' actions move it. Both are real; neither reaches the gradient.

     **THE CAUSE. The action is inert on 97% of decisions.** `decode_speed_bin`
     returns `free_flow + offset` for offsets of -10, -3 and 0 m/s with a 12 m/s
     floor, so at a 30 m/s limit the three values are 20, 27 and 30 m/s.
     `setSpeed` is an upper bound that SUMO's car-following dominates, so on a
     vehicle already slower than the commanded value all three do the same thing. Of
     the 17,296 decisions above, **511 -- 3.0% -- were taken on a vehicle moving fast
     enough for the command to bind.** On that subset the ratio is 1.605, but at
     about 170 decisions per seed the per-seed values are 0.416, 3.811 and 0.588,
     which is too few to conclude. The 15.6% figure in task 104 is over AV-STEPS; 3%
     is over AV-DECISIONS, which is the population the gradient is built from.

     **The seed-7 run of task 103 is reported as a null in advance of finishing.**
     Nine of 25 updates at the time of writing: entropy 2.1847, 2.1783, 2.1811,
     2.1875, 2.1835, 2.1821, 2.1832, 2.1854, 2.1869 against a maximum of 2.1972 --
     flat to within 0.009, which is 0.4% of the range -- and the score random-walking
     between -0.254 and +0.119. It was left running rather than killed, so the
     pre-registered gate is read on the run as specified rather than on a run
     stopped when its numbers were disliked.

     **What this does NOT say.** It does not say a decentralised policy cannot help
     here. It says that with an action whose three values are indistinguishable on 97%
     of the decisions taken, no reward decomposition, discount horizon, critic input
     or hold length can produce a gradient, and none of the changes in task 103 could
     have worked. The environment question -- whether there is headroom at this
     operating point -- is still open and separate.

     **THE DECISION THIS NEEDS, and it is the user's.** Rescaling the speed bins
     changes what `desired_speed_bin` means on both sides of the deployed contract:
     the phone executes it and the Jetson logs it. Task 86 recorded that as the
     user's call and it now blocks the simulation leg. The options, with what each
     costs:

     1. **Keep the offsets, lower the floor, and make the context the local
        conditions rather than the lane limit.** `decode_speed_bin` already takes a
        `free_flow_speed_mps` argument; the SUMO env passes the lane limit. Passing
        the segment's prevailing speed and dropping the 12 m/s floor makes `slow`
        bind wherever the vehicle is moving. The head keeps its three names and its
        meaning becomes relative rather than absolute.
     2. **Rescale the offsets to a congested range**, for instance multiplicative
        0.6 / 0.85 / 1.0 of the vehicle's current achievable speed. Same effect,
        expressed on the action rather than on the context.
     3. **Leave the contract alone and change the operating point** so the traffic
        runs near 20 m/s, where the existing bins bind. That means a demand below
        the capacity collapse, which is the regime where the metering oracle
        measured +7.6 +/- 15.2 -- the one positive figure on record.
     4. **Leave everything and report the simulation leg as a null**, with this
        measurement as the reason.

     Recommendation: option 1. It is the smallest change that makes the head
     functional, it is what the argument name already says the parameter is, and it
     leaves the three action names -- which is what the deployed executor and the
     logs carry -- untouched.

104. **The credit-assignment diagnosis is now measured, and the speed head is
     inert on 84% of decisions.** Measured 2026-09-10 by
     `scripts/measure_credit_signal.py` while the task 103 run was in flight. Both
     halves were found by the same measurement.

     **The local reward carries the signal the team reward does not.** The
     measurement is a counterfactual on identical traffic: run the warm-up, hold one
     AV at 20 m/s for a 20 s window, repeat from the same seed holding it at 30 m/s,
     and record EVERY tracked agent's reward in both. Three seeds, three agents each,
     nine own and eighteen cross pairs:

     | quantity | value |
     |---|---|
     | agent i's local reward moved by agent i's own action | 22.82 |
     | agent i's local reward moved by another agent's action | 1.28 |
     | ratio, own over other | **17.8** |
     | team reward moved by one agent's action | 2.06 |
     | team reward over the same window, uncommanded | 193.03 |

     So one agent's action moves the team reward by **1.07%** of its magnitude, and
     moves its own neighbourhood reward by 17.8 times what its neighbours' actions
     move it. Task 96 diagnosed this from the policy loss; this measures it directly,
     and it is the justification for the per-agent term.

     **The control that had to come first.** Two uncommanded runs at the same seed
     differ by 0.000e+00 in the summed team reward. Without that, every difference
     above would be unattributable rather than small.

     **THE SPEED HEAD IS EQUIVALENT ACROSS ITS VALUES ON MOST DECISIONS.**
     `decode_speed_bin` returns `free_flow + offset` with offsets of -10, -3 and 0
     m/s and a floor of 12 m/s, and the SUMO env passes the lane limit as the
     context, so at a 30 m/s limit the three values are **20, 27 and 30 m/s**.
     `setSpeed` is an upper bound that SUMO's car-following then dominates, so a
     value above the speed the vehicle would take anyway changes nothing. Measured
     over 114,889 AV-steps at this operating point, mean AV speed 7.33 m/s:

     | value | m/s | share of AV-steps where it binds |
     |---|---|---|
     | `slow` | 20 | **15.6%** |
     | `nominal` | 27 | 0.9% |
     | `fast` | 30 | 0.07% |
     | (stopped, below 0.1 m/s) | | 25.6% |

     **On 84% of AV-steps all three values do the same thing, and on 99% `nominal`
     and `fast` do the same thing.** This is how the first version of the credit
     measurement was caught: it selected the three lowest-numbered agent ids, which
     after a 300 s warm-up are the oldest vehicles and so the deepest in the queue,
     and every commanded speed from 0.5 to 30 m/s produced a bit-identical
     trajectory. Exact zeros, not small numbers.

     **What this does and does not mean.** It is NOT that the head is unwired -- that
     was task 86 and it is fixed. The 15.6% of AV-steps where `slow` binds are the
     vehicles still moving fast as they approach the queue, which is exactly where
     speed metering has to act, so the mechanism the project studies IS expressible.
     What is lost is the other 84%, where the choice cannot matter, and those
     decisions put pure noise into the gradient. A vehicle stopped in a queue cannot
     help by any speed command, so part of that 84% is the operating point rather
     than the action space.

     **Consequence for task 103's gate, re-derived rather than moved.** Criterion 1
     asked for summed entropy below 1.978, which is 90% of ln(9). With the speed head
     equivalent across its values on 84% of decisions, an optimal policy is
     indifferent there, so the achievable mean summed entropy is about
     0.84 x ln(3) + a deterministic headway head, which is roughly 0.92. The gate
     threshold is therefore still reachable and is NOT changed. The confound is that
     a FAILURE of criterion 1 would be ambiguous between "the policy did not learn"
     and "most of its decisions had nothing to choose between".

     **The run was not restarted.** Rescaling the speed bins changes what the
     deployed actor's heads mean, on both sides of the contract, and task 86 already
     recorded that as the user's decision rather than mine. The run in flight is
     still informative under the reading above, and it costs no human time.

103. **PRE-REGISTERED: one configuration, everything enabled, one seed.** Written
     2026-09-10 BEFORE the run. The plan is `plans/plan_task_103_local_credit.md`.

     **What is being tested.** Whether a decentralised policy learns anything on
     this environment when every change that attacks the diagnosed cause is enabled
     at once. NOT how large its effect is: one seed cannot detect a 5% effect, since
     the paired standard error across seeds is 5 to 10%.

     **The five changes, deliberately confounded.** A per-agent reward measured over
     the agent's own segment and the segments downstream of it, blended half and
     half with the team reward; one decision per simulated second, with the action
     held for the ten simulation steps in between; the reward re-weighted to price
     delay, stopping and jerk; AV penetration 0.25; `deployed_fidelity` false.
     Ablations come after something works.

     **A sixth change that is not a treatment.** `sumo_burst` is replaced by
     `sumo_capacity_drop` at 2400 veh/h. Measured with no control at dt 0.1 over a
     900 s episode after a 300 s warm-up, in 60 s buckets: at 3000 veh/h the warm-up
     alone gridlocks the road, so the episode opens at 3.0 m/s with 70% of vehicles
     below 0.1 m/s and every minute of it is post-collapse; at 2400 the episode opens
     at 6.6 m/s and 33 completions per minute, collapses between minutes 5 and 6,
     and settles near 2.3 m/s. The episode is shortened to 600 s to sit around that
     collapse.

     **Latent demand does not engage on this road, and that is a finding.** Pending
     vehicles measured 0.0 at 2400 veh/h and 0.45 at 3000. Two lanes on every
     approach absorb the excess onto the carriageway rather than refusing entry, so
     the queue forms on the road. It reaches 181 only at 4500 veh/h, where the
     network is gridlocked from the first observed step. The saturated design's
     "served plus on-road plus latent" accounting therefore cannot separate the arms
     here; completed trips and delay have to.

     **THE GATE, fixed now.** Read off the training curve of seed 7, and nothing
     else:

     1. **Entropy falls** below 90% of its maximum. The maximum is ln(9) = 2.197 for
        the 9-action `speed_headway` profile; 90% is 1.978. The 4.38 figures from
        tasks 96 and 101 were against ln(81) = 4.394 and are NOT comparable.
     2. **The score trends up** by more than the variation between consecutive
        updates. The score is not comparable across configurations -- the local term
        changes its magnitude -- so only its trend within this run counts.
     3. **The action distribution leaves uniform**: the modal action's share exceeds
        1/9 by a clear margin.

     Passing all three earns a five-seed run, reported on evaluation seeds disjoint
     from the training seeds, exactly as task 99 required. Failing means the next
     iteration is on the reward, not on more seeds and not on more environment
     changes.

     **What a null would mean, stated now.** The metering oracle on this road at
     `sumo_oversaturated` measured -0.8 and +0.8 against no control, which is
     indistinguishable from doing nothing (task 102), and the only positive figure
     in that whole progression is +7.6 +/- 15.2 at congestion onset, which is one
     standard error. So a null here does not separate "a decentralised policy cannot
     learn this" from "there is nothing at this operating point to learn". It would
     be reported as the first, qualified by the second.

     **The metrics the five-seed run would report, also fixed now.** Primary:
     completed trips per episode, paired against `no_av` on the same traffic seeds,
     with the bar at two standard errors. Secondary: mean delay of completed trips,
     stopped fraction, mean absolute jerk, the trough of the 60 s rolling mean speed,
     summed `rolling_roadblock_score`, and collisions, which must be zero. Delay,
     stopping and jerk did not exist as measurements before this task; the first
     three are the axes the predecessor paper's gains were largest on.

     **One reading guard on the delay metric.** `mean_delay_recent` is a mean over
     COMPLETED trips, so a controller that stops a vehicle from completing improves
     it. That is why completed trips is the primary metric and why `latent_demand`
     and `throughput_recent` are reported beside it.

102. **Two lanes everywhere, and the oracle progression that made the case.**
     Decided by the user 2026-09-10; the measurements that led there are below,
     and they are a sequence of removed harms rather than a found benefit.

     The metering oracle -- perfect state, the project's own backpressure
     mechanism, no learning -- against no control, in the order the setup was
     corrected:

     | setup | metering vs no control |
     |---|---|
     | permanent yield at the merge, Krauss fleet | −16.6 +/- 7.8 |
     | zipper merge, calibrated W99, at congestion onset | **+7.6 +/- 15.2** |
     | zipper merge, W99, at the capacity peak (2100 veh/h) | −21 to −38 |
     | zipper merge, W99, past the peak (2400 veh/h) | −4 to −30 |
     | oversaturated, single-lane leaves | −10 to −32, latent unchanged |
     | **oversaturated, two lanes everywhere** | **−0.8 and +0.8** |

     **Each correction has removed harm and none has produced benefit.** The final
     row is the important one: with two lanes everywhere, metering is
     indistinguishable from doing nothing (−0.8 and +0.8 against a standard
     deviation of about 16), where on single-lane leaves it cost 10 to 32 vehicles.

     **Why the lane count mattered.** On a single-lane approach a slow vehicle
     cannot be overtaken, so an in-stream AV that slows is a rolling roadblock and
     not a meter. Ramp metering works because the meter sits beside the road; an AV
     is in it. The six entry leaves were one lane, so the mechanism the project
     exists to study was structurally unavailable on the segments where it would
     have to act. Both topologies now have two lanes on every approach; the
     bottleneck variant keeps its single-lane drop, which is intended rather than
     accidental.

     **The saturated design, and what it measures.** `sumo_oversaturated` holds
     3000 veh/h against a peak served flow near 1300, so the accounting closes:
     every scheduled vehicle is served, on the road, or latent. A controller must
     raise served and lower latent, and moving vehicles from the road into the queue
     is visible as one improving while the other worsens. Measured with no AVs at
     3000 veh/h on single-lane leaves: served settles near 1150 veh/h, the road
     holds about 480 vehicles, and latent grows about 0.44 vehicles per second once
     the leaves saturate. With two lanes the road absorbs far more, so at the same
     1200 s warm-up latent is 61 rather than 506 -- the two lane counts are NOT at
     the same point in their saturation trajectory, and comparisons across them are
     not like for like. Comparisons within a lane count are.

     **Three measurement defects found and fixed while doing this**, all in my own
     scripts rather than the simulator: the runs were sequential on one core of ten;
     a shared working directory had parallel runs overwriting each other's network
     file mid-read; and `ProcessPoolExecutor` hangs because libsumo keeps the
     simulation in module-level state and a forked child inherits a copy of it. The
     measurement now runs one independent subprocess per seed and takes control
     decisions at 1 Hz rather than 10 Hz, which is also the more realistic rate.

101. **RESULT on the corrected road: the environment is now right and the
     learning is not.** Run 2026-09-10 per the amended pre-registration (task 99):
     zipper merges, the calibrated Wiedemann-99 fleet, `sumo_burst`, 900 s episodes
     at dt 0.1, 50 updates of three episodes each, five policies, evaluated on ten
     traffic seeds disjoint from training. Rows in
     `plans/result_task99_corrected_road.json`.

     | arm | arrivals | trough m/s | recovery s | roadblock | collisions |
     |---|---|---|---|---|---|
     | `no_av` | **249.5 +/- 10.3** | 11.22 | 70.2 | 10.4 | 0 |
     | `density_lookup` | 248.0 +/- 14.1 | 12.15 | 82.0 | 22.6 | 0 |
     | `mappo` | 236.3 +/- 8.9 | 9.46 | 83.6 | 338.5 | 3.6 |

     Paired against `no_av` on the same traffic:

     | arm | metric | difference | verdict |
     |---|---|---|---|
     | `density_lookup` | arrivals | **−1.50 +/- 5.17** | **no effect** |
     | `density_lookup` | trough speed | +0.93 +/- 1.72 | no effect |
     | `density_lookup` | recovery | +4.03 +/- 87.39 | no effect |
     | `density_lookup` | roadblock | +12.16 +/- 4.70 | real |
     | `mappo` | arrivals | **−13.22 +/- 3.51** | **real** |
     | `mappo` | trough speed | −1.76 +/- 0.93 | no effect |
     | `mappo` | recovery | +13.38 +/- 44.80 | no effect |
     | `mappo` | roadblock | +328.01 +/- 25.62 | real |

     **The road fix moved the non-learning baseline from harmful to neutral.**
     `density_lookup` was −17.40 +/- 7.97 arrivals on the network with the permanent
     yield and is −1.50 +/- 5.17 now: indistinguishable from doing nothing. That is
     the clearest evidence that the earlier failures were the environment. A local
     density-and-queue heuristic can now act without paying for it.

     **MAPPO is still worse than doing nothing, and now more clearly so.** −13.22
     +/- 3.51 arrivals, a 5.3% reduction that clears the bar comfortably at ten
     seeds. It also causes 3.6 collisions against zero for both baselines, which is
     possible at all only because the calibrated model can collide (task 100).

     **Because the policy still did not learn.** Over 50 updates the score rose from
     1.636 to about 1.71 and plateaued, and entropy ended at 4.3822 against a
     maximum of 4.394 -- 99.7% of maximum, essentially uniform. Tripling the
     episodes per update improved the gradient enough to show a trend where the
     previous run had none, and not enough to move the policy. So this measures a
     near-uniform policy over the action space, and a near-uniform policy on this
     action space slows AVs at random, holds lanes (roadblock 338.5 against 10.4)
     and crashes occasionally.

     **What is now isolated.** The environment supports the phenomenon: the
     fundamental diagram has a 24% capacity drop, a perfect-information oracle
     scores positive at the operating point (+7.6 +/- 15.2), and the non-learning
     baseline is neutral rather than harmed. What remains is the learning problem,
     and it is quantified rather than guessed: a team reward shared among about
     twelve agents moves less than its own noise under one agent's action, and even
     at three episodes per update this is 1/17th of the per-update trajectory count
     Flow's benchmarks use, at 1/10th of their iteration count.

     **The next thing to try, and it is a decision.** Three options, in increasing
     order of departure from the current design: raise the trajectory count per
     update towards Flow's 50, which is a pure compute cost; give each agent a
     reward component it can move on its own, which changes the objective; or adopt
     Flow's formulation of one policy emitting all AVs' actions jointly, which
     abandons decentralised execution and so the project's premise. The first is the
     only one that does not change what is being claimed.

100. **The calibrated driving model gives up the collision-free guarantee, and
     that is the same trade in both directions.** Measured 2026-09-10 at 1200 veh/h
     with no AVs, three 600 s runs: SUMO's default Krauss produces **0** collisions
     and the calibrated Wiedemann-99 produces **2**.

     This is not a defect in either model. Krauss computes a collision-free safe
     speed exactly and recovers from a disturbance immediately, which is why task 84
     chose SUMO over highway_env in the first place -- "SUMO's car-following cannot
     produce a collision" was the premise of the whole migration. The same property
     is why it has no capacity drop and nothing for a controller to recover: a model
     that never over-brakes cannot produce the stop-and-go instability the
     controller exists to damp. W99's psycho-physical thresholds do over-brake, on
     purpose (CC2 and CC6 "help introduce stop-and-go dynamics"), and the price is
     that a collision becomes possible.

     **Three consequences, recorded rather than resolved.**

     - Every "zero collisions" claim in this project is now conditional on the
       driving model. The claims stand for Krauss; under W99 the rate is small but
       not zero. `TestNoCollisionsEver` and the action-head collision guard both
       exercise the Krauss path, so they still pass and now cover less than their
       names suggest.
     - `crash_penalty` in the reward is no longer inert on SUMO. It was recorded as
       structurally absent because `crashed_agent_ids()` returns nothing and SUMO
       could not collide; under W99 the collision counter does move, so the -5.0
       collision weight can now charge a policy.
     - The safety layer matters again. It was reasonable to leave
       `apply_safety_layer` unrun on SUMO while the simulator's own model was the
       guarantee. It is not reasonable under W99, and whether to run it is a
       decision that now has consequences.

     The honest framing for the paper is that the simulator offers a choice between
     a fleet that cannot crash and a fleet that can congest, and the phenomenon
     under study requires the second.

99. **AMENDED PRE-REGISTRATION for the re-run on the corrected road.** Written
    2026-09-10, before training, superseding task 93. Amended for two reasons that
    are both about the instrument and neither about a result: the road and the
    driving model changed (task 98 and the W99 calibration), and the variance
    changed with them.

    **What changed in the setup.** Merge nodes are zipper junctions. The fleet uses
    the predecessor paper's calibrated Wiedemann-99 parameters, so the fundamental
    diagram has a capacity drop: served flow peaks at 1298 veh/h and falls 24%,
    where the old road rose monotonically to 1110 with no drop at all.
    `sumo_saturating` is 1200 veh/h, the congestion onset. `sumo_burst` keeps a
    sustainable 900 base doubled for 150 s, re-profiled as speed 23.0 to 10.2 m/s
    with the queue peaking at 23 and recovering to 19.5.

    **The evaluation seeds, named before the run: 57, 67, 77, 87, 97, 107, 117,
    127, 137, 147.** All ten are disjoint from the five the policies trained on
    (7, 17, 27, 37, 47). The trainer seeds each episode as `seed + update`, so a
    shared base seed does not reproduce a training episode exactly, but evaluating
    on unseen traffic realisations removes the question.

    **Seeds: 10 for the evaluation, not 5.** The oracle's paired standard deviation
    at this operating point is 15.2 arrivals against 7.8 on the old road, because
    the congestion is now genuinely stochastic. At five seeds the two-standard-error
    bar admits only effects above 13.6 arrivals, or 5.4%; at ten it is 9.5, or 3.8%.
    Training stays at five seeds because it costs hours where evaluation costs
    minutes, and the evaluation is where the power is needed.

    **Training budget: three episodes per update, 50 updates.** The previous run's
    policy never left its initialisation, and the diagnosis was gradient noise
    rather than step count: a team reward shared among twelve agents moves less than
    its own noise under one agent's action, and one episode per update is 1/50th of
    the gradient quality Flow's benchmarks use. Three episodes per update at 50
    updates costs the same wall clock as the previous 100 single-episode updates.

    **Everything else is unchanged from task 93** and is what will be reported:
    completed trips as the primary metric; trough speed, recovery time, roadblock
    sum and collisions as secondaries; `no_av` and `density_lookup` as comparators;
    the final checkpoint rather than the best; a two-standard-error bar paired on
    seed; no commanded-speed arm as a comparator; and a null reported as a null.

    **Recorded before the run, so it cannot be claimed afterwards:** the oracle on
    the corrected road at this operating point scores +7.6 +/- 15.2, which is about
    one standard error. The honest prior is that any effect available here is small
    enough that ten seeds may still not resolve it.

98. **Why nothing works: the network has an unintended permanent yield, and it
    sets capacity.** Asked 2026-09-10 why MAPPO fails here when Flow reports gains
    on similar networks, and whether the sensing model is the cause. It is not the
    sensing model. `inverted_tree` as generated is not the network it was meant to
    be.

    **The evidence, in the order it was found.**

    A perfect-information oracle also fails. `scripts/measure_oracle_metering.py`
    slows AVs only while the segment downstream of them is congested -- the
    project's own declared backpressure metering -- reading true simulator state
    rather than the sensing model, with no learning involved. Five seeds on
    `sumo_burst`: `no_av` 259.8 +/- 7.2 arrivals, metering at 8 m/s
    −16.6 +/- 7.8, at 12 m/s −6.4 +/- 4.4, at 16 m/s −1.2 +/- 2.6. Nothing beats
    inaction and the gentlest intervention is merely the least harmful. Whatever is
    wrong is upstream of both sensing and learning.

    Neither topology has a capacity drop at dt 0.1. Served flow rises
    monotonically with demand -- 890, 908, 950, 1100, 1110 veh/h on `inverted_tree`
    at 900 to 2400 veh/h offered, and 802 to 1068 on the bottleneck variant. A
    capacity drop, throughput FALLING past a critical point, is the inefficiency
    every mixed-autonomy control result exploits; Flow's bottleneck benchmark has
    one, and its paper describes the opportunity as arranging vehicles "so that
    they merge optimally without the sharp decelerations that eventually give rise
    to the bottleneck".

    **Then the segment profile gave it away.** At 1800 veh/h with no AVs:

    | segment | lanes | mean speed | queue | vehicles |
    |---|---|---|---|---|
    | `tree_middle_b1` | 2 | **1.74** | **64.4** | 69.3 |
    | `tree_middle_b2` | 2 | 22.76 | 0.0 | 5.9 |
    | `tree_trunk_c` | 2 | 22.87 | 0.0 | 12.1 |

    `b1` is jammed solid between neighbours that are both free-flowing, and the
    trunk it feeds is nearly empty. That is not congestion physics.

    **The cause is in the generated network.** Every edge is written with
    `priority="-1"`, so `netconvert` broke the tie at junction `c` by geometry. The
    built connections read `state="M"` for `tree_middle_b2` into the trunk and
    `state="m"` for `tree_middle_b1`: b1 yields permanently, so its three leaves
    (a1, a2, a3) starve behind a yield that never clears while the trunk runs
    empty.

    **Confirmed by fixing it.** Rebuilding the merge nodes as SUMO `zipper`
    junctions, which alternate between approaches, no AVs, three seeds, arrivals
    per 600 s:

    | veh/h | as built | zipper merge |
    |---|---|---|
    | 900 | 148.3 | 148.0 |
    | 1500 | 158.3 | **252.3** |
    | 1800 | 183.3 | **252.0** |
    | 2400 | 185.0 | **268.0** |

    Capacity goes from about 1110 veh/h to about 1600, and the two middles
    symmetrise: b1 rises from 1.02 to 11.95 m/s while b2 falls from 22.87 to 11.26,
    with the starved leaves clearing (a5 from 4.70 to 23.63). At 900 veh/h nothing
    changes, because that is below capacity either way.

    **What this invalidates.** Every capacity figure in this project measured the
    yield rather than the road, and both demand configs were chosen against it. It
    also means **branch fairness -- the objective `inverted_tree` exists to study --
    was structurally unattainable**: b1's three branches could never be served
    equally with b2's, whatever a controller did. And it explains why every
    controller tried so far does harm: the constraint is right-of-way, and slowing
    vehicles that are already yield-limited only reduces what arrives.

    **On Flow specifically, four differences beyond this one**, checked against
    Vinitsky et al. 2018 rather than recalled: its benchmarks are the figure-eight,
    merge, grid and bottleneck -- there is no tree; they use a single centralized
    policy emitting all AVs' continuous accelerations, not decentralized agents
    sharing one reward, so the policy's action moves the reward substantially; they
    train 500 iterations of 50 rollouts each, against our 100 updates of one
    rollout; and their merge reward is dense and normalised in every vehicle's
    velocity, where ours is led by a 60 s rolling throughput count whose response
    to an action arrives long after the GAE horizon. Their own merge benchmark
    "started to gradually degrade after certain iterations, suggesting that the
    problem is difficult to solve with existing optimization methods".

    **DECISION FOR THE USER, because it changes the research object.** Fixing the
    merge makes `inverted_tree` the network it was described as, but it invalidates
    every capacity table and both demand levels, and the operating point, the burst
    scenario and the dt-convergence result all need re-measuring on the corrected
    road. The alternative -- keeping it -- means studying a network whose capacity
    is set by an arbitrary tie-break and whose fairness objective is unreachable.

97. **RESULT of the pre-registered run: MAPPO does not beat doing nothing, and is
    measurably worse.** Executed 2026-09-09 exactly as task 93 fixed it in advance:
    five seeds (7, 17, 27, 37, 47), `sumo_burst`, 900 s episodes at dt 0.1, the
    final checkpoint of each seed, comparators `no_av` and `density_lookup`,
    completed trips as the primary metric, a two-standard-error bar paired on seed.
    Produced by `scripts/evaluate_burst_scenario.py`.

    | arm | arrivals | trough m/s | recovery s | roadblock | collisions |
    |---|---|---|---|---|---|
    | `no_av` | **259.8 +/- 7.2** | 8.80 | 264.5 | 32.9 | 0 |
    | `density_lookup` | 242.4 +/- 19.2 | 8.10 | 342.3 | 173.7 | 0 |
    | `mappo` | 242.8 +/- 17.4 | 8.27 | 213.6 | 1365.3 | 0 |

    Paired against `no_av` on the same seeds:

    | arm | metric | difference | verdict |
    |---|---|---|---|
    | `density_lookup` | arrivals | −17.40 +/- 7.97 | **real** |
    | `density_lookup` | trough speed | −0.70 +/- 0.53 | no effect |
    | `density_lookup` | recovery | +17.17 +/- 75.35 | no effect |
    | `density_lookup` | roadblock | +140.80 +/- 36.45 | **real** |
    | `mappo` | arrivals | **−17.00 +/- 7.55** | **real** |
    | `mappo` | trough speed | −0.53 +/- 1.27 | no effect |
    | `mappo` | recovery | −91.40 +/- 87.66 | no effect |
    | `mappo` | roadblock | +1332.38 +/- 134.75 | **real** |

    **The primary metric says both controllers reduce throughput.** MAPPO completes
    17.0 +/- 7.6 fewer trips than an uncontrolled fleet, a 6.5% reduction, and the
    project's existing non-learning baseline is indistinguishable from it at
    −17.4 +/- 8.0. Zero collisions in all fifteen runs.

    **Nothing else clears the bar.** The trough is unchanged for both. MAPPO's
    recovery is 91 s faster on average but the spread is 88, so it does not clear
    two standard errors; on the pre-registration that is no effect, and it is
    recorded here rather than promoted.

    **The one term that moves decisively is the wrong one.** MAPPO's
    `rolling_roadblock_score` is 1365 against 33 for uncontrolled traffic, 41 times
    higher and far outside the noise. A policy that emits `slow` about a third of
    the time holds lanes below free flow constantly, which is exactly what that
    term exists to detect.

    **The qualification, recorded BEFORE this result was measured (task 96): the
    policy did not learn.** Entropy ended at 4.3548 against a maximum of 4.394, so
    it finished at 99.1% of maximum entropy, and the score and throughput were flat
    across all 100 updates. This measures a near-uniform policy over the action
    space more than it measures what MAPPO can do. The honest reading is therefore
    narrow: **with a team reward shared among about twelve agents, 100 updates of
    MAPPO produced no learning signal, and acting near-randomly over this action
    space costs 6.5% of throughput.**

    **It is consistent with task 92.** There is no throughput effect at a converged
    step size for a policy to find, so a policy that finds nothing is the expected
    outcome, and one that acts anyway does harm. What this run does NOT establish
    is that a policy with a working learning signal would fail; that question needs
    the credit-assignment problem addressed first, and it is the natural next task.

96. **The policy is not learning, and the cause is the credit-assignment signal
    rather than a defect.** Diagnosed 2026-09-09 at update 26 of the pre-registered
    run, before spending the remaining four hours on it.

    **The symptom.** Over 26 updates and five seeds: score flat at 1.60 to 1.73,
    throughput flat at 16.7 to 16.9 arrivals per 60 s, and entropy 4.3714 to 4.3674
    against a maximum of ln(81) = 4.394. The policy moved 0.004 in entropy and is
    effectively frozen. Zero collisions throughout.

    **A wrong diagnosis, and the control that caught it.** The objective is 99.96%
    value loss: `loss` about 116, of which `value_coef * value_loss` is 115.9,
    against a policy loss of -0.0004 and an entropy bonus of 0.044. Measured
    gradient norms on one minibatch: the critic's is 79.7 and the actor's 0.378,
    and `ppo_update` clips ONE norm over both networks, so `max_grad_norm` 0.5
    scaled every parameter by 0.0063. That looks decisive -- the actor apparently
    trained at 0.6% of its intended rate -- and it is wrong.

    **Adam makes a uniform gradient rescaling irrelevant.** Its update is
    `lr * m / sqrt(v)`, and scaling every gradient by a constant scales both
    moments, so the step is unchanged. Measured directly: 20 Adam steps on the same
    problem move a parameter by 0.383268 with unscaled gradients and 0.383267 with
    gradients scaled by 0.0063. The separate-clipping change was reverted rather
    than shipped, because it changes shared code that every experiment in this
    project uses and it has no measurable effect under Adam. The test written for
    it passed against the un-separated control, which is how the wrong diagnosis
    was caught before it was acted on.

    **What is left, and it is not a bug.** With advantages normalised, a policy
    loss of -0.0004 means the ratio barely leaves 1, which means the advantages
    carry almost no information about the actions. That is the expected shape of a
    shared team reward divided among about twelve agents: one agent's choice moves
    the team reward by far less than the noise in it over the GAE horizon of 20 s.
    The signal is weak because the problem is configured that way, not because a
    knob is set wrong.

    **The run continues unchanged.** This is what the pre-registration exists for:
    it fixed the metric and the bar before any of this was visible, so a null gets
    reported as a null. Task 92 already retracted the evidence that a throughput
    effect exists here at a converged step size, and a policy that cannot find a
    signal that is not there is the consistent outcome rather than a surprise.

95. **Round 3 leftovers, recorded rather than fixed, with the reason.**

    - **The critic's `time` input is far outside its normalisation range.**
      `FIELD_SCALES["time"]` is 120.0, so at the 900 s episodes `mappo_sumo` now
      runs it reaches 7.5 while every other input sits near [0, 1]. It is one of
      only three top-level inputs among the critic's 115. NOT changed: `FIELD_SCALES`
      is shared with `sim_contract`, this is a conditioning problem rather than a
      correctness one, and a finite-horizon value function should see the clock. If
      it is changed it should become the episode duration so the feature spans
      [0, 1], and that must not be done while a pre-registered run is in flight.
    - **`get_segment_metrics(snapshots)` ignores its argument when the cache is
      warm.** No external caller passes it -- `base_ctde_env.py`, `run_baseline.py`,
      `evaluate_policy.py` and `validate_topology_baselines.py` all call it with no
      argument -- so it is latent, but the signature invites a caller to be silently
      ignored. The fix is to hold the step's snapshots on the environment and drop
      the parameter. NOT done while a training run is in flight, because it changes
      the environment the run is training against and the evaluation must use the
      same one.
    - **`all_lane_av_low_speed_occupancy` reaches the reward at weight zero.** It
      has no entry in `DEFAULT_REWARD_WEIGHTS` and no config overrides one. Not a
      defect: the per-segment version is one of the eleven fields the critic reads,
      and `rolling_roadblock_score`, which carries -2.0, is built from it. Now
      documented at the point it is computed so nobody reads it as a penalised
      quantity.
    - **The comment round 3 reported as misattributing the -2.0 weight does not
      reproduce.** At HEAD that comment sits above `rolling_roadblock_score` and
      describes it correctly.

94. **The action heads could cause collisions, and the first training run was
    invalid because of it.** Found 2026-09-09 by the collision counter, on the
    smoke evaluation of a two-update checkpoint: 30 collisions on the learned arm
    against 0 for every baseline. SUMO's car-following being unable to produce a
    collision is the premise of the whole migration, so this was a defect in the
    actuation I had just added, not in SUMO.

    `hold_lane` set the lane-change mode to 0. Bits 8-9 of a SUMO lane-change mode
    are the collision-avoidance bits, so 0 does not mean "no lane changes", it
    means "no lane changes and no safety". Separately, `changeLane` holds its
    choice for a duration, so a vehicle told to prefer a lane on one step and to
    hold its lane on the next carried on into the change with the checks removed.

    Measured over 3000 steps, actions varying per agent per step:

    | heads varied | collisions |
    |---|---|
    | speed alone, headway alone, lane alone, merge alone | 0 each |
    | speed + lane | 0 |
    | speed + merge | 0 |
    | headway + merge | 0 |
    | **lane + merge** | **646** |
    | all four | 703 |

    **Every single head was clean and the pair was not**, which is why the guard
    now varies the heads together. A single-head test would have passed throughout.

    Two fixes, and each is sufficient on its own -- confirmed by reverting them
    separately, where either alone keeps the count at zero and only the shipped
    combination fails. Both are kept because both are right independently:
    `hold_lane` now uses mode 1536, which clears the vehicle's own motivations to
    change lane while leaving collision avoidance on, and it cancels any pending
    change by requesting the current lane.

    **The first five training runs were killed.** They had been training against an
    environment in which the policy's own actions could cause collisions, which
    contradicts the premise and makes the crash penalty fire on the environment's
    defect rather than the policy's behaviour.

93. **PRE-REGISTERED: what the MAPPO run will be judged on.** Written 2026-09-09
    BEFORE the run, because the previous headline was a number chosen after the
    fact from a sweep of constant commanded speeds, and Ankit's instruction was
    "no cheating and cherry picking". Everything below is fixed in advance. If the
    run does not clear these bars it is reported as a null.

    **What is run.** `configs/training/mappo_sumo.yaml` unchanged: MAPPO, `full`
    action profile (all four heads now actuate), `deployed_fidelity: true`, the
    cited sensing noise, dt 0.1, 900 s episodes on `sumo_burst`, one rollout per
    episode, 100 updates. Seeds fixed in advance: **7, 17, 27, 37, 47.** These are
    NOT the seeds the retracted arm tables were measured on (3, 7, 11, 19, 23),
    deliberately.

    **What it is compared against.** `no_av` at the same seeds and the same
    scenario, and `density_lookup`, the project's existing non-learning baseline.
    No commanded-speed arm is a comparator: a constant speed chosen from a sweep is
    an oracle, not a controller, and the sweep that produced one is retracted.

    **Primary metric, decided now: completed trips per episode.** One number,
    reported as a mean over the five seeds with its standard deviation and the
    paired difference against `no_av` on the same seeds.

    **Secondary metrics, also decided now**, because the scenario has a shape a
    single total hides:
    - the minimum of the 60 s rolling mean speed, which is the depth of the trough;
    - the time from the end of the burst until the queue returns below 10 vehicles,
      which is the recovery;
    - `rolling_roadblock_score`, summed, which is whether the policy bought its
      result with behaviour the contract forbids;
    - collisions, which must be zero.

    **The bar.** A gain is reported as real only if the paired difference against
    `no_av` exceeds two standard errors over the five seeds. Anything smaller is
    reported as no effect. No seed is dropped, no arm is selected, and the metric
    is not chosen after the numbers are seen.

    **What a null would mean, stated now so it cannot be reinterpreted later.**
    Task 92 retracted the evidence that a throughput effect exists here at a
    converged step size, so a null is the expected outcome rather than a
    disappointment, and it is still worth reporting: it would say that under the
    deployment's own sensing model, on this network, a policy with speed, headway,
    lane and merge control does not beat doing nothing. The alternative reading --
    that the reward does not ask for the right thing -- is guarded against by
    reporting arrivals directly rather than the reward.

92. **RETRACTION: the throughput gain was an artefact of the simulation step.**
    Measured 2026-09-09 after the user asked for `dt: 0.1` so the deployment's
    measured sensing latency could be represented. It retracts task 86 and moots
    task 91.

    `scripts/measure_step_size_convergence.py`, 600 s episode after 300 s of
    warm-up, five seeds, `sumo_saturating`, arrivals:

    | dt | steps | uncommanded | AVs at 10 m/s | gain |
    |---|---|---|---|---|
    | 1.0 | 600 | 132.0 +/- 6.4 | 155.2 +/- 10.5 | **+17.6%** |
    | 0.5 | 1200 | 134.4 +/- 5.9 | 157.8 +/- 16.0 | +17.4% |
    | 0.2 | 3000 | 157.6 +/- 12.3 | 160.4 +/- 7.2 | +1.8% |
    | 0.1 | 6000 | 168.4 +/- 11.3 | 161.2 +/- 8.6 | **−4.3%** |
    | 0.05 | 12000 | 173.0 +/- 8.4 | 161.6 +/- 8.6 | **−6.6%** |

    **The commanded arm barely moves: 155.2, 157.8, 160.4, 161.2, 161.6, a drift of
    6.4 arrivals across a twentyfold change in step size and well inside its own
    standard deviation. The uncommanded arm rises 31%, from 132.0 to 173.0.** The
    treatment is invariant to the numerical parameter and the control is not, so
    the difference between them was never a property of the traffic.

    **Mechanism.** SUMO's `--step-length` is the physics step, and the migration
    passed `dt` straight to it. A vehicle travelling 24 m/s advances 24 m per step
    at dt 1.0, so junction gap acceptance and car following were resolved at 24 m
    granularity and the uncommanded fleet lost throughput to the integration. A
    fleet held at 10 m/s advances 10 m per step and loses much less. Commanding a
    lower speed was buying back numerical resolution, not damping waves.

    **The project already knew this and the migration lost it.**
    `HighwayTopologyEnv` integrates at `physics_substeps: 10`, with a comment at
    `src/envs/topology_env.py:233-236` saying why: "decisions happen once per dt,
    but the physics must integrate at a finer grid (highway_env is built for ~10-15
    Hz): a single 1 s Euler step drives IDM vehicles through each other and to
    negative speeds". The SUMO env has no equivalent, so from the first commit of
    the migration the SUMO fleet integrated at 1 s where the highway_env fleet it
    was compared against integrated at 0.1 s. Every SUMO capacity figure, including
    the "junction-limited at about 900 veh/h" that the demand configs were chosen
    against, was measured on the coarse integration.

    **What is retracted.** Task 86's throughput gain, in all three of its recorded
    magnitudes: 52% at one seed, 28% at five, 17.6% after the fleet correction. At
    a converged step size holding AVs at 10 m/s does not raise throughput, it
    lowers it by 4 to 7%. Task 91's finding that 48.5% of the gain came from
    behaviour the contract forbids is moot, because there is no gain. The metering
    exemption committed in `2d313cc` is kept: it is a correct refinement of a
    metric that could not tell metering from obstruction, and it stands on its own
    reasoning, but the measurement that motivated it is withdrawn.

    **What this does NOT establish.** That no controller can raise throughput here.
    The retracted evidence came from an oracle -- every AV commanded to the same
    speed for a whole episode, which no policy can express. A learned policy acting
    on local conditions might still find something. What is gone is the evidence
    that an effect was there to be found, which is what justified the training run.

    **Open, and the user's call: whether the simulation leg still has a question.**
    The paper is a deployment story and the simulation exists to replicate that
    MAPPO works under the deployment-measured sensing model. That replication can
    still be run and reported -- including as a null -- but it should be commissioned
    knowing that the uniform-speed oracle now shows no effect to find at this
    operating point and topology.

91. **MOOT after task 92: there is no gain to be admissible or not.** The metering
    exemption is kept on its own reasoning. Original text follows.

    ~~The measured throughput gain is largely inadmissible under the project's own
    contract.~~ Measured 2026-09-09 while diagnosing why the reward ranks the
    commanded-speed arms differently from arrivals. The reward is not
    mis-specified; it is correctly refusing a strategy the project forbids.

    Reward decomposed by term, three seeds, 600 steps, per step:

    | arm | arrivals | reward | `rolling_roadblock_score` contribution |
    |---|---|---|---|
    | uncommanded | 132.7 | +0.919 | −0.011 |
    | 10 m/s | **162.3** | +1.181 | **−0.741** |
    | 15 m/s | 151.7 | +1.130 | −0.363 |
    | 20 m/s | 146.0 | **+1.243** | −0.015 |

    Every other term ranks 10 m/s first. `rolling_roadblock_score` at weight −2.0
    is the whole inversion: remove it and 10 m/s scores 1.922 against 20 m/s at
    1.258, which is the arrival order.

    **What the term measures.** `_rolling_roadblock_score` fires only when AVs
    occupy every lane of a segment at a mean speed more than 8 m/s below free flow,
    AND the segment's `jam_fraction` is at most 0.25, AND its queue length is zero.
    It is deliberately narrow: slow AVs while the road around them is clear. That
    is the README's prohibition on rolling roadblocks, made measurable.

    **Where it fires.** Over 300 steps at 10 m/s it fires on 767 segment-steps
    against 7 for both uncommanded traffic and 20 m/s. Not only on the single-lane
    leaves, where one AV trivially holds "every lane": 159 of them are on the
    two-lane `tree_middle_b2` and `tree_trunk_c`.

    **Metering or obstruction?** The guard checks jam and queue on its own segment
    only, so a segment held slow *because the next one is jammed* would be scored
    as obstruction although it is metering. Classifying the 767 firings by the
    state of the downstream segment at the same step:

    | downstream state | segment-steps | share |
    |---|---|---|
    | jammed (`jam_fraction` > 0.25) — metering | 295 | 38.5% |
    | queued but not jammed | 0 | 0.0% |
    | the trunk, which has no downstream segment | 100 | 13.0% |
    | **clear — obstruction by the project's definition** | **372** | **48.5%** |

    So it is not simply a mis-specified metric. Nearly half the firings are AVs
    holding a clear segment with a clear road ahead.

    **Consequence for the headline number.** The 17.6% gain at 10 m/s is achieved
    substantially through behaviour the project's contract forbids. The arms that
    do not trigger the penalty do not clearly beat doing nothing: at five seeds,
    20 m/s gives 138.8 +/- 10.6 against 132.0 +/- 6.4 uncommanded, which is within
    noise.

    **Decision for the user.** Three readings, and they lead to different papers.
    (a) The contract stands: the admissible gain is what a policy achieves without
    triggering the term, and on current evidence that is not distinguishable from
    zero — a null worth reporting, and the reason to report it is that the
    unconstrained gain is large. (b) The term is too strict on a segment whose
    downstream is jammed; exempting those 38.5% would license metering while still
    forbidding the 48.5%. That is a defensible refinement of the metric, not a
    weakening, but it must be made before the training run rather than after seeing
    the result. (c) The prohibition itself is reconsidered for single-lane
    approaches, where "hold every lane" cannot distinguish a roadblock from an
    ordinary slow vehicle.

    My recommendation is (b) plus reporting under both, because the exemption has a
    stated principle -- a jammed downstream segment is a traffic reason for being
    slow, which is exactly what the guard's other two conditions are testing for --
    and because the 48.5% that remains forbidden is the part that would make a
    reviewer uncomfortable.

90. **The two unmeasured sensing parameters now carry citations.** Ankit,
    2026-09-09: use prior papers to fill the values that could not be measured, or
    HERE data if possible.

    HERE does not apply. It is a segment-level traffic feed, so it can characterise
    aggregate speed on a link but not the per-vehicle position and speed error of a
    forward monocular camera, which is what these two parameters model.

    `configs/training/mappo_sumo.yaml` now cites Song, Lu, Zhang and Li,
    "End-to-end Learning for Inter-Vehicle Distance and Relative Velocity
    Estimation in ADAS with a Monocular Camera", ICRA 2020 (arXiv:2006.04082),
    on the TuSimple velocity benchmark. It measures this exact pair of quantities
    for this exact sensor and reports, for its full model, position MSE 10.23 m^2
    and range-averaged velocity MSE 0.86 m^2/s^2 — RMSE 3.20 m and 0.93 m/s.

    | parameter | was | now | source |
    |---|---|---|---|
    | `position_noise_std` | 1.5 | **3.2** | position MSE 10.23 m^2 |
    | `speed_noise_std` | 0.15 | **0.93** | velocity MSE 0.86 m^2/s^2 |

    The old values were carried over from `shared_ppo_deploysense` with no source
    and are optimistic against this citation by roughly 2x on position and 6x on
    speed. The cited figures belong to a state-of-the-art learned method, while
    this rig runs YOLOv8n with pinhole geometry and vehicle-width priors, so they
    are a floor on the noise rather than an estimate of it — which is the
    conservative direction for the claim: a policy that works under them would work
    under a better sensor.

    Song's velocity error is range-resolved — RMSE 0.39 near (< 20 m), 0.58 medium
    (20-45 m), 1.45 far (> 45 m) — and the range-averaged figure is taken because
    most vehicles observed at a 100 m range are beyond 45 m. Corroborating for
    speed alone, from a roadside camera validated against a GNSS and IMU reference:
    Bell et al., ISPRS Annals V-2-2020, average RMSE 0.625 m/s over four
    experiments.

    Verified against a noise-free run on the same seed: the injected error on
    `leader_gap` has a standard deviation of 3.00 m and on `leader_relative_speed`
    0.70 m/s over 376 paired observations. A test pins that the configured noise
    reaches the observations.

    **Only `mappo_sumo.yaml` is changed.** `mappo_deploysense.yaml` and
    `shared_ppo_deploysense.yaml` still carry 1.5 and 0.15, because their recorded
    results were produced under those values and changing them would silently
    supersede those results. Any future run on those configs should adopt the cited
    values first.

    **Still open: `latency_s`.** This one was measured — end-to-end latency over
    22,929 ticks on 2026-09-08 was p50 96.7 ms, mean 112.4 ms, p95 172.9 ms — and
    the obstacle is the simulation's time resolution, not a missing measurement.
    `SensingBuffer.frame_for_latency` selects whole recorded frames and the buffer
    records once per step, so at `dt: 1.0` any value in (0, 1.0] imposes a full
    second and overstates the measured delay tenfold. It is pinned at 0, which
    understates it by 97 ms. **Decision for the user:** running at `dt: 0.1` would
    make `latency_s: 0.1` represent the measured p50 almost exactly and would put
    the policy's decision rate at 10 Hz, closer to the deployment's 30 Hz tick than
    1 Hz is. The cost is that every capacity and arrival figure would need
    re-measuring a third time, and each episode becomes 6,000 steps rather than
    600.

89. **What the simulation study still needs.** Asked 2026-09-09: are the results
    for the paper in hand? Scope corrected by the user in the same exchange, and
    the correction matters: **the paper is a deployment story.** The simulation
    exists to replicate that MAPPO works under the sensing model measured on the
    deployment, not to run the nine-figure study in `plan_simulations.md` section
    8. Against that narrower target the gap is one training run, not a programme.

    **In hand.**

    - The effect is real and measurable in this simulator: AVs commanded to hold
      10 m/s raise arrivals 17.6% over 600 steps (155.2 +/- 10.5 against
      132.0 +/- 6.4, five seeds) and 20.9% over an hour, with zero collisions.
    - The action space can express a usable part of it: `slow` decodes to 13.94 m/s
      and gives 143.0 +/- 11.7, about half of what is available.
    - Every prerequisite for a valid run is now in place. The critic receives its
      115 inputs rather than 2; the fleet is the one the demand config declares;
      the episode is 600 steps, the horizon at which the effect exists; and
      `mappo_sumo.yaml` carries the deployment sensing block with
      `deployed_fidelity: true`.

    **Missing: the run itself.** `outputs/checkpoints/` is empty, and every SUMO
    training run to date is invalid on grounds found this session. One MAPPO run at
    `mappo_sumo` plus its evaluation against `no_av` is the deliverable.

    **Two things to settle before spending it.**

    1. **The reward does not rank the commanded-speed arms the way arrivals do.**
       Over 600 steps arrivals peak at 10 m/s and the reward at 15 m/s.
       `throughput_recent` is a 60 s window rather than the episode total, and
       `jam_fraction` at weight -2.0 grows as the network fills. A policy
       maximising this reward is not guaranteed to show the effect, so a null
       result would be uninterpretable.
    2. **The sensing model is only partly measured.** `range_m: 100.0` is
       deployment-derived and end-to-end latency was measured over 22,929 ticks,
       but `latency_s` is pinned at 0 because at a 1 s step any positive value
       means a full second, and `position_noise_std` and `speed_noise_std` are
       marked in the config as NOT measured, carried over from an earlier config
       and needing a published characterisation of monocular bounding-box ranging
       before the paper describes them as measured. A claim that MAPPO works
       "under the deployment's sensing model" rests on all three.

88. **Lower-severity items from the same audit.** Recorded, not fixed.

    - **The demand config's speed distribution is ignored on SUMO.** FIXED
      2026-09-09. `_write_routes` read only `speed_distribution.max_mps` and
      hardcoded the desired-speed spread as `normc(1,0.1,0.8,1.2)` on the lane
      limit, so a config declaring a 24.0 m/s mean produced a fleet desiring about
      30 m/s: every capacity measurement belonged to a fleet no config described.
      Every edge this builder writes carries the topology's single
      `speed_limit_mps`, so the configured distribution maps onto the factor
      exactly. Measured after the fix, per distinct vehicle at free flow: declaring
      24.0 gives 23.68, declaring 18.0 gives 17.88.

      **`spawn_min_gap_m` is deliberately not mapped, and the original wording of
      this item was wrong about it.** It claimed the minimum gap "falls from the
      configured 12 m to SUMO's vType default of 2.5 m, which raises jam density
      roughly fourfold". Those are two different quantities. On `highway_env`
      `spawn_min_gap_m` gates insertion — `_lane_has_spawn_gap` refuses a lane
      holding a vehicle within that distance — and changes no car-following
      behaviour. SUMO enforces insertion feasibility itself through the
      car-following model, which is the stronger criterion. Mapping the field to a
      vType `minGap` would change the standstill gap, and so jam density, from a
      config field that on the other simulator changes no physics at all.

      `branch_split` and `burst` remain unmapped: `branch_split` is `{main: 1.0}` in
      both SUMO demand configs and the builder already splits the rate evenly across
      the six entries, and `burst.enabled` is false in both.
    - **`mean_speed` and `active_vehicle_count` exclude vehicles inside junctions.**
      Measured: 700 of 48,590 vehicle-steps (1.44%) were on internal lanes, and the
      reported `mean_speed` was 6.157 m/s against SUMO's own 6.358 over all
      vehicles — 3.2% low, because junction-crossing vehicles are moving.
    - **`mappo_sumo.yaml` does not set `duration_steps`** — FIXED 2026-09-09, and it
      mattered more than "lower-severity" suggested. It defaulted to 120 while
      `rollout_steps` is 512. At 120 steps the throughput effect this configuration
      exists to learn is absent: AVs holding 10 m/s produce 19.7 arrivals against
      22.0 uncommanded, where at 600 steps the same comparison is 143.0 against
      111.8. The reward ranking inverts with it, placing 30 m/s first and 10 m/s
      last, so training at the default would have optimised against the effect under
      study. Now 600, with the measurement in the config and in
      `plans/plan_task_84_sumo_simulator.md`.
    - **Threshold sources differ between the simulators.** The SUMO env reads
      `queue_speed_mps` from `config["sensing"]` and `throughput_window_s` from the
      top level; `highway_env` reads both from `config["metrics"]["thresholds"]`, so
      a config setting them there is silently ignored on SUMO. FIXED 2026-09-09,
      and the item was half stale as the second audit round reported: by then
      `queue_speed_mps` already came from the shared
      `metric_thresholds_from_config`, but `throughput_window_s` was still read
      from the top level of the config, where nothing writes it. Every experiment
      config sets it under `metrics.thresholds`, so the window silently stayed at
      the 60 s default on SUMO whatever a config declared.

86. **RETRACTED by task 92 — the gain was a discretisation artefact.** The text
    below is kept as the record of how it was measured and corrected twice before
    being withdrawn; every arrival figure in it was taken at dt 1.0.

    ~~A 17.6% throughput gain exists, and the action space cannot reach it.~~
    Measured 2026-09-09 on SUMO, then re-measured twice after defects found in the
    measurement itself. This is the control effect the project has been trying to
    measure, and the reason no policy has found it.

    Every AV given the same commanded speed, 600 steps after a 300-step warm-up, at
    `sumo_saturating` (1050 veh/h, 20% penetration), five seeds:

    | command | arrivals | mean team reward |
    |---|---|---|
    | none | 132.0 +/- 6.4 | +0.823 |
    | 5 m/s | 123.4 +/- 19.9 | +0.520 |
    | 8 m/s | 148.8 +/- 9.0 | +0.915 |
    | **10 m/s** | **155.2 +/- 10.5** | +0.930 |
    | 12 m/s | 150.6 +/- 17.9 | +0.823 |
    | 15 m/s | 148.0 +/- 17.1 | **+1.034** |
    | 20 m/s | 138.8 +/- 10.6 | +0.987 |
    | 24 m/s | 137.0 +/- 4.7 | +0.956 |

    **Holding AVs at 10 m/s raises arrivals from 132.0 to 155.2, a 17.6% gain with a
    standard error of about 5.** Zero collisions throughout. The relationship is
    non-monotonic: 5 m/s is worse than doing nothing, so this is a genuine operating
    point rather than "slower is better".

    **The 52% figure this item first recorded is superseded twice over.** 167
    arrivals against 110 was one seed, taken with the fairness denominator defect
    present and on a fleet desiring 30 m/s where the demand config declares 24. The
    direction has survived every correction; the magnitude has fallen from 52% to
    17.6%.

    **The effect also does not exist at 120 steps**, which is what
    `mappo_sumo.yaml` was implicitly training at. No commanded arm beats an
    uncommanded fleet over 120 steps: the best is 28.8 +/- 2.9 against 27.8 +/- 3.1.
    The config now declares 600.

    That is the speed-harmonisation result the replication targets, and it is
    reproducible in one 600-step run.

    **The action space reaches about half of it — CORRECTED 2026-09-09.** This item
    first recorded that the action space could not express the effect at all, and
    that was true of the fleet then being simulated. `decode_speed_bin` returns the
    free-flow speed plus an offset (`slow` −10, `nominal` −3, `fast` 0), and
    free-flow is the vehicle's own desired speed. While the route writer ignored the
    demand config and gave every vehicle the 30 m/s lane limit, the bins decoded to
    20, 27 and 30 m/s, all inside the flat region where the effect is gone. With the
    fleet the config declares, they decode near 14, 20.5 and 23.5.

    Measured through the real action path, 600 steps, five seeds:

    | bin | decodes to | arrivals |
    |---|---|---|
    | uncommanded | — | 132.0 +/- 6.4 |
    | `slow` | 13.94 m/s | **143.0 +/- 11.7** |
    | `nominal` | 20.48 m/s | 133.6 +/- 4.3 |
    | `fast` | 23.53 m/s | 137.4 +/- 9.0 |

    `slow` recovers 11.0 of the 23.2 arrivals available between doing nothing
    (132.0) and the best commanded speed (155.2 at 10 m/s): about half the effect,
    at roughly two standard errors. The earlier measurement of arrivals identical at
    137 across all three bins and no action was taken on the undeclared fleet, where
    every bin landed above 20 m/s.

    So the actions were nearly equivalent rather than exactly equivalent, and the
    flat training curves and task 69's null still follow — but the remedy is smaller
    than this item first implied.

    **This supersedes task 85.** The entropy bonus really was 66% of the reward and
    `reward_scale: 1.0` really does fix that ratio, but changing it moved nothing --
    entropy went from −0.025 to +0.021 across 400 updates -- because the binding
    constraint is that the actions are equivalent. Task 85's arithmetic stands; its
    implied conclusion does not.

    **The decision this needs, and it is the user's. The case for it is now weaker
    than when it was first put.** The bins are part of the action contract shared
    with the deployed system through `sim_contract`, so rescaling them changes what
    the Jetson's actor emits as well. On the corrected measurements the question is
    whether to capture the remaining half of a 17.6% effect, not to make an
    unreachable effect reachable. Three routes:
    make the bins fractions of the free-flow speed rather than offsets from it
    (`slow` 0.33x gives 7.9 m/s on the declared fleet, which measured 148.8; a 0.42x
    fraction would give the 10 m/s that measured 155.2); decode them against the
    local traffic speed rather than the edge limit, so `slow` means slow *for these
    conditions*;
    or leave the contract and add absolute low-speed bins. The first is the smallest
    change and the third is the most explicit.

85. **The entropy bonus is 66% of the reward, so no policy has ever converged.**
    Found 2026-09-09 on SUMO, but it is not a SUMO defect: it applies to every
    training run this project has done.

    Measured at the metrics the 400-update SUMO run actually produced —
    `mean_speed` 7.33, `throughput_recent` 11.53, `jam_fraction` 0.128:

    | quantity | value |
    |---|---|
    | team reward | +1.2835 |
    | after `reward_scale` 0.05 | **+0.0642** — what the agent optimises |
    | entropy bonus at `entropy_coef` 0.01 and entropy 4.24 | **+0.0424** |
    | entropy bonus as a share of the reward | **66%** |
    | policy entropy against a 4.68 maximum | **91% of uniform** |

    PPO's default hyperparameters — learning rate 3e-4, value coefficient 0.5,
    entropy coefficient 0.01 — assume a reward of order 1. `reward_scale: 0.05`
    crushes this one to 0.064, so the entropy term is two thirds as large as the
    entire objective and the policy has almost no incentive to become
    deterministic. It held 91% of maximum entropy after 400 updates, and entropy
    moved −0.025 across them.

    **This is why every training run has looked flat.** The 100-update
    `highway_env` run showed entropy 3.426 → 2.830, which looked like convergence;
    at the SUMO operating point the reward is smaller still and the same
    coefficient dominates it. Neither run was learning much.

    **The fix is to stop scaling the reward down.** `reward_scale: 1.0` makes the
    reward 1.28 and the entropy bonus 3% of it, which is the ratio the default
    coefficients were chosen for. `reward_clip` at 10 already bounds the magnitude,
    so the scale-down was not protecting anything.

    Changing one thing at a time, so the effect is attributable.

81. **The reweighting worked: throughput improved 30% during training.** Done.

    100 updates, `mappo_deploysense` with `throughput_recent` at 0.10 and
    `jam_fraction` at −2.0, `inverted_tree` at the `saturating` demand, 3600-step
    episodes:

    | metric | first half | second half | change |
    |---|---|---|---|
    | `throughput_recent` | 10.779 | 14.062 | **+3.283 (+30%)** |
    | `entropy` | 3.426 | 2.830 | −0.596, converging |
    | `mean_speed` | 20.355 | 20.288 | −0.066, flat |
    | `jam_fraction` | 0.004 | 0.005 | +0.001 |
    | `collision_count` | 0.053 | 0.068 | +0.015 |

    Under the old weights the agent raised its own speed and left throughput
    untouched; under the new ones it does the reverse. That is the objective
    change working as intended, and it is the first time throughput has moved
    during training at all.

    **This is a training-metric result, not the replication.** The evaluation is
    still blocked: the reference is bistable at capacity and the arms do not
    complete reliably. See the retraction under task 80 and task 77.

82. **The `score` column no longer measures what is being optimised.** Open, small.

    `src/rl/trainers.py:134` computes `score = mean_speed − jam_fraction` and uses
    it for `best_score` and for choosing which checkpoint is "best". Since the
    reward became configurable and throughput-led, that expression is not the
    objective: a run whose throughput improves 30% while speed stays flat shows a
    `score` delta of −0.067, i.e. it looks slightly worse.

    So `actor.pt` is selected on a quantity the trainer is no longer maximising.
    `latest_actor.pt` is unaffected. The fix is to score with
    `build_team_reward(metrics, self.config.reward_weights)`, which is the thing
    actually being maximised.

83. **`config_resolved.yaml` is written only when training completes.** Open, small.

    Observed mid-run: the checkpoint directory held `actor.pt`, `critic.pt`,
    `latest_actor.pt`, `latest_critic.pt`, `trainer_state.pt` and
    `training_metrics.csv`, but no `config_resolved.yaml`; it appeared at
    completion. Task 69's evaluation harness refuses a checkpoint whose sensing
    block it cannot read, by design, so an interrupted or still-running training
    run cannot be evaluated even though its weights are on disk. Writing it
    alongside the first checkpoint would cost nothing.

80. **Controllers DO have a large measurable effect, once the simulator is fixed
    and measured at the right operating point.** PRELIMINARY — two seeds.

    At the `saturating` demand of 2000 veh/h, 600-step runs, after the
    collision-free bound, the node-geometry fix and the capacity measurement:

    | controller | AV penetration | jam | speed | throughput |
    |---|---|---|---|---|
    | `no_av` | 0.00 | 0.1083 | 10.70 | 11.5 |
    | `backpressure` | 0.10 | **0.0000** | 19.96 | **25.0** |
    | `backpressure` | 0.20 | 0.1000 | 17.69 | **31.5** |

    **A hand-written baseline at 10% penetration removes the congestion entirely
    and more than doubles throughput.** That is the effect task 8 concluded did not
    exist — it measured "+0.12 to +0.14 m/s against a 1.0 m/s threshold" and failed
    68 of 72 cells on `baselines_separate`.

    **Why task 8 could not see it.** Three reasons, each now measured:
    - Its demand levels bracket capacity without hitting it. `medium` (1800 veh/h)
      free-flows so there is nothing to relieve; `high` (2700) gridlocks so there is
      no throughput left to compare. The effect lives at 2000, which no config had.
    - Collisions dominated the dynamics. Crash queueing was most of the measured
      congestion — `merge` entirely, `inverted_tree` 61% — so a controller's effect
      was buried under crash-induced jams it could do nothing about.
    - Its reference controller was `no_av`, which cannot terminate early and so
      passed `episodes_complete` by construction, making the criterion uninformative.

    **What this changes.** Section C's premise, that the simulator cannot show a
    control effect, is refuted. The replication has something to replicate. It also
    means the MAPPO result must be compared against these baselines rather than
    only against `no_av`, because the bar is now high: a learned policy has to beat
    a hand-written one that already doubles throughput.

    **RETRACTED at five seeds. Do not quote the table above.** The five-seed check:

    | controller | pen | jam | speed | throughput | completed |
    |---|---|---|---|---|---|
    | `no_av` | 0.00 | 0.214 ± **0.334** | 11.41 ± **8.72** | 17.2 ± **16.2** | 5/5 |
    | `backpressure` | 0.10 | 0.000 | 19.24 | 21.0 | **1/5** |
    | `cooperative_smoothing` | 0.10 | — | — | — | **0/5** |
    | `backpressure` | 0.20 | 0.000 | 19.39 | 39.0 | **1/5** |
    | `cooperative_smoothing` | 0.20 | 0.000 | 20.08 | 23.0 | **1/5** |

    Two independent reasons the effect cannot be measured here:

    - **The reference is bistable at capacity.** `no_av` throughput is 17.2 with a
      standard deviation of 16.2, and jam 0.214 ± 0.334. Some seeds free-flow and
      some gridlock, which is what near-capacity traffic does. Two seeds happened
      to draw two congested ones, which is why the preliminary table looked clean.
    - **The treatment arms mostly crash.** 1 of 5 completed for `backpressure`,
      0 of 5 for `cooperative_smoothing` at 0.10. So each percentage above rests on
      one surviving run, and survivors are selected for not having crashed. This is
      the selection effect that already invalidated task 69's comparison.

    **And it contradicts something reported earlier in task 72.** `backpressure` at
    `high`/0.20 over 120 steps was 0 collisions and 6/6 completed. At the saturating
    demand over 600 steps it crashes 4 of 5. **The collision-free bound does not
    hold at longer durations**, which is consistent with the truncation arithmetic
    in task 72 and means task 77's residual is larger than two events.

    **So the claim that survives is narrow:** at this operating point a controller
    can drive jam to zero on the runs where it survives, and `no_av` cannot. Whether
    that is a throughput improvement is unmeasured, and cannot be measured until the
    runs complete reliably.


79. **The 120-step episode was hiding that `inverted_tree`/high is over-saturated.**
    Open, and it decides what task 74 can measure.

    Measured `no_av`, penetration 0.10, before the geometry fix:

    | topology | demand | steps | jam | speed | throughput | collisions |
    |---|---|---|---|---|---|---|
    | `merge` | high | 120 / 360 / 900 | 0.00 / 0.00 / 0.00 | 19.3 / 18.7 / 19.6 | 20 / 41 / 40 | 0 |
    | `inverted_tree` | high | 120 | 0.149 | 14.75 | 11.5 | 3 |
    | `inverted_tree` | high | 360 | 0.681 | **1.88** | **0.0** | 7 |
    | `inverted_tree` | high | 900 | **1.000** | **0.00** | **0.0** | 17 |

    **`inverted_tree` at high demand gridlocks.** Jam fraction reaches 1.0, mean
    speed 0.00 and throughput 0.0 — demand exceeds capacity, the network fills, and
    it never recovers. The 120-step episode measured only the transient before
    saturation, so every number this project has taken from that cell describes a
    filling network rather than a steady state.

    **After the geometry fix and the collision-free bound**, `low` and `medium` are
    clean at every duration tested — zero collisions and zero jam, speed 19.7 to
    22.7, throughput scaling with duration as a 60 s rolling window should:

    | demand | steps | jam | speed | throughput |
    |---|---|---|---|---|
    | low | 120 / 600 / 1800 | 0.000 | 22.4 / 22.3 / 22.7 | 6.5 / 13.5 / 14.5 |
    | medium | 120 / 600 | 0.000 | 20.4 / 19.7 | 9.0 / 26.5 |

    **The problem for task 74.** An hour-long run needs a demand that is congested
    AND moving. `low` and `medium` are free-flowing, so a controller has nothing to
    improve; `high` gridlocks, so there is no throughput to compare. Neither is
    usable, and the existing demand levels bracket the operating point without
    hitting it.

    **RESOLVED 2026-09-09 by measurement.** `configs/demand/saturating.yaml`, at
    2000 veh/h. Swept on 600-step runs, 2 seeds, after the collision-free bound and
    the geometry fix:

    | veh/h | jam | speed | throughput | |
    |---|---|---|---|---|
    | 1800 (`medium`) | 0.0000 | 19.66 | 26.5 | free flow, nothing to control |
    | **2000** | **0.1083** | **10.70** | **11.5** | **congested and moving** |
    | 2200 | 0.3867 | 8.66 | 15.5 | congested, heavier |
    | 2700 (`high`) | 1.0000 | 0.00 | 0.0 | gridlock |

    Capacity is about 1800 veh/h. `medium` sits at it, `high` is far past it. At
    2000 the network congests while still moving, and the throughput deficit
    against free flow — 11.5 against 26.5 — is the headroom a controller has to
    recover. Task 8's advice to raise demand was right in direction and would have
    overshot into gridlock.

78. **The simulator must present the sensing model the rig actually has.**
    **DECIDED BY THE USER 2026-09-09.** Supersedes the narrower task 71 ordering
    question, and blocks 73, 68 and 69.

    **The instruction:** the best possible representation in the simulator of the
    sensing model we deployed, using HERE to compute whatever HERE can compute.

    **The audit.** `src/analysis/observation_parity.py` already classifies all 39
    observation slots:

    | class | slots | meaning |
    |---|---|---|
    | `identical` | 8 | both sides compute the same thing the same way |
    | `approximated` | 18 | both compute it, by different mechanisms |
    | `substituted` | 7 | the sim has a real value; the rig substitutes a constant |
    | `structurally_absent` | 6 | the rig cannot produce it at all |

    **All six absent fields are rear-facing** — `follower_gap`,
    `follower_relative_speed`, `left_lane_rear_gap`, `right_lane_rear_gap`,
    `target_lane_rear_gap`, `target_lane_rear_required_decel` — because the live
    vehicle list is forward-camera derived and there is no rear sensor. The seven
    substituted are `ego_lane`, `time_since_last_lane_change`,
    `lane_changes_last_km`, `distance_to_downstream_bottleneck` and the three
    `nearby_av_lane_distribution` slots.

    **So the actor currently trains on 13 of 39 fields the deployed vehicle either
    cannot sense or replaces with a constant.** That is a third of its input, and it
    is the largest single reason a trained policy would not transfer.

    **What HERE can restore, and what it cannot.**
    - **Can:** road identity and geometry, hence route awareness for assigning
      camera detections to the ego's own road — the substance of task 71. And
      `distance_to_downstream_bottleneck`, from `jamFactor` on the segments ahead,
      which is currently substituted with a constant 0.4.
    - **Cannot:** any per-vehicle quantity. HERE's traffic flow API reports
      aggregate speed, free-flow speed and jam factor per road segment. It has no
      individual vehicles in it, so it cannot give a leader distance, a follower
      gap, or a lane distribution. Leader distance stays a camera measurement;
      HERE's contribution is knowing which road the camera is looking down.

    **Steps, in order:**
    1. A sensing-fidelity mode that presents the absent six and substituted seven
       exactly as the rig does, so the actor's input is what the vehicle can produce.
    2. HERE-derived `distance_to_downstream_bottleneck` from downstream `jamFactor`,
       replacing the substituted constant on both sides.
    3. Route-aware assignment of camera detections via map matching (task 71), which
       is what lets the live `leader_gap` mean what the sim's means.
    4. Only then retrain, because every step above changes the actor's input
       distribution.

    **Open:** whether the absent rear fields are dropped from the encoding entirely
    or held at the rig's constants. Dropping changes the vector width and every
    checkpoint; holding keeps the width and wastes six inputs. Recommendation: hold
    at the rig's constants, because the width is baked into `sim_contract` and the
    Jetson's actor runtime, and a width change is a far larger blast radius than six
    dead inputs.

77. **Two collisions survive the collision-free bound and the geometry fix.** Open.

    With the arcs joined, at `inverted_tree`/high/penetration 0.20 over 6 runs:
    `no_av` 4 collisions and 6/6 completed, `cooperative_smoothing` 2 and 5/6,
    `backpressure` **0 and 6/6**. So an AV arm is now fully collision-free and the
    human-only arm is not, which is the reverse of where this task started.

    **DIAGNOSED 2026-09-09, and it is density, not duration.** Per seed, at the
    `saturating` demand over 600 steps:

    | controller | seed | steps | completed | collisions | jam | throughput |
    |---|---|---|---|---|---|---|
    | `no_av` | 7 | 600 | yes | 0 | 0.217 | **0.0** |
    | `no_av` | 17 | 600 | yes | 0 | 0.000 | 23.0 |
    | `no_av` | 27 | 600 | yes | 0 | 0.000 | 34.0 |
    | `no_av` | 37 | 600 | yes | 0 | 0.064 | 29.0 |
    | `no_av` | 47 | 600 | yes | **9** | 0.790 | **0.0** |
    | `backpressure` | 7 | **225** | no | 2 | 0.000 | 29.0 |
    | `backpressure` | 17 | 600 | yes | 0 | 0.000 | 21.0 |
    | `backpressure` | 27 | **290** | no | 3 | 0.111 | 25.0 |
    | `backpressure` | 37 | **319** | no | 7 | 0.228 | 37.0 |
    | `backpressure` | 47 | **171** | no | 2 | 0.000 | 30.0 |

    **Three findings.**

    - **`no_av` is bimodal, not noisy.** Seeds 7 and 47 gridlock to throughput 0.0;
      seeds 17, 27 and 37 flow at 23 to 34. The 17.2 ± 16.2 reported under task 80
      was averaging two distinct regimes. Any reference at this operating point must
      report the modes or the count in each, never a mean.
    - **The bound fails for humans too**, not just AVs: seed 47 has 9 human-human
      collisions with no AVs present at all. So this is not an AV-control problem.
    - **AV runs die early, at steps 171 to 319**, and carry *higher* throughput
      (25 to 37) than the reference right up to the crash. They were working.

    **Why the bound fails.** It caps speed on the nearest vehicle ahead within a
    2.6 m lateral window. Density rises with run length until steady state -- 2000
    veh/h over 600 s spawns 333 vehicles against 67 over 120 s -- and at higher
    density the collisions measured earlier were side-by-side, at 1.9 to 3.7 m
    lateral, outside or at the edge of that window. A forward-looking window cannot
    see a conflict that is currently beside the vehicle and converging.

    **The fix is predictive rather than a wider window.** Widening to 4 m would make
    every adjacent-lane vehicle a longitudinal constraint and over-brake multi-lane
    sections. What is needed is closest-point-of-approach: for each pair, project
    both velocities, and if the predicted miss distance is under a vehicle width
    within the braking horizon, treat it as a conflict. That covers converging paths
    the current test misses and does not constrain parallel traffic that never
    meets.

## Ordering correction 2026-09-09: task 71 precedes training

Raised by the user, and it is right. Task 71 puts a route-aware leader gap on the
Jetson, and that lands in the **sensing model**, which `src/envs/topology_env.py`
uses to build every agent observation. It is therefore the same dependency that
already put task 9 ahead of task 68, and it changes three things:

- **`range_m`** stops being an optics estimate (100.0 m, from a 1.8 m vehicle
  spanning 14.4 px) and becomes measurable from the replay.
- **New error terms with no current analogue.** A map-matched gap carries
  map-matching error, polyline resolution, and a failure mode the sim does not
  model at all: matching the wrong road.
- **The parity class for `leader_gap`**, which is `identical` today only because
  every parity scene is single-arc.

So training before 71 produces a policy tuned to a precision the vehicle will not
have — the objection recorded in the task 67 round 3 audit, which was then only
half-acted on. **Revised order: 71, then 9, then 73, then 68, then 69.**

**The one thing this does not block.** A replication is a simulator claim, so the
throughput number does not depend on the Jetson. Training may proceed on the
current sensing block if the block is described as provisional; it may not if the
trained policy is the one to be deployed. That distinction is the user's.

75. **The arcs do not join. Vehicles are teleported sideways at every node.**
    Open, and it is the root cause of what task 72 could not reach.

    Measured distance between the end of each lane and the start of the lane
    `next_lane` sends a vehicle to:

    | transition | lateral jump |
    |---|---|
    | `a1/a2/a3_entry` → `('b1','c',0)` | **4.00 m** |
    | `('b1','c',0)` → `('c','exit',0)` | 2.00 m |
    | `('b1','c',1)` → `('c','exit',1)` | 2.00 m |
    | `('b2','c',1)` → `('c','exit',1)` | **10.00 m** |
    | `a4/a5/a6_entry` → `('b2','c',1)` | 0.04–0.09 m (these do join) |

    A vehicle crossing node `c` from `('b2','c',1)` is moved **ten metres**
    sideways, which is two lanes. No car-following bound can prevent a collision
    caused by a vehicle being placed into occupied space, which is why task 72's
    residual is exactly six side-by-side events at 1.9–3.7 m lateral separation,
    and why tightening MOBIL's `LANE_CHANGE_MAX_BRAKING_IMPOSED` from 2.0 to 0.05
    changed the count not at all.

    **It also explains two findings deferred from the task 67 audit.** The static
    successor map disagreed with the `lane_index` vehicles acquire on 9 of 9
    `('b2','c',1)` → `('c','exit',0)` transitions — because after a 10 m jump the
    geometrically nearest lane is not the steering target. And the one colliding
    pair the successor filter wrongly excluded collided 5–7 m past node `c`.

    So this is one defect with four symptoms, and fixing it is what makes
    collision-freedom achievable. The fix is that each lane's end must coincide
    with the start of its successor.

    Also noticed: `next_lane` on `('c','exit',k)` returns that same lane, giving a
    900 m self-jump. Harmless today because nothing walks past the exit, but it
    means the terminal arc has no proper successor.

76. **Most of this simulator's congestion was crash-induced queueing.** Open.

    Measured at high demand, penetration 0.10, `no_av`, with and without task 72's
    collision-free bound:

    | topology | collisions allowed | collision-free |
    |---|---|---|
    | `inverted_tree` | jam 0.2523 | **0.0990** |
    | `merge` | jam 0.0580 | **0.0000** |

    A crashed vehicle stops permanently and everything behind it backs up, so what
    the health check read as congestion was substantially a crash queue. On `merge`
    it was **all** of it, and its mean speed rose 14.13 → 20.11 m/s.

    **What this costs.** Task 8's `congestion_reachable` criterion was largely
    measuring crashes, so its topology ranking needs re-deriving. More importantly,
    a replication needs congestion for a controller to have anything to improve,
    and `inverted_tree` retains only 0.0990 — which is why it was the right
    topology to pick, but not obviously enough on its own.

    **Consequence for task 74:** reaching genuine congestion now has to come from
    demand and duration rather than from crashes. That is exactly task 8's own
    recommended diagnostic — raise demand, episode length and penetration together
    — arrived at from the opposite direction.

72. **Collisions must be impossible by construction, as in PTV Vissim.**
    **DECIDED BY THE USER 2026-09-09.** Blocks 73 and 74.

    **The instruction:** collisions are not something you generally see on a road,
    and a traffic simulator should make them structurally impossible rather than
    penalise them. PTV Vissim and SUMO both guarantee this in their car-following
    models; this simulator does not.

    **It is a prerequisite, not a preference, and the arithmetic says so.** At the
    current 120-step episode the AV arms complete 30 of 54, so per-step survival is
    0.995114 and the expected time to the first AV crash is **205 steps, about 3.4
    minutes**. Extrapolated:

    | episode | P(no AV crash) |
    |---|---|
    | 120 steps (current) | 0.556 |
    | 600 steps (10 min) | 0.053 |
    | 1200 steps (20 min) | 0.0028 |
    | **3600 steps (one hour)** | **2.2e-8** |

    An hour of simulated driving is unmeasurable until this lands. Every
    hour-long number would come from the vanishing fraction of runs that happened
    not to crash, which is the selection effect task 69 already had to work around
    at 7 of 18.

    **What it dissolves.** Task 67's entire premise — vehicles colliding because
    they cannot see each other — stops being a thing to mitigate. The `-5.0`
    `collision_count` reward weight becomes inert. `terminated` stops firing, so
    truncation ceases to be a criterion at all, and task 8's `episodes_complete`
    becomes trivially satisfied rather than structurally broken.

    **Three mechanisms are needed, in this order:**
    1. A safe-velocity cap for the in-lane leader — the rear-end case. A Krauss or
       Gipps bound, `v_safe = -b*tau + sqrt((b*tau)^2 + v_lead^2 + 2*b*gap)`, is
       collision-free by construction given both parties decelerate at `b`.
    2. The same cap against the merge-projected leader, reusing task 67's
       projection, which covered 27 of 51 measured collisions.
    3. Gap acceptance on lane changes — the lateral case, 6 of 51.

    **Design question, open:** enforce it once in the environment's substep loop,
    where `sub_dt` is known and it can cover humans and AVs together, or inside
    each vehicle model. The former gives one invariant and one place to test; the
    latter keeps each model self-contained. Recommendation: the substep loop,
    because a guarantee that lives in two models is a guarantee that can disagree
    with itself.

73. **Reweight the team reward so throughput is not 1.3% of the signal.**
    **DECIDED BY THE USER 2026-09-09.** Waits on 72.

    Measured over the 100-update run: `mean_speed` contributes 69.3% of the
    positive reward (weight 0.05 against a mean of 20.86) and `throughput_recent`
    contributes **1.3%** (weight 0.02 against a mean of 0.95) — a ratio of 55 to 1,
    because the weights do not normalise for scale. `throughput_recent` is a count
    of completions in a 60 s rolling window, so its magnitude also depends on the
    traffic state: about 0.95 during training, 5 to 15 at an evaluation's final
    step.

    Waits on 72 because the `-5.0` collision weight is currently a live term and
    becomes inert once collisions are impossible, which changes what the remaining
    weights have to balance against.

74. **Widen the evaluation to an hour of simulated driving.**
    **DECIDED BY THE USER 2026-09-09.** Waits on 72.

    The present evaluation is 18 conditions per arm at 120 steps, of which only 7
    had both arms complete. `throughput_recent` is an **integer** count, and it read
    5 to 15, so the resolution is one vehicle — 7% to 20% of the value. The paired
    result was +0.57 vehicles with four of seven conditions exactly equal, which is
    what that resolution produces rather than a measurement of the controller.

    An hour per run at `dt = 1.0` is 3600 steps, about 75 s of wall time per run at
    the measured 0.02 s per step. Seeds are the cheap axis now that topology and
    demand are fixed.

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

---

# Three-arm attribution, resolved 2026-09-09

The validator's round 3 finding 1 asked which of two changes the +17pp completion
gain belonged to, since `MergeAwareIDMVehicle` carries both a merge rule and a
low-speed stopping floor. Measured on the 81-run distinct grid, three arms:

| arm | completion | collisions (9 `no_av` conditions) |
|---|---|---|
| A plain IDM | 43/81 (53%) | 155 |
| B merge rule only | 57/81 (70%) | 126 |
| C merge rule + stopping floor | 57/81 (70%) | 138 |

**The merge rule accounts for the entire +17pp. The floor accounts for none of it,
and costs 12 collisions.** The floor was added to stop vehicles reversing through
zero speed, which it does — no vehicle reaches a negative speed with it — but the
extra stopping distance it imposes gives back 12 of the 29 collisions the merge rule
saves. From 5 m/s, unrestricted braking at 6 m/s² travels 2.08 m to rest while the
floor's geometric decay travels 3.26 m.

So the honest attribution is: the merge rule is the improvement, and the floor is a
correctness fix for the reversing defect that costs 12 collisions. It should stay,
because a vehicle driving backwards through traffic is a worse defect than 12
collisions, but it must not be credited with any of the completion gain.

# Task 77 resolved 2026-09-09: closest-point-of-approach helps, and does not solve it

`CollisionFreeMixin` now asks where a vehicle is GOING as well as where it is. For
every pair it projects both velocities, computes the time and distance of closest
approach, and imposes a stopping limit when the predicted miss is under 2.6 m
within a 4 s horizon. Parallel traffic in an adjacent lane is untouched, which is
what a wider lateral window could not have achieved.

Measured at the `saturating` demand over 600 steps, 5 seeds each:

| | before | with CPA |
|---|---|---|
| runs completing | 6/10 | **7/10** |
| total collisions | 23 | **14** |

**Collisions fall 39%. Completion improves by one run in ten. Four seeds get
worse** — `no_av` seeds 27 and 37 go from 0 collisions to 4 and 2, and
`backpressure` seed 17 stops completing. Braking for a predicted conflict creates
work for the vehicle behind, so the rule trades one collision mode for another.

**So collision-freedom is not achieved, and two mechanisms have now been tried.**
The safe-velocity bound took human-only collisions from 98 to 6 at short durations;
the geometry fix took the worst node discontinuity from 10 m to 0 m; CPA takes the
remaining total from 23 to 14. Each helped and none finished the job.

**The decision this now needs.** Constraining `highway_env`'s kinematics from
outside has produced diminishing returns across three attempts. PTV Vissim and SUMO
do not constrain a car-following model — their models cannot produce a collision in
the first place, because the safe speed is the model rather than a cap applied to
it. Continuing to add constraints is one option; replacing the vehicle model with
one that is collision-free by construction is another; running the replication on
SUMO, which is what the target papers use, is a third. That is a modelling decision
rather than a defect to fix, and it is the user's.

**Cost note.** CPA adds a second O(n²) pass per substep. A 600-step run at the
saturating demand went from about 12 s to about 40 s, so the 81-run grid is now
roughly 25 minutes rather than 8.

# PAUSED 2026-09-09

## The blocking unknown

**The collision-free bound is forward-looking only, and fails as density rises.**
`CollisionFreeMixin.perceived_safe_speed` in `src/vehicles/safe_following.py` caps
speed on the nearest vehicle ahead within a 2.6 m lateral window. At the
`saturating` demand the AV arms die at steps 171, 225, 290 and 319 of 600, and
`no_av` seed 47 has 9 human-human collisions with no AVs present, so it is not an
AV-control problem. The collisions are side-by-side at 1.9 to 3.7 m lateral —
outside or at the edge of that window — because a forward-looking test cannot see a
conflict that is currently beside the vehicle and converging.

**How to settle it.** Replace the lateral window with closest-point-of-approach:
for each pair, project both velocities, compute the time of closest approach and
the miss distance, and treat it as a conflict when the miss distance is under a
vehicle width inside the braking horizon. **Do not simply widen `SAFE_LATERAL_M`** —
4 m makes every adjacent-lane vehicle a longitudinal constraint and over-brakes
multi-lane sections, which `tests/test_collision_free.py::test_traffic_still_moves`
exists to catch.

**A second fact that blocks evaluation independently.** `no_av` at this demand is
**bimodal**, not noisy: seeds 7 and 47 gridlock to throughput 0.0 while 17, 27 and
37 flow at 23 to 34. A reference mean is meaningless there. Any comparison must
report the modes, or per-seed values, or use a demand below the bistable region.

## Exactly where I stopped

- `dsrc`, branch `main`, **clean tree, 0 commits ahead of `origin/main`**, HEAD
  `137b125`. Everything is pushed.
- Suites at HEAD: **simulator 337 passed**, **Jetson 2319 passed / 24 skipped**.
- **A trained checkpoint exists but will be lost.** The session scratchpad holds
  `train2/mappo_inverted_tree_full_seed7` with `actor.pt`, `critic.pt` and
  `config_resolved.yaml`, trained 100 updates under the throughput-led reward.
  The scratchpad is session-local. Retraining is about 40 minutes.
- Nothing is running. Two background jobs completed; a third measurement of the
  reference across more seeds was never started.

## What was learned that would otherwise be re-derived

- **Collisions were most of this simulator's congestion.** `merge` entirely,
  `inverted_tree` 61%. Task 8's `congestion_reachable` was largely measuring crashes.
- **The 120-step episode hid that `high` demand is over-saturated**: jam 0.149 at
  120 steps, 0.681 at 360, 1.000 at 900. Capacity is about 1800 veh/h.
- **`no_av` cannot fail `episodes_complete`** — it has no AVs and `terminated` tests
  AV crashes — so that criterion was uninformative for every task 8 cell.
- **The arcs did not join.** A vehicle crossing node `c` from `('b2','c',1)` was
  moved 10 m sideways. Fixed; worst jump is now 0.00 m on all six topologies.
- **The reward is a shared team reward**, not ego speed. Nine weighted network
  metrics. My earlier claim otherwise is corrected in the plan.
- **The 55-to-1 speed-to-throughput ratio was demand-specific.** At `medium`
  throughput averages 0.95; at `saturating` it averages 11.5 and the shipped weights
  already gave it 19%.
- Successor map after the geometry fix: `a1+a3 → ('b1','c',0)`, `a2 → ('b1','c',1)`,
  `a5 → ('b2','c',0)`, `a4+a6 → ('b2','c',1)`. Merge-conflict fixtures need a
  converging pair, so `a4`+`a6` or `a1`+`a3`.

## Task order to resume

1. **Task 77** — closest-point-of-approach in the bound. Then re-measure per seed;
   the target is every arm completing 600 steps.
2. **Re-measure the reference** with enough seeds to bound the bistability, and
   report modes rather than means.
3. **Task 74** — the hour-long evaluation, using `scripts/evaluate_replication.py`,
   which reads the sensing block from the checkpoint and refuses if it is absent.
4. **Task 82** — score the checkpoint on `build_team_reward`, not
   `mean_speed − jam_fraction`.
5. **Task 78 steps 2 and 3** — Jetson-side HERE work, which gates a deployable
   policy but not a simulator result.

## Open decisions not yet made

- **Whether the replication number waits for task 78 step 3.** A simulator claim
  does not depend on the Jetson; a deployable policy does. Recorded under task 78.
- **Whether the six rear fields are dropped from the encoding or held at the rig's
  constants.** Recommendation on record: hold, because the vector width is baked
  into `sim_contract` on both sides.
- **Whether to evaluate at `saturating` (bimodal) or find a demand below the
  bistable region.** Not yet investigated; 2200 was measured as congested and
  heavier, and may be more stable than 2000.
