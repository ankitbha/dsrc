# Tasks 40, 42, 43, 47 — USB colocation and the first live checks

## The short version

The two devices are now cabled together. `ZY227VV4XC` reports `device` on
`usb:1-2.2` of `jetson-orin` at USB 2.0 high speed (480 Mb/s), so section H can start.
This plan covers the four tasks the user asked for, in the order they can be executed.

**Task 40 comes before task 42, and the requested order was 42, 40, 43, 47.** Task 42 is
a bench run over USB; there is no USB data path until task 40 builds one. The order here
is 40, 42, 43, 47.

**Task 40 is a new `Acceptor`, not a new `ByteConnection`.** The in-car path is
`adb reverse`, which makes the phone's device-local `127.0.0.1:47811` resolve to a port
on the Jetson, tunnelled over USB by `adbd`. Both ends are ordinary TCP sockets, so
`TcpConnection` already satisfies the `ByteConnection` contract on this kernel. What USB
adds is one piece of external state — the reverse mapping — which vanishes on a replug
or an `adb kill-server`, and whose absence is indistinguishable from the phone's side
from a Jetson that never started listening. `UsbAcceptor` owns that mapping's lifetime,
binds loopback rather than `0.0.0.0`, and inherits both existing conformance suites.
The wire contract and the golden frames are untouched.

**Task 42 measures a delta against a measured baseline, not a margin.** On the tailnet
run `run_20260902_183446`, `report.json → latency_ms` gives `e2e_ms` p95 **215.63 ms**
(n=1,229), `link_ms` p95 **185.38 ms** (n=1,204), `jetson_ms` p95 **32.98 ms** (n=1,229).
Jetson compute is effectively fixed (p50 31.31, p95 32.98), so every millisecond USB can
win comes out of `link_ms`, and task 42 is testing whether USB closes a **15.63 ms p95
gap the network backend does not close**. Two different 200 ms claims are in play: the
gate in code is `GATE_JETSON_P95_MS` (`eval_run.py:73`) on `jetson_ms.p95`, which the
tailnet run clears by 167 ms; the 215.63 ms is `e2e_ms`, which no gate reads. The plan
reports both and names which is which.

**Task 43's load-bearing half is on the phone, not the Jetson.** `command_for` is the
only reader of the mode in the codebase and uses it in one boolean expression, so the
Jetson-side equality is close to structural. The behavioural difference is
`ConfigApplier.kt` returning early on a shadow command — and its counters do not cross
the wire, so the Jetson's record cannot state whether the phone honoured shadow.

**Task 47's brief is false as worded and the plan says so.** "The observation vector
produced live matches the simulator's sensing model field for field" does not hold:
at least ten of the 39 encoded slots differ by construction, six of them because the
vehicle has no rear sensor. Nothing in the tree compares the two sensing models — the
existing `test_sim_contract` feeds one hand-written dict to two *encoders*. Task 47
therefore delivers a slot-by-slot parity ledger classifying all 39 slots as identical,
approximated, substituted or structurally absent, with the mechanism named for each.

### Scope boundary

In: a USB acceptor and its lifecycle; a `--usb` selection path in `run_demo.py`;
installing the phone app on `ZY227VV4XC`; one measured bench run with both devices on a
desk; a shadow-versus-live replay check; a live-versus-sim observation parity check.

Out: tasks 44 (live-mode verification — shadow correctness is not evidence for it),
45 (thermal soak), 46 (failure injection), 48 (in-car install); anything about a moving
vehicle; closing any of the sim-versus-live divergences task 47 finds; and any change to
`specs/transport_protocol.md`'s frame layout or header schema, or to
`specs/transport_golden_frames.json`. A backend that changes a byte is not a backend.

### Open items flagged for the user — not decided here

1. **The app is not installed on the phone.** `adb shell pm path com.dsrc.phone` is
   empty; the handset carries 12 third-party packages, none of them ours. Task 32 ran on
   this same handset, so it has been removed since. All four tasks need it back, and
   installing it is a write this planning stage did not perform.
2. **Where the APK is built.** Gradle project is on this laptop; the phone's only `adb`
   host is now `jetson-orin`. Recommended in D8; the user may prefer otherwise.
3. **`specs/transport_protocol.md:21` and `task_list.md`'s section D preamble name
   `adb forward`.** That subcommand inverts the direction the same paragraph mandates,
   and the phone has no production `ServerSocket`. `LinkConfig.kt` and `tailnet.py:116`
   say `adb reverse` and the code agrees. Correcting the spec changes no encoding, but it
   is a frozen document and the edit is not taken here.
4. **`eval_run.py:741` pools `link_ms` across estimator sources**, which
   `sensors/time_sync.py:64-68` says must not be done. 6 of 1,204 samples on the tailnet
   baseline; over USB the one-way bias is comparable to the quantity being measured.
5. **Task 47 is reinterpreted**, so what it delivers changes. Its wording in
   `task_list.md` may want changing to match.
6. **The phone's applier counters do not cross the wire.** Task 43 reads them from
   `logcat`; a `PhoneTelemetry` field would be better and is a wire change, so it is not
   taken here.
7. **Two recorded USB latency figures disagree by 3.5x** — 56.9 ms handshake round trip
   (`plan_task26:73`) and a 7.9 ms link segment (`phone_source.py:113`) — and neither is
   a percentile, so neither is carried forward as an expectation.

---

## Why 40 must precede 42

Task 42's brief is "bench loopback over USB with both devices on a desk". There is no
USB data path in the tree today: `PhoneLink` constructs a `TcpAcceptor` bound to
`0.0.0.0` (`sensors/phone_link.py:144`) and `run_demo.py --phone` passes it a host and
port (`run_demo.py:367`). Running task 42 before task 40 would mean either running it
over the tailnet — which is task 32, already done — or setting up an `adb reverse` by
hand outside any harness, which measures a configuration nothing records and which task
32 round 1 already showed produces a run record that cannot say which path it used.

Tasks 43 and 47 do not depend on 40 or 42 for correctness — both could run over the
tailnet — but both need a live session and the phone is now cabled to the Jetson, so
they run over the USB path once it exists, and each gains a second machine's
independence for free.

---

## What is already true, measured

Every number here was read out of the repository or off the devices during planning, not
restated from a plan's prose. The source is named for each.

### The devices

| Fact | Value | Read from |
|---|---|---|
| Handset | `ZY227VV4XC`, `moto g power`, `sofia_retail` | `ssh jetson adb devices -l` |
| USB port | `usb:1-2.2`, `transport_id:1` | same |
| USB link | 480 Mb/s, USB 2.00 | `/sys/bus/usb/devices/1-2.2/{speed,version}` |
| adb on Jetson | 1.0.41 (Debian 28.0.2), server on `127.0.0.1:5037` | `adb version`, `ss -lntp` |
| Reverses in place | none | `adb reverse --list` (empty) |
| Forwards in place | none | `adb forward --list` (empty) |
| `com.dsrc.phone` | **not installed** | `adb shell pm path com.dsrc.phone` (empty) |
| Jetson tailnet address | `100.106.45.8` | `ssh jetson tailscale ip -4` |

The tailnet address is the third value this project has recorded. `tcp.py`'s `dial`
docstring says it "has changed once already"; `plan_task32` used `100.90.108.88`; it is
`100.106.45.8` today. `127.0.0.1` does not move, which is a small independent argument
for the USB path being the one that survives a drive.

### The transport seam

`ByteConnection` (`transport/connection.py:47`) is four members — `peer`, `send_all`,
`recv_exact`, `close` — and deliberately no liveness query; task 13 removed the fifth
rather than adding it. Two requirements the seam cannot work around are stated there and
were measured on the Jetson: `recv_exact` must raise rather than return short or empty,
and `close()` must release a read already blocked in another thread, which on Linux
needs `shutdown()` before `close()` (measured: close alone never released at 6 s;
shutdown-then-close released in 0.001 s).

`Acceptor` (`transport/endpoint.py:45`) is two members, `accept(timeout)` and `close()`,
and its docstring already says "Loopback, TCP and USB each supply one".

Both seams have conformance suites parametrised over backends —
`tests/test_transport_backend_contract.py` (13 checks) and
`tests/test_transport_acceptor_contract.py` (9 checks) — and the acceptor suite's
docstring says "Task 40's USB acceptor is the next one written; it inherits this".

`PhoneLink.__init__` already takes an `acceptor` keyword and falls back to `TcpAcceptor`
only when it is None (`sensors/phone_link.py:129-144`). The injection point for task 40
exists and needs no change.

### The direction is `adb reverse`, and the code proves it

`LinkConfig.kt`'s class docstring: "The default host is loopback because the in-car path
is `adb reverse`, which puts the Jetson's listener on a device-local port."
`tailnet.py:116`: "`127.0.0.1` means the phone dialled loopback and the data crossed USB
via `adb reverse`".

The decisive evidence is not the prose. Searching `phone/` for `ServerSocket` finds it
only under `src/test` and `src/androidTest`; the sole production socket construction is
`SessionHolder.kt:352`, a dial. The phone cannot accept a connection, so `adb forward` —
which requires the phone to listen — is not implementable without new Kotlin. `adb
reverse` is the only direction the existing code supports, and it preserves the
phone-dials asymmetry that `specs/transport_protocol.md` mandates for the network path.

Consequence for configuration: under `adb reverse tcp:47811 tcp:47811` the phone's
**default** `LinkConfig` (host `127.0.0.1`, port 47811) is already correct. `link.json`
is the tailnet path's requirement, not the USB path's. A `link.json` left behind from a
tailnet run would silently send a "USB" run over the tailnet, so removing it is a step,
not an assumption.

### How latency is measured today

In `pipeline.py:279-288`:

- `t_arrival = frame.timebase.t_arrival_mono` — the Jetson's `time.monotonic()` at the
  instant the session's reader took the frame off the socket. Exact, local.
- `e2e_ms = (t4 - frame.t_mono) * 1000` — `t4` is the Jetson's `time.monotonic()` after
  the advisory is decoded and the headway target set. `frame.t_mono` is the phone's
  capture instant **converted onto the Jetson's monotonic clock** by the timebase
  estimate carried on that frame.
- `jetson_ms = (t4 - t_arrival) * 1000` — both terms on the Jetson's clock. Exact.
- `link_ms = frame.timebase.link_s * 1000` — arrival minus converted capture. `None`
  for a local camera and `None` whenever the stamp was proxied rather than converted.

So the only cross-clock term in any of the three is the capture instant, and it arrives
attached to its own `bound_s`, `estimate_id` and `source` (`sensors/time_sync.py:44-73`).
Nothing is subtracted across two machines by hand; the protocol forbids it.

`TimebaseStamp.source` is `round_trip`, `one_way` or `proxy`, and the field's own comment
says why it exists: "a round-trip-converted number must not be pooled with a
one-way-converted one — the two have different error semantics, so this is what lets a
consumer keep their samples in separate series."

The full capture-to-driver path needs the phone's log. `eval_run.py --phone-log` joins
`return` (advisory wire departure on the Jetson clock to phone receipt on the phone
clock, converted offline against the nearest persisted estimate, `eval_run.py:258`) and
`render` (phone receipt to the first `current()` that returned it, phone clock on both
ends, no conversion, `eval_run.py:293`). Those two stages are the only ones not visible
from the Jetson alone.

### The tailnet baseline, re-read

From `jetson:/home/edge/dsrc_logs/run_20260902_183446/report.json`, key `latency_ms`:

| segment | n | p50 | p95 | max | min | negative |
|---|---|---|---|---|---|---|
| `e2e_ms` | 1229 | 96.15 | **215.63** | 649.85 | 30.70 | 0 |
| `link_ms` | 1204 | 65.68 | 185.38 | 617.06 | 27.22 | 0 |
| `jetson_ms` | 1229 | 31.31 | 32.98 | 67.13 | 22.80 | 0 |

`summary.json`'s `stats` block reports the same run as `e2e_ms` p95 184.27 over
**n=300**. The two disagree because `pipeline_stats` is window-capped at 300 ticks and
`report.json` covers the whole run; task 16 recorded this and said to quote
`latency.*.n`. Task 42 quotes `report.json`.

**The 25 ticks with no `link_ms` are accounted for exactly.** Counting per-tick
`timebase.source` across `metadata.jsonl` for that run:

```text
round_trip, link_ms present    1198
one_way,    link_ms present       6
proxy,      link_ms absent       25
                               ----
                               1229
```

The 25 are the coordinator's `1229 − 1204`. Their reasons are not link ill-health: the
adapter's `proxy_reasons` for the whole run are `no samples` (7) and "only 1/2/3/4
samples in the offset window" (11 each) — every one of them the offset estimator filling
its window at the start of a session. On this run the proxied population is session
warm-up, and it correlates with session start rather than with a degraded link.

The direction of the bias matters and is the opposite of the intuitive one. Under the
proxy, `t_capture_mono` is set equal to `t_arrival_mono`, so `e2e_ms == jetson_ms` — about
31 ms. Including proxied ticks therefore pulls the pooled `e2e_ms` distribution
**down**, and the reported `min` of 30.70 ms against `jetson_ms`'s 22.80 ms is consistent
with the low tail being made of them. A p95 over 1,229 is not a worse statistic than one
over 1,204 — it is a different one, mixing 25 ticks whose e2e was never measured from
capture.

### Two recorded USB latency figures, and why neither is a prior

`plans/plan_task26_phone_backends.md:73` records "a 56.9 ms handshake round trip over
`adb reverse`" and infers "~28 ms" one-way. `sensors/phone_source.py:113` records the
link segment as "7.9 ms measured". These differ by a factor of 3.5, one is a handshake
round trip that includes app-side work and the other is a link segment, and neither is a
percentile over a population. Task 42 does not carry either forward as an expectation.

The same passage carries a warning that is directly load-bearing here: a one-way offset
estimate has an error equal to the one-way delay, "a bias rather than noise, so averaging
does not remove it". Over USB the one-way delay **is** the quantity being measured, so a
`link_ms` sample converted from a one-way estimate is biased by nearly its own magnitude.
The round-trip estimator that task 26 called "the right answer later" now exists
(`phone_link.py` builds a `TimebaseEstimator` from the previous-exchange trio and
`PhoneClockAdapter` tries it first), and on the tailnet baseline it carried 1,198 of
1,204 samples. Task 42's acceptance requires that it carry the USB run too.

---

## Decisions

Every row was taken by recommendation under `plan_dsrc_rec`. **None of these is user
sign-off.** Where a row was close, the rejected option is named so it can be revisited
cheaply.

| # | Decision | Taken because | Rejected |
|---|---|---|---|
| D1 | Task 40 ships a `UsbAcceptor` implementing `Acceptor`, wrapping a loopback-bound `TcpAcceptor`, plus the `adb reverse` lifecycle. No new `ByteConnection`. | `adb reverse` delivers ordinary TCP at both ends; `TcpConnection` is already proved against this exact kernel. A second byte-stream implementation would re-litigate the three socket requirements task 13 measured. | A raw `ByteConnection` over `adb exec-out` stdio or the adb server protocol: more code, no new capability, loses proven socket handling. |
| D2 | `UsbAcceptor` binds `127.0.0.1`, not `0.0.0.0`. | The adb server runs on the Jetson and dials the Jetson's own loopback, so loopback is sufficient. Binding `0.0.0.0` while Tailscale is up leaves both paths live at once, making "which path did this run use" a question rather than an impossibility. | Keeping `0.0.0.0` and relying on `path_for_address` to disambiguate after the fact — detection where prevention is available. |
| D3 | `UsbAcceptor` owns the reverse mapping: establishes it on construction, verifies it with `adb reverse --list`, and removes it on `close()`. | The mapping is external state that disappears on replug or `adb kill-server`, and its absence presents to the phone as ECONNREFUSED — identical to a Jetson that never listened. An object that owns it can say which happened. | Requiring the operator to run `adb reverse` by hand: the configuration then appears in no run record, which is the defect task 32 round 1 fixed. |
| D4 | The port stays 47811 on both sides of the reverse (`tcp:47811 tcp:47811`). | It is `LinkConfig.DEFAULT_PORT` and `tcp.DEFAULT_PORT`, so the phone needs no `link.json` at all. A different device-side port would require pushing one, reintroducing the file whose absence is the point. | Distinct ports to dodge the `ImuWireTest` collision (R4) — solved instead by not running that test concurrently, which is already enforced by `scripts/with_device.py`. |
| D5 | `UsbAcceptor` re-establishes a mapping it finds missing, counting each re-establishment, rather than failing the run. | A replug mid-drive is the expected in-car event and the phone's own reconnect never gives up; a Jetson side that gave up would make the phone's persistence useless. | Fail fast: correct on a bench, wrong in a car. |
| D6 | Selection is `run_demo.py --usb`, mutually exclusive with `--phone-host`. | `--phone` already selects the handset for both sensors; `--usb` selects the path. Making them exclusive means a command line cannot claim USB and name a tailnet address. | A `--phone-host 127.0.0.1` convention: indistinguishable from a genuine loopback test, and does not establish the reverse. |
| D7 | The run record gains a `usb` block (serial, `transport_id`, reverse spec, re-establishment count, `adb` version) beside the existing `network` block. | Task 32's lesson was that a record which cannot name its own path is not evidence. `path_for_address` already reports `127.0.0.1` as USB; that says which path, not which cable or how often it dropped. | Extending the `network` block: `network` answers a different question and mixing them is how the two-`LinkConfig` defect happened. |
| D8 | The app is built on this laptop with Gradle and installed from the Jetson: `./phone/gradlew :app:assembleDebug`, `scp` the APK to `jetson-orin`, `adb install -r` there. | The Jetson has no Android SDK and building one there is a day of work for no benefit. The Jetson is the only `adb` host the phone is cabled to. | Building on the Jetson (no SDK); re-cabling the phone to the laptop to install (undoes the colocation this section exists for). |
| D9 | Task 42 reports `e2e_ms` p95 over **all** ticks and over **converted-only** ticks, both, with the proxied count and its reasons beside them. | The two populations are different statistics and the proxy biases the pooled figure downward, not upward. Reporting one would be choosing which. | Excluding proxied ticks silently — a p95 that improves because 25 samples left. |
| D10 | Task 42 partitions `link_ms` by `timebase.source` and never pools `round_trip` with `one_way`. | `time_sync.py:64-68` says exactly this, and `eval_run.py:741` does not do it. Over USB the one-way bias is comparable to the quantity measured. | Pooling, as today: on this baseline it would contaminate 6 of 1,204 samples, and there is no guarantee a short USB session is as round-trip-dominated. |
| D11 | Task 42's run is 180 s of wall time at the planned sensor rates, matching task 32's run, and is repeated three times. | A single run cannot separate a path effect from the unattributed cadence anomaly that has now appeared twice (task 14, task 15) and reproduced neither time. Matching task 32's duration keeps the comparison like-for-like. | One long run: a single sample of a phenomenon known to be bimodal. |
| D12 | Task 43 compares by **replaying the logged input through the live gating function**, not by running two paths side by side. | Two live paths can differ for reasons that are not the gating decision — a different tick, a different sensor sample. A replay pins the input exactly, so any difference is the function. | Side-by-side live: cannot control the input, so an inequality is uninterpretable. |
| D13 | Task 47's parity check is over the **39 encoded slots** named by `sim_contract.encoded_slot_names()`, not the 33 flat observation fields. | `ARCHITECTURE.md`'s section 5 already states the map covers all 39, with `cooperation.*` and `nearby_av_lane_distribution.<lane>` dotted, "so the missingness metric is a statement about the whole encoded vector". | The 33 flat fields: leaves 6 slots unchecked, and they are the composite ones most likely to disagree. |
| D14 | Tasks 43 and 47 both run against the **same** live session artifact as task 42, not their own runs. | One run's `metadata.jsonl` carries the shadow decision log, the observation vector and the latency segments. Three separate runs would let 43 and 47 pass on inputs 42 never saw. | Separate runs per task: more compute, weaker evidence. |
| D15 | Nothing in `specs/transport_protocol.md` or `specs/transport_golden_frames.json` is edited by task 40. | A backend that changes the wire is not a backend. The golden vectors are the cross-language contract and are already proved byte-identical on two architectures. | (none — this is a constraint, recorded as a decision so it is checkable) |
| D16 | Task 43 builds **no mid-drive flip surface**. | `run_demo.py:506` fixes the mode at construction and never calls `flip_to`, so one drive is entirely one mode. A replay of the logged inputs answers "the same input" without one, and an operator flip surface is task 44's territory. | Adding `--flip-at`: scope creep into the task whose whole content is flipping the flag. |
| D17 | Task 43 compares **decoded wire objects**, not Python objects. | `shadowed.here == live.here` is two references to one `HereQuery` and passes even if `command_for` dropped it — a defect `test_shadow_mode.py:44-74` already had to guard with an identity check. Encoding through the real codec removes the class. | Comparing dataclasses: passes on aliasing. |
| D18 | Task 43 reads the phone's `applied`/`shadowed` counters from `logcat` at teardown. | They are the only direct witness that a shadow command was not acted on, and they are not `PhoneTelemetry` fields. Adding one is a wire change, which D15 forbids in this plan. | Adding a telemetry field (a wire change); inferring from `reference.achieved` alone (a windowed average, and `eval_run._comparable` already refuses that comparison on a shadow drive). |
| D19 | Task 47 delivers a **parity ledger**, not a match assertion. | The match assertion is false: at least ten of 39 slots differ by construction, six because there is no rear sensor. A check written to pass would pass on the 29 that agree and say nothing about the ten. | Asserting equality on a subset: chooses which divergences to hide. |
| D20 | Task 47's harness expresses one scene and **instantiates it separately** on each side, then encodes both through the same encoder. | A difference is then attributable to the sensing model rather than the encoder, and the harness cannot agree trivially — the failure `test_sim_contract` has today, where one dict is fed to two encoders. | Feeding one observation dict to both: proves nothing about sensing. |
| D21 | The ledger's per-slot classification is pinned by a test. | A slot silently moving from `identical` to `substituted` is exactly the change a later reader must not miss, and prose in a plan does not fail CI. | Leaving the ledger as an artifact: it rots the first time either producer changes. |

---

## Task 40 — USB transport backend

### What it is

`transport/usb.py`, one new class `UsbAcceptor`, satisfying `Acceptor`. It composes a
`TcpAcceptor` bound to `127.0.0.1:47811` with the `adb reverse` mapping that makes the
phone's device-local `127.0.0.1:47811` resolve to it. Connections it hands back are
`TcpConnection` instances, unchanged, so the `ByteConnection` contract is satisfied by
code already proved on this kernel.

Nothing above the seam changes: not `session.py`, not `frames.py`, not `messages.py`,
not `handshake.py`, not `timebase.py`, and not either spec file.

### Steps

| # | Step | Produces |
|---|---|---|
| 40.1 | `AdbReverse` helper: `establish()`, `verify()`, `remove()`, each shelling `adb -s <serial> reverse ...` with a bounded timeout, plus `list()` parsing `adb reverse --list`. Serial required, never inferred, because `adb` with two devices attached is ambiguous and silently picks none. | A unit-testable object with the subprocess call injectable. |
| 40.2 | `UsbAcceptor(serial, port=47811, device_port=None)`: constructs the `TcpAcceptor` on `127.0.0.1` first, then establishes the reverse to the port the acceptor actually bound (`acceptor.port`, resolved after bind, so `port=0` works in tests). | The acceptor, plus `address` and a `usb_record()`. |
| 40.3 | `accept(timeout)` delegates, and before returning `None` on a timeout re-verifies the mapping is still listed; a mapping found missing is re-established and counted in `reverses_reestablished`. | The one behaviour a TCP acceptor cannot have. |
| 40.4 | `close()` removes the reverse, then closes the `TcpAcceptor`. Idempotent, and tolerant of a device already unplugged — `adb reverse --remove` against an absent device fails, and failing to tear down must not raise out of a teardown path. | Clean teardown. |
| 40.5 | Register `usb` in `BACKENDS` in `tests/test_transport_backend_contract.py` and in `ACCEPTORS` in `tests/test_transport_acceptor_contract.py`, both gated on a device being attached, skipping with a stated reason when not. | Both conformance suites run against the third backend: 13 + 9 checks. |
| 40.6 | A fake-`adb` unit suite: mapping absent at construction, mapping vanishing mid-run, `adb` returning non-zero, `adb` not on `PATH`, two devices attached with no serial. | The failure paths, none of which need a phone. |
| 40.7 | `run_demo.py --usb [--usb-serial S]`, mutually exclusive with `--phone-host`; constructs `PhoneLink(acceptor=UsbAcceptor(...))`. Refuses when no device is attached rather than falling back to TCP. | The selection path. |
| 40.8 | `summary["usb"]` from `usb_record()`: serial, `transport_id`, reverse spec, `adb` version, `reverses_reestablished`, and the acceptor's bound address. | The run record can name its own cable. |
| 40.9 | Run the whole transport suite on the Mac and on `jetson-orin`, and diff the counts. | Two architectures, one number, as tasks 12–15 each did. |

### Acceptance

- `UsbAcceptor` passes all 13 `ByteConnection` checks and all 9 `Acceptor` checks with
  the phone attached, and the suite skips with a named reason when it is not.
- `isinstance(connection, ByteConnection)` holds for an accepted connection, and the
  contract suite's no-undeclared-surface check passes — task 13's rule that a backend
  implements four members, not five.
- `specs/transport_golden_frames.json` regenerates byte-identically
  (`scripts/generate_transport_golden_frames.py`), and `git diff` on both spec files is
  empty.
- The full transport suite passes on the Mac and on the Jetson with equal counts.
- With the phone attached and `run_demo.py --usb` running, `adb reverse --list` shows
  exactly one mapping and shows none after the process exits.
- A deliberate `adb kill-server` mid-run is followed by `reverses_reestablished >= 1` in
  the run record and an unbroken session, or by a recorded session end and a reconnect —
  not by a silent stall.

---

## Task 42 — Bench loopback over USB, latency against the 200 ms figure

### What is measured, between which instants, on which clock

This is the part a validator has to be able to check, so it is stated as three separate
quantities rather than one.

**`jetson_ms`** — from `t_arrival`, the Jetson's `time.monotonic()` when the session
reader took the frame off the socket, to `t4`, the Jetson's `time.monotonic()` after the
advisory is decoded. **Both terms on the Jetson's monotonic clock.** No conversion, no
cross-device subtraction. This is the quantity `GATE_JETSON_P95_MS = 200.0` gates.

**`link_ms`** — from the phone's capture instant, **converted onto the Jetson's
monotonic clock** by the timebase estimate carried on that frame, to `t_arrival` on the
Jetson's clock. One cross-clock term, carrying its own `bound_s` and `estimate_id`, and
its `source` is `round_trip` or `one_way`. Never reported as one pooled number:
`round_trip`- and `one_way`-converted samples are separate series, per
`sensors/time_sync.py:64-68`.

**`e2e_ms`** — the phone's converted capture instant to `t4`, i.e. `link_ms + jetson_ms`
by construction. Reported twice, over two stated populations (D9).

**The capture-to-driver path**, which none of the above covers, comes from
`eval_run.py --phone-log`: `return` is the advisory's wire departure on the Jetson clock
to its receipt on the phone clock, converted offline against the nearest persisted
estimate and carrying that estimate's bound; `render` is phone receipt to the first
`AdvisoryHolder.current()` that returned it, phone clock at both ends, no conversion.
Ten stages, joined on the exact nanosecond `t_capture_mono_ns` both sides carry.

### The baseline this is a delta against

`jetson:/home/edge/dsrc_logs/run_20260902_183446/report.json`, tailnet, 1,229 ticks:
`e2e_ms` p95 **215.63 ms**, `link_ms` p95 **185.38 ms** (n=1,204), `jetson_ms` p95
**32.98 ms**. The tailnet run **misses 200 ms on `e2e_ms` at p95 by 15.63 ms** and
clears the gate that exists (`jetson_ms` p95 32.98 against 200) by 167 ms.

### Steps

| # | Step | Produces |
|---|---|---|
| 42.1 | Build the debug APK on this laptop; `scp` to `jetson-orin`; `adb install -r` there; confirm `pm path com.dsrc.phone` is non-empty and record the version code. | The app on the handset, which it is not today. |
| 42.2 | Remove any `link.json` from the app's external files directory and confirm removal, so the phone uses the loopback default and cannot silently take the tailnet. | The USB path is the only path. |
| 42.3 | Rsync the tree to `jetson:~/dsrc-task42` per the established per-task convention; run the jetson-side suite there. | The Jetson runtime at this commit. |
| 42.4 | Three runs of `run_demo.py --phone --usb --usb-serial ZY227VV4XC`, 180 s each at the planned sensor rates, both devices on a desk (D11). Each writes its own `run_dir`. | Three `metadata.jsonl` + `summary.json` + `report.json`. |
| 42.5 | Pull the phone's session log per run with `scripts/run_device_session.py`'s existing `run-as … cat files/sessions/…` path, and run `eval_run.py --phone-log` on each. | The ten-stage table including `return` and `render`. |
| 42.6 | Partition `link_ms` by per-tick `timebase.source` in the report, and report the proxied ticks separately with their `proxy_reasons` (D10). | `latency_ms.link_ms` becomes `{round_trip: …, one_way: …}` plus a proxied count. |
| 42.7 | Report `e2e_ms` p95 over all ticks and over converted-only ticks, both (D9). | Two stated statistics rather than one ambiguous one. |
| 42.8 | Record `tailscale status --json` at start and end of each run, as task 32 did, and assert `path_for_address` reports `over_tailnet: false` with `session_peer` starting `127.0.0.1`. | Positive evidence the bytes crossed USB, from the accepted socket rather than from reachability. |

### Acceptance

Concrete enough to check, and deliberately not phrased as "beats 200 ms":

1. **Path.** In all three runs, `network.session_peer` begins `127.0.0.1:` and
   `over_tailnet` is `false`. A run that cannot say this is not a USB run and does not
   count, whatever its latency.
2. **The gate that exists.** `report.json → latency_ms.jetson_ms.p95 < 200.0` in all
   three runs, and `jetson_ms_source == "measured"`. Expected to pass with wide margin;
   it is here because it is the only threshold in the code.
3. **The delta being tested.** `report.json → latency_ms.link_ms`, partitioned by
   source, `round_trip` series only, p50 and p95 reported against the tailnet baseline's
   pooled 65.68 / 185.38 ms. The claim task 42 is testing is that the p95 of the USB
   `round_trip` series is **below 185.38 ms**; the size of the reduction is the result,
   not a threshold.
4. **The 200 ms figure on end-to-end.** `latency_ms.e2e_ms.p95` reported over both
   populations, against the tailnet baseline's 215.63 ms over 1,229 ticks, and against
   200 ms. Whether it clears 200 ms is the measurement, not the pass condition — a run
   that misses it and says by how much has done task 42's job.
5. **Conversion health.** Per run, converted fraction, `converted_by_source`, proxied
   count and `proxy_reasons`. A run in which `one_way` carries more than 5% of converted
   ticks is reported as such and its `link_ms` is not compared to the baseline, because
   the one-way bias is comparable in size to the quantity being measured.
6. **The account closes.** Per channel: `received`, `delivered`, `dropped_inbound`,
   `abandoned_inbound`, `heartbeats_received` — balancing on every channel including
   `control`, which is the identity task 32's first run got wrong.
7. **Reproducibility across the three runs.** The three `link_ms` p50 values are
   reported together. A spread larger than the within-run interquartile range is
   reported as unattributed rather than averaged away — the cadence anomaly has appeared
   twice and reproduced neither time.
8. **The ten-stage table** is produced for at least one run, so `return` and `render`
   are on the record and the capture-to-driver path has a number for the first time.

---

## Task 43 — Shadow-mode correctness

### What the grounding changed about this task

Three facts move it.

**The mode reaches exactly one function.** `command_for(decision, mode, *,
t_capture_mono_ns)` in `policy/shadow_mode.py:173-191` is the only reader of the mode in
the codebase, and it uses it in one expression, `shadow=(mode != LIVE)`.
`SensingController.decide` takes no mode argument and this is pinned structurally by
`inspect.signature` in `tests/test_shadow_mode.py:76`. So on the Jetson side, "the same
decision in both modes" is close to structural.

**The mode is fixed at construction in the only entry point that writes a log.**
`run_demo.py:506` builds `ModeHolder(LIVE if args.live_rates else SHADOW)` and never
calls `flip_to`; the sole production `flip_to` call site is
`scripts/run_phone_drive.py:185`, a loopback harness that writes no `metadata.jsonl`. So
one drive is entirely shadow or entirely live, and no tick exists in both modes.
"The same input" therefore has to mean *the logged input*, replayed — which is D12, and
which the machinery for already exists: `score_shadow._replay_incumbent` reconstructs a
`SensingController` on a `ReplayClock`, feeds `Inputs.from_record(...)` and the logged
`decided_at_mono` tick by tick, and requires `decide().to_record()` to equal the log.

**The half that is not structural is on the phone, and it does not cross the wire.**
`ConfigApplier.kt:58-83` returns at line 63 on `command.shadow` — before `applied++`,
before `current = command`, and before the four `setXRate` calls and `setHereQuery`.
That early return is the entire behavioural difference between the two modes. The
counters that witness it (`applied`, `shadowed`, `currentRates`) are printed at teardown
(`SensingService.kt:773`) and are **not** fields of `PhoneTelemetry`
(`transport/messages.py:595-644`), so the Jetson's run record cannot state whether the
phone honoured a shadow command. `reference.achieved` witnesses it only indirectly, as a
windowed average, and `eval_run._comparable` already refuses the commanded-versus-achieved
comparison outright on a pure-shadow drive.

### Steps

| # | Step | Produces |
|---|---|---|
| 43.1 | Run `score_shadow.py <run_dir>` on each of task 42's three run directories. | `shadow_score.json`; `replay_identity.status` and its mismatch count. |
| 43.2 | New check, offline: for every logged tick, take the replayed `Decision` and build **both** commands — `command_for(d, SHADOW, t_capture_mono_ns=tick["t_capture_mono_ns"])` and `command_for(d, LIVE, …)`. Encode each through the real `rate_cmd` codec and compare the **decoded** objects, not the Python objects. | A per-tick equality result over the whole drive. |
| 43.3 | Compare decoded-command fields: `rates` (all four), `trigger`, `here` (all of `in_`, `location_ref`, `lat`, `lon`, or both `None`), `t_capture_mono_ns`. Require equality on every one and inequality on `shadow` alone. | The property task 30 proved on 120 in-process ticks, now on a real drive's real inputs. |
| 43.4 | Assert the logged `sensing.shadow` equals `mode != LIVE` for the mode the drive ran in, on every tick — because `shadow` is the one logged key no replay can check (`score_shadow.py:190-201` reports it in `replay_identity.log_only`, and flipping the whole column still yields `status: ok`). | Closes the one column the existing gate is blind to. |
| 43.5 | Capture the phone's `ConfigApplier` teardown line from `logcat` at the end of each run and store it in the run directory. | `applied`, `shadowed`, `lastTrigger`, `currentRates`, `hereConfigured` on disk. |
| 43.6 | On the shadow drives, require `applied == 0` and `shadowed == commands_sent`, with `commands_sent` read from the Jetson's own `rate_cmd` channel counter. | The behavioural half, from the side that performs it. |
| 43.7 | Cross-check the same conclusion from the Jetson alone: `reference.achieved` never tracks a commanded rate on a shadow drive, and `sensing.mode.structurally_absent` lists `feed_congestion` and `source_disagreement`, with `attribution.rules` showing the disagreement rule `not_evaluable` on every tick. | Two independent witnesses of the same fact. |
| 43.8 | Pin 43.2–43.4 as a test over a small recorded log fixture, and mutate `command_for` (drop the query; copy `rates` by reference; return the same `shadow` for both modes) to confirm each mutation is killed. | The check cannot pass vacuously. |

### Acceptance

1. `replay_identity.status == "ok"` with **0 mismatched ticks** on all three run
   directories, over the full tick sequence from tick 0 — the replay reconstructs
   controller state by sequence, so a gap invalidates everything after it.
2. `tick_coverage` reports no missing ticks, and `_log_truncation` does not fire.
3. On every tick, the shadow and live commands built from the same replayed `Decision`
   decode to objects equal in `rates`, `trigger`, `here` and `t_capture_mono_ns`, and
   differing in `shadow`. Zero exceptions across all three runs.
4. On every tick, logged `sensing.shadow` is `True` on a shadow drive and `False` on a
   live drive — checked directly, since replay cannot check it.
5. On the shadow drives, the phone's applier reports `applied == 0` and `shadowed`
   equal to the number of `rate_cmd` messages the Jetson's channel counter says it sent.
6. Every mutation in 43.8 is killed by the test that names it.

**What this does not establish, stated because task 44 exists to establish it.** Shadow
predicts the decision function, not the trajectory. A live drive feeds its reduced
observations back into the next decision and a shadow drive never does, so the two
diverge after the first commanded change and no amount of shadow logging closes that
(`plan_task30_shadow_live.md:41-46`). Task 43 passing is not evidence that live mode
works.

---

## Task 47 — Sim-contract parity

### The brief cannot pass as literally worded, and that is the finding

Task 47 reads "the observation vector produced live matches the simulator's sensing
model field for field." It does not, it is not built to, and at least ten of the 39
encoded slots differ **by construction**. No test in the tree compares the two sensing
models: `deployment/jetson/tests/test_sim_contract.py` (25 tests, all passing, nothing
skipped) feeds *one hand-written dict* to both encoders and proves the two **encoders**
agree. `src.sensing.LocalObservationBuilder` and
`perception.observation_builder.ObservationBuilder` are never imported by the same
module anywhere in the repository.

The divergences found during grounding, each with its mechanism:

| Slot(s) | Sim | Live | Class |
|---|---|---|---|
| `local_density_bin`, `nearby_av_density` | one `range_m` for both | 2×80 m and 2×150 m respectively | different denominator |
| `active_vehicle_count_local`, `local_density_bin`, `local_queue_estimate` | every neighbour within `range_m`, all directions | forward camera detections within 80 m, **doubled** | different counted population |
| `local_mean_speed_bin` | every measured neighbour | only tracks with a valid relative speed; falls back to ego speed | different subset |
| `ego_lane` | `int(ego.lane_id)` | `int(cfg.assumed_lane)`, always 1 | substituted constant |
| `time_since_last_lane_change`, `lane_changes_last_km` | finite after a lane change | `INF` and `0` unconditionally | substituted constant |
| `distance_to_downstream_bottleneck` | `0.0` at a bottleneck, else `inf` | `INF` always | one-sided |
| `follower_gap`, `follower_relative_speed`, `left/right/target_lane_rear_gap`, `target_lane_rear_required_decel` | computed from rear neighbours | hardcoded `INF`/`0.0` | structurally absent (6 slots) |
| `left_lane_front_gap`, `right_lane_front_gap` | `inf` when the lane does not exist | a measured gap for any lateral detection | different domain |
| `uncongested_low_speed_flag` | segment density, per-vehicle free-flow speed | local ±80 m density, config 30.0 | different input quantity |
| `target_lane_front_gap` | recomputed from the actual target lane | aliased to `leader_gap` | approximation |
| `nearby_av_lane_distribution.*` | each peer's absolute `lane_id` | each peer's self-reported `assumed_lane` constant | substituted constant |

There is also a class of divergence with no counterpart at all: the simulator has no
notion of a substituted value, so two vectors can be bit-identical while one is measured
throughout and the other is entirely held or neutral. `field_sources` records that on
the live side and has no sim-side twin.

### What task 47 becomes

A **slot-by-slot parity ledger** over the 39 slots `sim_contract.encoded_slot_names()`
lists, classifying each into one of four states — `identical`, `approximated`,
`substituted`, `structurally_absent` — with the mechanism named for every non-identical
one, and with each classification cross-checked against the provenance class the live
builder assigns that slot. That is a statement a reader can act on. "Matches field for
field" is not, because it is false and the plan should say so rather than produce a
check that quietly passes on 29 slots.

### Steps

| # | Step | Produces |
|---|---|---|
| 47.1 | One scene description, expressed once, instantiated on both sides: a small set of vehicles with positions, speeds and lanes relative to an ego. Fed to `src.sensing.LocalObservationBuilder` and to `perception.observation_builder.ObservationBuilder` with `deployment/jetson/config.yaml`'s shipped values. | The first module in the repository that imports both producers. |
| 47.2 | Encode both dicts through the **same** `encode_local_observation`, so any difference is the sensing model and not the encoder. | Two `(39,)` float32 vectors. |
| 47.3 | Diff slot by slot at `atol=1e-5`, matching the 5-decimal rounding `metadata.jsonl` applies to `encoded`. | A per-slot equal/unequal result with both values. |
| 47.4 | Repeat across a scene set chosen to exercise the classes above: empty scene, leader only, leader plus follower, adjacent-lane traffic, single-lane road, a scene at a bottleneck segment. | Coverage of the slots a single scene leaves at their fallback. |
| 47.5 | Write the ledger: 39 rows, each with slot name, class, mechanism, sim value, live value, encoded difference, and the live `field_sources` provenance class. Emit JSON and markdown. | `outputs/validation/observation_parity/` artifacts. |
| 47.6 | Pin the ledger's classification for all 39 slots in a test, so a slot silently changing class fails. | A regression fence on the parity claim itself. |
| 47.7 | Assert the two independent constructions of the 39 slot names agree: `sim_contract.encoded_slot_names()` versus `src/analysis/observation_audit.py:encoded_field_names()`. They agree today and nothing compares them. | A one-line test closing a soft gap. |
| 47.8 | Read `"encoded"` out of task 42's `metadata.jsonl` — nothing in the tree reads it today — and assert every live tick's vector equals what re-encoding that tick's `obs` produces. | Proof the logged vector is the vector the policy saw, not a re-derivation. |
| 47.9 | For each `substituted` and `structurally_absent` slot, assert the live value equals the constant the ledger names, on every tick of task 42's runs. | The substitutions are constant in fact, not just in intent. |

### Acceptance

1. The ledger covers **all 39** slots, with no slot unclassified.
2. Every slot classed `identical` produces bit-equal encoded values, at `atol=1e-5`, on
   every scene in the set. A slot that is `identical` on one scene and not another is
   reclassified, not averaged.
3. Every slot not classed `identical` carries a named mechanism and a citation to the
   two producing lines.
4. Every slot classed `substituted` or `structurally_absent` carries a live
   `field_sources` provenance class in the `SUBSTITUTED` partition
   (`perception/provenance.py:71-74`). A slot substituted in fact but marked measured in
   provenance is a defect and fails the task.
5. `_covers_encoder` is true on every tick of task 42's runs — `field_sources` names all
   39 slots and no others.
6. 47.8 holds on every tick: the logged `encoded` equals a re-encode of the logged `obs`.
7. `sim_contract.encoded_slot_names()` and `observation_audit.encoded_field_names()` are
   equal as ordered sequences.
8. `test_sim_contract.py` still passes at 25 with nothing skipped, and
   `contract_fingerprint()` still returns `918ec57cf2f2e1db`.

**Named as out of scope, not fixed here:** closing any of the divergences. Six rear-half
slots need a rear sensor the vehicle does not have; `ego_lane` needs lane estimation
that does not exist. Task 47's job is to state the gap, not to close it, and the ledger
is what a later task would work from.

---

## Risks, and what reveals each early

| # | Risk | What reveals it early |
|---|---|---|
| R1 | **The app is not installed and the build does not reproduce.** The Gradle project has not been built on this laptop in this session; `local.properties` may point at an SDK path that has moved. Nothing else can start until it builds. | Step 42.1 is the first step of task 42 and would be the first step of the whole sequence but for task 40 being device-independent. Run `./phone/gradlew :app:assembleDebug` before anything else; it either produces an APK in minutes or it does not. |
| R2 | **The reverse mapping is silently absent and the phone retries forever.** `adb reverse` is cleared by a replug or an `adb kill-server`, and the phone's reconnect never gives up, so the symptom is a drive that produces no ticks and a listener that reports perfect health — the exact shape of the failure `PhoneLink` documents ("a phone that never dialled at all"). | `UsbAcceptor.accept()` re-verifies the mapping on every timeout (step 40.3), and `reverses_reestablished` is in the run record. Also: `adb reverse --list` before every run, asserted non-empty. |
| R3 | **The Jetson's USB-C port is a device-mode controller.** A phone plugged into it never appears on the bus at all — no error, no partial enumeration, nothing in `lsusb`, because the Orin presents itself as a device on that port (`tegra-xudc` behind `l4tbr0`). Two devices both waiting to be enumerated produce no diagnostic. | `adb devices -l` must show `usb:1-2.x`, a USB-A path. It currently shows `usb:1-2.2`. Any run whose serial is absent from `adb devices` stops before it starts, rather than falling back. |
| R4 | **Port collision between `adb reverse` and `ImuWireTest`.** `ImuWireTest.kt:85` binds a `ServerSocket` on the device's `127.0.0.1:47811` — the same address and port `adbd` must listen on for `tcp:47811`. Running the instrumented suite while a reverse is up, or establishing a reverse while it runs, fails one of them with `EADDRINUSE`. | `scripts/with_device.py` already serialises device-touching commands behind a lock; put every USB run behind it too. The tell is an instrumented test failing to bind, which is loud. |
| R5 | **A stale `link.json` sends a "USB" run over the tailnet.** The file is read once at service start and persists across installs in the app's external files directory. The run would succeed and its latency would be the tailnet's. | `path_for_address` reads the accepted socket's remote address; a tailnet-carried run reports a `100.x` peer and `over_tailnet: true`. Acceptance item 42.1 checks it, and step 42.2 removes the file first. |
| R6 | **`link_ms` is converted from a one-way estimate and is biased by the quantity being measured.** A one-way offset's error is the one-way delay itself, which over USB is what task 42 is measuring; averaging does not remove it because it is a bias. | `converted_by_source` per run. On the tailnet baseline, `one_way` carried 6 of 1,204 samples; acceptance item 42.5 refuses the comparison above 5%. |
| R7 | **The round-trip estimator does not converge on a short USB session.** The baseline's session 1 ended with `usable: false`, reason "newest sample is 55.7s old", and the run's proxied ticks were all offset-window fill at session start. A shorter or repeatedly-redialling USB session could proxy a larger fraction. | Converted fraction and `proxy_reasons` are in every run's `summary.json` already; check them on run 1 before running 2 and 3. |
| R8 | **The cadence anomaly recurs.** Twice a run has offered ~40% and ~12% of every commanded rate at once, uniformly, losing nothing, and reproduced neither time. It is phone-side scheduling, not transport. A single USB run hitting it would read as a USB result. | Three runs (D11), each recording achieved-versus-commanded per channel. The anomaly is uniform across channels, which distinguishes it from anything the link could cause. |
| R9 | **`score_shadow`'s replay is sequence-dependent and a mid-log gap corrupts everything after it silently.** Controller state (`_raised_since`, `_holding_until`, `_last`, `_last_active`, `_last_at`) is never serialised, so a missing tick reconstructs a wrong state that then propagates. `_log_truncation` is documented as structurally blind to an outage that reduces both counts. | Task 43 acceptance item 2 requires `tick_coverage` to report no missing ticks *before* the replay result is read. A replay that passes on an incomplete log is not evidence. |
| R10 | **Task 47's paired-scene harness could be built so both sides read the same code and agree trivially** — the failure mode `plan_task30` names for task 43 and the one `test_sim_contract` already has (one dict, two encoders). | Step 47.1 requires the scene to be expressed once and *instantiated separately* on each side, and step 47.2 uses one encoder deliberately so that a difference is attributable to the sensing model. The known divergence table is the check on the check: a harness that reports 39 of 39 identical is broken, because at least ten are not. |
| R11 | **Three runs on a desk say nothing about a car.** Both devices sit on one bench, mains-powered, thermally cold, with a stationary camera and no GPS motion. Every number here is a bench number. | Not mitigable within task 42; stated so it is not read as a drive result. Tasks 45, 46 and 49 exist for the rest. |

---

## What each task hands to the next

- **40 → 42.** A `UsbAcceptor` and a `--usb` flag. Without it, 42 has no data path that
  is not task 32's.
- **42 → 43 and 47.** Three run directories, each with `metadata.jsonl` carrying the
  15-key `sensing` block (including `decision_inputs`, all 17 fields unrounded, and
  `decided_at_mono`), the 39-slot `encoded` vector, `field_sources`, and a phone session
  log. No `metadata.jsonl` exists anywhere in the tree today, so task 42's runs are the
  first input either later task has.
- **43 and 47 → 44 and 50.** 43 establishes the decision-function half of the shadow
  property, which is what task 44 flips the flag against. 47's ledger is what a drive-set
  analysis has to read before quoting a live observation as a simulator observation.

---

## Open items, restated in one place

These are the points at which a design question would have gone to the user. Under
`plan_dsrc_rec` the recommended option was taken and work continues; each is recorded
here so none of it reads as approved.

1. **The app is absent from the handset.** Blocks all four tasks. Recommended path is
   D8: build here, install from the Jetson. The user may prefer to install it himself,
   and the install is a write to the device that planning did not perform.
2. **`specs/transport_protocol.md:21` and `plans/task_list.md` name `adb forward`.**
   The named subcommand inverts the direction the same paragraph mandates and the phone
   has no `ServerSocket` to support it. Recommended correction is one word in each,
   changing no encoding — but the spec is a frozen document and the edit is not taken
   here.
3. **`eval_run.py:741-742` pools `link_ms` across estimator sources**, which
   `sensors/time_sync.py:64-68` says must not be done. Six of 1,204 samples on the
   tailnet baseline. Task 42 fixes it in the report (D10); whether the fix belongs in
   `eval_run` for every run or only in task 42's reporting is a question this plan
   answered by putting it in `eval_run`, where every future run inherits it.
4. **The phone's applier counters do not cross the wire.** `applied`, `shadowed` and
   `currentRates` exist only in a teardown log line, so a drive's own record cannot
   state whether the phone honoured shadow mode. Task 43 reads them from `logcat`
   (step 43.5) rather than adding a `PhoneTelemetry` field, because that is a wire
   change and D15 forbids one here. A wire field is the better long-term answer and is
   not taken.
5. **Task 47's brief is false as worded.** The plan reinterprets it as a parity ledger
   rather than a match assertion. That is a change to what the task delivers, taken by
   recommendation, and the user may want the task renamed in `task_list.md`.
6. **Task 43's Jetson-side check is close to structural.** `command_for` is a pure
   function whose only use of the mode is one boolean expression, so the replay can
   almost only fail if `decide` stops being mode-blind — which is separately pinned by
   `inspect.signature`. The check is still worth running on real logged inputs, and the
   mutation step (43.8) is what keeps it from being a test that pins nothing. But the
   plan should not oversell it: the load-bearing half of task 43 is the phone-side
   check in 43.5–43.6.
7. **No mid-drive flip surface exists**, so task 43 cannot observe both modes in one
   drive. Building one was considered and rejected as task 44's territory. If the user
   wants task 43 to include a flip, the log format, `ModeHolder` and
   `score_shadow._segments` all already support it and only the operator surface is
   missing.
