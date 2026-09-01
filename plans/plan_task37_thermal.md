# Task 37 — Thermal and throttle-event log for both devices

> Written by `plan_dsrc_rec`: every decision below was taken by recommendation,
> without putting the question to the user. The decisions table says what was
> chosen and why; **none of it is user-approved**, and the "Open items" section
> holds every point where the recommendation was weak or where the code
> contradicted the first reading. This plan is the fixed target a validator
> audits the implementation against.

## The short version

Task 37 (plans/task_list.md:1331) asks for a thermal and throttle-event log for
both devices. The two devices are in opposite states and the task is a different
size on each.

**The phone's thermal input is already live, and this plan's first job was to
establish that.** `SensingService.kt:477-480` reads
`PowerManager.currentThermalStatus` and `ThermalZones.read()` once per second;
`TelemetryReporter` (PERIOD_MS = 1000) sends both on the `telemetry` channel;
`messages.PhoneTelemetry.from_wire` decodes them; `sensing_loop.inputs_from`
(:178-180) copies `thermal_status` and `skin_temp_c` and computes
`telemetry_age_s`; `SensingController._thermal_scale`
(sensing_controller.py:730-797) maps them to the multiplier that
`THERMAL_SCALED_KEYS` applies to `camera_hz` and `here_hz`. It is not a
placeholder, not a constant, and not read from something that never updates.
Task 34's device run confirmed the whole chain end to end: `thermal_backoff` was
**quiet on 899 of 899 ticks**, and quiet requires `scale == 1.0`, which is only
reachable when telemetry arrived, was fresh, said `nominal`, and carried a skin
temperature below 40 C. So this task does **not** make a live input out of a dead
one on the phone side.

**The Jetson has no thermal reading at all.** Nothing in
`deployment/jetson/` reads `/sys/class/thermal`, runs `tegrastats`, or otherwise
measures the Orin's temperature. The one thing that comes close,
`logio.metadata_logger.SystemStatsSampler` (:154-208), samples `Temp cpu` /
`Temp gpu` through `jtop` at 5 s -- and it **degrades silently to a no-op when
the `jtop` import fails** (:166-172), writes its records under `type: "system"`,
which `eval_run.load_records` (:78-95) does not collect, and has no test in
`tests/test_metadata_logger.py`. So a Jetson with no `jetson-stats` installed
produces no temperature records and no statement that it produced none. That is
this task's own defect class, already present: a device that cannot report is
recorded exactly like a cool one, which is to say not at all.

**So the Jetson half is new work and the phone half is a completion.** The plan
adds, on the Jetson, a 1 Hz thermal sampler over `/sys/class/thermal` with a
census of every readable zone and a throttle-event stream derived from the
cooling devices; on the phone, four absent-reason and event fields on the
existing telemetry frame plus a `PowerManager` thermal-status listener; and on
both, one per-tick block, one event record type, one 1 Hz sample record type, a
`summary["thermal"]` rollup and an `eval_run` section, because task 33's
experiment found the measurement present and the surface absent
(task_list.md:1110-1112) and task 36 repeated it (task_list.md:1320-1324).

**Does it change behaviour?** In exactly one place: the phone registers a
`PowerManager.OnThermalStatusChangedListener` at come-up and unregisters it at
teardown. **No commanded sensor rate changes, anywhere, in either direction.**
`Inputs` gains no field, `_thermal_scale` is not touched, and
`TelemetryReporter.Sample.thermalStatus` keeps reading `power.currentThermalStatus`
on the same poll rather than the listener's cached value -- so the number
`_thermal_scale` maps is byte-identical to today's. The boundary is asserted four
ways (§"What asserts the boundary"), including a test that exhibits a listener
holding `severe` while the poll holds `nominal` and asserts the frame and the
rates both say `nominal`. The Jetson's temperature deliberately does not reach
the controller in this task; the reason is that no threshold for it can be set
before this task's own experiment produces the data, and inventing one would be
a rate change with no measured basis.

**Scope boundary.** In: new `deployment/jetson/sensors/thermal.py`;
`logio/metadata_logger.py` (nothing, see below -- `SystemStatsSampler` is read,
not changed); `run_demo.py` (start/stop the sampler, one tick-record key, one
summary key); `eval_run.py` (`load_records` gains two record types, one result
block, one report section); `config.yaml` (two keys); `transport/messages.py`
(`PhoneTelemetry`, six absent-tolerant fields); `specs/transport_protocol.md`
and `specs/transport_golden_frames.json`; Kotlin
`transport/PhoneTelemetry.kt`, `sensors/ThermalReader.kt`,
`sensors/ThermalZones.kt`, `sensors/TelemetryReporter.kt`, `SensingService.kt`;
tests on both sides; pins in `scripts/remutate.py`; `ARCHITECTURE.md` §9.
Out: `policy/sensing_controller.py` and `policy/sensing_loop.py` (nothing --
`Inputs` is frozen at 17 fields on purpose, D6); `score_shadow.py` (nothing, for
the same reason); the `rate_cmd` path; any new channel; `replay_demo.py`,
`scripts/run_loopback_pipeline.py` and `scripts/run_phone_drive.py` (they build
records without a sampler and get the absent branch, named, not a crash);
`SystemStatsSampler`'s own behaviour; the failure-event log (task 38); the
session summary generator (task 39); and any threshold that would make
temperature a controller input (named, not built).

**Open items, in one line each** (details at the bottom): the Orin's zone and
cooling-device names are unconfirmed from the repo and the selection rule is
built not to depend on them; whether this moto g power supports
`getThermalHeadroom` is unknown and the plan records the refusal rather than
guessing; a cooling device whose state rises may be a fan rather than a throttle
and this task records the device name rather than classifying; two status
transitions inside one second are counted but the intermediate one is not
carried; and `jtop`'s `Temp cpu` and this task's zone census answer the same
question by two routes that are not reconciled here.

## Which of the three vocabularies this is

The section has built three and the brief forbids a fourth. Thermal is two
different problems and each is an instance of a different one of them. That is
the whole reason a temperature and a throttle event are separate records here.

**A temperature sample is an instance of task 33's stage timing.** Both are a
scalar physical quantity read off one device, whose record has to distinguish
"read now" from "not from this tick, published with a stated bound" from "not
readable, and here is why", and must never write a zero for the third.
`StageTiming` (sensors/time_sync.py:115-181) says exactly that with
`ms` / `basis` / `clock` / `bound_ms` / `reason`, and its docstring names the
defect class this task is another instance of. The thermal record reuses the
three-member basis and **two of its three words are literally the same
constants**: `sensors/thermal.py` re-exports `STAGE_BASIS_MEASURED` and
`STAGE_BASIS_ABSENT` from `time_sync` rather than retyping the strings, the way
`feed_fusion.SOURCE_FEED` became a re-export of `provenance.SOURCE_FEED` in
task 36. The middle member is `stale` rather than `converted`, because here the
number is displaced in time rather than across a clock, and the bound it carries
is `age_s` rather than `bound_ms`. That is the same three-state shape with one
member renamed to say what it actually is, not a fourth vocabulary.

`StageTiming` itself is not reused as the container: its value field is named
`ms` and its bound is `bound_ms`, and putting a temperature in a field called
`ms` would be a worse lie than a new record type.

**A throttle event is an instance of task 34's rule attribution.** The brief's
requirement -- "a throttle event that could not be observed must not look like an
absence of throttling" -- is word for word the distinction task 34 built
`RULE_QUIET` versus `RULE_NOT_EVALUABLE` for. So the event stream carries a
per-device status that is one of task 34's three words, imported from
`policy.sensing_controller` rather than restated:

- **`fired`** -- the source was readable and at least one event occurred; `count`
  says how many and the `thermal_event` lines say which.
- **`quiet`** -- the source was readable throughout and no event occurred.
- **`not_evaluable`** -- the source could not be read, with `missing` naming
  exactly what was missing: `["cooling_device_cur_state"]`, `["telemetry"]`, or
  `["thermal_status_changes"]`. Those three are different absences and the record
  never merges them.

That is how a reader tells "no throttle events occurred" from "throttle events
were not observable": the first is `{"status": "quiet", "count": 0}` and the
second is `{"status": "not_evaluable", "count": 0, "missing": [...]}`. `count`
is 0 in both, which is exactly why `count` alone is not the record.

**Task 36's eleven-member provenance vocabulary is deliberately not extended.**
It tags the 39 encoder slots and the identity `set(field_sources) ==
set(encoded_slot_names())` is pinned on every tick (task_list.md:1310). A
temperature is not an encoder slot; adding one would break that identity and
move the missingness denominator, which is a number already quoted in the paper.
Thermal reuses the *discipline* -- a closed set of named absence reasons -- and
its own closed sets live beside the readings they describe.

## What is true today, from reading the code

### The phone

- **Status.** `ThermalReader.statusName` (ThermalReader.kt:15-25) maps
  `PowerManager`'s seven constants to the wire's words plus `unknown` for an
  unrecognised integer. `SensingService.kt:477` calls
  `power.currentThermalStatus` inside `TelemetryReporter`'s `sample` lambda,
  which runs once per second on the `dsrc-telemetry` thread
  (TelemetryReporter.kt:52-72, PERIOD_MS = 1000).
- **Headroom.** `ThermalReader.headroomFrom` (:56-60) is guarded on
  `SDK_INT < R` and returns null there; `headroomOrNull` (:88-93) returns null
  for a non-finite value and for anything outside `[0, MAX_PLAUSIBLE_HEADROOM]`.
  **Four distinct causes collapse into one bare `null`** and no reason is carried:
  API below 30, `NaN` from the platform, negative, and above 10. The class's own
  docstring says `NaN` means "too soon after boot, too soon after the last call,
  or unsupported on the device" -- three further causes the platform does not
  separate, inside the second one.
- **Skin temperature.** `ThermalZones` (ThermalZones.kt) walks
  `/sys/class/thermal`, resolves one zone once per come-up against a `PREFERRED`
  name list, and returns `Reading(celsius, zone)` or null. **Five distinct causes
  collapse into one bare `null`**: the root would not list (SecurityException or
  absent), no `type` matched `PREFERRED`, the `temp` file would not read, the
  contents would not parse, and the value fell outside `[-40, 125]`. The class
  already carries the zone name with every reading, and its docstring explains
  why -- on the moto g power the HAL's `skin` sensor matched `xo_therm` to
  0.007 C while `quiet_therm`, the conventional name, read 1.2 C lower and is a
  different sensor.
- **The wire.** `PhoneTelemetry` (Kotlin) / `messages.PhoneTelemetry` (Python)
  carry `thermal_status` (required string), `thermal_headroom` (nullable number),
  `skin_temp_c` and `skin_temp_zone` (absent-tolerant, added after the first
  phones shipped -- specs/transport_protocol.md:391-397). Canonical JSON refuses
  a non-finite number on both sides, which is why `headroomOrNull` must return
  null rather than `NaN`: a `NaN` would fail the whole frame and take the thermal
  status down with it.
- **Nothing on the phone records a transition.** The status is sampled at 1 Hz
  and the sample overwrites the last one. There is no event, no count, and no
  `PowerManager.addThermalStatusListener` anywhere in `phone/`.
- **`thermal_headroom` is decoded and dead.** Its only reader on the Jetson is
  `PhoneLink.to_record()` (phone_link.py:913), which records the **last** value
  of the drive into `summary["phone"]["telemetry"]`. Same for `skin_temp_zone`
  (:915). Neither reaches `Inputs`, neither appears per tick, and neither has a
  time series anywhere.

### The Jetson

- **No thermal reading exists.** Verified by grep over `deployment/jetson/`:
  no `/sys/class/thermal`, no `tegrastats`, no `nvpmodel`, no thermal zone of any
  kind.
- **`SystemStatsSampler`** (logio/metadata_logger.py:154-208) is the only
  adjacent thing. It is constructed in `run_demo.py:388-389` under
  `config["logio"]["system_stats"]` (config.yaml:103-104, default true,
  interval 5.0 s) and writes `{"type": "system", ..., "temp_cpu_c": ...,
  "temp_gpu_c": ...}`. Three properties matter: it is **silent when `jtop` is
  missing** (`self.available = False` and `return` from `__init__`, :166-172,
  with `start()` a no-op), its records are **not read by `eval_run`**
  (`load_records` collects `tick`, `scenario` and `timebase_estimate` only,
  :86-93), and it has **no test** (`tests/test_metadata_logger.py` does not
  mention it). Its `_loop` also writes a `{"type": "system_error"}` record on an
  exception, which nothing reads either.
- **The controller's thermal path is fully attributed already.** Task 34's
  `RuleCheck` for `Trigger.THERMAL` carries, on every tick,
  `{"thermal_status", "skin_temp_c", "telemetry": "fresh"|"stale"|"absent",
  "telemetry_age_s", "scale", "cause"}` where `cause` is a `THERMAL_CAUSES`
  member (sensing_controller.py:158-166, :730-797). So the phone's status, skin
  temperature and report freshness are **already** in every tick record, on the
  same three-state discipline, sourced from the objects the controller actually
  decided on. This plan does not restate them (D5).

### The transport

Eight channels, table in `specs/transport_protocol.md:120-130` and
`Channels.kt`. `telemetry` is up / normal / reliable / depth 32, described as
"thermal, phone-side stats". One typed message per channel
(`PhoneTelemetry.CHANNEL`), so a second message shape on `telemetry` is a
protocol change and a new event message would be a new channel.

## Decisions taken (by recommendation -- not signed off by the user)

| # | Question | Options | Taken | Why |
|---|----------|---------|-------|-----|
| D1 | Does the Jetson's temperature become a controller input | (a) feed it into `_thermal_scale` beside the phone's; (b) record only | **(b)** | The task list names thermal headroom as one of the three binding costs (task_list.md:38-41), so (a) is where this eventually goes -- but a multiplier needs a threshold, and no measurement of this Orin's temperature under this workload exists yet, because this task is what produces it. (a) would change `camera_hz` and `here_hz` on real hardware on the strength of a guessed constant. Task 36's lesson is one behaviour change, stated precisely; this is the one place to *not* take. Named as a follow-on, not built. |
| D2 | Where the Jetson reads temperature | (a) `tegrastats` subprocess; (b) `jtop`; (c) `/sys/class/thermal` directly | **(c)** | (a) is a subprocess per sample and a text format that changes between JetPack versions. (b) is already present, already optional, and already silent when absent -- adopting it would make the whole task conditional on a package the device may not have. (c) is ordinary file reads, needs no service, is testable against a fixture directory with no device at all, and is the same source the phone side already uses, so the two devices' readings are the same kind of number. |
| D3 | Sample rate and where | (a) on the tick path at up to 5 Hz; (b) beside it on its own thread at 1 Hz | **(b)** | A temperature's time constant is tens of seconds; 5 Hz buys nothing and costs one `open`/`read`/`close` per zone per tick -- roughly 60 file reads a second against roughly 12. More importantly, the tick loop `continue`s without producing a record whenever the camera yields no frame (run_demo.py:490-494), so a thermal log that only exists on ticks goes silent exactly when a hot, throttling device has stopped delivering frames. 1 Hz also matches `TelemetryReporter.PERIOD_MS`, so the two devices' series are on the same cadence. |
| D4 | What a tick with no fresh sample records | (a) the last value, unmarked; (b) `basis: "stale"` with `age_s`; (c) `basis: "absent"` | **(b)**, with (c) for "never sampled" | This is task 33's question and gets task 33's answer: never a zero, never an unmarked carry-forward. The freshness bound is `2 x` the sampler period, **derived** from `thermal_interval_s` rather than typed, following `MAX_EVIDENCE_GAP_S`'s precedent (sensing_controller.py:117-122) -- a typed constant stops covering the rate the moment the rate changes. A stale reading is never collapsed to absent no matter how old, because this record decides nothing and `age_s` is strictly more informative than a refusal; that is a deliberate divergence from `MAX_TELEMETRY_AGE_S`, which exists because the controller *does* decide on it. |
| D5 | Whether the per-tick block restates the phone's status and skin temperature | (a) restate them for a self-contained block; (b) carry only what nothing else records, plus a join key | **(b)** | They are already on every tick in task 34's `thermal_backoff` evidence, from the objects the controller decided on. Restating them creates a second source of truth that can disagree, which is the failure `SessionLog`'s docstring refuses by design. The block carries `at_mono` -- the telemetry report's own arrival instant, the idiom task 35 established because "an age recomputed against a fresh `now` each time never reveals that the underlying report did not change" -- and a test asserts `thermal.phone.at_mono == sensing.reference.at_mono` on every tick where both exist. The block must not depend on `sensing`, which can be `None` (run_demo.py:444). |
| D6 | Whether anything joins `Inputs` | (a) add `jetson_temp_c` and friends; (b) nothing | **(b)** | `score_shadow`'s `REFUSAL_INPUTS_SCHEMA` compares `set(decision_inputs)` against `{f.name for f in fields(Inputs)}` on every sensing tick and refuses the whole log on a mismatch. One added field makes every task-35 and task-36 drive unscoreable by name, for a value no rule reads. It is also the mechanism by which D1 could happen by accident. `Inputs` stays at 17 fields and a test pins the count and the names. |
| D7 | Where the phone's throttle event goes | (a) a new channel; (b) a second message type on `telemetry`; (c) absent-tolerant fields on the existing `PhoneTelemetry` frame | **(c)** | (a) costs a row in the channel table, a priority and an overflow policy, both `Channels` tables, the golden frames and the spec, for an event that happens a handful of times per drive; and the `telemetry` channel is already up / normal / reliable / depth 32 at 1 Hz with headroom. (b) breaks the one-typed-message-per-channel rule the router dispatches on. (c) is the exact path `skin_temp_c` and `skin_temp_zone` already took, spec text and all (transport_protocol.md:391-397). **The channel decision is therefore: no channel changes, and `telemetry`'s policy is untouched.** |
| D8 | How a transition is detected on the phone | (a) infer it from the 1 Hz poll series on the Jetson; (b) `PowerManager.addThermalStatusListener` | **(b)** | (a) cannot see a transition that begins and ends between two polls, and cannot timestamp one to better than a second. (b) is API 29, which is `minSdk` (app/build.gradle.kts:18), so it needs no version guard beyond the one lint reads; it gives the transition its own `elapsedRealtimeNanos` stamp; and it is a count that is monotone whether or not a frame is dropped. It is also the one behaviour change in this task and is fenced accordingly. |
| D9 | What carries the transition over the wire | (a) a full event list per frame; (b) a monotone count plus the last transition | **(b)** | A list is unbounded and the frame is 1 Hz. A monotone `thermal_status_changes` means no transition is lost *in count* even across a dropped frame, and `thermal_change_{from,to,at_mono_ns}` names the most recent one. The bound is stated and derived: a count that rises by more than one between two frames means intermediate transitions were not carried, and the Jetson records that as `events.phone.count` rising with no matching `thermal_event` line for the gap. Absent until the first transition, so a nominal drive pays nothing for them. |
| D10 | What a Jetson throttle event *is* | (a) a trip-point crossing; (b) a cooling device's `cur_state` changing; (c) a CPU/GPU frequency cap | **(b)** | `cooling_device*/cur_state` is the only one of the three whose semantics is literally "a cooling action is in effect right now", it is one integer per device, and its transitions are events by construction. (a) needs per-zone trip tables whose meaning varies by vendor; (c) confounds a thermal cap with every other governor decision. The device's `type` string travels with every event -- the same rule `ThermalZones` applies to zone names, and for the same reason -- because a `pwm-fan` rising from 0 to 1 is a fan speeding up and not a throttle, and this task records the fact rather than classifying it (open item 3). |
| D11 | Which zone becomes the per-tick number | (a) a fixed name; (b) the hottest readable zone, re-picked every sample; (c) a preferred-name list, falling back to the hottest at the first sample, then held | **(c)** | (a) cannot be written without the device. (b) makes the series flip between sensors, so a trend is not a trend. (c) mirrors `ThermalZones.PREFERRED` exactly, degrades to *something* when none of the guessed names exist -- which is the real risk of guessing names from a plan -- and holds the choice for the drive so the series is one sensor. `selected_by` records which arm won, `zone` records the name, and the full census on every sample record makes a different choice recoverable offline. |
| D12 | Where the drive-level surface lives | (a) the tick log only; (b) `summary["thermal"]` and an `eval_run` section and report lines | **(b)** | Task 33 recorded the `stages` block on all 900 ticks and `report.md` printed none of it; task 36 printed a missingness mean with no spread and the printed mean occurred on zero ticks. A measurement with no surface is this project's most repeated mistake and it has now cost two experiments. `eval_run.load_records` gains the two new record types, `result["thermal"]` gains a block, and `report.md` gains a `## Thermal` section. |
| D13 | Whether a gate is added | (a) gate on peak temperature; (b) gate on the event stream being observable; (c) no gate | **(c)** | (a) needs a threshold this task does not have. (b) would fail every desk run with no phone attached, and `gate(..., ok=None)` for "not applicable" would make the gate pass vacuously on exactly the runs it exists for. Task 36's precedent: "a run recorded before this task is not a failed drive". The report states the three-word status in words instead, and open item 6 says what would have to be true to gate it later. |
| D14 | `SystemStatsSampler` | (a) fold jtop's `Temp cpu` into the new record; (b) replace it; (c) leave it alone and record whether it was available | **(c)** | (a) merges two temperature sources whose relationship on this device is unknown -- exactly the reconciliation open item 5 exists to name. (b) deletes power and utilization sampling this task has no business touching. (c) is one boolean: `summary["thermal"]["jetson"]["jtop_available"]` is `true` / `false` / `null`, where null means the sampler was never constructed. That closes the silent no-op where a reader looks, without changing a line of the sampler. |
| D15 | Where the new module lives, and the vocabulary import | (a) `logio/`; (b) `sensors/thermal.py` importing the three status words from `policy.sensing_controller` | **(b)** | It reads a device sensor and exposes `latest()`, which is the shape `camera_stream`, `gps_reader` and `phone_link` all already have. The import adds one new edge, `sensors -> policy`, which is acyclic: `policy/sensing_controller.py` imports only `math`, `dataclasses` and `typing`. The alternative -- restating `"fired"`, `"quiet"`, `"not_evaluable"` and pinning them with an equality test -- is precisely the drift task 36 gave the vocabulary one home to prevent. |

## The record, exactly

### `deployment/jetson/sensors/thermal.py` (new)

```python
from sensors.time_sync import STAGE_BASIS_ABSENT, STAGE_BASIS_MEASURED
from policy.sensing_controller import RULE_FIRED, RULE_NOT_EVALUABLE, RULE_QUIET

THERMAL_BASIS_MEASURED = STAGE_BASIS_MEASURED   # re-export, not a new string
THERMAL_BASIS_ABSENT   = STAGE_BASIS_ABSENT     # re-export, not a new string
THERMAL_BASIS_STALE    = "stale"                # displaced in time, bound = age_s
THERMAL_BASES = frozenset({...the three...})

#: Why a temperature is absent. Closed; every arm of the reader returns a member.
ABSENT_NO_THERMAL_ROOT  = "no_thermal_root"     # the directory would not list
ABSENT_NO_ZONE_READABLE = "no_zone_readable"    # listed, no plausible temp in any zone
ABSENT_NO_SAMPLE_YET    = "no_sample_yet"       # running, first pass not finished
ABSENT_SAMPLER_STOPPED  = "sampler_stopped"     # disabled in config, or the thread died
ABSENT_REASONS = frozenset({...the four...})

#: Why the phone's thermal fields are absent, beyond the phone's own reasons.
ABSENT_NO_TELEMETRY = "no_telemetry"            # spelled as reference_from's already is

#: What `events.<device>` can be missing. Closed.
MISSING_COOLING_STATE   = "cooling_device_cur_state"
MISSING_TELEMETRY       = "telemetry"
MISSING_STATUS_CHANGES  = "thermal_status_changes"

MIN_PLAUSIBLE_C = -40.0     # the same band ThermalZones.kt applies, and for the same
MAX_PLAUSIBLE_C = 125.0     # reason: zones report values that are not temperatures

PREFERRED_ZONE_TYPES = ("tj-thermal", "cpu-thermal", "CPU-therm",
                        "gpu-thermal", "GPU-therm", "soc0-thermal")
```

Types:

- `ThermalReading` (frozen): `celsius: float | None`, `zone: str | None`,
  `basis: str`, `age_s: float | None`, `reason: str | None`, `zones_n: int`.
  `to_record()` emits all six. `celsius` is `None` whenever
  `basis == "absent"`; a temperature is never a zero.
- `JetsonThermal(root=Path("/sys/class/thermal"))` -- pure, no threads,
  injectable root so every branch is reachable against a fixture directory.
  `read_zones() -> tuple[dict[str, float], str | None]` returns the census and an
  absence reason; `read_cooling() -> tuple[dict[str, int], str | None]` the same
  for `cooling_device*/{type,cur_state}`; `celsius_of(raw)` applies the
  millidegree/degree magnitude rule and the plausibility band, ported from
  `ThermalZones.celsiusOf` so both devices interpret a `temp` file the same way.
- `ThermalSampler(sink, phone=None, interval_s=1.0, clock=time.monotonic)` --
  the 1 Hz thread. Holds the last census, resolves the selected zone once,
  diffs `cooling` against the previous pass, writes a `thermal_sample` record
  every pass and a `thermal_event` record per transition through `sink`
  (`MetadataLogger.write`). Exposes `latest(now) -> dict` for the tick block,
  `to_record()` for the summary, `start()` / `stop()`.

### Per tick, on the tick record, key `"thermal"`

Written in `run_demo`'s tick loop beside `session_id` and `sensing`
(run_demo.py:519-527), which is where the two existing per-tick additions are
made. Typical shape, both devices healthy:

```json
"thermal": {
  "jetson": {"temp_c": 47.5, "zone": "cpu-thermal", "basis": "measured",
             "age_s": 0.412, "reason": null, "zones_n": 9},
  "phone":  {"headroom": null, "headroom_absent": "not_a_number",
             "skin_zone": "xo_therm", "skin_temp_absent": null,
             "at_mono": 1234.5678, "absent": null},
  "events": {"jetson": {"status": "quiet", "count": 0, "last": null},
             "phone":  {"status": "quiet", "count": 0, "last": null}}
}
```

And with neither device reporting -- the shape that must not be confusable with
the one above:

```json
"thermal": {
  "jetson": {"temp_c": null, "zone": null, "basis": "absent",
             "age_s": null, "reason": "no_zone_readable", "zones_n": 0},
  "phone":  {"headroom": null, "headroom_absent": null, "skin_zone": null,
             "skin_temp_absent": null, "at_mono": null, "absent": "no_telemetry"},
  "events": {"jetson": {"status": "not_evaluable", "count": 0, "last": null,
                        "missing": ["cooling_device_cur_state"]},
             "phone":  {"status": "not_evaluable", "count": 0, "last": null,
                        "missing": ["telemetry"]}}
}
```

`missing` is present only on a `not_evaluable` entry, the way task 34's
`RuleCheck.to_record` emits it. `last`, when non-null, is
`{"at_mono": 1234.5, "from": "nominal", "to": "light"}` for the phone and
`{"at_mono": 1234.5, "unit": "tegra-heavy", "from": 0, "to": 1}` for the Jetson.
`count` is cumulative for the drive so far, so a log truncated by an unflushed
buffer still says how many events preceded the cut.

Reading rules a validator audits against:

- `basis` is a member of `THERMAL_BASES`; `temp_c is None` if and only if
  `basis == "absent"`; `reason is not None` if and only if `basis == "absent"`
  and is a member of `ABSENT_REASONS`; `age_s is None` if and only if
  `basis == "absent"`.
- `events.<device>.status` is one of task 34's three constants;
  `status == "fired"` if and only if `count > 0`; `missing` is non-empty if and
  only if `status == "not_evaluable"`, and every member is one of the three
  `MISSING_*` constants.
- `thermal.phone.at_mono == sensing.reference.at_mono` on every tick where both
  are present.
- `sum(events.<device>.count)` over the drive equals the number of
  `thermal_event` lines for that device, except where the phone's count rose by
  more than one between two frames (D9), which is itself recorded.

### Per event, a whole record: `{"type": "thermal_event"}`

A discrete event is not a tick field. Written the moment the sampler observes
it, so it exists on a drive that produced no ticks at all:

```json
{"type": "thermal_event", "device": "jetson", "seq": 1,
 "t_wall": 1756700000.123, "t_mono": 1234.567,
 "clock": "jetson", "at_ns": 1234567000000,
 "source": "cooling_device", "unit": "tegra-heavy",
 "from": 0, "to": 1, "temp_c": 88.5,
 "zones": {"cpu-thermal": 88.5, "gpu-thermal": 84.0, ...}}
```

```json
{"type": "thermal_event", "device": "phone", "seq": 1,
 "t_wall": 1756700000.456, "t_mono": 1234.891,
 "clock": "phone", "at_ns": 987654321000,
 "source": "thermal_status", "unit": null,
 "from": "nominal", "to": "light", "temp_c": 41.2, "zones": null}
```

`clock` and `at_ns` are the task-33 discipline applied to an event: the phone's
transition instant is on the phone's `elapsedRealtimeNanos` and the Jetson's is
on the Jetson's monotonic clock, and **the two are not converted**. `t_mono` is
always this side's own observation instant, so a reader always has one stamp on
one clock; `at_ns` is the peer's own stamp, labelled with whose clock it is. A
status transition does not need millisecond accuracy, and running it through
task 33's timebase machinery would attach an estimate-dependent bound to a
number nobody differences. `seq` counts from 1 per device, so a gap in the log
is visible rather than inferred.

### Per sample, at 1 Hz: `{"type": "thermal_sample"}`

The device-level time series, at the sampler's own rate and independent of the
tick loop:

```json
{"type": "thermal_sample", "t_wall": 1756700000.123, "t_mono": 1234.567,
 "jetson": {"zones": {"cpu-thermal": 47.5, "gpu-thermal": 45.0, "soc0-thermal": 46.2,
                      "soc1-thermal": 46.0, "soc2-thermal": 46.4, "tj-thermal": 48.0,
                      "cv0-thermal": 44.8, "cv1-thermal": 44.6, "cv2-thermal": 44.9},
            "cooling": {"pwm-fan": 0, "tegra-heavy": 0, "tegra-crit": 0},
            "basis": "measured", "reason": null},
 "phone":  {"status": "nominal", "headroom": null, "headroom_absent": "not_a_number",
            "skin_temp_c": 33.2, "skin_zone": "xo_therm", "skin_temp_absent": null,
            "at_mono": 1234.5, "age_s": 0.618, "absent": null}}
```

This record does restate the phone's status and skin temperature, and D5 said
the tick block must not. The difference is the reason each exists: the tick
block says what was true when a decision was made and the controller already
records that; this record is the phone's thermal **series**, which exists
nowhere -- `PhoneLink.to_record()` keeps only the last value of the drive
(phone_link.py:910-919) -- and it has to survive a drive with no ticks. `at_mono`
is carried so a reader can drop the repeats that occur whenever the sampler and
the phone's reporter drift into step, which is the same reason task 35 put it on
`reference`.

### The wire: six absent-tolerant fields on `telemetry`

No channel is added and `telemetry`'s policy (up / normal / reliable / depth 32)
is unchanged. New header fields, all optional and absent-tolerant, following the
`skin_temp_c` precedent verbatim on both sides:

| field | type | when absent |
|---|---|---|
| `thermal_headroom_absent` | string, one of `api_too_old` / `not_a_number` / `out_of_band` | when `thermal_headroom` is present |
| `skin_temp_absent` | string, one of `no_zones_listed` / `no_preferred_zone` / `unreadable` / `implausible` | when `skin_temp_c` is present |
| `thermal_status_changes` | integer count, monotone within a service run | never, after this task; absent from an older phone build |
| `thermal_change_from` | string, a `THERMAL_SCALE` key | before the first transition |
| `thermal_change_to` | string, a `THERMAL_SCALE` key | before the first transition |
| `thermal_change_at_mono_ns` | integer, phone `elapsedRealtimeNanos` | before the first transition |

`thermal_headroom_absent` and `skin_temp_absent` are exactly one string each and
they are the whole point of the pair: a `null` headroom on this handset today
means one of four things and the record says which. `not_a_number` covers the
three causes the platform itself does not separate -- unsupported, too soon after
boot, and called again inside one second -- and the field's doc comment says so,
because at `PERIOD_MS = 1000` this app sits exactly on that rate limit.

Every one of them goes through the same finite/plausible guard the existing
fields do. A non-finite number here would not produce a wrong value, it would
fail canonical JSON and take the whole telemetry frame down -- the defect
`ThermalReader`'s docstring already names -- so the guard is a test, not a
comment.

### `summary["thermal"]`

Written from `ThermalSampler.to_record()` in `run_demo`'s `finally` block beside
`summary["sensing"]` and `summary["phone"]`:

```json
"thermal": {
  "jetson": {"samples": 180, "selected_zone": "cpu-thermal",
             "selected_by": "preferred_name",
             "zones_seen": ["cpu-thermal", "gpu-thermal", ...],
             "temp_c": {"min": 41.0, "mean": 47.1, "p50": 47.2, "p95": 49.9, "max": 50.5},
             "per_zone_max_c": {"cpu-thermal": 50.5, ...},
             "cooling_devices": ["pwm-fan", "tegra-heavy", "tegra-crit"],
             "basis_counts": {"measured": 180, "stale": 0, "absent": 0},
             "absent_reasons": {},
             "jtop_available": true},
  "phone":  {"samples": 178, "status_counts": {"nominal": 178},
             "skin_temp_c": {"min": 30.1, "mean": 33.0, "p50": 33.1, "p95": 35.4, "max": 35.9},
             "skin_zone": "xo_therm",
             "headroom_absent_counts": {"not_a_number": 178},
             "skin_temp_absent_counts": {},
             "absent_counts": {"no_telemetry": 2}},
  "events": {"jetson": {"status": "quiet", "count": 0, "missing": [], "by_unit": {}},
             "phone":  {"status": "quiet", "count": 0, "missing": []}}
}
```

`basis_counts` and `absent_reasons` sum to `samples`. `headroom_absent_counts`
and `skin_temp_absent_counts` are the phone's two closed reason sets counted
over the drive, so a handset that never answers headroom reads
`{"not_a_number": 178}` rather than a silent zero. `jtop_available` is
`true` / `false` / `null`, where null means `SystemStatsSampler` was not
constructed (D14).

### `eval_run.py`

`load_records` gains two record types, returned alongside the existing three:
`thermal_sample` and `thermal_event`. `result["thermal"]` carries the summary's
own block when `summary.json` has one, plus what only the records can say:
`ticks_by_basis` (from the tick blocks), `sample_gaps_s` (p50/p95/max of the
interval between consecutive `thermal_sample` records, which is how a stalled
sampler becomes visible), and `events` as a list of the `thermal_event` lines.
A run recorded before this task reports `null` and is **not** a failed drive --
the `jetson_ms_source` precedent (eval_run.py:447-451), and D13.

`report.md` gains, after `## Observation quality`:

```
## Thermal

- jetson cpu-thermal: p50 47.2 C, p95 49.9 C, max 50.5 C over 180 samples
  (measured 180, stale 0, absent 0; 9 zones read, hottest at peak tj-thermal 51.0 C)
- phone: nominal on 178 of 178 reports; skin xo_therm p50 33.1 C, max 35.9 C
- phone headroom: not reported on 178 of 178 reports (not_a_number)
- throttle events, jetson: quiet -- cooling devices readable throughout, 0 transitions
- throttle events, phone: quiet -- 0 status transitions in 178 reports
```

and, when a device could not be observed, the line that must not read like the
one above:

```
- throttle events, jetson: NOT EVALUABLE -- missing cooling_device_cur_state;
  this drive says nothing about whether the Jetson throttled
```

### `config.yaml`

```yaml
logio:
  thermal: true                      # /sys/class/thermal sampling + throttle events
  thermal_interval_s: 1.0            # freshness bound on the tick block is 2x this
```

## The work

1. **`sensors/thermal.py`** -- the constants, `ThermalReading`, `JetsonThermal`,
   `ThermalSampler`. The two basis words re-exported from `time_sync`, the three
   status words imported from `policy.sensing_controller`.
2. **`run_demo.py`** -- construct and `start()` the sampler under
   `config["logio"]["thermal"]`, beside `SystemStatsSampler` (:388-389); one
   `record["thermal"] = sampler.latest(now)` in the tick loop beside
   `record["session_id"]` (:519-527); `sampler.stop()` in the `finally` beside
   `stats_sampler.stop()` (:626-627), **before** `logger.close()` so its last
   records are flushed; `summary["thermal"] = sampler.to_record()` with
   `jtop_available` taken from `stats_sampler`.
3. **`config.yaml`** -- the two keys, and the `logio` block's schema check in
   `run_demo.load_config` if one exists for the sibling keys.
4. **Kotlin `ThermalReader.kt`** -- `headroomFrom` returns a
   `(Double?, String?)` pair, or a small `Headroom` data class, so the reason
   travels with the null. The API-30 guard stays written where lint can read it;
   the `headroomIfSupported` split and the boundary-agreement test stay exactly
   as they are.
5. **Kotlin `ThermalZones.kt`** -- `read()` returns a reading or a reason. The
   four reasons are the four existing null returns, named where they already
   are; no behaviour changes and the resolve-once caching is untouched.
6. **Kotlin `ThermalStatusWatcher`** (new, in `sensors/`) -- registers
   `PowerManager.addThermalStatusListener` on a named executor, holds
   `changes: Long` and the last transition, and unregisters on `stop()`. It is
   the one new resource, so it joins `SensingService`'s teardown census
   (`resourcesHeldAfterTeardown`, SensingService.kt:806-816) and its own
   `release(...)` step.
7. **Kotlin `TelemetryReporter.kt`** -- `Sample` gains
   `headroomAbsent`, `skinTempAbsent`, `statusChanges`, and the last transition's
   three parts, all defaulted and **appended last**, for the reason the class
   already documents: it is constructed positionally and inserting a parameter
   rebinds every existing caller's arguments to the wrong names.
8. **Kotlin `SensingService.kt`** -- build the watcher beside `ThermalZones()`
   (:464), fill the new `Sample` fields in the lambda, log its stats in the
   teardown census, null it in the `finally`.
9. **Kotlin + Python `PhoneTelemetry`** -- the six fields, absent-tolerant on
   both sides, encoded only when present, every number through the finite guard.
10. **`specs/transport_protocol.md`** -- the six fields in the `telemetry` row,
    marked `*`, and a paragraph on the two reason sets beside the existing
    `skin_temp_c` one. The channel table itself does not change.
11. **`specs/transport_golden_frames.json`** -- regenerate through
    `scripts/generate_transport_golden_frames.py`; add one frame carrying a
    transition and the two reason strings, so both sides' decoders are pinned
    against the same bytes.
12. **`eval_run.py`** -- `load_records`, `result["thermal"]`, the report section.
13. **`ARCHITECTURE.md` §9** -- the evaluation-hooks paragraph names
    `type: system` today; it gains `type: thermal_sample` and
    `type: thermal_event` and the per-tick `thermal` block.
14. **Tests and pins** (below).

## Tests, and what each one proves

Python, `deployment/jetson/tests/test_thermal.py` (new) unless stated:

1. **A zone that will not read is absent with a reason, not a zero.**
   `JetsonThermal` over a fixture root with `thermal_zone0/type` readable and
   `temp` unreadable returns `basis="absent"`, `reason="no_zone_readable"`,
   `celsius=None`. Proves the brief's first requirement directly.
2. **The four absence reasons are each reachable and distinct.** One fixture per
   arm: root missing, root present with no plausible zone, sampler started but no
   pass completed, sampler stopped. Asserts four different strings and that all
   four are in `ABSENT_REASONS`.
3. **A value that is not a temperature is refused.** A zone reading `100000`
   millidegrees and one reading `-351000` are both excluded from the census, and
   the selection does not pick them -- the `soc`/`ibat` case `ThermalZones.kt`
   documents on the phone, ported.
4. **Millidegrees and degrees are separated by magnitude.** `47500 -> 47.5`,
   `47 -> 47.0`, `999 -> 999.0` refused as implausible. Same rule as the Kotlin
   side; a second test asserts the two implementations agree on a shared table
   of raw strings (the table lives in the test, one copy, read by both suites is
   not possible across languages, so the Python test states the table and a
   Kotlin test states the same one -- a divergence shows as two different
   expected lists in review).
5. **A stale sample is stale, not measured, and carries its age.** Frozen clock;
   one sample; advance past `2 x interval`; `latest()` returns
   `basis="stale"`, `age_s` equal to the advance, `temp_c` unchanged.
6. **A stale sample is never collapsed to absent.** Advance 600 s; still
   `basis="stale"` with `age_s == 600`. Pins D4's deliberate divergence from
   `MAX_TELEMETRY_AGE_S`.
7. **The freshness bound follows the configured interval.** Construct at
   `interval_s=0.2`; a 0.5 s advance is stale. Pins that the bound is derived,
   not typed -- the `MAX_EVIDENCE_GAP_S` failure mode.
8. **`quiet` and `not_evaluable` are different records.** A drive whose cooling
   devices read 0 throughout gives `{"status": "quiet", "count": 0}` with no
   `missing`; a drive whose cooling directory is unreadable gives
   `{"status": "not_evaluable", "count": 0, "missing":
   ["cooling_device_cur_state"]}`. **This is the task's headline assertion** and
   it is written as one test with both halves so deleting either is visible.
9. **A cooling transition emits exactly one event, with the device name.**
   `pwm-fan` 0 to 1 across two passes emits one `thermal_event` with
   `unit="pwm-fan"`, `from=0`, `to=1`, and the zone census attached.
10. **An event is emitted with no tick loop running.** The sampler is driven
    directly with no pipeline; the sink receives sample and event records.
    Pins D3's reason for putting the sampler beside the tick path.
11. **The status words are the controller's own objects.** `assert
    thermal.RULE_QUIET is sensing_controller.RULE_QUIET` -- identity, not
    equality, because equal strings hide a copy (the `is` lesson).
12. **`Inputs` is unchanged.** `[f.name for f in fields(Inputs)]` equals the
    seventeen names, in order. Pins D6, and therefore pins that every task-35 and
    task-36 log stays scoreable.
13. **`score_shadow` still scores a task-36 log after this task.** Run the tool
    over a recorded fixture and assert `replay_identity` mismatched is 0 and no
    refusal. The second half of D6, from the reader's side.
14. **The tick block does not depend on `sensing`.** Build a tick record with
    `sensing = None`; `record["thermal"]` is present and complete.
15. **`at_mono` agrees with the reference witness.** On a tick with both,
    `record["thermal"]["phone"]["at_mono"] == record["sensing"]["reference"]["at_mono"]`.
16. **`eval_run` prints the thermal section, on real record shapes.** Task 33 and
    task 36 both shipped a measurement with no surface; the test builds a
    `metadata.jsonl` with ticks, samples and one event, runs `analyze` and
    `render_markdown`, and asserts the `## Thermal` heading, a temperature, and
    the word `NOT EVALUABLE` on a fixture where the cooling devices are missing.
17. **A pre-task-37 run does not crash and does not fail.** `analyze` over a log
    with no thermal records reports `result["thermal"] is None` and
    `overall_pass` is unchanged.

Kotlin, JVM (`phone/app/src/test/.../sensors/`):

18. **The listener cannot change what is reported.** `TelemetryReporter` built
    with a watcher whose last status is `severe` and a `sample` whose
    `thermalStatus` is `nominal` emits `nominal`. **This is the direction test
    for the one behaviour change**: the listener can only add records, so it
    cannot lower `thermal_scale`, so it cannot lower `camera_hz` or `here_hz`.
19. **Each headroom absence carries its own reason.** Four cases --
    `sdkInt = 29`, `NaN`, `-1f`, `11f` -- give `api_too_old`, `not_a_number`,
    `out_of_band`, `out_of_band`, and a present value gives a null reason.
20. **Each skin-temperature absence carries its own reason.** Four fixture roots
    give the four `skin_temp_absent` members.
21. **A transition increments the count and sets the three fields.** Two
    callbacks give `thermal_status_changes == 2` and the last transition's
    `from`/`to`/`at_mono_ns`.
22. **Two transitions inside one report period are counted, and the loss is
    visible.** The count rises by 2 across one frame while only one transition is
    carried -- pins D9's stated bound rather than leaving it as prose.
23. **The teardown census includes the watcher.**
    `resourcesHeldAfterTeardown == 0` after a come-up-and-destroy that registered
    a listener; and a second test asserts the listener was actually unregistered
    (a fake `PowerManager` counts add/remove calls), because nulling the field
    and releasing the registration are different things.
24. **A non-finite value never reaches the frame.** `Double.NaN` into every new
    numeric path yields an absent field, not a `NaN` -- the frame survives.
25. **Golden frames round-trip.** `GoldenFramesTest` and
    `tests/test_transport_golden.py` both encode and decode the new
    transition-carrying frame to the recorded `frame_sha256`.

The no-rate-change contract, asserted rather than argued (task 34's round-2
method): **replay 15,000 randomized `Inputs` through the pre-task and post-task
controller trees and assert byte-identical `Decision.to_record()` and the same
md5.** Since `Inputs` gains no field and `_thermal_scale` is untouched this must
be exact, and it is the single test that would catch D1 being taken by accident.

### What asserts the boundary of the one behaviour change

The change is: the phone registers and unregisters a thermal-status listener.
Four things fence it, and each names a different way it could escape.

1. Test 18 -- the listener's value is not what is reported, so it cannot move
   `thermal_scale`.
2. Test 12 -- `Inputs` has seventeen fields, so nothing new can reach `decide`.
3. The 15,000-decision replay -- the controller's output is byte-identical.
4. Test 23 -- the registration is released, so the change does not leak a
   platform callback past teardown.

### Mutations to pin in `scripts/remutate.py`

Each names a defect a specific test above is supposed to catch. Anchors are
single, unambiguous lines.

- `thermal: an unreadable zone reports 0.0 instead of absent` -- the absent arm
  returns `celsius=0.0, basis="measured"`. Caught by 1.
- `thermal: a stale reading is reported as measured` -- drop the age comparison
  in `latest()`. Caught by 5 and 7.
- `thermal: an unobservable event stream is reported as quiet` -- the
  `not_evaluable` arm returns `RULE_QUIET` with an empty `missing`. **The task's
  own defect class turned on the task's own output**, in the shape task 34 used.
  Caught by 8.
- `thermal: the freshness bound is a typed 2.0 rather than 2 x interval` --
  caught by 7 and only by 7.
- `thermal: the event count is derived by subtraction rather than counted` --
  `count = len(samples) - quiet_samples`. This is task 34's `superseded =
  received - shown - expired` defect and the replay pilot's "count outcomes from
  records, never by subtraction"; caught by 9 and 22.
- `thermal: the tick block reads sensing.reference instead of the sampler` --
  caught by 14.
- `phone: the reported status comes from the listener, not the poll` -- the one
  mutation that would turn this task into a rate change. Caught by 18.
- `phone: a headroom absence reports one reason for all four causes` -- caught
  by 19.
- `phone: the watcher is nulled but not unregistered` -- caught by the second
  half of 23, and **not** by the census test alone, which is why 23 is two
  assertions and not one.

A note the validator should hold this plan to: **a fix is new code.** Every
round of this section has found its defects inside the previous round's fix, so
each fix gets its own mutation, not a re-run of the mutation that found it.

## Byte cost

Measured by `json.dumps` with the default separators `MetadataLogger.write` uses
(`", "` and `": "` -- they are not compact, and omitting that is a way to be
wrong by about 10 per cent), and by canonical compact JSON for the wire. **Every
field is enumerated, including the ones that feel too small to count**, because
task 36's estimate came out 26 per cent below the measurement by omitting one
field entirely.

**Base record.** Task 35 measured a mean tick record of 8,511 and 8,552 B; task
36 added a measured 852 B. So the base this is a fraction of is about
**9,363 B**.

**Per tick -- the `thermal` block, 21 named fields.** The block key itself;
`jetson.{temp_c, zone, basis, age_s, reason, zones_n}` (6);
`phone.{headroom, headroom_absent, skin_zone, skin_temp_absent, at_mono,
absent}` (6); `events.jetson.{status, count, last}` and
`events.phone.{status, count, last}` (6); plus `events.<device>.missing` on the
two not-evaluable variants (2). The three container keys `jetson`, `phone`,
`events` are counted in the measurement below rather than listed as fields.

| variant | bytes, inline including the leading `, ` |
|---|---|
| both devices healthy | **409** |
| both devices absent, both `missing` lists present | **491** |

409 B on 9,363 B is **4.37 per cent**; the absent variant is 5.24 per cent. Over
900 ticks: **368,100 B**.

**Per 1 Hz sample record**, with 9 zones and 3 cooling devices: **594 B** per
line. A 180 s drive writes 180 of them: **106,920 B**. Nine zones is a guess
(open item 1); each additional zone costs about 24 B per sample, so a 12-zone
Orin is about 666 B per line and 119,880 B per drive.

**Per event record**: phone **249 B**, Jetson **434 B** (the Jetson's carries the
zone census). At a handful of events per drive this is under 3 KB and it is
listed so that it is not omitted rather than because it matters.

**`summary["thermal"]`**: **1,086 B**, once per run.

**Per telemetry frame, on the wire** (canonical, compact, sorted). The current
frame header is 366 B. With the two always-present additions
(`thermal_headroom_absent`, `thermal_status_changes`) it is 434 B, **+68 B,
+18.6 per cent**. While a transition is being carried, adding
`thermal_change_from`, `thermal_change_to`, `thermal_change_at_mono_ns` and
`skin_temp_absent`, it is 559 B, **+193 B, +52.7 per cent**. In absolute terms
that is **+68 B/s** on a link also carrying a 5 Hz JPEG stream, so it is under a
tenth of a per cent of the uplink; the relative figure is large only because the
telemetry frame is small. The `telemetry` queue is depth 32 at 1 Hz and is
untouched.

**Drive total**, 900 ticks over 180 s with no events:
368,100 + 106,920 + 1,086 = **476,106 B, about 465 KiB**, against a base
metadata log of about 8.0 MiB -- **5.7 per cent**.

Two things this estimate could still get wrong, named in advance so the
experiment can check them rather than discover them: the zone count on this Orin
is unconfirmed and drives the sample record linearly; and the base record has
grown three times in this section, so the 4.37 per cent figure is a ratio against
a moving denominator and the absolute 409 B is the number to compare against.

## Open items

They are allowed to stay open. Unclosed open items are not defects.

1. **The Orin's zone and cooling-device names are unconfirmed.** Nothing in the
   repo records what `/sys/class/thermal` holds on `jetson-orin`, and this plan
   deliberately did not ssh to find out. `PREFERRED_ZONE_TYPES` is a guess; D11's
   fallback exists so that a wrong guess degrades to the hottest plausible zone
   rather than to nothing, and `selected_by` records which arm ran. The
   experiment settles it; if no arm produces a zone, the record says
   `no_zone_readable` and the drive says nothing about the Jetson's temperature,
   which is the correct outcome rather than a bug.
2. **Whether this moto g power supports `getThermalHeadroom` is unknown.**
   `ThermalZones.kt`'s docstring states as fact that "the moto g power this was
   written against is such a device: its thermal HAL is connected and reporting,
   and `getThermalHeadroom` still never answers" -- but that claim is prose in a
   source file, with no recorded measurement behind it in this repo.
   `thermal_headroom_absent` is what turns it into a measured fact, and if the
   drive reports `{"not_a_number": 178}` the claim is confirmed by count for the
   first time.
3. **A cooling device is not necessarily a throttle.** A `pwm-fan` rising from 0
   to 1 is a fan speeding up, which is the opposite of a throttle in effect and
   identical in shape. This task records the device `type` with every event and
   does **not** classify, because classifying needs the device. The consequence
   is that `events.jetson.status == "fired"` may mean "the fan changed speed",
   and the report line must therefore name the units rather than say "throttled".
   Building the classification is the follow-on this task's data enables.
4. **Two status transitions inside one telemetry period lose the intermediate
   one.** D9 bounds it: the count rises by more than one and no `thermal_event`
   line covers the gap, so the loss is visible in the record rather than silent.
   Closing it needs either a transition list on the frame or a faster telemetry
   period, and neither is worth it before a drive shows the case occurring.
5. **`jtop`'s `Temp cpu` and this task's zone census are not reconciled.** Both
   claim to be the Jetson's temperature and they are read by different routes at
   different rates. `jtop_available` records whether the other one was even
   running. If both produce numbers on the same drive, comparing them is free and
   is the first thing to do with the result; this task does not merge them.
6. **No gate.** D13. A gate would need either a peak-temperature threshold this
   task cannot set, or an observability gate that would fail every desk run. What
   would have to be true to gate it later: one drive establishing that the event
   stream is observable on both devices, after which
   `thermal_events_observable` becomes a legitimate gate whose failure means the
   drive cannot answer the question.
7. **The Jetson's temperature reaches no decision.** D1. If a later task makes it
   a controller input, the change lands in `_thermal_scale` and moves
   `camera_hz` and `here_hz`, and it must be planned as a rate change with its
   own direction claim -- not folded into a logging task.
8. **`SystemStatsSampler` remains untested and silent.** D14 records its
   availability and changes nothing else. Its `type: system` and
   `type: system_error` records still have no reader.
9. **The phone's transition stamp is on the phone's clock and is not
   converted.** A reader wanting the Jetson-clock instant of a phone transition
   has `t_mono` (this side's observation, up to one report period late) and
   `at_ns` on the phone's clock, and joining them needs task 33's timebase
   estimate. Deliberate: a status transition does not need millisecond accuracy,
   and converting would attach an estimate-dependent bound to a number nobody
   differences.

## Scope boundary -- what this task does not do

- It does not change any commanded sensor rate, in either direction, on either
  device. `Inputs` keeps its seventeen fields, `_thermal_scale` is not touched,
  and the 15,000-decision replay asserts the controller is byte-identical.
- It does not make the Jetson's temperature a controller input (D1, open item 7).
- It does not add, remove or re-policy a transport channel; `telemetry` stays up
  / normal / reliable / depth 32, and the change is six absent-tolerant header
  fields on a message that already carries thermal (D7).
- It does not add an encoder slot or a `field_sources` entry, so the 39-slot
  coverage identity and the missingness denominator are unchanged.
- It does not touch `score_shadow.py`; every task-35 and task-36 log stays
  scoreable, and a test proves it rather than the diff arguing it.
- It does not change `SystemStatsSampler`'s behaviour, its record shape, or its
  silence when `jtop` is missing -- it records that the silence happened.
- It does not classify a cooling device as a throttle or a fan (open item 3).
- It does not add a gate (D13, open item 6).
- It does not touch `replay_demo.py`, `scripts/run_loopback_pipeline.py` or
  `scripts/run_phone_drive.py`. They build records with no sampler and take the
  named-absent branch; a test covers one of them so the branch is exercised
  rather than assumed.
- It is not the failure-event log (task 38) and not the session summary
  generator (task 39). A GPS dropout, a HERE quota exhaustion and a transport
  stall are task 38's records even where a hot device caused them.
- It does not run the drive. The experiment is `experiment_dsrc`, separately, and
  the numbers above are estimates until it does.
