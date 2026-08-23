# Plan: Task 17 — Android project skeleton

> Kotlin, CameraX, permissions, foreground service.

## Short version

Stand up the Android project that tasks 18–25 will fill in: a Gradle build that
resolves on this laptop, two modules, the manifest and permission model, and a
foreground service that starts, survives, and stops cleanly. Nothing captures
anything yet. The deliverable is a project that **builds, tests and installs**,
because until that is true no later task can be verified at all.

The one decision here with consequences past task 17 is the module split:

- **`:transport`** — pure Kotlin/JVM, no Android dependency. It will hold the wire
  protocol. Created and compiling in this task, filled in with task 18, its first
  consumer. No Android in it means its tests run at JVM speed on this laptop and
  can be checked against the Python implementation directly.
- **`:app`** — the Android half. Manifest, permissions, foreground service, and
  later the sensors and UI.

This split is worth making now rather than later because it is structural: it is
what makes the cross-language contract testable without a device, and retrofitting
it after the sensors are written would mean moving every file.

### Scope boundary

**In:** Gradle wrapper and pinned toolchain, `:transport` and `:app` modules, the
manifest, the runtime-permission flow, a foreground service with its lifecycle and
notification, CameraX dependencies declared and resolving, JVM unit tests for the
logic this task introduces, an emulator install-and-launch check.

**Out:** any sensor capture, any wire format, any network connection, any UI beyond
what the service notification requires. The `:transport` module is created empty on
purpose — writing the protocol here would be task 18's work done early and
unverifiable, since nothing yet sends a frame.

### Open items

- **O1. `minSdk` is a guess about the handset.** `specs/transport_protocol.md`'s
  hello example says `moto-g-power`, a name spanning several Android versions.
  This build takes `minSdk 29`: it covers CameraX, foreground-service types and
  everything else used here, and the only system image installed is API 31 so
  tests run above it. If the phone in hand is older, this is the number to change,
  and it is the only place the guess appears.
- **O2. Foreground-service type.** Android 14+ requires a declared type and a
  matching permission. This service will eventually hold camera, location and a
  network connection at once. It declares `camera|location` and requests both
  permissions. If the handset targets below 14 this is inert; if the app is ever
  submitted to Play, `camera` is the type that draws scrutiny, and that is a
  packaging problem rather than a code one.

---

## 1. Grounding

### Toolchain, as measured on this laptop

| thing | state |
|---|---|
| Android SDK | `/opt/homebrew/share/android-commandlinetools`, licences accepted |
| platform | `android-35` only |
| build-tools | `34.0.0`, `35.0.0` |
| platform-tools | present, `adb` available |
| system image | `android-31` / `google_apis` / `arm64-v8a` |
| AVD | `dsrc_test` already exists (API 31, arm64) |
| JDK | Temurin 17.0.20 arm64 — the only JDK installed |
| Gradle on PATH | 9.7.0, launcher JVM 26 |
| device attached | none |

Two facts from that table drive the build configuration. Only platform 35 is
installed, so `compileSdk` is 35 and not a matter of preference. And the Gradle on
PATH runs on JVM 26, which no current Android Gradle Plugin supports — so the build
must use its own **wrapper** with a pinned Gradle and an explicit JDK 17, never the
system Gradle.

### What the phone will owe the protocol

Not implemented in this task, but it is why the module split exists, and it is the
reason `:transport` must not depend on Android. Read off
`specs/transport_protocol.md`: framing with big-endian lengths and pre-allocation
limit checks; canonical JSON with recursively sorted keys and no non-ASCII
escaping; five required header keys with `n` equal to the payload length; three
reserved extension keys; the phone opening the connection and both sides sending a
hello before either reads; a 1.0 s keepalive and a 5.0 s stall measured on
completed reads; strict priority across three tiers with round-robin inside one;
two overflow policies with `seq` assigned before the overflow decision; eight
channels each with exactly one message type; and a closed vocabulary of refusal
reasons where a bad message costs one message but a bad frame ends the session.

### Two conformance traps found while grounding

Recorded here because they are the evidence for D2, which is decided in this task
even though the code lands in task 18.

1. `specs/transport_golden_frames.json` case `large_ints` carries
   `9007199254740993` (2^53+1) and `-9007199254740993`. Gson and
   kotlinx.serialization both parse JSON numbers into `Double` by default, which
   silently returns `9007199254740992`. The frame then re-encodes to different
   bytes, and `seq` arithmetic on the peer is wrong.
2. Kotlin's `Double.toString()` and Python's `json.dumps` agree on **every float in
   the golden vectors** — which is what makes this dangerous rather than safe. They
   diverge outside the range those vectors sample: Kotlin emits `1.5E-5` where
   Python emits `1.5e-05`, and `1.0E7` where Python emits `10000000.0`. A near-zero
   IMU gyro reading sits squarely in the first range.

So `:transport` will carry a hand-written canonical JSON codec rather than a
library. Deciding that now is what keeps a JSON dependency out of the module's
build file today, where it would be awkward to remove later.

---

## 2. Decisions

Taken by recommendation under `plan_dsrc_rec`. **None is user sign-off.**

| # | decision | why | runner-up |
|---|---|---|---|
| D1 | Two modules: `:transport` (JVM-only) and `:app` (Android) | Makes the wire contract testable at JVM speed with no emulator, and structurally prevents Android APIs leaking into the wire format | single module; every conformance test would then need a device |
| D2 | `:transport` will use a hand-written canonical JSON codec, no JSON library | The two traps above — one corrupts 2^53+1, the other formats doubles differently from Python | kotlinx.serialization; fails both |
| D3 | Wrapper Gradle 8.9, AGP 8.7.x, Kotlin 2.0.x, JDK 17 | JDK 17 is the only one installed and the system Gradle's JVM 26 is unsupported by AGP | AGP 9 on system Gradle 9.7; unsupported launcher JVM |
| D4 | `compileSdk`/`targetSdk` 35, `minSdk` 29 | 35 is the only platform installed; 29 covers everything used and sits below the API-31 image | `minSdk` 31 to match the image exactly; needlessly narrows the handset range |
| D5 | `phone/` at the repo root, not under `deployment/jetson/` | It is not deployed to the Jetson | `deployment/phone/`; pairs oddly with a Jetson-specific sibling |
| D6 | Foreground service owns the whole sensing lifecycle | One place starts and stops everything, so there is one answer to "is sensing running" | per-sensor services; multiplies lifecycle bugs by four |
| D7 | Permission state modelled as pure logic, adapter reads the platform | Makes the grant/deny/rationale flow unit-testable on the JVM, which is where its bugs are | call `checkSelfPermission` inline; untestable off-device |
| D8 | Service exposes an observable state enum, no business logic in the Activity | The UI reads state; it does not own it. Task 23 will attach a real UI to the same enum | Activity drives the service; the two then disagree during rotation |

---

## 3. Steps

| # | step | done when |
|---|---|---|
| 1 | `phone/` skeleton: `settings.gradle.kts`, wrapper, `gradle.properties` pinning JDK 17; the SDK path lives in `local.properties`, which is machine-specific and untracked | `./gradlew projects` lists `:transport` and `:app` |
| 2 | `:transport` module, JVM-only, empty but compiling with a test source set | `./gradlew :transport:test` succeeds |
| 3 | `:app` module, AGP configured, CameraX dependencies declared | `./gradlew :app:assembleDebug` produces an APK |
| 4 | Manifest: permissions, foreground-service type, launcher activity | `aapt2`/`badging` shows the expected permissions and service |
| 5 | `PermissionModel` — pure logic over required/granted/denied/rationale | unit tests cover every transition including permanent denial |
| 6 | `SensingService` — foreground service, notification, state enum, start/stop | the state machine is unit-tested; the service itself is covered by instrumented tests, since every platform call in a JVM test returns a default |
| 7 | `MainActivity` — requests permissions, starts/stops the service, shows state | installs and launches on the AVD |
| 8 | Emulator check: boot `dsrc_test`, install, launch, start service, stop | logcat shows the service reaching `RUNNING` and returning to `IDLE` |

---

## 4. Tests

**Unit — input/output**
- `PermissionModel`: every combination of required vs granted, first denial versus
  permanent denial, and the resulting next action.
- `SensingService` state machine: the whole state x event table, written out
  explicitly rather than derived. `Stop` reaches `IDLE` from every state except
  `STOPPING`, where a teardown is already in flight.
- Build configuration: `minSdk` and `targetSdk` are the intended numbers, asserted
  from the merged manifest rather than restated in a test. `compileSdk` never appears
  in a merged manifest, so it is read from the APK's badging instead.
- Every runtime permission `PermissionModel.required()` names is declared in the
  manifest. Without this the permission constants assert against themselves, and a
  typo'd string is permanently denied at runtime with no dialog shown.

**Sanity — behaviour**
- Starting twice does not start two sessions and is not an error.
- Stopping when not running is a no-op, not a crash.
- A required permission missing when sensing starts drives the service to a named
  stopped state rather than leaving it half-running. Revocation *while already
  running* has no producer yet — the gate is checked at start only, and closing that
  gap belongs with the capture tasks that have something to tear down.
- The service survives an Activity going away — the point of a foreground service.
  Not currently asserted: it needs an Activity harness, and nothing yet launches
  `MainActivity` in a test.

**Instrumented (emulator)**
- App installs, launches, starts the service, and the notification appears.
- The real service reaches `RUNNING` in the foreground and returns to `IDLE`.
- A duplicate stop and an unknown action each leave no resident service. These have no
  JVM equivalent: a plain unit test can construct the service, but every platform call
  returns a default, so it would exercise branches the phone never takes.

---

## 5. Experiments

What this task can measure. It is a build task, so the measurements are about the
build.

1. `./gradlew :transport:test :app:check` — test count and wall time. `:app:check`
   rather than `:app:testDebugUnitTest`, because the manifest gate hangs off `check`
   and the narrower task skips it.
2. `./gradlew :app:assembleDebug` — APK produced, its size, and cold vs warm build
   time. Cold build time matters because it is paid once per task after this.
3. `aapt2 badging` on the APK — the declared permissions, `minSdk`, `targetSdk`
   and the service entry, read back from the artifact rather than from the source.
4. Emulator: install, launch, start service, stop service, uninstall — with logcat
   evidence for each.

**Not measured:** anything about sensing. Nothing senses yet.

---

## 6. Risks

- **The pinned AGP/Gradle/JDK combination may not resolve.** Most likely failure in
  this task. If it does not, the fallback is installing a second JDK rather than
  loosening the Android target, because only platform 35 is available.
- **First build downloads a lot.** AGP, Kotlin and CameraX come from Google's Maven
  and Maven Central. A network failure mid-resolve looks like a build error; it is
  worth distinguishing before debugging the build file.
- **The emulator is not a phone.** It proves the app installs, starts and holds a
  foreground service. It proves nothing about camera or sensors, and this task does
  not claim otherwise.
- **`minSdk 29` (O1) is unverified** against the actual handset.

---

## 6b. Known gaps, accepted with reasons

Found by validation and deliberately not closed in task 17. Recorded so they are
choices rather than oversights.

- **`MainActivity`'s wiring is unpinned.** The permission split is a pure, tested
  function, but the two lines that *call* it are not: reverting them — recording a
  granted permission as refused, which the split's own comment calls silent and
  expensive — passes both suites. Closing it needs an Activity harness, and nothing
  yet launches `MainActivity` in a test. Rolled into task 23, which builds the real
  UI. The same gap covers the Settings-grant reconcile and the `startRequested`
  save/restore.
- **Nothing asserts that the app launches, or that the notification appears.** The
  emulator install-and-launch check is manual. The notification assertion was
  replaced by reading `RunningServiceInfo.foreground`, which is the stronger fact but
  a different one — a service can be in the foreground with the notification
  suppressed by the user.
- **Two test seams ship in the release APK.** `permissionOverride` and
  `enterForegroundOverride` are `internal`, which is public in JVM bytecode, and are
  not gated on `BuildConfig.DEBUG`. They exist because the paths they unlock —
  a permission refusal and a failed foreground transition — are unreachable on a
  healthy emulator, and both are the fix for a real crash. Acceptable for a
  sideloaded research app; it would not be for a distributed one.
- **`stopForegroundCompat()` survives deletion.** Destroying the service also drops
  it out of the foreground. Unreachable as a lasting state today because `release()`
  is only ever called from `onStartCommand`'s call tree; it becomes load-bearing the
  moment a capture task calls `release()` from a sensor callback or a coroutine.

## 7. Needs sign-off

1. **O1** — the real `minSdk` for the phone in hand.
2. **O2** — the foreground-service type, if this app is ever to be distributed
   rather than sideloaded.
