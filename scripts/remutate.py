"""Re-apply every mutation used as a pin, and check each is still caught.

A mutation test proves something on the day it is run and not after. The reason
this exists is a case where it stopped being true: a test written for the
Failed-from-RUNNING route was driven by a throwing status listener, and a later,
unrelated fix -- containing listener exceptions -- closed the door that trigger
used. The test kept passing. Mutating the teardown away left all 51 instrumented
tests green, and nothing announced that the pin had lapsed.

So each entry below names a defect that a specific test is supposed to catch. Run
it after landing a batch of fixes, not only after landing one.

    python3 scripts/remutate.py

An anchor that no longer matches is reported as a failure rather than skipped:
the code moved, and whether the pin survived the move is exactly the open
question. Every mutation is restored in a `finally`, including on Ctrl-C -- but
this edits files in place, so do not run it with uncommitted work you would mind
losing, and never point it at a tree a validator is reading.
"""
import pathlib, shutil, subprocess, sys
from xml.etree import ElementTree

ROOT = pathlib.Path(".")
GRADLE = ["env", "JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home",
          "./phone/gradlew", "-p", "phone"]

# Deliberately absent: "deliver decodes a control frame twice"
# (`if (frame.channel != Channels.CONTROL)` -> `if (true)`).
#
# It has no behavioural observable, and I looked for one rather than assuming. Both
# routes refuse the same frames with the same reasons, because checkInbound's control
# entry *is* TimeSyncMessage.fromWire. The two things checkInbound would add back --
# checkAllFinite over the whole extension map, and the payload rule -- are respectively
# unreachable inbound (both parsers refuse every non-finite literal at the framing layer,
# so no frame carrying one can arrive) and already enforced by fromWire itself.
#
# What restoring the double decode actually does is make handleTimeSync's own
# `catch (e: MessageError)` dead, because checkInbound's refusal fires first. So it is
# guarded here, indirectly but genuinely: the entry below named "timebase: a decode
# failure is passed through, not refused" starts SURVIVING the moment the double decode
# comes back. An entry that asserts nothing directly is worse than a note saying which
# entry covers it.
MUTATIONS = [
    ("inbound: a delivery failure is filed as delivered",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     '            lastDeliveryFailure = "${t.javaClass.name}: ${t.message}"\n            deliveryFailures.incrementAndGet()\n            inbound.countFailed(frame.channel)',
     '            lastDeliveryFailure = "${t.javaClass.name}: ${t.message}"\n            deliveryFailures.incrementAndGet()\n            inbound.countDelivered(frame.channel)',
     "transport"),
    # One entry, not two. The other spelling of this deleted a declaration that is still
    # referenced below it, so it never compiled -- and a build that fails to compile exits
    # non-zero, which the old exit-code scoring counted as CAUGHT. It measured the Kotlin
    # compiler and reported a pin.
    ("stats: the per-channel map is read inline, after the totals",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "            inboundChannels = inboundChannels,",
     "            inboundChannels = inbound.counters(),",
     "transport"),
    # Replaced the write-ordering entry. That one was inherently probabilistic -- the
    # window it opened is nanoseconds wide, so it reported SURVIVED one run and CAUGHT the
    # next with nothing changed, and a harness cannot tell a lapsed pin from an unlucky
    # one. The two fields are one volatile record now, so there is no ordering left to get
    # wrong; what is worth pinning is that the count is reported at all.
    ("stats: an outbound framing refusal is counted but never reported",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "            outboundFramingRefusals = framing?.count ?: 0,",
     "            outboundFramingRefusals = 0,",
     "transport"),
    ("inbound: depth() returns the total across channels",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Queues.kt",
     "    fun depth(channel: String): Long = synchronized(lock) { queues.getValue(channel).size.toLong() }",
     "    fun depth(channel: String): Long = synchronized(lock) { queues.values.sumOf { it.size.toLong() } }",
     "transport"),
    ("timebase: a decode failure is passed through, not refused",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "        } catch (e: MessageError) {\n            countInboundRefusal(frame.channel, e.reason.wire)\n            return TimeSyncOutcome.REFUSED\n        }",
     "        } catch (e: MessageError) {\n            return TimeSyncOutcome.NOT_OURS\n        }",
     "transport"),
    ("timebase: the wrong-direction pong gets a decode reason",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "            countInboundRefusal(frame.channel, RefusalReason.UNKNOWN_VALUE.wire)\n            return TimeSyncOutcome.REFUSED\n        }\n        // Wire-stamped",
     "            countInboundRefusal(frame.channel, RefusalReason.WRONG_TYPE.wire)\n            return TimeSyncOutcome.REFUSED\n        }\n        // Wire-stamped",
     "transport"),
    ("camera: a rejected submit escapes offer",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/CameraPipeline.kt",
     "            abandoned.incrementAndGet()\n            return false\n        }\n        return true",
     "            throw t\n        }\n        return true",
     "app"),
    ("frame: hashCode is a constant",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/CapturedFrame.kt",
     "    override fun hashCode(): Int {",
     "    override fun hashCode(): Int {\n        if (true) return 0",
     "app"),
    ("status: listener delivery is unguarded",
     "phone/app/src/main/kotlin/com/dsrc/phone/SensingStatus.kt",
     "        try {\n            listener.onState(state)\n        } catch (t: Throwable) {",
     "        try {\n            listener.onState(state)\n        } catch (t: NoSuchElementException) {",
     "app"),
    ("imu: an unpaired accelerometer event is not counted",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPipeline.kt",
     "        unpaired++\n    }",
     "    }",
     "app"),
    ("imu: the gyro age keeps the last instead of the max",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPipeline.kt",
     "gyroAgeMaxNs = maxOf(gyroAgeMaxNs, reading.gyroAgeNs)",
     "gyroAgeMaxNs = reading.gyroAgeNs",
     "app"),
    ("imu: an accelerometer axis is transposed on the way to the sample",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPipeline.kt",
     "            az = reading.az,", "            az = reading.ay,",
     "app"),
    ("imu: a gyro axis is transposed on the way to the reading",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "                gx = gx, gy = gy, gz = gz,", "                gx = gx, gy = gz, gz = gz,",
     "app"),
    ("imu: the gyro stream is not gated on the timebase",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "        if (!checkTimebase(captureNs, appNowNs, monoNowNs)) {\n            refusedWrongTimebase.incrementAndGet()\n            return\n        }",
     "", "app"),
    ("imu: the delivery bound moves",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "        const val MAX_PLAUSIBLE_DELIVERY_NS = 2_000_000_000L",
     "        const val MAX_PLAUSIBLE_DELIVERY_NS = 1_000_000_000L", "app"),
    ("imu: the timebase gate is skipped entirely",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "        if (!checkTimebase(captureNs, appNowNs, monoNowNs)) {",
     "        if (false) {",
     "app"),
    ("imu: the moot branch uses the delivery bound again",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "if (clockGapNs <= MAX_TOLERABLE_CLOCK_GAP_NS) {",
     "if (clockGapNs <= maxDeliveryNs) {",
     "app"),
    ("imu: the first gyro reading is kept, not the latest",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "        gyroNs = captureNs\n        hasGyro = true",
     "        if (!hasGyro) gyroNs = captureNs\n        hasGyro = true",
     "app"),
    ("imu: a negative gyro age keeps its sign",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPairing.kt",
     "                gyroAgeNs = if (age < 0) -age else age,",
     "                gyroAgeNs = age,",
     "app"),
    ("imu: the monotonic baseline advances only on accepted",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPipeline.kt",
     "        val previous = lastCaptureNs.getAndSet(reading.captureMonoNs)",
     "        val previous = lastCaptureNs.get()",
     "app"),
    ("frames: json accepts bare NaN again",
     "deployment/jetson/transport/frames.py",
     "            parse_constant=_reject_json_constant,",
     "",
     "python"),
    ("messages: require_str loses its null clause",
     "deployment/jetson/transport/messages.py",
     '    if not isinstance(value, str):\n        raise MessageError(f"{field} is {type(value).__name__}, expected str", REASON_WRONG_TYPE)',
     '    if value is None or not isinstance(value, str):\n        raise MessageError(f"{field} is {type(value).__name__}, expected str", REASON_WRONG_TYPE)',
     "python"),
]

RESULTS = {
    "transport": [ROOT / "phone/transport/build/test-results"],
    "app": [ROOT / "phone/app/build/test-results"],
    "python": [ROOT / "build/pytest-results"],
}


def failing_tests(kind):
    """Names of the tests that failed, from the JUnit XML the run just wrote."""
    # Parsed as XML, not by regex over the attributes. Gradle writes `name` first and
    # pytest writes `classname` first, so a pattern that fixes the order silently matches
    # nothing on one of the two -- which is how the first version of this reported both
    # Python pins as SURVIVED while they were being caught perfectly well. A harness that
    # reports a false SURVIVED is the same failure as one that reports a false CAUGHT.
    names = []
    for base in RESULTS[kind]:
        for report in base.rglob("*.xml"):
            try:
                root = ElementTree.parse(report).getroot()
            except ElementTree.ParseError:
                continue
            for case in root.iter("testcase"):
                if case.find("failure") is None and case.find("error") is None:
                    continue
                cls = (case.get("classname") or "").rsplit(".", 1)[-1]
                names.append(f"{cls}.{case.get('name')}" if cls else str(case.get("name")))
    return names


BUILD_ERROR = ["<the mutation did not compile>"]


def run(kind):
    """Run the suite and report *which* tests failed, not merely that something did.

    Keying on the process exit code was not enough, and the reason is the whole
    argument for this script existing. A run reported SURVIVED for the outbound
    framing-refusal pin while the same mutation, applied by hand and run against the
    same test, failed 5 times out of 5. Either verdict might have been the true one and
    the harness could not say which -- so a tool built because pins lapse silently was
    itself producing a verdict nobody could check.

    Naming the failing test settles it. A mutation that is caught says which assertion
    caught it, and one that is "caught" by an unrelated failure is now visible as such
    instead of counting as a pass.
    """
    for base in RESULTS[kind]:
        if base.exists():
            shutil.rmtree(base)          # or a previous run's failures count as this one's
    if kind == "python":
        subprocess.run(
            [".venv/bin/python3", "-m", "pytest", "-q", "deployment/jetson/tests/",
             "-p", "no:cacheprovider", f"--junit-xml={RESULTS['python'][0]}/results.xml"],
            capture_output=True, text=True,
        )
    else:
        target = ":transport:test" if kind == "transport" else ":app:test"
        result = subprocess.run(GRADLE + [target, "--rerun-tasks"], capture_output=True, text=True)
        if "e: file://" in result.stdout or "e: file://" in result.stderr:
            return BUILD_ERROR
    return failing_tests(kind)

# A killed run used to leave its mutation in the tree. The `finally` below restores on an
# exception and on Ctrl-C, and not on a SIGKILL -- and one of those left
# `UNKNOWN_VALUE -> WRONG_TYPE` applied to Session.kt, which then surfaced as two unrelated
# tests failing in a later run. Chasing that as a regression is exactly the wrong trail.
#
# So the pristine text goes to a sidecar first. If it exists on startup the previous run
# died mid-mutation, and this restores it before doing anything else rather than mutating
# an already-mutated file.
SIDECAR = pathlib.Path(".remutate-restore")

if SIDECAR.exists():
    saved = SIDECAR.read_text().split("\n", 1)
    target = ROOT / saved[0]
    print(f"  a previous run died mid-mutation; restoring {saved[0]}")
    target.write_text(saved[1])
    SIDECAR.unlink()

survived = []
for name, rel, old, new, kind in MUTATIONS:
    path = ROOT / rel
    keep = path.read_text()
    if old not in keep:
        print(f"  SKIP  {name}  (anchor not found -- the code moved)")
        survived.append(name + " [anchor moved]")
        continue
    # For require_str the null clause must also go, or the mutation is a no-op.
    mutated = keep.replace(old, new, 1)
    if "require_str" in name:
        mutated = mutated.replace(
            '    if value is None:\n        raise MessageError(f"{field} must not be null", REASON_NULL_NOT_ALLOWED)\n    if value is None or not isinstance(value, str):',
            '    if value is None or not isinstance(value, str):', 1)
    SIDECAR.write_text(f"{rel}\n{keep}")
    try:
        path.write_text(mutated)
        failed = run(kind)
    finally:
        path.write_text(keep)
        SIDECAR.unlink(missing_ok=True)
    if failed == BUILD_ERROR:
        # Distinct from both verdicts. A mutation that does not compile proves nothing
        # about any test, and reporting it as CAUGHT is how one of these entries came to
        # measure the compiler for weeks.
        survived.append(name + " [did not compile]")
        print(f"  DID NOT COMPILE    {name}")
    elif failed:
        print(f"  CAUGHT ({len(failed)})         {name}")
        print(f"                     by {failed[0]}")
    else:
        survived.append(name)
        print(f"  *** SURVIVED ***   {name}")

print()
print("survived:", len(survived), survived if survived else "(none)")
sys.exit(1 if survived else 0)
