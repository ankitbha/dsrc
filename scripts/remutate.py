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
import pathlib, subprocess, sys

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
    ("stats: the per-channel map is read after the session totals",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "        val inboundChannels = inbound.counters()\n        val channels = queues.counters()",
     "        val channels = queues.counters()",
     "transport"),
    ("stats: the per-channel map is read inline, after the totals",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     "            inboundChannels = inboundChannels,",
     "            inboundChannels = inbound.counters(),",
     "transport"),
    ("stats: an outbound framing refusal counts before it describes",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     '        lastOutboundFramingRefusal = "${cause.javaClass.simpleName}: ${cause.message}"\n        outboundFramingRefusals.incrementAndGet()',
     '        outboundFramingRefusals.incrementAndGet()\n        lastOutboundFramingRefusal = "${cause.javaClass.simpleName}: ${cause.message}"',
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
     "        unpaired.incrementAndGet()\n    }",
     "    }",
     "app"),
    ("imu: the gyro age keeps the last instead of the max",
     "phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuPipeline.kt",
     "gyroAgeMaxNs.getAndUpdate { maxOf(it, reading.gyroAgeNs) }",
     "gyroAgeMaxNs.set(reading.gyroAgeNs)",
     "app"),
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

def run(kind):
    if kind == "python":
        return subprocess.run([".venv/bin/python3", "-m", "pytest", "-q", "deployment/jetson/tests/",
                               "-p", "no:cacheprovider"], capture_output=True, text=True).returncode
    target = ":transport:test" if kind == "transport" else ":app:test"
    return subprocess.run(GRADLE + [target, "--rerun-tasks"], capture_output=True, text=True).returncode

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
    try:
        path.write_text(mutated)
        code = run(kind)
    finally:
        path.write_text(keep)
    verdict = "CAUGHT" if code != 0 else "*** SURVIVED ***"
    if code == 0:
        survived.append(name)
    print(f"  {verdict:18} {name}")

print()
print("survived:", len(survived), survived if survived else "(none)")
sys.exit(1 if survived else 0)
