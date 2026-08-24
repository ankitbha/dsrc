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

MUTATIONS = [
    ("inbound: a delivery failure is filed as delivered",
     "phone/transport/src/main/kotlin/com/dsrc/transport/Session.kt",
     '            inbound.countFailed(frame.channel)\n            lastDeliveryFailure = "${t.javaClass.name}: ${t.message}"',
     '            inbound.countDelivered(frame.channel)\n            lastDeliveryFailure = "${t.javaClass.name}: ${t.message}"',
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
    ("frames: json accepts bare NaN again",
     "deployment/jetson/transport/frames.py",
     "header = json.loads(text, parse_constant=_reject_json_constant)",
     "header = json.loads(text)",
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
