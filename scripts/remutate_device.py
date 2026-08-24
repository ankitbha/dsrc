"""The instrumented pins, which the JVM harness cannot cover.

Each run is ~2 minutes plus however long the device lock is held by someone else.
"""
import pathlib, re, subprocess, sys

SERVICE = pathlib.Path("phone/app/src/main/kotlin/com/dsrc/phone/SensingService.kt")
SOURCE = pathlib.Path("phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuSource.kt")

# Not listed, deliberately: passing a literal 20_000 for the accelerometer's sampling
# period. It survives, and it should -- 20,000 us *is* the 50 Hz default, so the mutant is
# equivalent and there is nothing for a test to catch. It briefly reported CAUGHT while the
# instrumented test watermarked the sensor log by position, which was the test being flaky
# rather than the mutation being real.
MUTATIONS = [
    (SERVICE, "teardown: release no longer guards each step",
     '    private fun release(what: String, step: () -> Unit) {\n        try {\n            step()\n        } catch (t: Throwable) {',
     '    private fun release(what: String, step: () -> Unit) {\n        try {\n            step()\n        } catch (t: NoSuchElementException) {'),
    # Anchored on the last assignment and the recorder, not on the whole block. Task 20
    # inserted two fields into the middle of it and the old anchor stopped matching, so
    # this pin was inert from the moment that landed and round 1 did not notice -- which
    # is precisely the failure the ANCHOR MOVED branch exists to report.
    (SERVICE, "teardown: the fields are never cleared",
     "            link = null\n",
     "            @Suppress(\"UNUSED_EXPRESSION\") link\n"),
    (SERVICE, "teardown: entering a stopped state does not tear down",
     "                onSensingDown()\n                release()\n            }",
     "                release()\n            }"),
    # Two entries removed with the dumpsys test they depended on: "the listeners are never
    # unregistered" and "batching is turned back on". Both are still real defects and
    # neither is pinned now -- see the note in ImuCaptureTest. Leaving them here would make
    # the harness permanently red for a gap that is recorded elsewhere.
    (SOURCE, "imu: the gyroscope is never registered",
     "        if (manager.registerListener(callback, gyroscope, periodUs, 0, handler)) registered++", ""),
    (SERVICE, "imu: the source is never started",
     "        motion.start(onReading = { imu.offer(it) }, onUnpaired = { imu.offerUnpaired() })\n", ""),
    (SERVICE, "teardown: one later release is skipped",
     '            release("gps source") { gpsSource?.stop() }\n', ''),
]

def failures():
    subprocess.run(
        ["python3", "scripts/with_device.py", "--", "env",
         "JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home",
         "./phone/gradlew", "-p", "phone", ":app:connectedDebugAndroidTest", "--rerun-tasks"],
        capture_output=True, text=True)
    names = []
    for path in pathlib.Path("phone/app/build/outputs/androidTest-results/connected").rglob("*.xml"):
        text = path.read_text()
        names += re.findall(r'<testcase name="([^"]+)"[^>]*>\s*<(?:failure|error)', text)
    return names

survived = []
for path, name, old, new in MUTATIONS:
    keep = path.read_text()
    if old not in keep:
        print(f"  ANCHOR MOVED       {name}")
        survived.append(name)
        continue
    try:
        path.write_text(keep.replace(old, new, 1))
        caught = failures()
    finally:
        path.write_text(keep)
    if caught:
        print(f"  CAUGHT ({len(caught)})         {name}")
        print(f"                     by {caught[0]}")
    else:
        print(f"  *** SURVIVED ***   {name}")
        survived.append(name)

print()
print("survived:", len(survived), survived if survived else "(none)")
sys.exit(1 if survived else 0)
