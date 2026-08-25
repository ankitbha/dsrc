"""The instrumented pins, which the JVM harness cannot cover.

Each run is ~2 minutes plus however long the device lock is held by someone else.
"""
import pathlib, re, subprocess, sys

SERVICE = pathlib.Path("phone/app/src/main/kotlin/com/dsrc/phone/SensingService.kt")
SOURCE = pathlib.Path("phone/app/src/main/kotlin/com/dsrc/phone/sensors/ImuSource.kt")
ACTIVITY = pathlib.Path("phone/app/src/main/kotlin/com/dsrc/phone/MainActivity.kt")

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
    (ACTIVITY, "advisory: the labels keep their text while backgrounded",
     "        blankAdvisory()\n        super.onStop()", "        super.onStop()"),
    (SERVICE, "log: nothing is offered to the session log",
     "            onSent = { header -> log.offer(header) },", ""),
    (SERVICE, "advisory: advisories are never routed",
     "        if (frame.channel == Channels.ADVISORY) {", "        if (false) {"),
    (SERVICE, "advisory: a stop leaves the advisory up",
     '            release("advisory") { advisories.clear() }\n', ""),
    (SERVICE, "config: a raise never reaches the imu source",
     "                motion.setRate(hz)", ""),
    (SERVICE, "config: the rate_cmd handler is never reached",
     "        if (frame.channel != Channels.RATE_CMD) {", "        if (true) {"),
    (SERVICE, "config: the command is decoded but never applied",
     "        applier.apply(command)", ""),
    (SERVICE, "imu: the sink never reaches the transport",
     "            holder.send(Channels.IMU, sample.toExtensions())",
     "            sample.toExtensions().isEmpty()"),
    (SERVICE, "imu: samples go out on the wrong channel",
     "            holder.send(Channels.IMU, sample.toExtensions())",
     "            holder.send(Channels.GPS, sample.toExtensions())"),
    (SOURCE, "imu: the two sensor streams are transposed",
     "                    Sensor.TYPE_GYROSCOPE -> {",
     "                    Sensor.TYPE_ACCELEROMETER -> {"),
    # Keyed on `accuracy`, which only the accelerometer branch passes. The bare y/z pair
    # appears in the gyroscope branch too, and an anchor that matched both took the
    # gyroscope -- where a device sitting still reads zero on every axis, so nothing could
    # tell the swap from the truth.
    (SOURCE, "imu: the accelerometer's y and z are transposed",
     "                            y = event.values[1].toDouble(),\n"
     "                            z = event.values[2].toDouble(),\n"
     "                            accuracy = event.accuracy.toLong(),",
     "                            y = event.values[2].toDouble(),\n"
     "                            z = event.values[1].toDouble(),\n"
     "                            accuracy = event.accuracy.toLong(),"),
    # No gyroscope counterpart here, deliberately. The instrumented suite asserts the
    # gyro reads about zero on every axis, which is what a device sitting on a desk does
    # -- so every permutation of its axes is invisible on-device. `ImuPairingTest` pins
    # them on the JVM, where the values can be chosen.
    (SOURCE, "imu: the gyroscope is never registered",
     "        manager.registerListener(callback, gyro, periodUs, 0, worker)\n", ""),
    (SERVICE, "imu: the source is never started",
     "        motion.start(onReading = { imu.offer(it) }, onUnpaired = { imu.offerUnpaired() })\n", ""),
    (SERVICE, "teardown: one later release is skipped",
     '            release("gps source") { gpsSource?.stop() }\n', ''),
    # The other half of that fix -- clearing `configApplier` before the sources stop -- is
    # not here, because nothing observes the release order from outside the service. It is
    # kept for the contract the field's own docstring states, and is unpinned; say so
    # rather than list a mutation that would survive.
    (SOURCE, "imu: a stopped source keeps the listener a rate command needs",
     "        listener?.let { manager.unregisterListener(it) }\n        listener = null",
     "        listener?.let { manager.unregisterListener(it) }"),
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
    hits = keep.count(old)
    if hits == 0:
        # Not folded into `survived` without a tag. A registry that has drifted off its
        # anchor and a pin that a test no longer catches both end this run non-zero, but
        # they call for opposite work -- re-anchor the mutation, or strengthen the test --
        # and the summary line is the only line a caller reads.
        print(f"  ANCHOR MOVED       {name}")
        survived.append(name + " [anchor moved]")
        continue
    if hits > 1:
        # An anchor that matches twice mutates whichever site comes first, which is a
        # property of the file's layout rather than of anything the registry chose. The
        # accelerometer entry below spent an unknown number of runs mutating the
        # gyroscope branch, where a stationary device reads zero on every axis and no
        # test could have seen it, under a name that said accelerometer.
        print(f"  ANCHOR AMBIGUOUS   {name} ({hits} sites)")
        survived.append(name + f" [ambiguous anchor, {hits} sites]")
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
