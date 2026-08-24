"""The instrumented pins, which the JVM harness cannot cover.

Each run is ~2 minutes plus however long the device lock is held by someone else.
"""
import pathlib, re, subprocess, sys

SERVICE = pathlib.Path("phone/app/src/main/kotlin/com/dsrc/phone/SensingService.kt")

MUTATIONS = [
    ("teardown: release no longer guards each step",
     '    private fun release(what: String, step: () -> Unit) {\n        try {\n            step()\n        } catch (t: Throwable) {',
     '    private fun release(what: String, step: () -> Unit) {\n        try {\n            step()\n        } catch (t: NoSuchElementException) {'),
    ("teardown: the seven fields are never cleared",
     "            cameraSource = null\n            gpsSource = null\n            pipeline = null\n            gpsPipeline = null\n            frameSender = null\n            encodeExecutor = null\n            link = null\n            resourcesHeldAfterTeardown",
     "            resourcesHeldAfterTeardown"),
    ("teardown: entering a stopped state does not tear down",
     "                onSensingDown()\n                release()\n            }",
     "                release()\n            }"),
    ("teardown: one later release is skipped",
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
for name, old, new in MUTATIONS:
    keep = SERVICE.read_text()
    if old not in keep:
        print(f"  ANCHOR MOVED       {name}")
        survived.append(name)
        continue
    try:
        SERVICE.write_text(keep.replace(old, new, 1))
        caught = failures()
    finally:
        SERVICE.write_text(keep)
    if caught:
        print(f"  CAUGHT ({len(caught)})         {name}")
        print(f"                     by {caught[0]}")
    else:
        print(f"  *** SURVIVED ***   {name}")
        survived.append(name)

print()
print("survived:", len(survived), survived if survived else "(none)")
sys.exit(1 if survived else 0)
