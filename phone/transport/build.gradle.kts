plugins {
    alias(libs.plugins.kotlin.jvm)
}

// Deliberately no Android dependency. This module is the phone's half of the
// cross-language wire contract in specs/transport_protocol.md, and keeping the
// platform out of it is what lets its tests run on a laptop at JVM speed and be
// checked directly against the Python implementation.
//
// No JSON library either. specs/transport_golden_frames.json carries 2^53+1, which
// Gson and kotlinx.serialization both round to 2^53 because they parse numbers as
// Double; and neither formats doubles the way Python's json.dumps does. The codec
// is written by hand in this module for both reasons.

kotlin {
    jvmToolchain(17)
}

dependencies {
    testImplementation(libs.junit)
    testImplementation(kotlin("test"))
}

// The spec is a real input to these tests: ProtocolSpecTest exists to fail when the
// spec and the Kotlin constants drift apart. Without declaring it, Gradle sees no
// change when only the spec is edited and reports the task UP-TO-DATE -- so the one
// direction the test exists for passed silently. Passing the path as a system
// property also removes a directory walk that depended on the working directory.
val protocolSpec = rootProject.layout.projectDirectory.file("../specs/transport_protocol.md")
val goldenFrames = rootProject.layout.projectDirectory.file("../specs/transport_golden_frames.json")

tasks.test {
    inputs.file(protocolSpec).withPropertyName("protocolSpec")
    systemProperty("dsrc.protocolSpec", protocolSpec.asFile.absolutePath)
    inputs.file(goldenFrames).withPropertyName("goldenFrames")
    systemProperty("dsrc.goldenFrames", goldenFrames.asFile.absolutePath)

    // Forwarded, not read from the test JVM's own properties: a -D on the Gradle command
    // line reaches the daemon, not the forked test process, so the measurement class
    // silently skipped and the run passed with no output at all.
    systemProperty("dsrc.measure", providers.systemProperty("dsrc.measure").getOrElse("false"))
    testLogging { showStandardStreams = true }

    // The interop test spawns the real Python peer. Both the script and the Python
    // transport package it imports are declared inputs, so a change to either re-runs
    // the one test that can catch the two implementations drifting apart.
    val repoRoot = rootProject.layout.projectDirectory.dir("..")
    systemProperty("dsrc.repoRoot", repoRoot.asFile.absolutePath)
    inputs.file(repoRoot.file("scripts/interop_jetson_peer.py")).withPropertyName("interopPeer")
    // The third Python input, and it was missed when round 6 added it. Without it,
    // `:transport:test` reports UP-TO-DATE after this script changes -- so an edit that
    // breaks the reconciliation passes the ordinary Gradle path in 332 ms having executed
    // nothing. That is the same "a filtered run reported BUILD SUCCESSFUL without running"
    // trap, on the default path rather than in a mutation loop.
    inputs.file(repoRoot.file("scripts/refusal_reasons.py")).withPropertyName("refusalReasons")
    inputs.dir(repoRoot.dir("deployment/jetson/transport")).withPropertyName("pythonTransport")
    testLogging {
        events("failed")
    }
}
