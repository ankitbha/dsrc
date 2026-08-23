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

tasks.test {
    inputs.file(protocolSpec).withPropertyName("protocolSpec")
    systemProperty("dsrc.protocolSpec", protocolSpec.asFile.absolutePath)
    testLogging {
        events("failed")
    }
}
