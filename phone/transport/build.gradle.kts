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

tasks.test {
    testLogging {
        events("failed")
    }
}
