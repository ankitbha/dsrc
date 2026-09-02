package com.dsrc.phone.log

import com.dsrc.transport.Json
import com.dsrc.transport.JsonValue
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * `FailureKinds.ALL` against the Python registry's own closed set for the
 * phone's offline-only kinds, `logio.failure_log.PHONE_OFFLINE_KINDS`.
 *
 * Executed, not asserted by hand -- the mechanism `:transport:test`'s
 * `DifferentialTest` already uses for the wire's refusal reasons, so a kind
 * added on one side and not the other fails a test rather than drifting
 * silently until an offline join finds a string neither side recognises.
 */
class FailureKindsInteropTest {

    @Test
    fun `every kind is a member of the Python registry's phone rows, and only those`() {
        val root = System.getProperty("dsrc.repoRoot") ?: error("dsrc.repoRoot is not set")
        val venv = File(root, ".venv/bin/python3")
        val interpreter = if (venv.canExecute()) venv.absolutePath else "python3"

        val snippet = """
            import json, sys
            sys.path.insert(0, "deployment/jetson")
            from logio.failure_log import PHONE_OFFLINE_KINDS
            print(json.dumps(sorted(PHONE_OFFLINE_KINDS)))
        """.trimIndent()

        val process = ProcessBuilder(interpreter, "-c", snippet)
            .directory(File(root))
            .redirectErrorStream(true)
            .start()
        val output = process.inputStream.bufferedReader().readText()
        check(process.waitFor(30, TimeUnit.SECONDS)) { "python did not finish within 30 s" }
        check(process.exitValue() == 0) { "python failed: $output" }

        val payload = output.trim().lines().last { it.startsWith("[") }
        val pythonKinds = (Json.decode(payload) as JsonValue.Arr).items
            .map { (it as JsonValue.Text).value }
            .toSet()

        assertEquals(
            "Kotlin's FailureKinds.ALL and Python's PHONE_OFFLINE_KINDS have drifted apart",
            pythonKinds, FailureKinds.ALL,
        )
    }
}
