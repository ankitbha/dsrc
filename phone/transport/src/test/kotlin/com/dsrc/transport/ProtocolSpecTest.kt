package com.dsrc.transport

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The Kotlin constants and specs/transport_protocol.md drifting apart is a real
 * failure, not a docs nit -- the Python side has the same test for the same reason.
 * Two implementations of one wire format only stay compatible if both are pinned to
 * the written contract rather than to each other.
 */
class ProtocolSpecTest {

    private val spec: String by lazy {
        var dir: File? = File(".").absoluteFile
        while (dir != null) {
            val candidate = File(dir, "specs/transport_protocol.md")
            if (candidate.isFile) return@lazy candidate.readText()
            dir = dir.parentFile
        }
        error("could not find specs/transport_protocol.md above ${File(".").absolutePath}")
    }

    @Test
    fun `protocol version matches the spec`() {
        assertTrue(
            spec.contains("`PROTOCOL_VERSION = ${Protocol.VERSION}`"),
            "spec does not state PROTOCOL_VERSION = ${Protocol.VERSION}",
        )
    }

    @Test
    fun `frame limits match the spec`() {
        assertEquals(numberAfter("MAX_PAYLOAD_BYTES"), Protocol.MAX_PAYLOAD_BYTES)
        assertEquals(numberAfter("MAX_HEADER_BYTES"), Protocol.MAX_HEADER_BYTES)
    }

    @Test
    fun `payload limit is four mebibytes and header limit is under the wire field`() {
        assertEquals(4 * 1024 * 1024, Protocol.MAX_PAYLOAD_BYTES)
        // header_len is a uint16, so a limit at or above 65536 would be unrepresentable.
        assertTrue(Protocol.MAX_HEADER_BYTES < 65536)
    }

    @Test
    fun `read chunk and session timers match the spec`() {
        assertTrue(
            spec.contains("**${Protocol.MAX_READ_BYTES} bytes**"),
            "spec does not state the ${Protocol.MAX_READ_BYTES}-byte read bound",
        )
        assertTrue(spec.contains("**${fmt(Protocol.KEEPALIVE_INTERVAL_S)} s**"))
        assertTrue(spec.contains("**${fmt(Protocol.STALL_TIMEOUT_S)} s**"))
    }

    @Test
    fun `the stall timeout is a multiple of the keepalive interval`() {
        // Otherwise a session could be declared stalled with a keepalive in flight.
        val ratio = Protocol.STALL_TIMEOUT_S / Protocol.KEEPALIVE_INTERVAL_S
        assertTrue(ratio >= 2.0, "stall timeout must allow at least two keepalives")
        assertEquals(ratio, Math.floor(ratio))
    }

    private fun fmt(value: Double): String =
        if (value == Math.floor(value)) "%.1f".format(value) else value.toString()

    private fun numberAfter(name: String): Int {
        val match = Regex("""$name\s+(\d+)""").find(spec)
            ?: error("spec does not define $name")
        return match.groupValues[1].toInt()
    }
}
