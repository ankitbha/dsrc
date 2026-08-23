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
        // Supplied by the build, which also declares it as a task input so an edit to
        // the spec alone re-runs these tests. A directory walk from the working
        // directory would find the file but leave the task UP-TO-DATE.
        val path = System.getProperty("dsrc.protocolSpec")
            ?: error("dsrc.protocolSpec is not set; the build must pass the spec path")
        val file = File(path)
        require(file.isFile) { "spec not found at $path" }
        file.readText()
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

    @Test
    fun `the channel table matches the spec's table row for row`() {
        // Eight channels, each with a direction, priority, overflow policy and depth. A
        // table that drifted from the spec would put the two implementations' queues out
        // of step in a way no single-sided test could see.
        for (policy in Channels.ALL) {
            val pattern = Regex("""\|\s*`""" + policy.id + """`\s*\|([^\n]*)""")
            val row = pattern.find(spec)?.groupValues?.get(1)
                ?: error("spec has no row for channel '" + policy.id + "'")
            val cells = row.split('|').map { it.trim() }
            assertEquals(policy.direction.name.lowercase(), cells[0], policy.id + " direction")
            assertEquals(policy.priority.name.lowercase(), cells[1], policy.id + " priority")
            assertEquals(policy.overflow.name.lowercase(), cells[2], policy.id + " overflow")
            assertEquals(policy.depth.toString(), cells[3], policy.id + " depth")
        }
    }

    @Test
    fun `the spec names no channel the table is missing`() {
        // The other direction: a channel added to the spec and not here would be a
        // protocol error on every frame that used it.
        val rows = Regex("""\n\|\s*`([a-z_]+)`\s*\|\s*(?:up|down|both)\s*\|""")
            .findAll(spec)
            .map { it.groupValues[1] }
            .toSet()
        assertEquals(Channels.ALL.map { it.id }.toSet(), rows)
        assertEquals(8, rows.size)
    }

    private fun fmt(value: Double): String =
        if (value == Math.floor(value)) "%.1f".format(value) else value.toString()

    private fun numberAfter(name: String): Int {
        val match = Regex("""$name\s+(\d+)""").find(spec)
            ?: error("spec does not define $name")
        return match.groupValues[1].toInt()
    }
}
