package com.dsrc.transport

import java.io.BufferedReader
import java.io.File
import java.net.InetAddress
import java.net.Socket
import java.util.concurrent.TimeUnit
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The Kotlin transport against the live Python one.
 *
 * This is the half of the acceptance criterion the golden vectors cannot reach. They
 * pin bytes; they say nothing about whether the handshake ordering, keepalive cadence,
 * sequence numbering and drop accounting *interact* the same way on both sides. A mock
 * peer would be no help either — it would agree with whatever this side happens to do.
 * So the peer here is `scripts/interop_jetson_peer.py`, running the real
 * `deployment/jetson/transport` Session and reporting what it saw.
 */
class InteropTest {

    private var process: Process? = null
    private var socket: Socket? = null
    private var session: Session? = null

    @AfterTest
    fun cleanup() {
        runCatching { session?.close() }
        runCatching { socket?.close() }
        process?.let { p ->
            if (!p.waitFor(5, TimeUnit.SECONDS)) p.destroyForcibly()
        }
        session = null; socket = null; process = null
    }

    private val repoRoot: File by lazy {
        val path = System.getProperty("dsrc.repoRoot")
            ?: error("dsrc.repoRoot is not set; the build must pass the repo root")
        File(path).also { require(it.isDirectory) { "repo root not found at $path" } }
    }

    /**
     * Minimal JSON reading, since the codec under test is the thing being validated.
     *
     * A JSON `null` maps to Kotlin null. Returning the string "null" instead made every
     * healthy run fail an `assertNull` on the peer's `drain_error` field -- the tests
     * were red while the interop was working perfectly.
     */
    private fun field(line: String, key: String): String? {
        val match = Regex(""""$key"\s*:\s*("([^"]*)"|[^,}\s]+)""").find(line) ?: return null
        val quoted = match.groupValues[2]
        val raw = match.groupValues[1]
        if (quoted.isEmpty() && raw == "null") return null
        return quoted.ifEmpty { raw }
    }

    private class Peer(val process: Process, val stdout: BufferedReader, val port: Int)

    private fun startPeer(seconds: Double = 30.0, echoAdvisory: Boolean = false): Peer {
        val command = buildList {
            add("python3")
            add("scripts/interop_jetson_peer.py")
            add("--seconds"); add(seconds.toString())
            if (echoAdvisory) add("--echo-advisory")
        }
        val builder = ProcessBuilder(command).directory(repoRoot)
        builder.redirectErrorStream(false)
        val process = builder.start().also { this.process = it }
        val stdout = process.inputStream.bufferedReader()

        val listening = stdout.readLine()
            ?: error("the peer produced no output; stderr: ${process.errorStream.bufferedReader().readText()}")
        val port = field(listening, "port")?.toIntOrNull()
            ?: error("could not read the port from: $listening")
        return Peer(process, stdout, port)
    }

    private fun connect(peer: Peer, onFrame: (Frame) -> Unit = {}): Session {
        val socket = Socket(InetAddress.getLoopbackAddress(), peer.port).also { this.socket = it }
        val session = Session(
            input = socket.getInputStream(),
            output = socket.getOutputStream(),
            deviceId = "interop-phone",
            role = "phone",
            monoClock = { System.nanoTime() },
            wallClock = { System.currentTimeMillis() * 1_000_000 },
            onFrame = onFrame,
        ).also { this.session = it }
        session.start()
        // The peer prints "ready" once its own handshake has completed.
        val ready = peer.stdout.readLine() ?: error("the peer never reported ready")
        assertTrue(ready.contains("\"ready\""), "expected ready, got: $ready")
        assertEquals("interop-phone", field(ready, "peer_device_id"))
        assertEquals("phone", field(ready, "peer_role"))
        return session
    }

    /** Drain the queues, then wait for the peer's summary line. */
    private fun finishAndSummarise(peer: Peer, session: Session, settleMs: Long = 1_500): String {
        val deadline = System.currentTimeMillis() + 20_000
        while (session.outboundPending() > 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(20)
        }
        Thread.sleep(settleMs)
        session.close()
        val lines = mutableListOf<String>()
        while (true) {
            val line = peer.stdout.readLine() ?: break
            lines.add(line)
            if (line.contains("\"summary\"")) break
        }
        return lines.lastOrNull { it.contains("\"summary\"") }
            ?: error("no summary from the peer; saw: $lines\nstderr: ${peer.process.errorStream.bufferedReader().readText()}")
    }

    @Test
    fun `the two implementations complete a handshake`() {
        val peer = startPeer(seconds = 15.0)
        val session = connect(peer)
        assertEquals("jetson", session.peer?.role)
        assertEquals(Protocol.VERSION.toLong(), session.peer?.protocolVersion)
        assertEquals("interop-jetson", session.peer?.deviceId)
    }

    @Test
    fun `gps records cross to the python side intact and in order`() {
        val peer = startPeer(seconds = 25.0)
        val session = connect(peer)

        val count = 40
        repeat(count) { i ->
            val record = GpsRecord(
                captureMonoNs = 1_000_000L + i,
                valid = true,
                latitude = 51.5074,
                longitude = -0.1278,
                speedMps = 13.4,
                headingDeg = 91.2,
                fixQuality = 1,
                satellites = 9,
                hdop = 0.9,
                altitudeM = 35.0,
                utcEpochNs = 1_755_648_000_000_000_000,
            )
            assertTrue(session.send(Channels.GPS, record.toExtensions()), "send $i was refused")
        }

        val summary = finishAndSummarise(peer, session)
        assertNull(field(summary, "drain_error"), "the peer's drain failed: $summary")

        val sent = session.stats().channels.getValue(Channels.GPS).sent
        val received = field(summary, "frames_received")?.toIntOrNull() ?: -1
        assertEquals(sent, received.toLong(), "python received $received of $sent written: $summary")
        assertEquals(count.toLong(), sent, "nothing should be dropped at depth 64: $summary")

        // Sequence numbers are the peer's evidence about drops, so they matter as much as
        // the payloads.
        assertTrue(summary.contains("\"monotonic_seq\": {\"gps\": true}"), summary)
        assertTrue(summary.contains("\"first_seq\": {\"gps\": 0}"), summary)
    }

    @Test
    fun `python decodes our gps fields, not merely our bytes`() {
        // A frame can arrive intact and still be rejected by the far side's message
        // layer. The peer's counters distinguish the two: `received` counts arrivals,
        // `delivered` counts messages its decoder accepted.
        val peer = startPeer(seconds = 20.0)
        val session = connect(peer)
        repeat(10) { i -> session.send(Channels.GPS, GpsRecord.noFix(i.toLong()).toExtensions()) }
        val summary = finishAndSummarise(peer, session)

        assertNull(field(summary, "drain_error"), summary)
        val gps = Regex(""""gps"\s*:\s*\{([^}]*)}""").findAll(summary)
            .map { it.groupValues[1] }
            .firstOrNull { it.contains("received") }
            ?: error("no gps channel counters in: $summary")
        assertTrue(gps.contains("\"received\": 10"), "python counted: $gps")
        assertTrue(gps.contains("\"delivered\": 10"), "python's decoder rejected some: $gps")
        assertTrue(gps.contains("\"dropped_inbound\": 0"), "python dropped some: $gps")
    }

    @Test
    fun `a camera payload crosses byte for byte`() {
        val peer = startPeer(seconds = 20.0)
        val session = connect(peer)
        val payload = ByteArray(40_960) { ((it * 37 + 11) % 256).toByte() }
        assertTrue(
            session.send(
                Channels.CAMERA,
                mapOf(
                    Fields.CAPTURE_KEY to JsonValue.Num(1_000_000_000),
                    "frame_id" to JsonValue.Num(1841),
                    "width" to JsonValue.Num(1280),
                    "height" to JsonValue.Num(720),
                    "format" to JsonValue.Text("jpeg"),
                    "quality" to JsonValue.Num(85),
                ),
                payload,
            )
        )
        val summary = finishAndSummarise(peer, session)
        assertNull(field(summary, "drain_error"), summary)
        assertTrue(summary.contains("\"camera\": 1"), "the camera frame did not arrive: $summary")
    }

    @Test
    fun `keepalives are exchanged in both directions`() {
        val peer = startPeer(seconds = 20.0)
        val session = connect(peer)
        // Long enough for both sides to fire at a 1 s interval.
        Thread.sleep(3_000)
        val summary = finishAndSummarise(peer, session, settleMs = 300)

        assertTrue(
            (field(summary, "heartbeats_received")?.toIntOrNull() ?: 0) >= 2,
            "python saw too few of our keepalives: $summary",
        )
        assertTrue(
            session.stats().heartbeatsReceived >= 2,
            "we saw ${session.stats().heartbeatsReceived} of python's keepalives",
        )
    }

    @Test
    fun `an advisory from python is delivered to us`() {
        // The downlink direction, which nothing else here exercises.
        val received = mutableListOf<Frame>()
        val peer = startPeer(seconds = 20.0, echoAdvisory = true)
        val session = connect(peer) { synchronized(received) { received.add(it) } }
        repeat(5) { i -> session.send(Channels.GPS, GpsRecord.noFix(i.toLong()).toExtensions()) }

        val deadline = System.currentTimeMillis() + 10_000
        while (System.currentTimeMillis() < deadline && synchronized(received) { received.size } < 1) {
            Thread.sleep(20)
        }
        val advisories = synchronized(received) { received.filter { it.channel == Channels.ADVISORY } }
        assertTrue(advisories.isNotEmpty(), "no advisory arrived; got ${received.map { it.channel }}")
        // And it decodes: the fields python sent are the ones the spec names.
        val header = advisories.first().header.entries
        assertTrue("rec_speed_mps" in header, "advisory header: ${header.keys}")
        assertEquals("mph", (header["units"] as? JsonValue.Text)?.value)
    }
}
