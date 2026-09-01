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

    /**
     * The peer's summary, parsed rather than pattern-matched.
     *
     * The substring checks this replaces were the weak link. A `contains` on
     * `"first_seq": {"gps": 0}` depends on Python's default `json.dumps` spacing, says
     * nothing about any other channel, and cannot reach a nested counter at all -- so the
     * "counters compared field by field" the plan promised was never happening.
     *
     * Parsed with our own decoder, which is not circular: the golden vectors pin that
     * decoder against frozen Python bytes in both directions, so it is validated
     * independently of anything this file asserts.
     */
    private fun summaryOf(line: String): Map<String, JsonValue> =
        (Json.decode(line) as? JsonValue.Obj)?.entries ?: error("summary is not an object: $line")

    private fun Map<String, JsonValue>.obj(key: String): Map<String, JsonValue> =
        (this[key] as? JsonValue.Obj)?.entries ?: error("'$key' is not an object in $this")

    private fun Map<String, JsonValue>.num(key: String): Long =
        (this[key] as? JsonValue.Num)?.value ?: error("'$key' is not an integer in $this")

    private fun Map<String, JsonValue>.text(key: String): String? =
        when (val value = this[key]) {
            is JsonValue.Text -> value.value
            null, is JsonValue.Null -> null
            else -> error("'$key' is not a string in $this")
        }

    /** One channel's counters, from the peer's own ChannelStats. */
    private fun Map<String, JsonValue>.channel(name: String): Map<String, JsonValue> =
        obj("channels").obj(name)

    private class Peer(val process: Process, val stdout: BufferedReader, val port: Int)

    private fun startPeer(
        seconds: Double = 30.0,
        echoAdvisory: Boolean = false,
        quietDrain: Boolean = false,
        sendMalformed: Boolean = false,
        sendFramingError: Boolean = false,
    ): Peer {
        val command = buildList {
            add("python3")
            add("scripts/interop_jetson_peer.py")
            add("--seconds"); add(seconds.toString())
            if (echoAdvisory) add("--echo-advisory")
            if (quietDrain) add("--quiet-drain")
            if (sendMalformed) add("--send-malformed")
            if (sendFramingError) add("--send-framing-error")
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
            onFrame = { frame, _, _ -> onFrame(frame) },
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
    fun `a thousand gps records cross with both sides counters agreeing field by field`() {
        // The plan's step 8 promised a thousand frames and a field-by-field comparison. It
        // was forty frames and three substring checks against Python's json.dumps spacing.
        val peer = startPeer(seconds = 90.0, quietDrain = true)
        val session = connect(peer)

        val count = 1_000
        repeat(count) { i ->
            // Paced. `gps` is reliable at depth 64 and a thousand unpaced sends outrun the
            // writer by a wide margin -- the first attempt dropped 126 of them -- so an
            // unpaced run cannot make a claim about clean delivery. Overflow gets its own
            // test, deliberately.
            while (session.outboundPending() > 32) Thread.sleep(1)
            // And a breath every twenty-five, because pacing our own queue says nothing
            // about the peer's: all thousand arrived, and 616 of them were shed by
            // *Python's* inbound queue at depth 64 before its drain reached them.
            if (i > 0 && i % 25 == 0) Thread.sleep(5)
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

        val summary = summaryOf(finishAndSummarise(peer, session, settleMs = 3_000))
        assertNull(summary.text("drain_error"), "the peer's drain failed: $summary")

        val ours = session.stats().channels.getValue(Channels.GPS)
        val theirs = summary.channel(Channels.GPS)

        assertEquals(count.toLong(), ours.sent, "we did not write them all: $ours")
        assertEquals(0L, ours.dropped, "we dropped some: $ours")

        // The interop claim: every frame we wrote arrived, and none was lost on the wire.
        assertEquals(ours.sent, theirs.num("received"), "arrivals disagree: $ours vs $theirs")
        assertEquals(0L, theirs.num("missing_seqs"), "frames were lost in flight: $theirs")
        assertEquals(0L, theirs.num("seq_gaps"), "a gap with nothing dropped: $theirs")

        // Deliberately *not* asserted: that `delivered` equals `received`. The gap between
        // them is Python's own inbound queue shedding at depth 64 because this harness's
        // drain thread is slower than the arrival rate -- 616 at full speed, 11 once the
        // send was paced. That is the harness's backpressure, not an interop property, and
        // driving it to zero means tuning a sleep until a number appears, which buys the
        // metric rather than measuring it. What must hold is that the peer's own accounting
        // adds up, so nothing vanishes unattributed on its side either.
        assertEquals(
            theirs.num("received"),
            theirs.num("delivered") + theirs.num("dropped_inbound"),
            "python's own inbound accounting does not balance: $theirs",
        )
    }

    @Test
    fun `a deliberate overflow is counted the same on both sides`() {
        // `camera` is latest_wins at depth one, so a burst is guaranteed to displace.
        //
        // The first version asserted `gaps > 0` straight after the burst and was a race
        // dressed as a property: a gap is only *visible* to the receiver if at least two
        // frames arrive with drops between them, and a 200-deep burst sometimes delivers
        // exactly one. It failed on unmutated HEAD, and it failed under mutations that
        // could not affect it -- which is worse than useless, because a red test that
        // moves on its own teaches people to ignore the suite.
        //
        // Bracketed now: one frame delivered, then the burst, then one more delivered. The
        // bracket is what makes a gap necessary rather than likely.
        val peer = startPeer(seconds = 60.0, quietDrain = true)
        val session = connect(peer)

        val payload = ByteArray(40_960) { ((it * 37 + 11) % 256).toByte() }
        fun frame(id: Int) = mapOf(
            Fields.CAPTURE_KEY to JsonValue.Num(1_000_000_000L + id),
            "frame_id" to JsonValue.Num(id.toLong()),
            "width" to JsonValue.Num(1280),
            "height" to JsonValue.Num(720),
            "format" to JsonValue.Text("jpeg"),
            "quality" to JsonValue.Num(85),
        )

        // The opening bracket, drained before anything else is offered.
        assertTrue(session.send(Channels.CAMERA, frame(0), payload))
        val drained = System.currentTimeMillis() + 10_000
        while (session.outboundPending() > 0 && System.currentTimeMillis() < drained) Thread.sleep(2)

        val burst = 200
        repeat(burst) { i -> session.send(Channels.CAMERA, frame(i + 1), payload) }

        // The closing bracket. Its sequence is far above the opening one, so whatever was
        // shed in between is a gap the peer must see.
        val settled = System.currentTimeMillis() + 15_000
        while (session.outboundPending() > 0 && System.currentTimeMillis() < settled) Thread.sleep(2)
        assertTrue(session.send(Channels.CAMERA, frame(burst + 1), payload))
        val closed = System.currentTimeMillis() + 10_000
        while (session.outboundPending() > 0 && System.currentTimeMillis() < closed) Thread.sleep(2)

        val summary = summaryOf(finishAndSummarise(peer, session, settleMs = 3_000))
        assertNull(summary.text("drain_error"), summary.toString())

        val ours = session.stats().channels.getValue(Channels.CAMERA)
        val theirs = summary.channel(Channels.CAMERA)

        assertEquals((burst + 2).toLong(), ours.enqueued, "not every frame was offered: $ours")
        assertTrue(ours.dropped > 0, "a 200-deep burst on a depth-1 queue dropped nothing: $ours")
        // Everything offered either went out or was counted as dropped.
        assertEquals(ours.enqueued, ours.sent + ours.dropped, "our own accounting: $ours")
        assertEquals(
            ours.sent,
            theirs.num("received"),
            "we wrote ${ours.sent} and python saw ${theirs.num("received")}",
        )

        // The two views of one fact. The peer's count is a *lower* bound: a receiver cannot
        // know about drops after the last frame it received, which is why the bracket
        // exists and why this is not an equality.
        val gaps = theirs.num("missing_seqs")
        assertTrue(gaps > 0, "python saw no gaps despite ${ours.dropped} drops: $theirs")
        assertTrue(gaps <= ours.dropped, "python found more gaps ($gaps) than we dropped (${ours.dropped})")

        // And the peer's own accounting adds up, so nothing vanishes unattributed there
        // either. Python's inbound queue is latest_wins at depth one too, so `delivered`
        // and `received` answer different questions.
        assertEquals(
            theirs.num("received"),
            theirs.num("delivered") + theirs.num("dropped_inbound"),
            "python's own inbound accounting does not balance: $theirs",
        )
    }

    @Test
    fun `a malformed frame from python costs one message, and our counter names it`() {
        // The plan promised this and it was never provoked. It also replaces an assertion
        // that read `dropped_inbound: 0` off the *peer's* counters and presented it as
        // ours -- we had no inbound counter to check, so it covered nothing it appeared to.
        val peer = startPeer(seconds = 40.0, sendMalformed = true)
        val session = connect(peer)

        val sent = peer.stdout.readLine() ?: error("the peer never reported sending it")
        assertTrue(sent.contains("sent_malformed"), "unexpected line: $sent")

        val deadline = System.currentTimeMillis() + 10_000
        while (System.currentTimeMillis() < deadline &&
            session.stats().inboundRefusals[Channels.GPS] == null
        ) {
            Thread.sleep(20)
        }
        val refusals = session.stats().inboundRefusals[Channels.GPS]
            ?: error("we did not refuse it: ${session.stats().inboundRefusals}")
        assertEquals(
            1L,
            refusals[RefusalReason.OUT_OF_RANGE.wire],
            "refused for the wrong reason: $refusals",
        )
        // One bad record costs one record. At 50 Hz, reconnecting over a single malformed
        // IMU sample would be far worse than dropping it.
        assertTrue(session.isRunning, "a malformed message ended the session")
        assertTrue(
            session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions()),
            "the session stopped carrying traffic",
        )
    }

    @Test
    fun `a framing error from python ends the session, unlike a malformed message`() {
        // The other half of the recoverability rule: a framing error means the byte stream
        // has desynchronised and there is no delimiter to hunt for.
        val ends = mutableListOf<SessionEnd>()
        val peer = startPeer(seconds = 40.0, sendFramingError = true)
        val socket = Socket(InetAddress.getLoopbackAddress(), peer.port).also { this.socket = it }
        val session = Session(
            input = socket.getInputStream(),
            output = socket.getOutputStream(),
            deviceId = "interop-phone",
            role = "phone",
            monoClock = { System.nanoTime() },
            wallClock = { System.currentTimeMillis() * 1_000_000 },
            onFrame = { _, _, _ -> },
            onEnd = { reason, _ -> synchronized(ends) { ends.add(reason) } },
        ).also { this.session = it }
        session.start()

        val deadline = System.currentTimeMillis() + 15_000
        while (System.currentTimeMillis() < deadline && synchronized(ends) { ends.isEmpty() }) {
            Thread.sleep(20)
        }
        assertEquals(
            listOf(SessionEnd.FRAMING_ERROR),
            synchronized(ends) { ends.toList() },
            "an over-size header did not end the session as a framing error",
        )
        assertTrue(!session.isRunning)
    }

    @Test
    fun `python answers our ping and echoes the wire stamp we wrote`() {
        // The timebase exchange across the language boundary, which nothing exercised: the
        // responder was reachable from no code path, so a ping was delivered to an
        // application with no handler and no pong was ever produced.
        val received = mutableListOf<Frame>()
        val peer = startPeer(seconds = 40.0)
        val session = connect(peer) { synchronized(received) { received.add(it) } }

        assertTrue(session.sendTimeSyncPing(exchangeId = 4242))

        val deadline = System.currentTimeMillis() + 10_000
        while (System.currentTimeMillis() < deadline &&
            synchronized(received) { received.none { it.channel == Channels.CONTROL } }
        ) {
            Thread.sleep(20)
        }
        val pong = synchronized(received) { received.firstOrNull { it.channel == Channels.CONTROL } }
            ?: error("no pong arrived; got ${synchronized(received) { received.map { it.channel } }}")

        val decoded = TimeSyncMessage.fromWire(pong.header.entries, pong.payload)
        assertEquals(4242, decoded.exchangeId, "python changed the exchange id")
        assertTrue(!decoded.isPing, "python answered with a ping")
        // The echoed t1 makes the offset computable, and a constant zero here would look
        // like a working exchange while making every estimate wrong.
        assertTrue(
            (decoded.peerWireMonoNs ?: 0L) > 0L,
            "our wire stamp came back as ${decoded.peerWireMonoNs}",
        )
        assertTrue(
            decoded.peerRecvMonoNs != null && decoded.peerRecvWallNs != null,
            "a pong with a partial peer triple: $decoded",
        )
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
        val summary = summaryOf(finishAndSummarise(peer, session))
        assertNull(summary.text("drain_error"), summary.toString())
        val theirs = summary.channel(Channels.CAMERA)
        assertEquals(1L, theirs.num("received"), "the camera frame did not arrive: $theirs")
        assertEquals(1L, theirs.num("delivered"), "python's decoder refused it: $theirs")
        // The payload is in bytes_received along with its header, so the floor is what
        // matters: fewer bytes than the image means the image did not cross whole.
        assertTrue(
            theirs.num("bytes_received") >= payload.size.toLong(),
            "only ${theirs.num("bytes_received")} bytes for a ${payload.size}-byte image",
        )
    }

    @Test
    fun `keepalives are exchanged in both directions`() {
        val peer = startPeer(seconds = 20.0)
        val session = connect(peer)
        // Long enough for both sides to fire at a 1 s interval.
        Thread.sleep(3_000)
        val summary = finishAndSummarise(peer, session, settleMs = 300)

        assertTrue(
            summaryOf(summary).num("heartbeats_received") >= 2,
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
