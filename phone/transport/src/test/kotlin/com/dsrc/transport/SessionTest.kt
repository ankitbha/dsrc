package com.dsrc.transport

import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Two sessions against each other over a loopback socket pair.
 *
 * A real socket rather than piped streams: the behaviours worth testing here are the
 * ones that only appear with a genuine peer -- the handshake ordering, keepalive cadence,
 * stall detection on completed reads, and what a framing error does to a live session.
 * Piped streams have a small fixed buffer and deadlock in exactly the cases that matter.
 */
class SessionTest {

    private val sockets = mutableListOf<Socket>()
    private val servers = mutableListOf<ServerSocket>()
    private val sessions = mutableListOf<Session>()

    @AfterTest
    fun cleanup() {
        sessions.forEach { runCatching { it.close() } }
        sockets.forEach { runCatching { it.close() } }
        servers.forEach { runCatching { it.close() } }
        sessions.clear(); sockets.clear(); servers.clear()
    }

    private class Peer(
        val session: Session,
        val received: MutableList<Frame>,
        val ends: MutableList<Pair<SessionEnd, Throwable?>>,
        val frameLatch: () -> CountDownLatch,
    )

    /** A connected pair, each side already through the handshake. */
    private fun pair(
        clock: () -> Long = { System.nanoTime() },
        onPhoneFrame: ((Frame) -> Unit)? = null,
    ): Pair<Peer, Peer> {
        val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).also { servers.add(it) }
        val clientSocket = Socket(InetAddress.getLoopbackAddress(), server.localPort).also { sockets.add(it) }
        val serverSocket = server.accept().also { sockets.add(it) }

        fun build(socket: Socket, role: String, onFrame: (Frame) -> Unit): Peer {
            val received = mutableListOf<Frame>()
            val ends = mutableListOf<Pair<SessionEnd, Throwable?>>()
            var latch = CountDownLatch(1)
            val session = Session(
                input = socket.getInputStream(),
                output = socket.getOutputStream(),
                deviceId = "test-$role",
                role = role,
                monoClock = clock,
                wallClock = { 1_755_648_000_000_000_000 },
                onFrame = { frame ->
                    synchronized(received) { received.add(frame) }
                    onFrame(frame)
                    latch.countDown()
                },
                onEnd = { reason, cause -> synchronized(ends) { ends.add(reason to cause) } },
            ).also { sessions.add(it) }
            return Peer(session, received, ends) { latch }
        }

        val phone = build(clientSocket, "phone", onPhoneFrame ?: {})
        val jetson = build(serverSocket, "jetson") {}

        // Both send their hello before either reads, which is why starting them on two
        // threads cannot deadlock here.
        val phoneStart = Thread { phone.session.start() }
        val jetsonStart = Thread { jetson.session.start() }
        phoneStart.start(); jetsonStart.start()
        phoneStart.join(5_000); jetsonStart.join(5_000)
        return phone to jetson
    }

    private fun awaitFrames(peer: Peer, count: Int, timeoutMs: Long = 5_000): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (synchronized(peer.received) { peer.received.size } >= count) return true
            Thread.sleep(10)
        }
        return false
    }

    private fun awaitEnd(peer: Peer, timeoutMs: Long = 8_000): SessionEnd? {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            synchronized(peer.ends) { peer.ends.firstOrNull() }?.let { return it.first }
            Thread.sleep(10)
        }
        return null
    }

    // -- handshake -----------------------------------------------------------

    @Test
    fun `both sides learn the peer's identity from the hello`() {
        val (phone, jetson) = pair()
        assertEquals("jetson", phone.session.peer?.role)
        assertEquals("phone", jetson.session.peer?.role)
        assertEquals(Protocol.VERSION.toLong(), phone.session.peer?.protocolVersion)
    }

    @Test
    fun `the hello is not delivered to the application`() {
        // It is transport traffic, consumed rather than delivered.
        val (phone, _) = pair()
        Thread.sleep(200)
        assertEquals(0, synchronized(phone.received) { phone.received.size })
    }

    @Test
    fun `control traffic starts at sequence one because the hello spent zero`() {
        val (phone, jetson) = pair()
        assertTrue(phone.session.send(Channels.CONTROL, mapOf("probe" to JsonValue.Num(1))))
        assertTrue(awaitFrames(jetson, 1), "the control frame never arrived")
        assertEquals(1L, synchronized(jetson.received) { jetson.received.first().sequence })
    }

    @Test
    fun `a version mismatch is refused before any data frame is read`() {
        val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).also { servers.add(it) }
        val client = Socket(InetAddress.getLoopbackAddress(), server.localPort).also { sockets.add(it) }
        val accepted = server.accept().also { sockets.add(it) }

        // A hand-rolled peer claiming the wrong version, plus a data frame behind it that
        // must never be read.
        val badHello = Framing.header(
            Channels.CONTROL, 0, 1, 2,
            mapOf(Session.HELLO to JsonValue.Obj(mapOf(
                "protocol_version" to JsonValue.Num(Protocol.VERSION + 1L),
                "device_id" to JsonValue.Text("wrong"),
                "role" to JsonValue.Text("jetson"),
            ))),
            allowReserved = setOf(Session.HELLO),
        )
        Framing.write(badHello, ByteArray(0), accepted.getOutputStream())
        Framing.write(
            Framing.header(Channels.GPS, 0, 1, 2, mapOf("should" to JsonValue.Text("not be read"))),
            ByteArray(0), accepted.getOutputStream(),
        )

        val received = mutableListOf<Frame>()
        val session = Session(
            client.getInputStream(), client.getOutputStream(), "phone", "phone",
            { System.nanoTime() }, { 0 },
            onFrame = { synchronized(received) { received.add(it) } },
        ).also { sessions.add(it) }

        val error = runCatching { session.start() }.exceptionOrNull()
        assertTrue(error is FramingError, "expected a framing error, got $error")
        assertTrue(error.message!!.contains("version"), error.message!!)
        Thread.sleep(200)
        assertEquals(0, synchronized(received) { received.size }, "no data frame may be read")
    }

    @Test
    fun `a first frame that is not a hello ends the session`() {
        val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).also { servers.add(it) }
        val client = Socket(InetAddress.getLoopbackAddress(), server.localPort).also { sockets.add(it) }
        val accepted = server.accept().also { sockets.add(it) }
        Framing.write(
            Framing.header(Channels.GPS, 0, 1, 2, mapOf("k" to JsonValue.Num(1))),
            ByteArray(0), accepted.getOutputStream(),
        )
        val session = Session(
            client.getInputStream(), client.getOutputStream(), "phone", "phone",
            { System.nanoTime() }, { 0 }, onFrame = {},
        ).also { sessions.add(it) }
        assertTrue(runCatching { session.start() }.exceptionOrNull() is FramingError)
    }

    // -- delivery ------------------------------------------------------------

    @Test
    fun `a message crosses the wire intact`() {
        val (phone, jetson) = pair()
        val record = GpsRecord.noFix(1_234_567)
        assertTrue(phone.session.send(Channels.GPS, record.toExtensions()))
        assertTrue(awaitFrames(jetson, 1), "the gps frame never arrived")
        val frame = synchronized(jetson.received) { jetson.received.first() }
        assertEquals(Channels.GPS, frame.channel)
        assertEquals(record, GpsRecord.fromWire(frame.header.entries, frame.payload))
    }

    @Test
    fun `a payload crosses intact`() {
        val (phone, jetson) = pair()
        val payload = ByteArray(4096) { ((it * 37 + 11) % 256).toByte() }
        assertTrue(
            phone.session.send(
                Channels.CAMERA,
                mapOf(
                    Fields.CAPTURE_KEY to JsonValue.Num(1),
                    "frame_id" to JsonValue.Num(1),
                    "width" to JsonValue.Num(64),
                    "height" to JsonValue.Num(64),
                    "format" to JsonValue.Text("jpeg"),
                    "quality" to JsonValue.Num(85),
                ),
                payload,
            )
        )
        assertTrue(awaitFrames(jetson, 1), "the camera frame never arrived")
        assertTrue(synchronized(jetson.received) { jetson.received.first().payload }.contentEquals(payload))
    }

    @Test
    fun `every message is delivered or counted as dropped, never neither`() {
        // `gps` is depth 64 and reliable, so bursting 200 past a writer that cannot keep
        // up *should* drop some. The property is not lossless delivery -- that would be
        // the wrong claim for this channel -- but that every message is accounted for
        // under exactly one heading.
        val (phone, jetson) = pair()
        repeat(200) { phone.session.send(Channels.GPS, GpsRecord.noFix(it.toLong()).toExtensions()) }

        val deadline = System.currentTimeMillis() + 15_000
        while (phone.session.outboundPending() > 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(20)
        }
        Thread.sleep(500)   // let the last frames land on the far side

        val sender = phone.session.stats().channels.getValue(Channels.GPS)
        assertEquals(200, sender.enqueued)
        assertEquals(
            sender.enqueued,
            sender.dropped + sender.sent + sender.pending,
            "sender accounting must balance: $sender",
        )

        val arrived = synchronized(jetson.received) { jetson.received.size }
        assertEquals(
            sender.sent,
            arrived.toLong(),
            "everything written should arrive on a loopback socket",
        )
        assertTrue(arrived > 0, "nothing arrived at all")
    }

    @Test
    fun `a dropped message leaves a gap in the sequence numbers the peer sees`() {
        // The whole reason sequence numbers are assigned before the overflow decision:
        // a gap is the receiver's only evidence that the sender dropped something.
        val (phone, jetson) = pair()
        repeat(300) { phone.session.send(Channels.GPS, GpsRecord.noFix(it.toLong()).toExtensions()) }
        val deadline = System.currentTimeMillis() + 15_000
        while (phone.session.outboundPending() > 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(20)
        }
        Thread.sleep(500)

        val sender = phone.session.stats().channels.getValue(Channels.GPS)
        val sequences = synchronized(jetson.received) { jetson.received.map { it.sequence } }
        assertTrue(sequences.isNotEmpty(), "nothing arrived")
        assertEquals(sequences, sequences.sorted(), "sequence numbers must arrive in order")
        assertEquals(sequences.size, sequences.distinct().size, "no sequence number may repeat")
        if (sender.dropped > 0) {
            val span = sequences.last() - sequences.first() + 1
            assertTrue(
                span > sequences.size,
                "with ${sender.dropped} dropped there must be a gap: span $span over ${sequences.size} frames",
            )
        }
    }

    // -- the sender rule -----------------------------------------------------

    @Test
    fun `a reserved key on a caller's message is refused before it is queued`() {
        // A message carrying one would be consumed by the peer as transport traffic and
        // never delivered -- lost with no drop counted and no sequence gap to show it.
        val (phone, _) = pair()
        assertFalse(phone.session.send(Channels.GPS, mapOf("heartbeat" to JsonValue.Bool(true))))
        assertEquals(1, phone.session.stats().outboundRefusals["reserved_key"])
    }

    @Test
    fun `an unknown channel is refused`() {
        val (phone, _) = pair()
        assertFalse(phone.session.send("nonsense", emptyMap()))
        assertTrue(phone.session.stats().outboundRefusals.isNotEmpty())
    }

    @Test
    fun `our own refusals are counted apart from the peer's`() {
        // One is a bug here and one is a bug there; a total that added them would hide
        // both.
        val (phone, _) = pair()
        phone.session.send(Channels.GPS, mapOf("hello" to JsonValue.Bool(true)))
        val stats = phone.session.stats()
        assertEquals(1, stats.outboundRefusals["reserved_key"])
        assertTrue(stats.inboundRefusals.isEmpty(), "nothing arrived to refuse")
    }

    @Test
    fun `a malformed message from the peer costs one message, not the session`() {
        val refusals = AtomicLong(0)
        val (phone, jetson) = pair(onPhoneFrame = {
            // The router's job in production; here it stands in for one.
            refusals.incrementAndGet()
            throw MessageError(RefusalReason.WRONG_TYPE, "pretend this field is wrong")
        })
        assertTrue(jetson.session.send(Channels.ADVISORY, mapOf("k" to JsonValue.Num(1))))
        assertTrue(awaitFrames(phone, 1), "the frame never arrived")
        Thread.sleep(200)
        assertTrue(phone.session.isRunning, "the session must stay open")
        assertEquals(1, phone.session.stats().inboundRefusals["wrong_type"])

        // And it keeps working afterwards.
        assertTrue(jetson.session.send(Channels.ADVISORY, mapOf("k" to JsonValue.Num(2))))
        assertTrue(awaitFrames(phone, 2), "the session stopped delivering")
    }

    // -- keepalive and stall -------------------------------------------------

    @Test
    fun `keepalives flow without any application traffic`() {
        val (phone, jetson) = pair()
        // Two intervals plus slack: the writer only sends one when its queue is empty.
        Thread.sleep(((Protocol.KEEPALIVE_INTERVAL_S * 2 + 1) * 1000).toLong())
        assertTrue(phone.session.stats().heartbeatsSent >= 2, "phone sent ${phone.session.stats().heartbeatsSent}")
        assertTrue(jetson.session.stats().heartbeatsReceived >= 1)
    }

    @Test
    fun `a keepalive is absorbed rather than delivered`() {
        // The transport generates them, so it also absorbs them.
        val (phone, _) = pair()
        Thread.sleep(((Protocol.KEEPALIVE_INTERVAL_S * 2 + 1) * 1000).toLong())
        assertEquals(0, synchronized(phone.received) { phone.received.size })
        assertTrue(phone.session.stats().heartbeatsReceived >= 1)
    }

    @Test
    fun `a session with no read progress ends stalled`() {
        // The peer here is a raw socket that sends its hello and then goes silent. A live
        // peer will not do: it shares the injected clock, so advancing time also fires
        // *its* keepalive, which arrives as read progress and resets the very timer under
        // test. That made this the one flaky test in the file -- green or red depending on
        // which thread won.
        val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).also { servers.add(it) }
        val client = Socket(InetAddress.getLoopbackAddress(), server.localPort).also { sockets.add(it) }
        val silent = server.accept().also { sockets.add(it) }
        Framing.write(
            Framing.header(
                Channels.CONTROL, 0, 1, 2,
                mapOf(Session.HELLO to JsonValue.Obj(mapOf(
                    "protocol_version" to JsonValue.Num(Protocol.VERSION.toLong()),
                    "device_id" to JsonValue.Text("silent"),
                    "role" to JsonValue.Text("jetson"),
                ))),
                allowReserved = setOf(Session.HELLO),
            ),
            ByteArray(0), silent.getOutputStream(),
        )

        val now = AtomicLong(0)
        val ends = mutableListOf<Pair<SessionEnd, Throwable?>>()
        val session = Session(
            client.getInputStream(), client.getOutputStream(), "phone", "phone",
            monoClock = { now.get() }, wallClock = { 0 }, onFrame = {},
            onEnd = { reason, cause -> synchronized(ends) { ends.add(reason to cause) } },
        ).also { sessions.add(it) }
        session.start()

        Thread.sleep(200)
        assertTrue(session.isRunning, "should still be alive before the timeout")
        now.addAndGet(((Protocol.STALL_TIMEOUT_S + 1) * 1e9).toLong())

        val deadline = System.currentTimeMillis() + 8_000
        while (System.currentTimeMillis() < deadline && synchronized(ends) { ends.isEmpty() }) {
            Thread.sleep(10)
        }
        assertEquals(SessionEnd.STALLED, synchronized(ends) { ends.firstOrNull()?.first })
        assertFalse(session.isRunning)
    }

    @Test
    fun `a keepalive from the peer is read progress and defers a stall`() {
        // The other half of the same mechanism, asserted rather than left as the accident
        // that made the test above flaky: a keepalive counts as read progress, which is
        // exactly what "no read progress" in the spec means.
        val now = AtomicLong(0)
        val (phone, _) = pair(clock = { now.get() })
        Thread.sleep(200)
        // Advance just under the timeout, repeatedly. The peer's keepalives fire on the
        // same clock and keep arriving, so the session must survive well past one timeout
        // of wall-clock-equivalent time.
        repeat(4) {
            now.addAndGet((Protocol.STALL_TIMEOUT_S * 0.5 * 1e9).toLong())
            Thread.sleep(250)
        }
        assertTrue(phone.session.isRunning, "keepalives should have deferred the stall: ${phone.ends}")
        assertTrue(phone.session.stats().heartbeatsReceived > 0, "no keepalive arrived")
    }

    @Test
    fun `traffic keeps a session alive past the stall timeout`() {
        val now = AtomicLong(0)
        val (phone, jetson) = pair(clock = { now.get() })
        // Advance in steps under the timeout, with a frame arriving each time.
        repeat(4) {
            jetson.session.send(Channels.ADVISORY, mapOf("k" to JsonValue.Num(it.toLong())))
            Thread.sleep(120)
            now.addAndGet((Protocol.STALL_TIMEOUT_S * 0.6 * 1e9).toLong())
            Thread.sleep(120)
        }
        assertTrue(phone.session.isRunning, "ended: ${phone.ends}")
    }

    // -- ending --------------------------------------------------------------

    @Test
    fun `a local close reports itself as such`() {
        val (phone, _) = pair()
        phone.session.close()
        assertEquals(SessionEnd.CLOSED_LOCAL, awaitEnd(phone))
        assertFalse(phone.session.isRunning)
    }

    @Test
    fun `the peer closing is noticed`() {
        val (phone, jetson) = pair()
        jetson.session.close()
        val end = awaitEnd(phone)
        assertTrue(
            end == SessionEnd.PEER_CLOSED || end == SessionEnd.TRANSPORT_ERROR,
            "expected a peer-closed style end, got $end",
        )
    }

    @Test
    fun `a session ends once, with one reason`() {
        val (phone, _) = pair()
        phone.session.close()
        phone.session.close()
        Thread.sleep(200)
        assertEquals(1, synchronized(phone.ends) { phone.ends.size })
    }

    @Test
    fun `sending after a close is refused rather than silently dropped`() {
        val (phone, _) = pair()
        phone.session.close()
        assertFalse(phone.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions()))
    }

    @Test
    fun `a framing error from the peer ends the session`() {
        val (phone, jetson) = pair()
        // A prefix declaring a payload past the limit: refused before anything is
        // allocated, and not recoverable, since there is no delimiter to resynchronise on.
        val socket = sockets.first { it.isConnected && it.port == sockets[1].localPort || true }
        val raw = byteArrayOf(0x7F, -1, -1, -1, 0x00, 0x02)
        synchronized(jetson) {
            val out = sockets[1].getOutputStream()
            out.write(raw); out.flush()
        }
        val end = awaitEnd(phone)
        assertTrue(
            end == SessionEnd.FRAMING_ERROR || end == SessionEnd.TRANSPORT_ERROR,
            "expected the session to end, got $end",
        )
    }

    // -- backpressure --------------------------------------------------------

    @Test
    fun `outbound pending is observable`() {
        val (phone, _) = pair()
        assertEquals(0, phone.session.outboundPending())
    }

    @Test
    fun `channel counters are reported per channel`() {
        val (phone, jetson) = pair()
        repeat(5) { phone.session.send(Channels.GPS, GpsRecord.noFix(it.toLong()).toExtensions()) }
        assertTrue(awaitFrames(jetson, 5), "frames did not arrive")
        val gps = phone.session.stats().channels.getValue(Channels.GPS)
        assertEquals(5, gps.enqueued)
        assertEquals(0, gps.dropped)
    }

    @Test
    fun `a camera frame displaces an unsent one and the drop is counted`() {
        // latest_wins at depth one. Enqueued faster than the writer can drain, which is
        // the normal condition for this channel.
        val (phone, _) = pair()
        repeat(50) {
            phone.session.send(
                Channels.CAMERA,
                mapOf(
                    Fields.CAPTURE_KEY to JsonValue.Num(it.toLong()),
                    "frame_id" to JsonValue.Num(it.toLong()),
                    "width" to JsonValue.Num(64), "height" to JsonValue.Num(64),
                    "format" to JsonValue.Text("jpeg"), "quality" to JsonValue.Num(85),
                ),
                ByteArray(64_000),
            )
        }
        val camera = phone.session.stats().channels.getValue(Channels.CAMERA)
        assertEquals(50, camera.enqueued)
        assertTrue(camera.dropped > 0, "expected displacements at depth 1, got $camera")
        assertTrue(
            camera.enqueued == camera.dropped + camera.sent + camera.pending,
            "counters must balance: $camera",
        )
    }
}
