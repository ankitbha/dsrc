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

    /**
     * An output stream that holds the writer until a latch opens.
     *
     * The only way to make "at enqueue" and "at write" distinguishable: without parking
     * the writer, the two instants are microseconds apart and a test cannot tell which
     * one a stamp came from. The handshake happens before the gate is armed, so the hello
     * is not held.
     */
    /**
     * Records what was true *at the moment* each frame was written.
     *
     * Sampling after the fact does not work for a claim about the writer's state: a
     * producer thread refills the queue between the write and the observation, so
     * "pending was non-zero when I looked" is a race rather than a proof. The hook runs on
     * the writer thread, inside the write, where the question is decidable.
     */
    private class ObservingOutputStream(
        private val inner: java.io.OutputStream,
        private val onFrame: (ByteArray) -> Unit,
        /** Slows the writer so a producer can keep the queue saturated. */
        private val perFrameDelayMs: Long = 0,
    ) : java.io.OutputStream() {
        override fun write(b: Int) = inner.write(b)

        override fun write(b: ByteArray, off: Int, len: Int) {
            onFrame(b.copyOfRange(off, off + len))
            if (perFrameDelayMs > 0) Thread.sleep(perFrameDelayMs)
            inner.write(b, off, len)
        }

        override fun flush() = inner.flush()
    }

    private class GatedOutputStream(
        private val inner: java.io.OutputStream,
        private val gate: CountDownLatch,
    ) : java.io.OutputStream() {
        @Volatile
        var armed = false

        override fun write(b: Int) {
            if (armed) gate.await()
            inner.write(b)
        }

        override fun write(b: ByteArray, off: Int, len: Int) {
            if (armed) gate.await()
            inner.write(b, off, len)
        }

        override fun flush() = inner.flush()
    }

    /** A connected pair, each side already through the handshake. */
    private fun pair(
        clock: () -> Long = { System.nanoTime() },
        onPhoneFrame: ((Frame) -> Unit)? = null,
        phoneOutputGate: CountDownLatch? = null,
        onPhoneWrite: ((ByteArray) -> Unit)? = null,
        phoneWriteDelayMs: Long = 0,
        phoneInput: ((java.io.InputStream) -> java.io.InputStream)? = null,
    ): Pair<Peer, Peer> {
        val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).also { servers.add(it) }
        val clientSocket = Socket(InetAddress.getLoopbackAddress(), server.localPort).also { sockets.add(it) }
        val serverSocket = server.accept().also { sockets.add(it) }

        val gates = mutableListOf<GatedOutputStream>()

        fun build(socket: Socket, role: String, onFrame: (Frame) -> Unit): Peer {
            val received = mutableListOf<Frame>()
            val ends = mutableListOf<Pair<SessionEnd, Throwable?>>()
            var latch = CountDownLatch(1)
            var output: java.io.OutputStream = socket.getOutputStream()
            if (role == "phone" && phoneOutputGate != null) {
                output = GatedOutputStream(output, phoneOutputGate).also { gates.add(it) }
            }
            if (role == "phone" && onPhoneWrite != null) {
                output = ObservingOutputStream(output, onPhoneWrite, phoneWriteDelayMs)
            }
            val input = if (role == "phone" && phoneInput != null) {
                phoneInput(socket.getInputStream())
            } else {
                socket.getInputStream()
            }
            val session = Session(
                input = input,
                output = output,
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
        // Armed only after the handshake, so the hello is never held.
        gates.forEach { it.armed = true }
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

    /** A valid time-sync ping, which is what `control` actually carries. */
    private fun ping(exchangeId: Long = 1) = TimeSyncMessage(
        captureMonoNs = 1_000,
        exchangeId = exchangeId,
        wireMonoNs = 0,
        peerRecvMonoNs = null,
        peerRecvWallNs = null,
        peerWireMonoNs = null,
    ).toExtensions()

    @Test
    fun `control traffic starts at sequence one because the hello spent zero`() {
        val (phone, jetson) = pair()
        // A real ping, not an arbitrary payload: the sender rule now runs the typed
        // decoder, and `control`'s typed message is the time-sync exchange.
        assertTrue(phone.session.send(Channels.CONTROL, ping(), wantsWireStamp = true))
        assertTrue(awaitFrames(jetson, 1), "the control frame never arrived")
        assertEquals(1L, synchronized(jetson.received) { jetson.received.first().sequence })
    }

    @Test
    fun `an arbitrary control message is refused before it goes out`() {
        // The sender rule in action: control carries the time-sync exchange and nothing
        // else, so a made-up payload is our bug to catch rather than the peer's.
        val (phone, _) = pair()
        assertFalse(phone.session.send(Channels.CONTROL, mapOf("probe" to JsonValue.Num(1))))
        assertTrue(
            phone.session.stats().outboundRefusals.isNotEmpty(),
            "the refusal must be counted: ${phone.session.stats().outboundRefusals}",
        )
    }

    @Test
    fun `a message our own decoder would refuse never reaches the wire`() {
        // Six of the nine refusal reasons were unreachable outbound before the sender rule
        // ran the typed decoder, so a bad record went out and came back as the peer's drop
        // counter.
        val (phone, jetson) = pair()
        val valid = GpsRecord(
            captureMonoNs = 1, valid = true, latitude = 51.5074, longitude = -0.1278,
            speedMps = 13.4, headingDeg = 91.2, fixQuality = 1, satellites = 9,
            hdop = 0.9, altitudeM = 35.0, utcEpochNs = 1,
        ).toExtensions()

        val cases = mapOf(
            "out_of_range (lat)" to valid + ("lat" to JsonValue.Real(200.0)),
            "out_of_range (count)" to valid + ("num_sats" to JsonValue.Num(-1)),
            "wrong_type" to valid + ("valid" to JsonValue.Text("yes")),
            "null_not_allowed" to valid + ("num_sats" to JsonValue.Null),
            "missing_field" to valid - "lat",
            "non_finite" to valid + ("speed_mps" to JsonValue.Real(Double.NaN)),
        )
        for ((name, extensions) in cases) {
            assertFalse(phone.session.send(Channels.GPS, extensions), "$name was sent")
        }
        assertEquals(
            cases.size,
            phone.session.stats().outboundRefusals.values.sum().toInt(),
            "every refusal must be counted: ${phone.session.stats().outboundRefusals}",
        )
        Thread.sleep(300)
        assertEquals(0, synchronized(jetson.received) { jetson.received.size }, "nothing may arrive")
    }

    @Test
    fun `a non-finite value is counted, not thrown at the caller`() {
        // Doubles.format raises IllegalArgumentException, which is neither MessageError nor
        // FramingError, so it used to propagate out of send() -- on the phone, into a
        // sensor callback.
        val (phone, _) = pair()
        val refused = runCatching {
            phone.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions() + ("extra" to JsonValue.Real(Double.POSITIVE_INFINITY)))
        }
        assertTrue(refused.isSuccess, "send threw ${refused.exceptionOrNull()}")
        assertEquals(false, refused.getOrNull())
        assertEquals(1, phone.session.stats().outboundRefusals["non_finite"])
    }

    @Test
    fun `a caller cannot set the wire stamp itself`() {
        // Reserved on the Python side and not here, so a caller could put a value in the
        // field the peer's timebase reads as our departure stamp, with no stamping done.
        val (phone, _) = pair()
        assertFalse(
            phone.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions() + ("t_wire_mono_ns" to JsonValue.Num(7))),
        )
        assertEquals(1, phone.session.stats().outboundRefusals["reserved_key"])
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

        // *Where* the gap falls is timing-dependent, and two earlier versions of this
        // test each assumed one shape. If the whole burst is enqueued before the writer
        // wakes, the oldest are dropped and the survivors are one contiguous run at the
        // end. If the writer interleaves, an early run gets out before the queue fills and
        // the gap lands in the middle. Both are correct behaviour for drop-oldest.
        //
        // What holds either way is the count: the sequence numbers the peer never saw is
        // exactly what the sender dropped. That is the invariant worth asserting, and it
        // is the one the receiver can actually act on.
        assertEquals(0, sender.pending, "the burst should have fully drained: $sender")
        val highest = sequences.last()
        val missing = (highest + 1) - sequences.size
        assertEquals(
            sender.dropped,
            missing,
            "the peer must be able to see every drop as a missing sequence number: " +
                "highest $highest, received ${sequences.size}, $sender",
        )
        if (sender.dropped > 0) {
            assertTrue(missing > 0, "with ${sender.dropped} dropped there must be a visible gap")
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

    @Test
    fun `a write-time framing error ends the session instead of killing the writer`() {
        // The worst failure the validator found. A header that fit at the size probe could
        // grow past MAX_HEADER_BYTES once the real sequence and clocks were substituted,
        // and that throw escaped the writer's catch: the thread died, isRunning stayed
        // true, send() kept returning true, the stall check died with the writer, and
        // every later message was discarded for the rest of the drive.
        val (phone, jetson) = pair()

        // A header padded to sit just under the limit, so the widest real values push it
        // over. The size probe should refuse this outright now.
        val padding = "x".repeat(Protocol.MAX_HEADER_BYTES - 220)
        val accepted = phone.session.send(
            Channels.TELEMETRY,
            mapOf(
                Fields.CAPTURE_KEY to JsonValue.Num(1),
                "pad" to JsonValue.Text(padding),
            ),
        )
        if (!accepted) {
            // Refused at the probe, which is the intended outcome.
            assertTrue(
                phone.session.stats().outboundRefusals.isNotEmpty(),
                "a refusal must be counted",
            )
            assertTrue(phone.session.isRunning, "refusing one message must not end the session")
            // And the session still works.
            assertTrue(phone.session.send(Channels.GPS, GpsRecord.noFix(2).toExtensions()))
            assertTrue(awaitFrames(jetson, 1), "the session stopped working after a refusal")
            return
        }
        // If it was accepted, the writer must still not die silently: either the frame
        // goes out, or the session ends with a reason.
        Thread.sleep(1_000)
        assertTrue(
            !phone.session.isRunning || synchronized(jetson.received) { jetson.received.isNotEmpty() },
            "the writer accepted a frame it could not write and said nothing",
        )
    }

    @Test
    fun `a wedged peer is detected as stalled even with work queued`() {
        // The stall check used to live in the writer's idle branch, so it never ran while
        // there was work queued -- and never at all when the writer was blocked in
        // output.write() because the peer had stopped reading, which is the exact case the
        // timeout exists for.
        val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).also { servers.add(it) }
        val client = Socket(InetAddress.getLoopbackAddress(), server.localPort).also { sockets.add(it) }
        val wedged = server.accept().also { sockets.add(it) }
        Framing.write(
            Framing.header(
                Channels.CONTROL, 0, 1, 2,
                mapOf(Session.HELLO to JsonValue.Obj(mapOf(
                    "protocol_version" to JsonValue.Num(Protocol.VERSION.toLong()),
                    "device_id" to JsonValue.Text("wedged"),
                    "role" to JsonValue.Text("jetson"),
                ))),
                allowReserved = setOf(Session.HELLO),
            ),
            ByteArray(0), wedged.getOutputStream(),
        )

        val now = AtomicLong(0)
        val ends = mutableListOf<Pair<SessionEnd, Throwable?>>()
        val session = Session(
            client.getInputStream(), client.getOutputStream(), "phone", "phone",
            monoClock = { now.get() }, wallClock = { 0 }, onFrame = {},
            onEnd = { reason, cause -> synchronized(ends) { ends.add(reason to cause) } },
        ).also { sessions.add(it) }
        session.start()

        // Keep the queue full so the idle branch is never reached.
        repeat(300) {
            session.send(
                Channels.CAMERA,
                mapOf(
                    Fields.CAPTURE_KEY to JsonValue.Num(it.toLong()),
                    "frame_id" to JsonValue.Num(it.toLong()),
                    "width" to JsonValue.Num(64), "height" to JsonValue.Num(64),
                    "format" to JsonValue.Text("jpeg"), "quality" to JsonValue.Num(85),
                ),
                ByteArray(256 * 1024),
            )
        }
        now.addAndGet(((Protocol.STALL_TIMEOUT_S + 5) * 1e9).toLong())

        val deadline = System.currentTimeMillis() + 10_000
        while (System.currentTimeMillis() < deadline && synchronized(ends) { ends.isEmpty() }) {
            Thread.sleep(20)
        }
        assertEquals(
            SessionEnd.STALLED,
            synchronized(ends) { ends.firstOrNull()?.first },
            "a wedged peer must be detected even with work queued",
        )
    }

    @Test
    fun `a heartbeat does not steal a queued control message`() {
        // enqueue-then-poll returned whatever was already at the head, destroying an
        // application control message with `dropped` still zero, then writing the
        // heartbeat twice with a duplicate sequence number.
        val (phone, jetson) = pair()
        assertTrue(phone.session.send(Channels.CONTROL, ping(exchangeId = 99), wantsWireStamp = true))
        // Long enough for at least one keepalive to fire alongside it.
        Thread.sleep(((Protocol.KEEPALIVE_INTERVAL_S * 2 + 1) * 1000).toLong())

        assertTrue(awaitFrames(jetson, 1), "the control message was destroyed by a heartbeat")
        val exchanges = synchronized(jetson.received) {
            jetson.received.filter { it.channel == Channels.CONTROL }
                .map { (it.header.entries["exchange_id"] as? JsonValue.Num)?.value }
        }
        assertTrue(99L in exchanges, "our ping never arrived; got $exchanges")

        val sequences = synchronized(jetson.received) { jetson.received.map { it.sequence } }
        assertEquals(sequences.size, sequences.distinct().size, "a sequence number was reused: $sequences")
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

    // -- the two enqueue stamps ---------------------------------------------

    @Test
    fun `t_mono_ns is the clock at enqueue, not the clock at write`() {
        // The spec's header table defines both clocks as the sender's values *at enqueue*,
        // and names `t_mono_ns - t_capture_mono_ns` as the queueing latency. Reading them
        // on the writer thread instead folded the queueing delay into the field defined as
        // excluding it, so that subtraction measured capture-to-write: the queue's own
        // depth, reported as if it were the sensor's.
        //
        // Three frames are enqueued while the clock reads ENQUEUE_NS, then the clock jumps
        // before the writer is allowed to run. A write-time stamp shows the jumped value.
        val clock = AtomicLong(ENQUEUE_NS)
        val release = CountDownLatch(1)
        val (phone, jetson) = pair(clock = { clock.get() }, phoneOutputGate = release)

        repeat(3) {
            assertTrue(phone.session.send(Channels.GPS, GpsRecord.noFix(ENQUEUE_NS - 1_000).toExtensions()))
        }
        clock.set(ENQUEUE_NS + 2_000_000_000L)
        release.countDown()

        assertTrue(awaitFrames(jetson, 3), "frames never arrived")
        val stamps = jetson.received.filter { it.channel == Channels.GPS }
            .map { (it.header.entries.getValue(Framing.KEY_MONO) as JsonValue.Num).value }
        assertEquals(listOf(ENQUEUE_NS, ENQUEUE_NS, ENQUEUE_NS), stamps, "stamped at write, not at enqueue")
    }

    @Test
    fun `the wire stamp is never earlier than the enqueue stamp`() {
        // The wire stamp exists to be the later of the two -- the spec makes it the
        // instant before the bytes leave, and the difference between them is the queueing
        // delay a timebase estimate has to remove. It used to be built before the header's
        // own clock call, so it was deterministically *earlier* and the delay it exposed
        // came out negative on every stamped frame.
        val clock = AtomicLong(0)
        val (phone, jetson) = pair(clock = { clock.addAndGet(1_000) })

        repeat(8) { exchange ->
            assertTrue(phone.session.send(
                Channels.CONTROL,
                ping(exchangeId = exchange.toLong()),
                wantsWireStamp = true,
            ))
        }
        assertTrue(awaitFrames(jetson, 8), "timebase frames never arrived")

        val pairs = jetson.received
            .filter { it.header.entries.containsKey(Session.WIRE_STAMP) }
            .map {
                val mono = (it.header.entries.getValue(Framing.KEY_MONO) as JsonValue.Num).value
                val wire = (it.header.entries.getValue(Session.WIRE_STAMP) as JsonValue.Num).value
                wire - mono
            }
        assertEquals(8, pairs.size, "no stamped frames arrived")
        assertTrue(pairs.all { it >= 0 }, "wire stamp precedes the enqueue stamp: $pairs")
    }

    @Test
    fun `a header accepted by send always survives the write`() {
        // The invariant, stated so it can fail. The size check runs before the sequence
        // number exists and before the writer adds the wire stamp, so both are substituted
        // at their widest; anything `send` accepts must therefore still fit once the real
        // values arrive.
        //
        // An earlier version of this test searched for the largest accepted padding and
        // asserted that one more byte was refused. That passes for *any* threshold,
        // including a wrong one -- setting the substitution back to 0 left it green,
        // because the search simply found the new boundary. It proved a boundary existed,
        // not that it was in the right place.
        //
        // A wire-stamped message is what exposes the difference: a probe reserving one
        // digit for a stamp that arrives with nineteen is short by eighteen bytes, so the
        // message passes the check and then throws on the writer thread.
        val clock = AtomicLong(1_000_000_000_000_000_000L)
        val (phone, jetson) = pair(clock = { clock.get() })

        val fits = longestPaddingAccepted(phone.session)
        assertTrue(fits > 0, "no padding fits at all")

        val before = jetson.received.size
        assertTrue(
            phone.session.send(Channels.CONTROL, paddedPing(fits), wantsWireStamp = true),
            "the largest fitting header was refused",
        )
        assertTrue(
            awaitFrames(jetson, before + 1),
            "a header that send() accepted never arrived: it grew past the limit at write time",
        )
        assertTrue(phone.session.isRunning, "the session died writing a header send() had accepted")

        // And one byte more is refused here, where the caller can see it, rather than on
        // the writer thread where it cannot.
        assertFalse(
            phone.session.send(Channels.CONTROL, paddedPing(fits + 1), wantsWireStamp = true),
            "a header one byte over the limit was accepted",
        )
        assertTrue(phone.session.stats().outboundRefusals.isNotEmpty(), "the refusal was not counted")
    }

    /** The largest padding length this session accepts on a wire-stamped control message. */
    private fun longestPaddingAccepted(session: Session): Int {
        var low = 0
        var high = Protocol.MAX_HEADER_BYTES
        // Binary search on a monotone predicate: one more byte of padding is one more byte
        // of header, so acceptance is monotone in the length.
        while (low < high) {
            val mid = (low + high + 1) / 2
            if (session.send(Channels.CONTROL, paddedPing(mid), wantsWireStamp = true)) low = mid else high = mid - 1
        }
        return low
    }

    /** A valid ping plus padding: extensions are additive, so an extra key is legal. */
    private fun paddedPing(length: Int): Map<String, JsonValue> =
        ping() + ("pad" to JsonValue.Text("x".repeat(length)))

    private companion object {
        const val ENQUEUE_NS = 1_000_000_000L
    }


    // -- the reader thread, and the keepalive under load ---------------------

    @Test
    fun `a handler that throws costs one frame, not the session`() {
        // The reader had no guard for this. A non-MessageError out of the application's
        // router killed the reader thread with `ended` still false, so `isRunning`
        // reported true, `send()` kept returning true, later inbound frames were consumed
        // by nobody with no counter moving, and the session finally ended as STALLED with
        // a null cause -- blaming the network for an application fault.
        val (phone, jetson) = pair(onPhoneFrame = { throw IllegalStateException("a router bug") })

        repeat(3) { index ->
            assertTrue(jetson.session.send(Channels.GPS, GpsRecord.noFix(index.toLong()).toExtensions()))
        }
        assertTrue(awaitCondition { phone.session.stats().deliveryFailures >= 3 },
            "delivery failures were not counted: ${phone.session.stats()}")

        assertTrue(phone.session.isRunning, "an application bug ended the session")
        assertEquals(3, phone.session.stats().deliveryFailures)
        assertTrue(
            phone.session.stats().lastDeliveryFailure!!.contains("IllegalStateException"),
            "the cause was not recorded: ${phone.session.stats().lastDeliveryFailure}",
        )
        // Counted apart from an inbound refusal: one is a bug here, the other is a bad
        // record from the peer, and a total that added them would hide both.
        assertTrue(phone.session.stats().inboundRefusals.isEmpty(), "counted as a refusal")

        // And the link still carries traffic the other way.
        val before = jetson.received.size
        assertTrue(phone.session.send(Channels.GPS, GpsRecord.noFix(9).toExtensions()))
        assertTrue(awaitFrames(jetson, before + 1), "the session stopped carrying traffic")
    }

    @Test
    fun `a keepalive is sent while the queue is busy, not only when it is idle`() {
        // The spec: each side sends a keepalive every 1.0 s. It used to live in the
        // writer's idle branch, so a phone under sustained camera traffic sent none -- and
        // the pre-existing test only ever exercised an idle link, as its name conceded.
        //
        // Getting this test to actually detect that took three wrong attempts, all of the
        // same shape: a *race* dressed up as a property.
        //
        //   1. Sample `outboundPending()` after the heartbeat arrives -- the producer
        //      refills between the write and the observation.
        //   2. Sample it inside the write, on the writer thread -- better, but the idle
        //      branch decides `poll() == null` and only *then* writes, and the producer
        //      refills in that gap too.
        //
        // What works is making the premise true by construction: the writer is slowed to
        // 5 ms a frame, so the producer keeps the queue permanently saturated and `poll()`
        // never returns null at all. Under the old code the idle branch is unreachable, so
        // no keepalive is written and the first assertion fails.
        val clock = AtomicLong(0)
        val heartbeats = java.util.concurrent.atomic.AtomicLong(0)
        val backlogWhenWritten = java.util.concurrent.ConcurrentLinkedQueue<Long>()
        val sessionHolder = arrayOfNulls<Session>(1)

        val (phone, _) = pair(
            clock = { clock.get() },
            onPhoneWrite = { bytes ->
                if (String(bytes, Charsets.UTF_8).contains(Session.HEARTBEAT)) {
                    heartbeats.incrementAndGet()
                    backlogWhenWritten.add(sessionHolder[0]?.outboundPending() ?: -1L)
                }
            },
            phoneWriteDelayMs = 5,
        )
        sessionHolder[0] = phone.session

        val producing = java.util.concurrent.atomic.AtomicBoolean(true)
        val producer = Thread {
            var n = 0L
            while (producing.get()) {
                phone.session.send(Channels.GPS, GpsRecord.noFix(n++).toExtensions())
            }
        }
        producer.isDaemon = true
        producer.start()
        try {
            // Let the producer saturate the depth-64 queue before the clock moves, so the
            // writer has never once seen it empty.
            assertTrue(
                awaitCondition { phone.session.outboundPending() > 0 },
                "the producer never got ahead of the writer",
            )
            clock.set(2_000_000_000L)   // past the 1 s interval

            assertTrue(
                awaitCondition { heartbeats.get() >= 1 },
                "a saturated writer sent no keepalive at all",
            )
            assertTrue(
                backlogWhenWritten.first() > 0,
                "the queue was empty when it fired, so this proves nothing: ${backlogWhenWritten.first()}",
            )
        } finally {
            producing.set(false)
            producer.join(2_000)
        }
    }

    @Test
    fun `an input stream that throws something unexpected ends the session with the cause`() {
        // readLoop caught EOFException, FramingError and IOException. Anything else -- a
        // StackOverflowError out of deeply nested JSON, an OutOfMemoryError inside a read,
        // a bug in this class -- killed the thread with `ended` still false, leaving a
        // session that reported itself healthy while nothing arrived. A stream that throws
        // a RuntimeException stands in for the whole family.
        //
        // Armed only after the handshake. Counting reads instead put the throw inside
        // readPeerHello, which has its own correct handler, so the test passed for the
        // wrong reason and pinned a path that was never broken.
        val thrown = IllegalStateException("a stream implementation bug")
        val armed = java.util.concurrent.atomic.AtomicBoolean(false)
        val (phone, jetson) = pair(phoneInput = { real ->
            object : java.io.InputStream() {
                override fun read(): Int = real.read()
                override fun read(b: ByteArray, off: Int, len: Int): Int {
                    if (armed.get()) throw thrown
                    return real.read(b, off, len)
                }
            }
        })

        armed.set(true)
        // The reader is parked in a read that was issued before the flag flipped, so it
        // needs a byte to wake on before it can reach the poisoned call.
        assertTrue(jetson.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions()))

        assertTrue(
            awaitCondition { !phone.session.isRunning },
            "the session still reports itself running",
        )
        val end = phone.ends.firstOrNull()
        assertEquals(SessionEnd.TRANSPORT_ERROR, end?.first, "ended as ${end?.first}")
        assertEquals(thrown, end?.second, "the cause was discarded")
    }

    /** Poll a condition with a deadline, so a failure is a failure and not a hang. */
    private fun awaitCondition(timeoutMs: Long = 5_000, condition: () -> Boolean): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return true
            Thread.sleep(5)
        }
        return false
    }

}
