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
        /** The socket this side reads from, so a test can inject a raw frame at it. */
        val socket: Socket? = null,
        /** The socket the *other* side writes on. */
        var peerSocket: Socket? = null,
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
        onJetsonFrame: ((Frame) -> Unit)? = null,
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
            return Peer(session, received, ends, { latch }, socket)
        }

        val phone = build(clientSocket, "phone", onPhoneFrame ?: {})
        val jetson = build(serverSocket, "jetson", onJetsonFrame ?: {})

        // Both send their hello before either reads, which is why starting them on two
        // threads cannot deadlock here.
        val phoneStart = Thread { phone.session.start() }
        val jetsonStart = Thread { jetson.session.start() }
        phoneStart.start(); jetsonStart.start()
        phoneStart.join(5_000); jetsonStart.join(5_000)
        phone.peerSocket = jetson.socket
        jetson.peerSocket = phone.socket
        // Armed only after the handshake, so the hello is never held.
        gates.forEach { it.armed = true }
        return phone to jetson
    }

    /**
     * Write a frame straight onto the wire, bypassing the sending session.
     *
     * Needed because the sender rule now refuses a wrong-direction timebase message, which
     * is correct and which means `send` can no longer produce one. Two of these tests
     * previously relied on that hole: they exercised the *receiver's* direction rule by
     * having a peer send what a peer should never be able to send.
     *
     * Safe only with a frozen clock, so no keepalive interleaves with this write on the
     * same stream.
     */
    private fun injectRaw(into: Peer, extensions: Map<String, JsonValue>, sequence: Long = 900) {
        val socket = into.peerSocket ?: error("no peer socket recorded")
        val header = Framing.header(
            channel = Channels.CONTROL,
            sequence = sequence,
            monoNs = 1_000,
            wallNs = 2_000,
            extensions = extensions,
            allowReserved = setOf(Session.WIRE_STAMP),
        )
        socket.getOutputStream().write(Framing.encode(header, ByteArray(0)))
        socket.getOutputStream().flush()
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
    /** A valid advisory. The channel has a typed decoder now, so a stub no longer passes. */
    private fun advisory(captureMonoNs: Long = 1_000) = AdvisoryMessage(
        captureMonoNs = captureMonoNs,
        recSpeedMps = 13.4,
        recSpeedDisplay = 30.0,
        currentSpeedDisplay = 28.0,
        units = "mph",
        headwayTargetS = 2.0,
        laneText = "keep",
        mergeText = "normal",
        trafficText = "clear",
        confidence = 0.87,
        confidenceLabel = "high",
        action = mapOf(
            "desired_speed_bin" to "nominal",
            "desired_headway_bin" to "normal",
            "lane_preference" to "keep",
            "merge_mode" to "normal",
        ),
    ).toExtensions()

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
        val wire = WireLog()
        val (phone, _) = pair(onPhoneWrite = wire::record)
        // A real ping, not an arbitrary payload: the sender rule runs the typed decoder,
        // and `control`'s typed message is the time-sync exchange.
        assertTrue(phone.session.sendTimeSyncPing(exchangeId = 1))
        assertTrue(awaitCondition { wire.on(Channels.CONTROL).any { it.sequence == 1L } },
            "the ping did not spend sequence 1: ${wire.on(Channels.CONTROL).map { it.sequence }}")
        // Zero belongs to the hello, and nothing else may reuse it.
        assertEquals(
            listOf(0L, 1L),
            wire.on(Channels.CONTROL).map { it.sequence }.distinct().sorted().take(2),
        )
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

        // "Arrive" now means two different things, and the distinction is the inbound
        // queue's. Everything written reaches the peer's *transport*; what reaches its
        // application is that minus whatever its own queue shed, and both are counted.
        // Asserting the old single equality would blame the sender for the receiver's
        // shedding -- it failed at 65 written against 64 delivered.
        val receiver = jetson.session.stats().inboundChannels.getValue(Channels.GPS)
        assertEquals(
            sender.sent,
            receiver.received,
            "everything written should reach the peer's transport on a loopback socket",
        )
        val delivered = synchronized(jetson.received) { jetson.received.size }.toLong()
        assertEquals(
            receiver.received,
            delivered + receiver.dropped + receiver.refused,
            "receiver accounting must balance: delivered $delivered, $receiver",
        )
        assertTrue(delivered > 0, "nothing was delivered at all")
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

        // Loss now has two sites and both are counted, which is the point. The sender's
        // queue drops on overflow, and the receiver's *inbound* queue sheds when delivery
        // falls behind -- so what the application never saw is the sum of the two, not the
        // sender's drops alone. Before the inbound queue existed this was one term, and
        // asserting it as one term now would blame the sender for the receiver's shedding.
        val receiver = jetson.session.stats().inboundChannels.getValue(Channels.GPS)
        assertEquals(
            sender.sent,
            receiver.received,
            "everything written must arrive: sent ${sender.sent}, arrived ${receiver.received}",
        )

        val highest = sequences.last()
        val missing = (highest + 1) - sequences.size
        assertEquals(
            sender.dropped + receiver.dropped + receiver.refused,
            missing,
            "every sequence number the application never saw must be accounted for: " +
                "highest $highest, delivered ${sequences.size}, sender $sender, receiver $receiver",
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
    fun `an unknown channel is refused as a framing condition`() {
        // The spec's framing table: "`ch` not in the channel table -> framing error,
        // session ends", and the read path already treats it that way. On the send path it
        // must not end our own session, but it must not be filed as `no_typed_message`
        // either -- that reason means something narrower, a channel that is in the table
        // and has no typed message.
        //
        // The previous version asserted only that *some* refusal was counted, which is
        // blind to the reason being wrong.
        val (phone, _) = pair()
        assertFalse(phone.session.send("not_a_channel", emptyMap()))

        assertEquals(1, phone.session.stats().outboundFramingRefusals)
        assertTrue(
            phone.session.stats().outboundRefusals.isEmpty(),
            "filed under a refusal reason: ${phone.session.stats().outboundRefusals}",
        )
        assertTrue(phone.session.isRunning, "refusing to send ended our own session")
    }

    @Test
    fun `a header that cannot be encoded is a framing refusal, not a non-finite one`() {
        // A lone surrogate is not valid text, so the header cannot be encoded at all --
        // the spec's framing table lists "header is not valid UTF-8". JsonError extends
        // IllegalArgumentException, so before the dedicated clause it fell through to the
        // one below and a malformed string was counted as `non_finite`: a counter naming
        // the wrong cause, which is the entire point of counting by reason.
        val (phone, _) = pair()
        val lonelyHighSurrogate = "\uD800"

        assertFalse(
            phone.session.send(
                Channels.GPS,
                GpsRecord.noFix(1).toExtensions() + ("note" to JsonValue.Text(lonelyHighSurrogate)),
            ),
            "an unencodable header was accepted",
        )
        assertEquals(1, phone.session.stats().outboundFramingRefusals)
        assertTrue(
            RefusalReason.NON_FINITE.wire !in phone.session.stats().outboundRefusals,
            "counted as non_finite: ${phone.session.stats().outboundRefusals}",
        )
        assertTrue(
            phone.session.stats().outboundRefusals.isEmpty(),
            "filed under a refusal reason: ${phone.session.stats().outboundRefusals}",
        )
    }

    @Test
    fun `every outbound refusal reason is in the spec's closed vocabulary`() {
        // "The reasons are a closed vocabulary -- exactly the second column above, so an
        // implementation reads off which to emit rather than guessing." A tenth key gives
        // a consumer keying on RefusalReason a bucket it cannot name; "framing" was one.
        val (phone, _) = pair()
        val vocabulary = RefusalReason.entries.map { it.wire }.toSet()

        // One of each thing a caller can get wrong, so the map is populated rather than
        // trivially empty.
        phone.session.send("not_a_channel", emptyMap())
        phone.session.send(Channels.CONTROL, mapOf("probe" to JsonValue.Num(1)))
        phone.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions() - "valid")
        phone.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions() + ("hdop" to JsonValue.Real(Double.NaN)))
        phone.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions() + (Session.HEARTBEAT to JsonValue.Bool(true)))
        phone.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions(), payload = byteArrayOf(1))
        phone.session.send(Channels.CONTROL, paddedPing(Protocol.MAX_HEADER_BYTES))

        val reported = phone.session.stats().outboundRefusals.keys
        assertTrue(reported.isNotEmpty(), "nothing was refused, so this proves nothing")
        assertTrue(
            reported.all { it in vocabulary },
            "outside the vocabulary: ${reported - vocabulary}",
        )
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
        assertTrue(jetson.session.send(Channels.ADVISORY, advisory(captureMonoNs = 1)))
        assertTrue(awaitFrames(phone, 1), "the frame never arrived")
        Thread.sleep(200)
        assertTrue(phone.session.isRunning, "the session must stay open")
        assertEquals(1, phone.session.stats().inboundRefusals[Channels.ADVISORY]?.get("wrong_type"),
            "not attributed to the advisory channel: ${phone.session.stats().inboundRefusals}")

        // And it keeps working afterwards.
        assertTrue(jetson.session.send(Channels.ADVISORY, advisory(captureMonoNs = 2)))
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
        val wire = WireLog()
        val (phone, _) = pair(onPhoneWrite = wire::record)
        assertTrue(phone.session.sendTimeSyncPing(exchangeId = 99))
        // Long enough for at least one keepalive to fire alongside it.
        Thread.sleep(((Protocol.KEEPALIVE_INTERVAL_S * 2 + 1) * 1000).toLong())

        val control = wire.on(Channels.CONTROL)
        val exchanges = control.map { (it.header.entries["exchange_id"] as? JsonValue.Num)?.value }
        assertTrue(99L in exchanges, "our ping was destroyed by a heartbeat; got $exchanges")

        val sequences = control.map { it.sequence }
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
        val wire = WireLog()
        val (phone, _) = pair(clock = { clock.addAndGet(1_000) }, onPhoneWrite = wire::record)

        repeat(8) { exchange ->
            assertTrue(phone.session.sendTimeSyncPing(exchangeId = exchange.toLong()))
        }
        assertTrue(
            awaitCondition { wire.snapshot().count { it.header.entries.containsKey(Session.WIRE_STAMP) } >= 8 },
            "timebase frames were never written",
        )

        val pairs = wire.snapshot()
            .filter { it.header.entries.containsKey(Session.WIRE_STAMP) }
            .map {
                val mono = (it.header.entries.getValue(Framing.KEY_MONO) as JsonValue.Num).value
                val wire = (it.header.entries.getValue(Session.WIRE_STAMP) as JsonValue.Num).value
                wire - mono
            }
        assertEquals(8, pairs.size, "no stamped frames were written")
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
        val wire = WireLog()
        val (phone, _) = pair(clock = { clock.get() }, onPhoneWrite = wire::record)

        val fits = longestPaddingAccepted(phone.session)
        assertTrue(fits > 0, "no padding fits at all")

        val before = wire.snapshot().size
        assertTrue(
            phone.session.send(Channels.CONTROL, paddedPing(fits), wantsWireStamp = true),
            "the largest fitting header was refused",
        )
        assertTrue(
            awaitCondition { wire.snapshot().size > before },
            "a header that send() accepted was never written: it grew past the limit at write time",
        )
        assertTrue(phone.session.isRunning, "the session died writing a header send() had accepted")

        // And one byte more is refused here, where the caller can see it, rather than on
        // the writer thread where it cannot.
        // A delta, because the binary search above refuses on its way to the boundary.
        val framingBefore = phone.session.stats().outboundFramingRefusals
        assertFalse(
            phone.session.send(Channels.CONTROL, paddedPing(fits + 1), wantsWireStamp = true),
            "a header one byte over the limit was accepted",
        )
        // The framing counter, not the refusal map: an over-size header is a framing
        // condition and the refusal reasons are a closed vocabulary of nine.
        assertEquals(
            framingBefore + 1,
            phone.session.stats().outboundFramingRefusals,
            "the refusal was not counted",
        )
        assertTrue(
            phone.session.stats().outboundRefusals.isEmpty(),
            "a framing condition was filed as a refusal reason: ${phone.session.stats().outboundRefusals}",
        )
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

    /**
     * Frames the phone wrote, decoded from the bytes as they went out.
     *
     * The right instrument for any claim about what the *sender* produced. Reading the
     * peer's delivered frames was never quite that, and stopped working entirely once the
     * transport began absorbing the control channel: a responder answers a ping instead of
     * handing it up, so an assertion about the ping's sequence number could no longer see
     * it at all.
     */
    private class WireLog {
        private val frames = mutableListOf<Frame>()

        fun record(bytes: ByteArray) {
            val frame = runCatching {
                Framing.read(java.io.ByteArrayInputStream(bytes))
            }.getOrNull() ?: return
            synchronized(frames) { frames.add(frame) }
        }

        fun snapshot(): List<Frame> = synchronized(frames) { frames.toList() }

        fun on(channel: String): List<Frame> = snapshot().filter { it.channel == channel }
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


    // -- the timebase exchange ----------------------------------------------

    @Test
    fun `a ping is answered and the pong reaches the initiator`() {
        // The whole exchange, which could not complete at all: TimeSyncResponder existed,
        // had tests, and was reachable from no code path, so a ping was delivered to an
        // application that has no handler for it and no pong was ever produced.
        val clock = AtomicLong(1_000)
        val (phone, jetson) = pair(clock = { clock.addAndGet(1_000) })

        assertTrue(phone.session.sendTimeSyncPing(exchangeId = 42))
        assertTrue(awaitFrames(phone, 1), "no pong came back")

        val pong = phone.received.single { it.channel == Channels.CONTROL }
        val decoded = TimeSyncMessage.fromWire(pong.header.entries, pong.payload)
        assertFalse(decoded.isPing, "the answer was itself a ping")
        assertEquals(42, decoded.exchangeId, "the pong changed the exchange id")
        // All three peer fields are set together, per the spec.
        assertTrue(decoded.peerRecvMonoNs != null && decoded.peerRecvWallNs != null &&
            decoded.peerWireMonoNs != null, "a pong with a partial peer triple: $decoded")

        // t1 echoed back. Substituting the responder's own clock here would replace the
        // initiator's departure with a reading from a different device, and the offset
        // would be wrong by the whole link delay.
        val ping = jetson.received.firstOrNull { it.channel == Channels.CONTROL }
        if (ping != null) {
            val sent = TimeSyncMessage.fromWire(ping.header.entries, ping.payload)
            assertEquals(sent.wireMonoNs, decoded.peerWireMonoNs)
        }
        assertTrue(phone.session.stats().inboundRefusals.isEmpty(),
            "the pong was refused: ${phone.session.stats().inboundRefusals}")
    }

    @Test
    fun `a phone refuses a ping, because the phone is the initiator`() {
        // The spec: "The phone initiates and the Jetson only ever answers... A Jetson
        // receiving a pong, or a phone receiving a ping, is a protocol error", counted as
        // unknown_value, "because the alternative is treating one as the other and
        // silently producing an offset with the sign inverted".
        // A frozen clock, so no keepalive interleaves with the raw write.
        val (phone, _) = pair(clock = { 1_000 })
        val before = phone.received.size

        // Injected raw: the sender rule now refuses a Jetson sending a ping, which is
        // correct, so a peer's `send` can no longer produce one.
        injectRaw(phone, ping(exchangeId = 7))
        assertTrue(
            awaitCondition {
                phone.session.stats().inboundRefusalsByReason[RefusalReason.UNKNOWN_VALUE.wire] == 1L
            },
            "a ping to the initiator was not refused as unknown_value: " +
                "${phone.session.stats().inboundRefusals}",
        )
        assertEquals(before, phone.received.size, "the ping was delivered as well as counted")
        assertTrue(phone.session.isRunning, "a wrong-direction message ended the session")
    }

    @Test
    fun `a responder refuses a pong`() {
        val (_, jetson) = pair(clock = { 1_000 })
        val pong = TimeSyncMessage(
            captureMonoNs = 1_000,
            exchangeId = 11,
            wireMonoNs = 0,
            peerRecvMonoNs = 2_000,
            peerRecvWallNs = 1_755_648_000_000_000_000,
            peerWireMonoNs = 1_500,
        )
        injectRaw(jetson, pong.toExtensions())
        assertTrue(
            awaitCondition {
                jetson.session.stats().inboundRefusalsByReason[RefusalReason.UNKNOWN_VALUE.wire] == 1L
            },
            "a pong to a responder was not refused: ${jetson.session.stats().inboundRefusals}",
        )
    }

    @Test
    fun `only the initiator may send a ping`() {
        val (_, jetson) = pair()
        // A responder that pinged would put two responders in a loop, each answering the
        // other's answer.
        val refused = runCatching { jetson.session.sendTimeSyncPing(exchangeId = 1) }
        assertTrue(refused.isFailure, "a responder was allowed to initiate")
        assertTrue(
            refused.exceptionOrNull() is IllegalArgumentException,
            "wrong failure: ${refused.exceptionOrNull()}",
        )
    }


    // -- how a session ends -------------------------------------------------

    @Test
    fun `a peer that connects and says nothing is peer_closed, not a framing error`() {
        // The same EOFException is PEER_CLOSED once the session is running. Reporting it
        // as a framing error during the handshake made the reason depend on when it
        // happened rather than on what happened, and the spec lists peer_closed as its own
        // row.
        val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).also { servers.add(it) }
        val clientSocket = Socket(InetAddress.getLoopbackAddress(), server.localPort).also { sockets.add(it) }
        val silent = server.accept().also { sockets.add(it) }

        val ends = mutableListOf<Pair<SessionEnd, Throwable?>>()
        val session = Session(
            input = clientSocket.getInputStream(),
            output = clientSocket.getOutputStream(),
            deviceId = "test-phone",
            role = "phone",
            monoClock = { System.nanoTime() },
            wallClock = { 0 },
            onFrame = {},
            onEnd = { reason, cause -> ends.add(reason to cause) },
        ).also { sessions.add(it) }

        silent.close()
        runCatching { session.start() }

        assertEquals(listOf(SessionEnd.PEER_CLOSED), ends.map { it.first })
        assertFalse(session.isRunning)
    }

    @Test
    fun `a hello that cannot be written ends the session instead of leaving it half-open`() {
        // readPeerHello ended the session on every failure path and the hello write did
        // not, so a failed write left `running` true with no onEnd ever fired: send() kept
        // enqueueing into a queue with no writer, and the sequence counters had already
        // moved for a handshake that never happened.
        val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).also { servers.add(it) }
        val clientSocket = Socket(InetAddress.getLoopbackAddress(), server.localPort).also { sockets.add(it) }
        server.accept().also { sockets.add(it) }

        val ends = mutableListOf<Pair<SessionEnd, Throwable?>>()
        val session = Session(
            input = clientSocket.getInputStream(),
            output = object : java.io.OutputStream() {
                override fun write(b: Int) = throw java.io.IOException("cable yanked")
                override fun write(b: ByteArray, off: Int, len: Int) = throw java.io.IOException("cable yanked")
            },
            deviceId = "test-phone",
            role = "phone",
            monoClock = { System.nanoTime() },
            wallClock = { 0 },
            onFrame = {},
            onEnd = { reason, cause -> ends.add(reason to cause) },
        ).also { sessions.add(it) }

        val started = runCatching { session.start() }
        assertTrue(started.isFailure, "start() reported success with no hello written")
        assertEquals(listOf(SessionEnd.TRANSPORT_ERROR), ends.map { it.first })
        assertFalse(session.isRunning, "the session still reports itself running")
        assertFalse(session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions()),
            "a dead session accepted a message")
    }

    @Test
    fun `closing a healthy session reports a local close`() {
        val (phone, _) = pair()
        phone.session.close()
        assertTrue(awaitCondition { phone.ends.isNotEmpty() }, "the session never ended")
        assertEquals(listOf(SessionEnd.CLOSED_LOCAL), phone.ends.map { it.first })
    }

    @Test
    fun `a session ends exactly once, whoever gets there first`() {
        // Round 2 reported a "phantom STALLED" when close() races the watchdog. It does not
        // hold, and the reason is worth writing down: finish() is a compare-and-set, so a
        // caller that loses cannot overwrite the reason. A flag to make close() win was
        // tried and removed -- removing it again survives 200 rounds, because the CAS had
        // already decided, and shipping a guard nothing can observe is worse than shipping
        // none.
        //
        // If the watchdog does win, it won by detecting a timeout that had genuinely
        // expired before close() arrived, so STALLED is not a misnaming. What must hold is
        // that there is exactly one end and one reason, and that is what this asserts --
        // the clock is forced past the timeout precisely so both contenders are live.
        repeat(200) {
            val clock = AtomicLong(0)
            val (phone, _) = pair(clock = { clock.get() })
            clock.set((Protocol.STALL_TIMEOUT_S * 2e9).toLong())
            phone.session.close()
            assertTrue(awaitCondition(2_000) { phone.ends.isNotEmpty() }, "the session never ended")
            Thread.sleep(1)   // give a losing contender room to try
            assertEquals(1, phone.ends.size, "two ends recorded: ${phone.ends.map { it.first }}")
            assertTrue(
                phone.ends.first().first in setOf(SessionEnd.CLOSED_LOCAL, SessionEnd.STALLED),
                "unexpected reason ${phone.ends.first().first}",
            )
            cleanup()
        }
    }


    @Test
    fun `inbound refusals are attributed to their own channel`() {
        // The spec asks for both axes: "the drop is counted per channel and per reason".
        // Keyed by reason alone, a summary could not tell a refused advisory -- a display
        // glitch -- from a refused rate_cmd, which leaves the phone sensing at the wrong
        // rate for the rest of the drive.
        val (phone, jetson) = pair(onPhoneFrame = { frame ->
            when (frame.channel) {
                Channels.ADVISORY -> throw MessageError(RefusalReason.WRONG_TYPE, "pretend")
                Channels.GPS -> throw MessageError(RefusalReason.OUT_OF_RANGE, "pretend")
                else -> Unit
            }
        })

        assertTrue(jetson.session.send(Channels.ADVISORY, advisory(captureMonoNs = 1)))
        // Waited for, not fired back to back: `advisory` is latest_wins at depth 1, so a
        // second send would displace the first before the writer ever saw it and only one
        // would arrive. The queue is doing its job; the test was asking the wrong thing.
        assertTrue(awaitCondition { phone.session.stats().inboundRefusals.containsKey(Channels.ADVISORY) })
        assertTrue(jetson.session.send(Channels.ADVISORY, advisory(captureMonoNs = 2)))
        assertTrue(jetson.session.send(Channels.GPS, GpsRecord.noFix(3).toExtensions()))

        assertTrue(
            awaitCondition {
                val refusals = phone.session.stats().inboundRefusals
                refusals[Channels.ADVISORY]?.get("wrong_type") == 2L &&
                    refusals[Channels.GPS]?.get("out_of_range") == 1L
            },
            "not attributed per channel: ${phone.session.stats().inboundRefusals}",
        )
        // And the by-reason view still totals correctly across channels.
        val byReason = phone.session.stats().inboundRefusalsByReason
        assertEquals(2L, byReason["wrong_type"])
        assertEquals(1L, byReason["out_of_range"])
    }


    @Test
    fun `a near-limit header still fits once the sequence number is three digits`() {
        // The size probe substitutes `seq` and the wire stamp at their widest, and the two
        // substitutions were only *jointly* pinned: setting the seq one back to 0 survived
        // the whole suite. The arithmetic is why. Long.MIN_VALUE is 20 characters and a
        // real wire stamp is 19, so the stamp reserves one surplus byte -- enough to absorb
        // a two-digit sequence. The other boundary test never gets past a one-digit control
        // sequence, so it could not see half of what it was named for.
        //
        // Past 100 the surplus is gone and the header grows after the check. In production
        // that needs about a hundred seconds of keepalives and a near-cap header: mid-drive,
        // not in a test, and it kills the session on the writer thread.
        val clock = AtomicLong(1_000_000_000_000_000_000L)
        val (phone, _) = pair(clock = { clock.get() })

        // Burn the control sequence past three digits.
        repeat(140) { assertTrue(phone.session.sendTimeSyncPing(exchangeId = it.toLong())) }
        val deadline = System.currentTimeMillis() + 10_000
        while (phone.session.outboundPending() > 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(2)
        }
        val control = phone.session.stats().channels.getValue(Channels.CONTROL)
        assertTrue(control.enqueued >= 100, "the sequence never reached three digits: $control")

        val fits = longestPaddingAccepted(phone.session)
        assertTrue(fits > 0, "no padding fits at all")
        assertTrue(
            phone.session.send(Channels.CONTROL, paddedPing(fits), wantsWireStamp = true),
            "the largest fitting header was refused",
        )
        // The point: what send() accepted must still fit when a three-digit sequence and a
        // real wire stamp replace the substitutions.
        val settle = System.currentTimeMillis() + 5_000
        while (phone.session.outboundPending() > 0 && System.currentTimeMillis() < settle) {
            Thread.sleep(2)
        }
        assertTrue(
            phone.session.isRunning,
            "the session died writing a header send() accepted: " +
                "framing refusals ${phone.session.stats().outboundFramingRefusals}",
        )
        assertEquals(
            listOf<SessionEnd>(),
            phone.ends.map { it.first },
            "the session ended: ${phone.ends}",
        )
    }


    // -- the inbound side ----------------------------------------------------

    @Test
    fun `a handler that blocks does not stall the session`() {
        // Delivery used to run on the reader thread. A handler that blocked froze
        // lastReadProgressNs, and the watchdog then ended a perfectly healthy link as
        // STALLED with a null cause: the phone tearing itself down over its own slowness,
        // then reconnecting and displacing itself. The peer is not at fault and the reason
        // said it was.
        val gate = CountDownLatch(1)
        val clock = AtomicLong(0)
        val (phone, jetson) = pair(clock = { clock.get() }, onPhoneFrame = { gate.await() })
        try {
            assertTrue(jetson.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions()))
            // Wait until the handler is genuinely stuck, so the premise is active.
            assertTrue(
                awaitCondition { phone.received.isNotEmpty() },
                "the handler was never reached, so nothing is blocked",
            )

            // Past the stall timeout, then keep the peer talking so the *reader* still
            // makes progress. Under the old arrangement the reader was inside the blocked
            // handler and could not.
            clock.set((Protocol.STALL_TIMEOUT_S * 3e9).toLong())
            val deadline = System.currentTimeMillis() + 1_500
            var sent = 2L
            while (System.currentTimeMillis() < deadline) {
                jetson.session.send(Channels.GPS, GpsRecord.noFix(sent++).toExtensions())
                clock.addAndGet(1_000_000)
                Thread.sleep(20)
            }

            assertTrue(
                phone.session.isRunning,
                "a blocked handler ended the session as ${phone.ends.map { it.first }}",
            )
            assertEquals(emptyList(), phone.ends.map { it.first }, "the session ended")
        } finally {
            gate.countDown()
        }
    }

    @Test
    fun `inbound frames shed at the channel's depth and are counted`() {
        // The spec: "Inbound queues use the same policies and depths." There were none, so
        // there was nothing to count and the interop test's balance assertion was checked
        // only against Python's numbers.
        val gate = CountDownLatch(1)
        val (phone, jetson) = pair(onPhoneFrame = { gate.await() })
        try {
            val depth = Channels.policy(Channels.GPS).depth
            val offered = depth * 4
            repeat(offered) { i ->
                // Paced on the *sender's* queue. Firing them unpaced filled the jetson's
                // own depth-64 outbound queue instead, so 64 arrived and none were shed --
                // the test was measuring outbound overflow while claiming inbound.
                while (jetson.session.outboundPending() > 0) Thread.onSpinWait()
                jetson.session.send(Channels.GPS, GpsRecord.noFix(i.toLong()).toExtensions())
            }
            assertTrue(
                awaitCondition {
                    val gps = phone.session.stats().inboundChannels[Channels.GPS]
                    gps != null && gps.received >= offered.toLong()
                },
                "not everything arrived: ${phone.session.stats().inboundChannels}",
            )
            val gps = phone.session.stats().inboundChannels.getValue(Channels.GPS)
            assertTrue(gps.dropped > 0, "a queue ${depth} deep took ${offered} frames: $gps")
            assertTrue(phone.session.isRunning, "shedding ended the session")
        } finally {
            gate.countDown()
        }
    }

    @Test
    fun `the pong carries the reader's receipt stamp, not the handler's clock`() {
        // The spec states this as a requirement rather than a check: the receipt must be
        // the stamp the transport took on arrival. The initiator computes the responder's
        // service interval as t3 - t2, so a t2 read when a handler got round to the message
        // makes that difference arbitrary -- and nothing on the wire can detect it.
        //
        // It was equivalent only while delivery was synchronous. With a delivery queue the
        // two instants are genuinely different, which is what this test forces.
        val clock = AtomicLong(1_000)
        val gate = CountDownLatch(1)
        val pongs = java.util.concurrent.ConcurrentLinkedQueue<Frame>()
        val (phone, _) = pair(
            clock = { clock.get() },
            onPhoneFrame = { pongs.add(it) },
            // The responder's own delivery thread is blocked behind this.
            onJetsonFrame = { gate.await() },
        )

        // A data frame first, to occupy the responder's delivery thread.
        assertTrue(phone.session.send(Channels.GPS, GpsRecord.noFix(1).toExtensions()))
        Thread.sleep(200)
        // The ping is read and stamped now, at this clock.
        val stampedAt = clock.get()
        assertTrue(phone.session.sendTimeSyncPing(exchangeId = 5))
        Thread.sleep(200)
        // Then the clock moves before the responder gets to it -- by a second, which is
        // enormous next to a real receipt stamp and still comfortably inside the 5 s stall
        // timeout. A 9 s jump killed both sessions instead: `pair` gives the two peers the
        // *same* injected clock, so advancing it past the timeout stalls them both, which
        // is the trap that made an earlier stall test flaky.
        val jump = 1_000_000_000L
        clock.set(stampedAt + jump)
        gate.countDown()

        assertTrue(awaitCondition { pongs.isNotEmpty() }, "no pong arrived")
        val decoded = TimeSyncMessage.fromWire(pongs.first().header.entries, pongs.first().payload)
        // A handling-time reading would be the jumped value.
        assertTrue(
            decoded.peerRecvMonoNs!! < stampedAt + jump,
            "the receipt was read at handling time: ${decoded.peerRecvMonoNs}, " +
                "which is the jumped clock rather than the ~$stampedAt the reader saw",
        )
    }

    @Test
    fun `messages queued when the session ends are counted as abandoned`() {
        // They were in no counter at all: an orphaned message showed up only in the derived
        // `pending`, on a session that had ended, which is indistinguishable from a
        // counting bug. Python's own comment for the equivalent counter says exactly that.
        val gate = CountDownLatch(1)
        val (phone, _) = pair(phoneOutputGate = gate)
        repeat(40) { i ->
            assertTrue(phone.session.send(Channels.GPS, GpsRecord.noFix(i.toLong()).toExtensions()))
        }
        phone.session.close()
        gate.countDown()

        val gps = phone.session.stats().channels.getValue(Channels.GPS)
        assertEquals(40, gps.enqueued)
        assertTrue(gps.abandoned > 0, "nothing was counted as abandoned: $gps")
        // Every message is under exactly one heading, and `pending` is no longer the only
        // place a loss appears.
        assertEquals(
            gps.enqueued,
            gps.sent + gps.dropped + gps.abandoned,
            "the accounting does not add up: $gps",
        )
        assertEquals(0, gps.pending, "pending should be zero once nothing is queued: $gps")
    }


    @Test
    fun `the sender rule covers exchange direction, which the table cannot state`() {
        // The refusal table's conditions are role-blind, so nothing in it can express "a
        // phone must not send a pong". Only the session knows its role, and the sender rule
        // is supposed to be the same table -- so this was a message we would refuse on
        // arrival and happily send. Two of the tests above relied on the hole.
        val (phone, jetson) = pair()

        val pong = TimeSyncMessage(
            captureMonoNs = 1_000,
            exchangeId = 3,
            wireMonoNs = 0,
            peerRecvMonoNs = 2_000,
            peerRecvWallNs = 1_755_648_000_000_000_000,
            peerWireMonoNs = 1_500,
        )
        assertFalse(
            phone.session.send(Channels.CONTROL, pong.toExtensions(), wantsWireStamp = true),
            "a phone sent a pong",
        )
        assertEquals(
            1L,
            phone.session.stats().outboundRefusals[RefusalReason.UNKNOWN_VALUE.wire],
            "refused for the wrong reason: ${phone.session.stats().outboundRefusals}",
        )

        assertFalse(
            jetson.session.send(Channels.CONTROL, ping(exchangeId = 4), wantsWireStamp = true),
            "a jetson sent a ping",
        )
        assertEquals(
            1L,
            jetson.session.stats().outboundRefusals[RefusalReason.UNKNOWN_VALUE.wire],
            "refused for the wrong reason: ${jetson.session.stats().outboundRefusals}",
        )

        // And the right direction still goes out.
        assertTrue(phone.session.sendTimeSyncPing(exchangeId = 5))
    }

}
