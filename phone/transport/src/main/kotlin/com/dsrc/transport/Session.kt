package com.dsrc.transport

import java.io.EOFException
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/** Why a session ended. The closed set from `specs/transport_protocol.md`. */
enum class SessionEnd {
    CLOSED_LOCAL,
    PEER_CLOSED,
    DISPLACED,
    STALLED,
    FRAMING_ERROR,
    TRANSPORT_ERROR,
}

/** What the peer said about itself in its hello. */
data class PeerHello(val protocolVersion: Long, val deviceId: String, val role: String)

/** Counts kept per session. The two refusal counters stay apart on purpose. */
data class SessionStats(
    val framesSent: Long,
    val framesReceived: Long,
    val heartbeatsSent: Long,
    val heartbeatsReceived: Long,
    /**
     * Frames the application's own handler threw on. Not a link failure, and not a bad
     * record from the peer: a bug in our own router, which is why it is neither of the
     * refusal maps.
     */
    val deliveryFailures: Long,
    val lastDeliveryFailure: String?,
    /**
     * Messages the peer sent that we refused: a bug there.
     *
     * Keyed channel then reason, because the spec asks for both and because the two
     * inbound channels fail in unrelated ways -- a refused `advisory` is a display glitch,
     * a refused `rate_cmd` is the phone running at the wrong rate for the rest of the drive.
     */
    val inboundRefusals: Map<String, Map<String, Long>>,
    /** Messages we refused to send: a bug here. */
    val outboundRefusals: Map<String, Long>,
    /**
     * Our own framing refusals: an over-size header, or a channel not in the table.
     *
     * Kept out of [outboundRefusals] because that map is the spec's closed vocabulary and
     * a framing condition is not one of its nine reasons.
     */
    val outboundFramingRefusals: Long,
    val lastOutboundFramingRefusal: String?,
    val channels: Map<String, ChannelCounters>,
    /** Per-channel inbound counters, the mirror of [channels]. */
    val inboundChannels: Map<String, InboundCounters>,
) {
    /**
     * Inbound refusals totalled across channels.
     *
     * Computed, not stored: a second stored copy is a second thing to keep in agreement,
     * and the per-channel map is the one the spec requires.
     */
    val inboundRefusalsByReason: Map<String, Long>
        get() = inboundRefusals.values
            .flatMap { it.entries }
            .groupingBy { it.key }
            .fold(0L) { total, entry -> total + entry.value }
}

/**
 * One connection, with the reader and writer on their own threads.
 *
 * Blocking IO on dedicated threads rather than NIO, for two reasons. It mirrors the
 * Python session one-for-one, so the two implementations stay comparable; and the spec
 * states stall detection in terms of *completed reads*, which is natural here and
 * awkward otherwise.
 *
 * The phone always opens the connection and the Jetson always listens. That is not a
 * preference: the Jetson's Tegra kernel is built without `CONFIG_NF_CONNTRACK_MARK`, so
 * Tailscale cannot install its connmark rules and the Jetson cannot originate ordinary
 * IP traffic to tailnet peers. Inbound works, and both directions work once the phone
 * has connected.
 */
class Session(
    private val input: InputStream,
    private val output: OutputStream,
    private val deviceId: String,
    private val role: String,
    private val monoClock: () -> Long,
    private val wallClock: () -> Long,
    /** Delivered messages, minus transport traffic. Called on the reader thread. */
    private val onFrame: (Frame) -> Unit,
    /** Called once, with the reason, when the session ends. */
    private val onEnd: (SessionEnd, Throwable?) -> Unit = { _, _ -> },
) {

    init {
        // A free string here decided which half of the timebase protocol this session runs:
        // `handleTimeSync` branches on `role == ROLE_PHONE`, so "Phone" or "jetson " became
        // a *responder* silently, and `sendTimeSyncPing` then threw at the caller. The spec
        // names exactly two roles and Python has an enum.
        require(role in ROLES) { "role '$role' is not one of $ROLES" }
    }

    private val queues = OutboundQueues()

    /**
     * Arriving frames, so a handler cannot stop the reader.
     *
     * Delivery used to happen on the reader thread itself. A handler that blocked froze
     * `lastReadProgressNs`, and the watchdog then ended a perfectly healthy session as
     * `STALLED` with a null cause -- the phone tearing down a working link over its own
     * slowness, then reconnecting and displacing itself. The spec asks for these queues in
     * so many words: "Inbound queues use the same policies and depths."
     */
    private val inbound = InboundQueues()

    /**
     * Answers pings, for a session in the Jetson's role.
     *
     * Was reachable from nothing at all: the class existed, had tests, and no code path
     * led to it, so the timebase exchange could not complete in either direction. It is
     * the Jetson's half -- retained here because a Kotlin session plays the Jetson in the
     * loopback tests, and because the phone's half is `sendTimeSyncPing`.
     */
    private val timeSync = TimeSyncResponder(monoClock, wallClock)
    private val running = AtomicBoolean(true)
    private val ended = AtomicBoolean(false)

    private val framesSent = AtomicLong(0)
    private val framesReceived = AtomicLong(0)
    private val heartbeatsSent = AtomicLong(0)
    private val heartbeatsReceived = AtomicLong(0)
    private val outboundFramingRefusals = AtomicLong(0)

    @Volatile
    private var lastOutboundFramingRefusal: String? = null

    private val deliveryFailures = AtomicLong(0)

    @Volatile
    private var lastDeliveryFailure: String? = null

    private val inboundRefusals = mutableMapOf<String, MutableMap<String, Long>>()
    private val outboundRefusals = mutableMapOf<String, Long>()
    private val refusalLock = Any()

    /** Last time a read completed, for the stall timer. */
    private val lastReadProgressNs = AtomicLong(0)

    @Volatile
    var peer: PeerHello? = null
        private set

    private var readerThread: Thread? = null
    private var writerThread: Thread? = null
    private var watchdogThread: Thread? = null
    private var deliveryThread: Thread? = null

    val isRunning: Boolean get() = running.get()

    /**
     * Send the hello, read the peer's, then start the reader and writer.
     *
     * Both sides send before either reads. The hello is small enough that no send can
     * block on a socket buffer, so this cannot deadlock — which is the only reason the
     * ordering is safe to state so plainly.
     */
    fun start() {
        val helloSequence = queues.reserveHelloSequence()
        val hello = JsonValue.Obj(
            mapOf(
                "protocol_version" to JsonValue.Num(Protocol.VERSION.toLong()),
                "device_id" to JsonValue.Text(deviceId),
                "role" to JsonValue.Text(role),
            )
        )
        val header = Framing.header(
            channel = Channels.CONTROL,
            sequence = helloSequence,
            monoNs = monoClock(),
            wallNs = wallClock(),
            extensions = mapOf(HELLO to hello),
            allowReserved = setOf(HELLO),
        )
        try {
            Framing.write(header, ByteArray(0), output)
        } catch (e: Exception) {
            // readPeerHello ends the session on every failure path and this did not, so a
            // hello that failed to go out left `running` true with no `onEnd` ever fired:
            // `send()` kept enqueueing into a queue with no writer, and the sequence
            // counters had already moved for a handshake that never happened.
            finish(SessionEnd.TRANSPORT_ERROR, e)
            throw e
        }
        framesSent.incrementAndGet()

        readPeerHello()

        lastReadProgressNs.set(monoClock())
        readerThread = Thread({ readLoop() }, "dsrc-reader").also { it.isDaemon = true; it.start() }
        writerThread = Thread({ writeLoop() }, "dsrc-writer").also { it.isDaemon = true; it.start() }
        // A third thread, and it has to be one. Both the reader and the writer spend
        // their time blocked -- the reader in a read, the writer in a write once the
        // peer's receive window closes -- so neither can be trusted to notice that
        // nothing has arrived. Checking on the writer looked sufficient and covered
        // exactly the case that does not matter: an idle link. A wedged peer, which is
        // what the timeout is for, blocks the writer inside output.write() and the check
        // never runs again.
        deliveryThread = Thread({ deliveryLoop() }, "dsrc-delivery").also {
            it.isDaemon = true
            it.start()
        }
        watchdogThread = Thread({ watchdogLoop() }, "dsrc-watchdog").also {
            it.isDaemon = true
            it.start()
        }
    }

    private fun readPeerHello() {
        val frame = try {
            Framing.read(input) { lastReadProgressNs.set(monoClock()) }
        } catch (e: EOFException) {
            // A peer that connected and closed without saying anything. The same
            // EOFException is PEER_CLOSED once the session is running, and reporting it as
            // a framing error here made the reason depend on when it happened rather than
            // on what happened.
            finish(SessionEnd.PEER_CLOSED, e)
            throw e
        } catch (e: Exception) {
            finish(SessionEnd.FRAMING_ERROR, e)
            throw e
        }
        framesReceived.incrementAndGet()

        // The first frame in each direction MUST be a hello. A non-hello first frame is a
        // protocol error, and no data frame is read from a peer whose version we have not
        // agreed on.
        val hello = frame.header.entries[HELLO] as? JsonValue.Obj
            ?: FramingError("first frame is not a hello").also { finish(SessionEnd.FRAMING_ERROR, it) }
                .let { throw it }

        val version = (hello.entries["protocol_version"] as? JsonValue.Num)?.value
            ?: throw FramingError("hello has no protocol_version").also { finish(SessionEnd.FRAMING_ERROR, it) }
        if (version != Protocol.VERSION.toLong()) {
            val error = FramingError("peer speaks version $version, we speak ${Protocol.VERSION}")
            finish(SessionEnd.FRAMING_ERROR, error)
            throw error
        }
        val peerRole = (hello.entries["role"] as? JsonValue.Text)?.value ?: ""
        if (peerRole !in ROLES) {
            val error = FramingError("peer claims role '$peerRole', not one of $ROLES")
            finish(SessionEnd.FRAMING_ERROR, error)
            throw error
        }
        peer = PeerHello(
            protocolVersion = version,
            deviceId = (hello.entries["device_id"] as? JsonValue.Text)?.value ?: "",
            role = (hello.entries["role"] as? JsonValue.Text)?.value ?: "",
        )
    }

    /**
     * Enqueue a message.
     *
     * Validated here against the same rules a receiver applies, because a receiver rule
     * alone leaves the sender free to emit garbage and learn about it as someone else's
     * drop counter. Our own refusals are counted separately from the peer's: one is a bug
     * here and one is a bug there, and a total that added them would hide both.
     */
    fun send(
        channel: String,
        extensions: Map<String, JsonValue>,
        payload: ByteArray = ByteArray(0),
        wantsWireStamp: Boolean = false,
    ): Boolean {
        if (!running.get()) return false
        val allowed = if (wantsWireStamp) setOf(WIRE_STAMP) else emptySet()
        // Read once, here, because this is enqueue: the spec's header table defines both
        // as the sender's clocks at enqueue, and the writer thread may not run for
        // milliseconds.
        val monoNs = monoClock()
        val wallNs = wallClock()
        try {
            // Every rule a receiver would apply, including the typed decoder. Checking
            // only reserved keys and the channel name left six of the nine refusal
            // reasons unreachable outbound, so a message our own decoder would refuse
            // went out and came back as the peer's drop counter.
            MessageValidation.check(channel, extensions, payload, allowed)

            // Direction, which the table's conditions do not cover because they are
            // role-blind and only the session knows its role. The spec makes the wrong
            // direction a protocol error, and the sender rule is meant to be the same table
            // -- so a phone emitting a pong, or a Jetson a ping, was a message we would
            // refuse on arrival and happily send.
            checkTimeSyncDirection(channel, extensions)

            // Then the size. Only two fields are still unknown here: the sequence
            // number, which `enqueue` assigns under its own lock a moment from now, and
            // the wire stamp, which the writer adds last. Both are substituted at their
            // widest so a header that fits this check cannot grow past
            // MAX_HEADER_BYTES later -- that throw would land on the writer thread, past
            // any caller's reach.
            //
            // The clocks are no longer substituted, because they are no longer guessed:
            // stamping them at enqueue is what made the real header knowable here.
            val probe = Framing.header(
                channel = channel,
                sequence = WIDEST_LONG,
                monoNs = monoNs,
                wallNs = wallNs,
                extensions = if (wantsWireStamp) extensions + (WIRE_STAMP to JsonValue.Num(WIDEST_LONG)) else extensions,
                allowReserved = allowed,
            )
            Framing.checkSizes(probe, payload.size)
        } catch (e: MessageError) {
            countOutboundRefusal(e.reason.wire)
            return false
        } catch (e: FramingError) {
            // Counted apart from the refusal reasons, not folded in as "framing". The
            // spec calls the reasons a closed vocabulary -- "exactly the second column
            // above" -- so an invented tenth key gives a consumer keying on
            // RefusalReason a bucket it cannot name. These are framing conditions: an
            // over-size header, or a channel not in the table.
            countOutboundFramingRefusal(e)
            return false
        } catch (e: JsonError) {
            // A header that cannot be encoded at all -- a lone surrogate in a string
            // value, say. Also framing: the spec's framing table lists "header is not
            // valid UTF-8". JsonError extends IllegalArgumentException, so without this
            // clause it fell through to the one below and a malformed string was counted
            // as `non_finite`, which names the wrong cause -- the entire point of
            // counting by reason.
            countOutboundFramingRefusal(e)
            return false
        } catch (e: IllegalArgumentException) {
            // Doubles.format's own refusal: a non-finite value that slipped checkAllFinite.
            // A backstop rather than the main path, and counted rather than thrown at a
            // sensor callback.
            countOutboundRefusal(RefusalReason.NON_FINITE.wire)
            return false
        }
        queues.enqueue(channel, extensions, payload, monoNs, wallNs, wantsWireStamp, allowed)
        return true
    }

    private fun writeLoop() {
        var lastHeartbeatNs = monoClock()
        try {
            while (running.get()) {
                // Checked every pass, not only when the queue is empty. The spec says
                // each side sends a keepalive every 1.0 s and Python's timer thread does
                // so unconditionally; keeping it in the idle branch meant a phone under
                // sustained camera traffic sent none at all. The consequence was masked --
                // data frames are read progress, so nothing stalled -- but the two
                // implementations disagreed, and the moment the camera pauses mid-drive
                // the peer's clock estimate has a hole with no keepalive to bridge it.
                val now = monoClock()
                if (now - lastHeartbeatNs >= (Protocol.KEEPALIVE_INTERVAL_S * 1e9).toLong()) {
                    sendHeartbeat()
                    lastHeartbeatNs = now
                }

                val message = queues.poll()
                if (message == null) {
                    Thread.sleep(WRITER_IDLE_MS)
                    continue
                }
                writeMessage(message)
            }
        } catch (e: InterruptedException) {
            // Closing interrupts the writer; not an error.
        } catch (e: IOException) {
            finish(SessionEnd.TRANSPORT_ERROR, e)
        } catch (t: Throwable) {
            // Anything else -- a FramingError from a header that grew past the limit at
            // write time, or a bug in this class -- used to kill the thread without
            // ending the session. `isRunning` stayed true, `send()` kept returning true,
            // the stall check died with the writer, and every subsequent message was
            // silently discarded for the rest of the drive. A session that cannot write
            // is over.
            finish(SessionEnd.TRANSPORT_ERROR, t)
        }
    }

    private fun hasStalled(): Boolean =
        monoClock() - lastReadProgressNs.get() >= (Protocol.STALL_TIMEOUT_S * 1e9).toLong()

    /**
     * Watches for a peer that has stopped reading progress, from outside both IO threads.
     *
     * Ending the session closes the streams, which is what unblocks a writer parked in a
     * write and a reader parked in a read.
     */
    private fun watchdogLoop() {
        try {
            while (running.get()) {
                if (hasStalled()) {
                    finish(SessionEnd.STALLED, null)
                    return
                }
                Thread.sleep(WATCHDOG_INTERVAL_MS)
            }
        } catch (e: InterruptedException) {
            // Closing interrupts it; not an error.
        }
    }

    private fun writeMessage(message: Outbound) {
        val extensions = if (message.wantsWireStamp) {
            // Stamped immediately before the bytes leave, which is the whole point of the
            // field: `t_mono_ns` is the enqueue stamp and so excludes the time the frame
            // waited behind others, and the difference between the two *is* the queueing
            // delay a timebase estimate has to remove.
            //
            // Necessarily read after the enqueue stamps, which were taken in `send`, so
            // the wire stamp cannot precede them. It used to: the stamped extension map
            // was built before the header's own clock call, so every stamped frame
            // reported a negative queueing delay.
            message.extensions + (WIRE_STAMP to JsonValue.Num(monoClock()))
        } else {
            message.extensions
        }
        val header = Framing.header(
            channel = message.channel,
            sequence = message.sequence,
            monoNs = message.monoNs,
            wallNs = message.wallNs,
            extensions = extensions,
            allowReserved = message.allowReserved,
        )
        synchronized(output) {
            Framing.write(header, message.payload, output)
        }
        framesSent.incrementAndGet()
    }

    private fun sendHeartbeat() {
        // A sequence number drawn directly, not by enqueueing and immediately polling.
        // `enqueue` appends to the tail and `poll` takes the head, so that round trip
        // returned whatever was already queued -- destroying an application's control
        // message with `dropped` still zero, and then writing the heartbeat twice with a
        // duplicate sequence number. Reachable by any send(control, ...) landing between
        // the writer finding the queue empty and this call.
        val sequence = queues.nextSequenceFor(Channels.CONTROL)
        val header = Framing.header(
            channel = Channels.CONTROL,
            sequence = sequence,
            monoNs = monoClock(),
            wallNs = wallClock(),
            extensions = mapOf(HEARTBEAT to JsonValue.Bool(true)),
            allowReserved = setOf(HEARTBEAT),
        )
        synchronized(output) {
            Framing.write(header, ByteArray(0), output)
        }
        framesSent.incrementAndGet()
        heartbeatsSent.incrementAndGet()
    }

    /**
     * The timebase exchange, which the transport owns the way it owns keepalives.
     *
     * Direction is not symmetric and the spec is explicit about it: *the phone initiates
     * and the Jetson only ever answers*. So the role decides which half runs, and the
     * wrong half arriving is a protocol error rather than something to interpret --
     * "treating one as the other produces an offset with the sign inverted", a plausible
     * number that is exactly wrong. Both wrong-direction cases are counted as
     * `unknown_value`, which the spec names for them.
     *
     * @return true if the frame was consumed here and must not be delivered.
     */
    /**
     * Refuse a timebase message going the wrong way.
     *
     * The phone initiates and the Jetson only ever answers, so a pong from a phone or a
     * ping from a Jetson is the protocol error the receiver counts as `unknown_value`. It
     * is counted here under the same reason, because "before a message goes out, it must
     * satisfy the same table" and this is the one condition the table cannot state on its
     * own.
     */
    private fun checkTimeSyncDirection(channel: String, extensions: Map<String, JsonValue>) {
        if (channel != Channels.CONTROL) return
        if (HELLO in extensions || HEARTBEAT in extensions) return
        val message = TimeSyncMessage.fromWire(extensions, ByteArray(0))
        val wrongWay = if (role == ROLE_PHONE) !message.isPing else message.isPing
        if (wrongWay) {
            throw MessageError(
                RefusalReason.UNKNOWN_VALUE,
                "a $role must not send ${if (message.isPing) "a ping" else "a pong"}",
            )
        }
    }

    private fun handleTimeSync(message: Received): Boolean {
        val frame = message.frame
        val decoded = try {
            TimeSyncMessage.fromWire(frame.header.entries, frame.payload)
        } catch (e: MessageError) {
            countInboundRefusal(frame.channel, e.reason.wire)
            return true
        }

        if (role == ROLE_PHONE) {
            if (decoded.isPing) {
                // Nobody should be pinging the initiator.
                countInboundRefusal(frame.channel, RefusalReason.UNKNOWN_VALUE.wire)
                return true
            }
            // A pong is the answer to our own ping, and the estimate is built above the
            // transport, so it is delivered rather than absorbed.
            return false
        }

        val reply = timeSync.reply(decoded, message.recvMonoNs, message.recvWallNs)
        if (reply == null) {
            // A responder receiving a pong: the other wrong direction.
            countInboundRefusal(frame.channel, RefusalReason.UNKNOWN_VALUE.wire)
            return true
        }
        // Wire-stamped, because t3 is the pong's departure and the enqueue stamp would
        // carry this session's own queueing delay into the peer's offset.
        send(Channels.CONTROL, reply.toExtensions(), wantsWireStamp = true)
        return true
    }

    /**
     * Send a timebase ping. The phone's half of the exchange.
     *
     * Exposed rather than driven from here: the cadence and the estimator belong above the
     * transport, which does not know what the samples are for.
     */
    fun sendTimeSyncPing(exchangeId: Long): Boolean {
        require(role == ROLE_PHONE) { "only the initiator sends pings; this session is '$role'" }
        val ping = TimeSyncMessage(
            captureMonoNs = monoClock(),
            exchangeId = exchangeId,
            // The writer stamps the real departure; zero is the spec's placeholder.
            wireMonoNs = 0,
            peerRecvMonoNs = null,
            peerRecvWallNs = null,
            peerWireMonoNs = null,
        )
        return send(Channels.CONTROL, ping.toExtensions(), wantsWireStamp = true)
    }

    /**
     * Deliver arriving frames on a thread of their own.
     *
     * Everything a handler can do slowly or wrongly happens here rather than on the reader:
     * the typed decode, the timebase exchange, and the application callback. The reader's
     * only job is to read and enqueue, so its progress -- which is what the stall timeout
     * watches -- cannot be held up by anything above the transport.
     */
    private fun deliveryLoop() {
        try {
            while (running.get()) {
                val message = inbound.poll()
                if (message == null) {
                    Thread.sleep(DELIVERY_IDLE_MS)
                    continue
                }
                deliver(message)
            }
        } catch (e: InterruptedException) {
            // Closing interrupts it; not an error.
        } catch (t: Throwable) {
            // The last line of defence. If this thread dies, nothing is delivered while
            // the session still reports itself healthy -- the shape the reader and writer
            // both had before.
            finish(SessionEnd.TRANSPORT_ERROR, t)
        }
    }

    private fun deliver(message: Received) {
        val frame = message.frame
        // The receiving half of the refusal table. Without it a malformed frame was handed
        // to the application unchecked, and `inboundRefusals` moved only when a handler
        // happened to throw -- so a deliberately bad record from the live Python peer
        // crossed and was counted nowhere.
        try {
            MessageValidation.checkInbound(frame)
        } catch (e: MessageError) {
            countInboundRefusal(frame.channel, e.reason.wire)
            inbound.countRefused(frame.channel)
            return
        }

        if (frame.channel == Channels.CONTROL && handleTimeSync(message)) {
            inbound.countDelivered(frame.channel)
            return
        }

        try {
            onFrame(frame)
            inbound.countDelivered(frame.channel)
        } catch (e: MessageError) {
            // Drop and count. The session stays open: one bad record costs one record.
            countInboundRefusal(frame.channel, e.reason.wire)
            inbound.countRefused(frame.channel)
        } catch (t: Throwable) {
            // A bug in the application's router, not in the link. It must not tear the
            // session down -- a NullPointerException in an advisory handler is no reason to
            // stop collecting GPS -- but it must not vanish either. With no clause here it
            // killed the delivering thread while `running` stayed true, so `isRunning`
            // lied, `send()` kept returning true, and every later frame was consumed by
            // nobody with no counter moving.
            deliveryFailures.incrementAndGet()
            lastDeliveryFailure = "${t.javaClass.name}: ${t.message}"
        }
    }

    private fun readLoop() {
        try {
            while (running.get()) {
                val frame = Framing.read(input) { lastReadProgressNs.set(monoClock()) }
                framesReceived.incrementAndGet()

                // The transport generates keepalives, so it also absorbs them. The
                // reserved key is honoured on `control` only: the same key arriving on a
                // data channel is a caller's message and MUST be delivered.
                if (frame.channel == Channels.CONTROL &&
                    (frame.header.entries[HEARTBEAT] as? JsonValue.Bool)?.value == true
                ) {
                    heartbeatsReceived.incrementAndGet()
                    continue
                }

                // Both clocks at one instant, here, where the frame arrived. The
                // timebase depends on it: a receipt stamp taken when a handler got round
                // to the message makes the responder's service interval arbitrary.
                inbound.offer(Received(frame, monoClock(), wallClock()))
            }
        } catch (e: EOFException) {
            finish(SessionEnd.PEER_CLOSED, e)
        } catch (e: FramingError) {
            finish(SessionEnd.FRAMING_ERROR, e)
        } catch (e: IOException) {
            finish(if (running.get()) SessionEnd.TRANSPORT_ERROR else SessionEnd.CLOSED_LOCAL, e)
        } catch (t: Throwable) {
            // Everything else the read path can raise: a StackOverflowError out of deeply
            // nested JSON, an OutOfMemoryError inside a read, a bug in this class. Without
            // it the thread died with `ended` still false -- the same shape the writer had
            // before round 1, a session reporting itself healthy while nothing arrives.
            finish(SessionEnd.TRANSPORT_ERROR, t)
        }
    }

    fun close() = finish(SessionEnd.CLOSED_LOCAL, null)

    /**
     * End the session once, recording the first reason offered.
     *
     * The compare-and-set is the whole ordering guarantee, and it is stronger than it
     * looks: a caller that loses the race cannot overwrite the reason, so a `close()` and a
     * watchdog firing at the same instant produce exactly one `onEnd` and one reason. An
     * extra flag to make `close()` win was tried and removed -- it changed nothing that any
     * test could observe, because the CAS had already decided. If the watchdog wins, it
     * won by detecting a genuinely-expired timeout before `close()` arrived, which is not
     * a misnamed shutdown.
     */
    private fun finish(reason: SessionEnd, cause: Throwable?) {
        if (!ended.compareAndSet(false, true)) return
        running.set(false)
        // Closing the streams is what unblocks a reader parked in a read.
        runCatching { input.close() }
        runCatching { output.close() }
        // The writer and the watchdog both sleep, so an interrupt reaches them. The
        // reader does not: it is parked inside a stream read, and Thread.interrupt() does
        // not unblock a socket read at all -- the flag is set and consumed by nothing.
        // `input.close()` above is what actually releases it.
        //
        // An interrupt was added here for symmetry and is removed for the same reason the
        // phantom-STALLED guard was: the justification was tidiness rather than behaviour,
        // deleting it survived the suite, and a line that reads as load-bearing while doing
        // nothing is worse than its absence.
        writerThread?.interrupt()
        watchdogThread?.interrupt()
        deliveryThread?.interrupt()
        // Counted, not left to a derived `pending` on a dead session.
        queues.abandonAll()
        inbound.abandonAll()
        onEnd(reason, cause)
    }

    /**
     * Count an inbound refusal against its channel and its reason.
     *
     * The spec asks for both: "the drop is counted per channel and per reason". Keyed by
     * reason alone, four thousand drops could not be attributed to a channel, and the
     * `advisory` and `rate_cmd` paths would be indistinguishable in a summary -- one is a
     * display glitch and the other is the phone running at the wrong rate for the rest of
     * the drive.
     */
    private fun countInboundRefusal(channel: String, reason: String) = synchronized(refusalLock) {
        val perChannel = inboundRefusals.getOrPut(channel) { mutableMapOf() }
        perChannel[reason] = (perChannel[reason] ?: 0) + 1
    }

    private fun countOutboundFramingRefusal(cause: Throwable) {
        outboundFramingRefusals.incrementAndGet()
        lastOutboundFramingRefusal = "${cause.javaClass.simpleName}: ${cause.message}"
    }

    private fun countOutboundRefusal(reason: String) = synchronized(refusalLock) {
        outboundRefusals[reason] = (outboundRefusals[reason] ?: 0) + 1
    }

    fun stats(): SessionStats = synchronized(refusalLock) {
        SessionStats(
            framesSent = framesSent.get(),
            framesReceived = framesReceived.get(),
            heartbeatsSent = heartbeatsSent.get(),
            heartbeatsReceived = heartbeatsReceived.get(),
            deliveryFailures = deliveryFailures.get(),
            lastDeliveryFailure = lastDeliveryFailure,
            inboundRefusals = inboundRefusals.mapValues { (_, byReason) -> byReason.toMap() },
            outboundFramingRefusals = outboundFramingRefusals.get(),
            lastOutboundFramingRefusal = lastOutboundFramingRefusal,
            outboundRefusals = outboundRefusals.toMap(),
            channels = queues.counters(),
            inboundChannels = inbound.counters(),
        )
    }

    /** Messages enqueued and not yet written, for a caller checking backpressure. */
    fun outboundPending(): Long = queues.pending()

    companion object {
        const val ROLE_PHONE = "phone"
        const val ROLE_JETSON = "jetson"

        /** The only two roles the spec defines. */
        val ROLES = setOf(ROLE_PHONE, ROLE_JETSON)
        const val HELLO = "hello"
        const val HEARTBEAT = "heartbeat"
        const val WIRE_STAMP = "t_wire_mono_ns"

        /**
         * The widest decimal a `Long` field can occupy, used for every transport-owned
         * field in the size probe.
         *
         * `Long.MIN_VALUE` rather than MAX: it is one character longer, because of the
         * sign.
         */
        const val WIDEST_LONG = Long.MIN_VALUE

        private const val WRITER_IDLE_MS = 2L
        private const val DELIVERY_IDLE_MS = 2L

        /**
         * How often the watchdog looks.
         *
         * Well under the stall timeout, so the detection delay is a small fraction of it
         * rather than a second timeout in disguise.
         */
        private const val WATCHDOG_INTERVAL_MS = 100L
    }
}
