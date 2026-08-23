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
    /** Messages the peer sent that we refused: a bug there. */
    /** Frames the application's own handler threw on. Not a link failure. */
    val deliveryFailures: Long,
    val lastDeliveryFailure: String?,
    val inboundRefusals: Map<String, Long>,
    /** Messages we refused to send: a bug here. */
    val outboundRefusals: Map<String, Long>,
    val channels: Map<String, ChannelCounters>,
)

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

    private val queues = OutboundQueues()
    private val running = AtomicBoolean(true)
    private val ended = AtomicBoolean(false)

    private val framesSent = AtomicLong(0)
    private val framesReceived = AtomicLong(0)
    private val heartbeatsSent = AtomicLong(0)
    private val heartbeatsReceived = AtomicLong(0)
    private val deliveryFailures = AtomicLong(0)

    @Volatile
    private var lastDeliveryFailure: String? = null

    private val inboundRefusals = mutableMapOf<String, Long>()
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
        Framing.write(header, ByteArray(0), output)
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
        watchdogThread = Thread({ watchdogLoop() }, "dsrc-watchdog").also {
            it.isDaemon = true
            it.start()
        }
    }

    private fun readPeerHello() {
        val frame = try {
            Framing.read(input) { lastReadProgressNs.set(monoClock()) }
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
            OutboundValidation.check(channel, extensions, payload, allowed)

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
            Framing.encode(probe, payload)
        } catch (e: MessageError) {
            countOutboundRefusal(e.reason.wire)
            return false
        } catch (e: FramingError) {
            countOutboundRefusal("framing")
            return false
        } catch (e: IllegalArgumentException) {
            // The JSON encoder's own refusals -- a non-finite value that slipped the
            // check above. Counted rather than thrown at a sensor callback.
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
                try {
                    onFrame(frame)
                } catch (e: MessageError) {
                    // Drop and count. The session stays open: one bad record costs one
                    // record.
                    countInboundRefusal(e.reason.wire)
                } catch (t: Throwable) {
                    // A bug in the application's router, not in the link. It must not tear
                    // the session down -- a NullPointerException in an advisory handler is
                    // no reason to stop collecting GPS -- but it must not vanish either.
                    // With no clause here it killed the reader thread while `running`
                    // stayed true, so `isRunning` lied, `send()` kept returning true, and
                    // every later inbound frame was consumed by nobody with no counter
                    // moving. The session then ended as STALLED with a null cause,
                    // blaming the network for an application fault.
                    deliveryFailures.incrementAndGet()
                    lastDeliveryFailure = "${t.javaClass.name}: ${t.message}"
                }
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

    private fun finish(reason: SessionEnd, cause: Throwable?) {
        if (!ended.compareAndSet(false, true)) return
        running.set(false)
        // Closing the streams is what unblocks a reader parked in a read.
        runCatching { input.close() }
        runCatching { output.close() }
        writerThread?.interrupt()
        watchdogThread?.interrupt()
        onEnd(reason, cause)
    }

    private fun countInboundRefusal(reason: String) = synchronized(refusalLock) {
        inboundRefusals[reason] = (inboundRefusals[reason] ?: 0) + 1
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
            inboundRefusals = inboundRefusals.toMap(),
            outboundRefusals = outboundRefusals.toMap(),
            channels = queues.counters(),
        )
    }

    /** Messages enqueued and not yet written, for a caller checking backpressure. */
    fun outboundPending(): Long = queues.pending()

    companion object {
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

        /**
         * How often the watchdog looks.
         *
         * Well under the stall timeout, so the detection delay is a small fraction of it
         * rather than a second timeout in disguise.
         */
        private const val WATCHDOG_INTERVAL_MS = 100L
    }
}
