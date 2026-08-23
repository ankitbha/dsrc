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
        try {
            Fields.checkReserved(extensions)
            if (!Channels.isKnown(channel)) {
                throw MessageError(RefusalReason.NO_TYPED_MESSAGE, "unknown channel '$channel'")
            }
            // Encoded here, with the widest possible wire stamp substituted, so the
            // caller's own encoder proves the header fits before it is ever queued.
            val probe = Framing.header(
                channel = channel,
                sequence = 0,
                monoNs = monoClock(),
                wallNs = wallClock(),
                extensions = if (wantsWireStamp) extensions + (WIRE_STAMP to JsonValue.Num(WIRE_STAMP_RESERVE)) else extensions,
                allowReserved = if (wantsWireStamp) setOf(WIRE_STAMP) else emptySet(),
            )
            Framing.encode(probe, payload)
        } catch (e: MessageError) {
            countOutboundRefusal(e.reason.wire)
            return false
        } catch (e: FramingError) {
            countOutboundRefusal("framing")
            return false
        }
        queues.enqueue(channel, extensions, payload, wantsWireStamp)
        return true
    }

    private fun writeLoop() {
        var lastHeartbeatNs = monoClock()
        try {
            while (running.get()) {
                val message = queues.poll()
                if (message == null) {
                    val now = monoClock()
                    if (now - lastHeartbeatNs >= (Protocol.KEEPALIVE_INTERVAL_S * 1e9).toLong()) {
                        sendHeartbeat()
                        lastHeartbeatNs = now
                    }
                    // Checked here rather than on the reader, so a reader blocked in a
                    // read cannot delay noticing that nothing has arrived.
                    val since = now - lastReadProgressNs.get()
                    if (since >= (Protocol.STALL_TIMEOUT_S * 1e9).toLong()) {
                        finish(SessionEnd.STALLED, null)
                        return
                    }
                    Thread.sleep(WRITER_IDLE_MS)
                    continue
                }
                writeMessage(message)
            }
        } catch (e: InterruptedException) {
            // Closing interrupts the writer; not an error.
        } catch (e: IOException) {
            finish(SessionEnd.TRANSPORT_ERROR, e)
        }
    }

    private fun writeMessage(message: Outbound) {
        val extensions = if (message.wantsWireStamp) {
            // Stamped immediately before the bytes leave, not at enqueue: `t_mono_ns` is
            // an enqueue stamp and so includes however long the frame waited behind
            // others, which for a timebase estimate is the dominant error, larger than
            // the network.
            message.extensions + (WIRE_STAMP to JsonValue.Num(monoClock()))
        } else {
            message.extensions
        }
        val header = Framing.header(
            channel = message.channel,
            sequence = message.sequence,
            monoNs = monoClock(),
            wallNs = wallClock(),
            extensions = extensions,
            allowReserved = message.allowReserved + if (message.wantsWireStamp) setOf(WIRE_STAMP) else emptySet(),
        )
        synchronized(output) {
            Framing.write(header, message.payload, output)
        }
        framesSent.incrementAndGet()
    }

    private fun sendHeartbeat() {
        val sequence = queues.enqueue(
            Channels.CONTROL,
            mapOf(HEARTBEAT to JsonValue.Bool(true)),
            ByteArray(0),
            allowReserved = setOf(HEARTBEAT),
        ).sequence
        val header = Framing.header(
            channel = Channels.CONTROL,
            sequence = sequence,
            monoNs = monoClock(),
            wallNs = wallClock(),
            extensions = mapOf(HEARTBEAT to JsonValue.Bool(true)),
            allowReserved = setOf(HEARTBEAT),
        )
        // Taken straight back off the queue: it was enqueued only to draw a sequence
        // number, since a keepalive consumes one like any other frame.
        queues.poll()
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
                }
            }
        } catch (e: EOFException) {
            finish(SessionEnd.PEER_CLOSED, e)
        } catch (e: FramingError) {
            finish(SessionEnd.FRAMING_ERROR, e)
        } catch (e: IOException) {
            finish(if (running.get()) SessionEnd.TRANSPORT_ERROR else SessionEnd.CLOSED_LOCAL, e)
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
         * Substituted at validation time so the caller's own encoder sizes the header
         * against the widest stamp it could ever carry.
         *
         * The Python side hit exactly this: a header sized for a placeholder overflowed
         * `MAX_HEADER_BYTES` once a 19-digit stamp replaced it, which killed the writer
         * thread while `send()` still returned true and nothing was transmitted.
         */
        const val WIRE_STAMP_RESERVE = Long.MAX_VALUE

        private const val WRITER_IDLE_MS = 2L
    }
}
