package com.dsrc.transport

/**
 * One message type, in both directions.
 *
 * **One**, not a ping type and a pong type: the channel is the discriminator for every
 * other message on this wire, and a second type on one channel would need a `kind` field
 * to tell them apart, which is exactly what this protocol refuses. The null convention
 * carries it instead — a ping has the three peer fields null, a pong has them set.
 */
data class TimeSyncMessage(
    val captureMonoNs: Long,
    val exchangeId: Long,
    /** Written by the transport just before the bytes leave. Zero until then. */
    val wireMonoNs: Long,
    /** Null on a ping; on a pong, when the responder read the ping. */
    val peerRecvMonoNs: Long?,
    /** Null on a ping; on a pong, the responder's wall clock at the same instant. */
    val peerRecvWallNs: Long?,
    /** Null on a ping; on a pong, the ping's own wire stamp echoed back. */
    val peerWireMonoNs: Long?,
) {
    /** A ping has no peer fields; a pong has all three. */
    val isPing: Boolean get() = peerRecvMonoNs == null && peerRecvWallNs == null && peerWireMonoNs == null

    fun toExtensions(): Map<String, JsonValue> = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
        KEY_EXCHANGE to JsonValue.Num(exchangeId),
        KEY_WIRE to JsonValue.Num(wireMonoNs),
        KEY_PEER_RECV_MONO to Fields.toWire(peerRecvMonoNs),
        KEY_PEER_RECV_WALL to Fields.toWire(peerRecvWallNs),
        KEY_PEER_WIRE to Fields.toWire(peerWireMonoNs),
    )

    companion object {
        const val KEY_EXCHANGE = "exchange_id"
        const val KEY_WIRE = "t_wire_mono_ns"
        const val KEY_PEER_RECV_MONO = "t_peer_recv_mono_ns"
        const val KEY_PEER_RECV_WALL = "t_peer_recv_wall_ns"
        const val KEY_PEER_WIRE = "t_peer_wire_mono_ns"

        fun fromWire(extensions: Map<String, JsonValue>, payload: ByteArray): TimeSyncMessage {
            Fields.checkNoPayload(payload, Channels.CONTROL)

            val peerRecvMono = Fields.optionalInt(extensions, KEY_PEER_RECV_MONO)
            val peerRecvWall = Fields.optionalInt(extensions, KEY_PEER_RECV_WALL)
            val peerWire = Fields.optionalInt(extensions, KEY_PEER_WIRE)

            // All-or-nothing. A partially filled pong would let a consumer compute an
            // offset from a mixture of set and missing terms, which is worse than
            // refusing: the arithmetic would run and produce a number.
            val set = listOf(peerRecvMono, peerRecvWall, peerWire).count { it != null }
            if (set != 0 && set != 3) {
                throw MessageError(
                    RefusalReason.MISSING_FIELD,
                    "a pong needs all three peer stamps or none; $set of 3 are set",
                )
            }

            return TimeSyncMessage(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                exchangeId = Fields.requireInt(extensions, KEY_EXCHANGE),
                wireMonoNs = Fields.requireInt(extensions, KEY_WIRE),
                peerRecvMonoNs = peerRecvMono,
                peerRecvWallNs = peerRecvWall,
                peerWireMonoNs = peerWire,
            )
        }
    }
}

/**
 * The phone's whole role in the shared timebase: answer, and record nothing.
 *
 * Task 16 established that the converting side must be the initiating side, and the
 * Jetson is the side that converts — a responder sees only t2 and t3 and has no path to
 * the offset. So there is no estimator here, no offset, no history and no conversion, and
 * that is deliberate rather than unfinished.
 *
 * The pong echoes the ping's wire stamp because the initiator needs its own t1 back to
 * pair with t4, and the *wire* stamp is the one that matters: `t_mono_ns` is an enqueue
 * stamp and so includes however long the frame waited behind others, which for a timebase
 * estimate is the dominant error, larger than the network.
 */
class TimeSyncResponder(
    private val monoClock: () -> Long,
    private val wallClock: () -> Long,
) {
    var pongsSent: Long = 0
        private set

    var pingsIgnored: Long = 0
        private set

    /**
     * Build the reply to a ping, or null if this was not a ping.
     *
     * A pong arriving here is not an error — the peer may be answering a ping we never
     * sent, or a duplicate — but it is not ours to answer, and replying to it would put
     * two responders in a loop.
     */
    fun reply(message: TimeSyncMessage, recvMonoNs: Long, recvWallNs: Long): TimeSyncMessage? {
        if (!message.isPing) {
            pingsIgnored++
            return null
        }
        pongsSent++
        return TimeSyncMessage(
            captureMonoNs = monoClock(),
            exchangeId = message.exchangeId,
            // Written by the writer at send time; zero is the placeholder the spec uses.
            wireMonoNs = 0,
            peerRecvMonoNs = recvMonoNs,
            peerRecvWallNs = recvWallNs,
            // The ping's own wire stamp, straight back. Substituting our own clock here
            // would silently replace the initiator's t1 with a value from a different
            // device, and the offset it computed would be wrong by the whole link delay.
            peerWireMonoNs = message.wireMonoNs,
        )
    }

    /** Convenience: read a frame, and produce the reply to send, if any. */
    fun replyTo(frame: Frame): TimeSyncMessage? {
        val message = TimeSyncMessage.fromWire(frame.header.entries, frame.payload)
        return reply(message, monoClock(), wallClock())
    }
}
