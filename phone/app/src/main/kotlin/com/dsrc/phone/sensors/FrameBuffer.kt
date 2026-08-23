package com.dsrc.phone.sensors

/**
 * Holds the newest frame, and counts the ones it displaced.
 *
 * Depth one and latest-wins, because that is the `camera` channel's policy in
 * `specs/transport_protocol.md`: a newer frame is worth more than an older one, and
 * a replaced frame counts as dropped. Matching the policy here rather than upstream
 * of the transport means the camera and the wire agree about what loss means, and
 * there is one place that counts it.
 */
class FrameBuffer {

    private val lock = Any()
    private var held: CapturedFrame? = null
    private var accepted = 0L
    private var dropped = 0L
    private var drained = 0L

    /** Offer a frame; returns the frame it displaced, if any. */
    fun offer(frame: CapturedFrame): CapturedFrame? = synchronized(lock) {
        val displaced = held
        if (displaced != null) dropped++
        held = frame
        accepted++
        displaced
    }

    /** Take the held frame, or null. Never blocks. */
    fun drain(): CapturedFrame? = synchronized(lock) {
        val frame = held
        held = null
        if (frame != null) drained++
        frame
    }

    fun clear() = synchronized(lock) {
        // Discarding on shutdown is not a drop: nothing displaced it and nothing was
        // owed it. Counting it would make the totals disagree with the frames that
        // were actually lost in flight.
        held = null
    }

    val stats: Stats get() = synchronized(lock) { Stats(accepted, dropped, drained, held != null) }

    data class Stats(
        val accepted: Long,
        val dropped: Long,
        val drained: Long,
        val holding: Boolean,
    ) {
        /** Every accepted frame was dropped, drained, or is still held. */
        val balances: Boolean get() = accepted == dropped + drained + if (holding) 1L else 0L
    }
}
