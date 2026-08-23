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
    private var discarded = 0L
    private var closed = false

    /**
     * Offer a frame; returns the frame it displaced, if any.
     *
     * Refused once [close] has been called. Compressing a 720p frame takes tens of
     * milliseconds, so a frame whose encode began before a stop finishes after it; a
     * late arrival into a buffer nobody drains would sit there until the next session
     * and be counted as delivered.
     */
    fun offer(frame: CapturedFrame): CapturedFrame? = synchronized(lock) {
        accepted++
        if (closed) {
            discarded++
            return null
        }
        val displaced = held
        if (displaced != null) dropped++
        held = frame
        displaced
    }

    /** Take the held frame, or null. Never blocks. */
    fun drain(): CapturedFrame? = synchronized(lock) {
        val frame = held
        held = null
        if (frame != null) drained++
        frame
    }

    /**
     * Discard the held frame and refuse anything further.
     *
     * Discarding at shutdown is counted separately from a drop: a drop is a frame a
     * newer one displaced, which is the channel's normal loss, while a discard is a
     * frame abandoned because sensing ended. Folding them together would make the
     * in-flight loss look worse than it is; leaving discards uncounted -- as an
     * earlier version did -- breaks the balance identity after every stop, so frames
     * abandoned at shutdown were counted nowhere at all.
     */
    fun close() = synchronized(lock) {
        closed = true
        if (held != null) discarded++
        held = null
    }

    val stats: Stats
        get() = synchronized(lock) { Stats(accepted, dropped, drained, discarded, held != null) }

    data class Stats(
        val accepted: Long,
        val dropped: Long,
        val drained: Long,
        val discarded: Long,
        val holding: Boolean,
    ) {
        /** Every accepted frame was dropped, drained, discarded, or is still held. */
        val balances: Boolean
            get() = accepted == dropped + drained + discarded + if (holding) 1L else 0L
    }
}
