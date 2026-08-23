package com.dsrc.phone.sensors

/**
 * Decides which frames to keep to hit a target rate.
 *
 * Two failure modes it is built to avoid, which pull in opposite directions:
 *
 * Scheduling each slot from *now* undershoots. At a 30 Hz source and a 10 Hz target,
 * accepting at t=0 and then waiting 100 ms means the next candidate is the frame at
 * 133 ms, so the achieved rate is 7.5 Hz -- a quarter low, and low in a way that
 * looks like the camera underperforming rather than the gate miscounting.
 *
 * Scheduling each slot from the *previous slot* hits the target exactly, but carries
 * a deficit: after a stall the next slot is far in the past, so every missed slot is
 * paid back at once as a burst. On a thermally throttled phone the stall and the
 * burst arrive in that order, which is the worst possible pairing.
 *
 * So slots advance from the previous slot, and are reset to now whenever they have
 * fallen a whole period behind. Exact rate while keeping up, no catch-up burst after
 * falling behind.
 */
class RateGate(hz: Double) {

    var hz: Double = hz
        private set

    private var periodNs: Long = periodNsFor(hz)

    /** No slot has been issued yet, so the first candidate is always accepted. */
    private var nextSlotNs: Long = Long.MIN_VALUE

    /** When the last frame was accepted, so a rate change can be anchored to it. */
    private var lastAcceptNs: Long = Long.MIN_VALUE

    val periodNanos: Long get() = periodNs

    fun accept(nowNs: Long): Boolean {
        if (nowNs < nextSlotNs) return false
        val advanced = nextSlotNs + periodNs
        // `advanced <= nowNs` means a whole period has already been missed. Advancing
        // again from there would issue one slot per missed period as fast as frames
        // arrive; starting from now drops the backlog instead.
        nextSlotNs = if (advanced <= nowNs || nextSlotNs == Long.MIN_VALUE) nowNs + periodNs else advanced
        lastAcceptNs = nowNs
        return true
    }

    /**
     * Change the target rate.
     *
     * The pending slot is re-anchored to one *new* period after the last accepted
     * frame, which is the only way to satisfy both of the things a rate change has to
     * do. Leaving the old slot alone means a rise from 0.2 Hz to 10 Hz waits out the
     * remaining five seconds of the old period before honouring the new rate --
     * indistinguishable from the command being ignored. Clearing the slot instead
     * would let repeated commands emit a frame each, driving the camera faster than
     * any rate ever asked for.
     *
     * Anchoring to the last acceptance is what makes it safe in both directions: the
     * next frame can come sooner than the old schedule allowed after an increase, and
     * is pushed out after a decrease, but never comes sooner than the new rate does.
     */
    fun setRate(newHz: Double) {
        val newPeriod = periodNsFor(newHz)
        if (lastAcceptNs != Long.MIN_VALUE) {
            // Exactly one new period after the last accepted frame, in both directions.
            // Taking the earlier of the old and new slots honours an increase but
            // ignores a decrease, which would keep emitting at the old faster rate.
            nextSlotNs = lastAcceptNs + newPeriod
        }
        periodNs = newPeriod
        hz = newHz
    }

    companion object {
        /** The wire's range for a rate, from `specs/transport_protocol.md`. */
        const val MIN_HZ_EXCLUSIVE = 0.0
        const val MAX_HZ = 1000.0

        fun periodNsFor(hz: Double): Long {
            require(hz > MIN_HZ_EXCLUSIVE && hz <= MAX_HZ) {
                // A zero is read as a period, so the field that meant "10 Hz" would say
                // "never" -- the exact case the protocol's sender rule calls out.
                "rate $hz is outside (${MIN_HZ_EXCLUSIVE}, $MAX_HZ] Hz"
            }
            require(hz.isFinite()) { "rate must be finite, was $hz" }
            return (1_000_000_000.0 / hz).toLong().coerceAtLeast(1)
        }
    }
}
