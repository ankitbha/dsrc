package com.dsrc.phone.sensors

/**
 * Decides which frames to keep to hit a target rate.
 *
 * Three failure modes it is built against, and they pull against each other.
 *
 * Scheduling each slot from *now* undershoots. At a 30 Hz source and a 10 Hz target,
 * accepting at t=0 and then waiting 100 ms means the next candidate is the frame at
 * 133 ms, so the achieved rate is 7.5 Hz -- a quarter low, and low in a way that looks
 * like the camera underperforming rather than the gate miscounting.
 *
 * Scheduling each slot from the *previous slot* hits the target exactly, but carries a
 * deficit: after a stall the next slot is far in the past, so every missed slot is paid
 * back at once as a burst. On a thermally throttled phone the stall and the burst
 * arrive in that order, which is the worst possible pairing.
 *
 * And a slot computed by addition overflows. At the bottom of the wire's legal range a
 * period is larger than `Long.MAX_VALUE` nanoseconds, so `now + period` wraps negative
 * and the gate stands permanently open -- a command meaning "almost never" producing
 * full-rate capture, which is the exact inversion of what a rate limit is for.
 *
 * So: slots advance from the previous slot, are reset to now once a whole period has
 * been missed, and every addition saturates instead of wrapping.
 */
class RateGate(hz: Double) {

    private val lock = Any()

    private var periodNs: Long = periodNsFor(hz)
    private var currentHz: Double = hz

    /**
     * Whether any frame has been accepted.
     *
     * A boolean rather than a sentinel timestamp: `Long.MIN_VALUE` is a legal value in
     * the domain, so a sentinel could be indistinguishable from a real stamp.
     */
    private var hasAccepted = false
    private var nextSlotNs: Long = 0
    private var lastAcceptNs: Long = 0

    val hz: Double get() = synchronized(lock) { currentHz }

    val periodNanos: Long get() = synchronized(lock) { periodNs }

    /**
     * Whether to keep the frame captured at [nowNs].
     *
     * Synchronized because the analyzer thread calls this while any thread can call
     * [setRate]; without it a rate change can stay invisible to the analyzer
     * indefinitely, and two concurrent callers can both be admitted in one slot.
     */
    fun accept(nowNs: Long): Boolean = synchronized(lock) {
        if (hasAccepted && nowNs < nextSlotNs) return false

        val advanced = addSaturating(nextSlotNs, periodNs)
        // `advanced <= nowNs` means a whole period has already been missed. Advancing
        // again from there would issue one slot per missed period as fast as frames
        // arrive; starting from now drops the backlog instead.
        nextSlotNs = if (!hasAccepted || advanced <= nowNs) addSaturating(nowNs, periodNs) else advanced
        lastAcceptNs = nowNs
        hasAccepted = true
        return true
    }

    /**
     * Change the target rate.
     *
     * A change re-anchors the pending slot to one *new* period after the last accepted
     * frame, which is the only way to satisfy both things a change has to do. Leaving
     * the old slot means a rise from 0.2 Hz to 10 Hz waits out the remaining five
     * seconds before honouring the new rate -- indistinguishable from the command being
     * ignored. Clearing it instead would let repeated commands emit a frame each,
     * driving the camera faster than any rate ever asked for. Anchoring to the last
     * acceptance is safe in both directions: the next frame can come sooner than the old
     * schedule allowed after an increase and is pushed out after a decrease, but never
     * comes sooner than the new rate permits.
     *
     * **An unchanged rate is a no-op**, and that is not a micro-optimisation. Re-anchoring
     * unconditionally converts "schedule from the previous slot" into "schedule from
     * now", which is the undershoot above -- so a peer that simply repeats the current
     * rate would quietly lose a quarter of the frame rate.
     */
    fun setRate(newHz: Double) = synchronized(lock) {
        val newPeriod = periodNsFor(newHz)
        currentHz = newHz
        if (newPeriod == periodNs) return
        periodNs = newPeriod
        if (hasAccepted) nextSlotNs = addSaturating(lastAcceptNs, newPeriod)
    }

    companion object {
        /** The wire's range for a rate, from `specs/transport_protocol.md`. */
        const val MIN_HZ_EXCLUSIVE = 0.0
        const val MAX_HZ = 1000.0

        /**
         * Nanoseconds per frame at [hz].
         *
         * Saturates rather than overflowing: the wire admits rates low enough that a
         * period exceeds `Long.MAX_VALUE` nanoseconds, and a saturated period means
         * "accept once, then never", which is what such a rate asks for.
         *
         * The saturation is `Double.toLong()`'s, not ours. There used to be an explicit
         * `if (period >= Long.MAX_VALUE.toDouble())` here, and replacing the whole
         * expression with a bare `period.toLong()` changed no result any test could see --
         * because Kotlin/JVM already clamps a `Double` past the range instead of wrapping.
         * The docstring credited the branch with preventing an overflow the language does
         * not have. A guard nothing can reach is a claim a reader has to disprove, so it is
         * gone and the mechanism is named instead.
         */
        fun periodNsFor(hz: Double): Long {
            // NaN fails `> 0` and an infinity fails `<= MAX_HZ`, so this one check
            // covers the non-finite cases too.
            require(hz > MIN_HZ_EXCLUSIVE && hz <= MAX_HZ) {
                // A zero is read as a period, so the field that should say "10 Hz"
                // instead says "never" -- the case the protocol's sender rule calls out.
                "rate $hz is outside ($MIN_HZ_EXCLUSIVE, $MAX_HZ] Hz"
            }
            return (1_000_000_000.0 / hz).toLong()
        }

        /**
         * `a + b`, clamped at [Long.MAX_VALUE] instead of wrapping.
         *
         * The addend is always a period from [periodNsFor], which is positive by
         * construction, so the `b >= 0` half of the old test was unreachable -- dropping it
         * left every test passing. But dropping it outright would silently return
         * `Long.MAX_VALUE` for a negative addend, where the old branch returned the correct
         * smaller sum: removing an unreachable guard must not change what happens if it
         * ever becomes reachable. So the precondition is stated as a precondition. A
         * negative addend now fails loudly at its caller instead of being clamped to
         * "never" and looking like a rate.
         *
         * Overflow with a positive addend is exactly a sum that came out below the base.
         */
        internal fun addSaturating(a: Long, b: Long): Long {
            require(b >= 0) { "addSaturating takes a non-negative addend; got $b" }
            val sum = a + b
            return if (sum < a) Long.MAX_VALUE else sum
        }
    }
}
