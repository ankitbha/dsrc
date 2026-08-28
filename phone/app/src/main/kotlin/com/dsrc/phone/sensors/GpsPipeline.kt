package com.dsrc.phone.sensors

import com.dsrc.phone.config.SensingConfig
import com.dsrc.transport.GpsRecord
import java.util.concurrent.atomic.AtomicLong

/**
 * The GPS path: rate-gate the readings, hand the survivors to a sink, count everything.
 *
 * Deliberately the same shape as [CameraPipeline], including reusing [RateGate], because
 * "at the commanded rate" has to mean the same thing on every modality. A second rate
 * implementation would drift from the first, and the gate is the piece that took four
 * attempts to get right.
 *
 * Unlike the camera there is no buffer here. `gps` is a `reliable` channel at depth 64, so
 * the transport's own queue is the buffer and adding another in front of it would mean two
 * places dropping for different reasons — and only one of them counted where the peer can
 * see it as a sequence gap.
 */
class GpsPipeline(
    config: SensingConfig,
    /** Where a kept reading goes. Returning false means the transport refused it. */
    private val sink: (GpsReading) -> Boolean,
) {
    private val gate = RateGate(config.gpsHz)

    private val seen = AtomicLong(0)
    private val accepted = AtomicLong(0)
    private val gated = AtomicLong(0)
    private val refusedStopped = AtomicLong(0)
    private val refusedBySink = AtomicLong(0)
    private val delivered = AtomicLong(0)
    private val invalidFixes = AtomicLong(0)
    private val nonMonotonic = AtomicLong(0)

    @Volatile
    private var running = true

    /**
     * Whether this pipeline has been stopped.
     *
     * Exists so the service can assert it is reading the stats *after* the stop. Three of
     * the counters -- `abandoned`, `refusedStopped` and the buffer's `discarded` -- can
     * only move once this is true, so a stats line read before it is structurally all
     * zeroes.
     */
    val isStopped: Boolean get() = !running

    private val lastFixNs = AtomicLong(Long.MIN_VALUE)

    fun offer(reading: GpsReading): Boolean {
        seen.incrementAndGet()
        if (!running) {
            refusedStopped.incrementAndGet()
            return false
        }

        // A fix stamp that goes backwards is worth counting rather than hiding: it means
        // the platform handed us an out-of-order update, and the receiver's freshness
        // arithmetic is built on these stamps being monotonic.
        //
        // Against the previous *delivery*, and advanced on every one, not only on the
        // fixes the gate kept. The baseline used to advance after the gate, so a
        // reversal between two gated fixes was invisible: at 1 Hz, offers at 0, 500 ms
        // and 400 ms counted nothing, though the platform had plainly handed us 500
        // before 400. What is being detected is a property of the delivery, and the
        // gate has no business in it. `ImuPipeline` says it counts this "for the same
        // reason GPS counts it" and carries the corrected form; this is the original it
        // was crediting.
        val previous = lastFixNs.getAndSet(reading.fixMonoNs)
        if (previous != Long.MIN_VALUE && reading.fixMonoNs < previous) {
            nonMonotonic.incrementAndGet()
        }

        if (!gate.accept(reading.fixMonoNs)) {
            gated.incrementAndGet()
            return false
        }
        accepted.incrementAndGet()
        if (!reading.record.valid) invalidFixes.incrementAndGet()

        return if (sink(reading)) {
            delivered.incrementAndGet()
            true
        } else {
            // The transport refused it -- a full queue, or a session that has ended.
            // Counted apart from a gate rejection because the causes are unrelated.
            refusedBySink.incrementAndGet()
            false
        }
    }

    fun setRate(hz: Double) = gate.setRate(hz)

    val rateHz: Double get() = gate.hz

    fun stop() {
        running = false
    }

    val stats: Stats
        get() = Stats(
            seen = seen.get(),
            accepted = accepted.get(),
            gated = gated.get(),
            refusedStopped = refusedStopped.get(),
            delivered = delivered.get(),
            refusedBySink = refusedBySink.get(),
            invalidFixes = invalidFixes.get(),
            nonMonotonicFixes = nonMonotonic.get(),
        )

    data class Stats(
        val seen: Long,
        val accepted: Long,
        val gated: Long,
        val refusedStopped: Long,
        val delivered: Long,
        val refusedBySink: Long,
        val invalidFixes: Long,
        val nonMonotonicFixes: Long,
    ) {
        /** Every reading the platform gave us is under exactly one heading. */
        val balances: Boolean get() = seen == accepted + gated + refusedStopped

        /** And every accepted one either went out or was refused by the transport. */
        val acceptedBalances: Boolean get() = accepted == delivered + refusedBySink
    }

    companion object {
        /**
         * Build a record from a reading's parts, mapping "no fix" to the all-null form.
         *
         * The single place a platform location becomes a wire record, so the null
         * convention is applied once rather than at every call site.
         */
        fun record(
            fixMonoNs: Long,
            latitude: Double?,
            longitude: Double?,
            speedMps: Double?,
            headingDeg: Double?,
            satellites: Long,
            hdop: Double?,
            altitudeM: Double?,
            utcEpochNs: Long?,
        ): GpsRecord {
            // A fix is valid when it has a position. Everything else can be missing --
            // speed and heading are absent when stationary on many devices, and altitude
            // needs more satellites than a 2D fix has.
            val valid = latitude != null && longitude != null
            if (!valid) return GpsRecord.noFix(fixMonoNs)
            return GpsRecord(
                captureMonoNs = fixMonoNs,
                valid = true,
                latitude = latitude,
                longitude = longitude,
                speedMps = speedMps,
                headingDeg = headingDeg,
                // 1 rather than 0: fix_quality 0 is the wire's "no fix", so a valid
                // record must not carry it.
                fixQuality = 1,
                satellites = satellites,
                hdop = hdop,
                altitudeM = altitudeM,
                utcEpochNs = utcEpochNs,
            )
        }
    }
}
