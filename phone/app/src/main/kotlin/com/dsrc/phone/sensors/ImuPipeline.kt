package com.dsrc.phone.sensors

import com.dsrc.phone.config.SensingConfig
import com.dsrc.transport.ImuSample
import java.util.concurrent.atomic.AtomicLong

/**
 * One accelerometer reading paired with the most recent gyroscope reading.
 *
 * The pairing is the approximation this whole modality rests on, so it carries its own
 * error term: the two sensors fire independently and at rates that drift against each
 * other, and [gyroAgeNs] is how stale the gyro half was when the accelerometer fired.
 * Zero would be a claim the two were simultaneous, which they never are.
 */
data class ImuReading(
    val captureMonoNs: Long,
    val ax: Double,
    val ay: Double,
    val az: Double,
    val gx: Double,
    val gy: Double,
    val gz: Double,
    val accuracy: Long?,
    val gyroAgeNs: Long,
)

/**
 * The IMU path: rate-gate the paired samples, hand the survivors to a sink, count
 * everything.
 *
 * Same shape as [GpsPipeline] and [CameraPipeline], reusing [RateGate], because "at the
 * commanded rate" has to mean the same thing on every modality. As with GPS there is no
 * buffer: `imu` is a transport channel with its own queue, and a second buffer in front
 * would mean two places dropping for different reasons with only one of them visible to
 * the peer as a sequence gap.
 *
 * `unpaired` is the one heading GPS and camera do not have. An accelerometer event that
 * arrives before any gyroscope event cannot be made into a sample: zeros are a reading,
 * and the message has no null for those axes. Dropping it is the only honest answer, and
 * counting it is what stops a startup gap from looking like a rate that came up slow.
 */
class ImuPipeline(
    config: SensingConfig,
    /** Where a kept sample goes. Returning false means the transport refused it. */
    private val sink: (ImuSample) -> Boolean,
) {
    private val gate = RateGate(config.imuHz)

    private val seen = AtomicLong(0)
    private val accepted = AtomicLong(0)
    private val gated = AtomicLong(0)
    private val refusedStopped = AtomicLong(0)
    private val unpaired = AtomicLong(0)
    private val refusedBySink = AtomicLong(0)
    private val delivered = AtomicLong(0)
    private val nonMonotonic = AtomicLong(0)
    private val staleGyro = AtomicLong(0)
    private val gyroAgeTotalNs = AtomicLong(0)
    private val gyroAgeMaxNs = AtomicLong(0)

    @Volatile
    private var running = true

    /** Whether this pipeline has been stopped. See [GpsPipeline.isStopped]. */
    val isStopped: Boolean get() = !running

    private val lastCaptureNs = AtomicLong(Long.MIN_VALUE)

    /**
     * An accelerometer event with no gyroscope reading yet.
     *
     * Separate from [offer] so the caller does not have to invent axes it does not have.
     */
    fun offerUnpaired() {
        seen.incrementAndGet()
        if (!running) {
            refusedStopped.incrementAndGet()
            return
        }
        unpaired.incrementAndGet()
    }

    fun offer(reading: ImuReading): Boolean {
        seen.incrementAndGet()
        if (!running) {
            refusedStopped.incrementAndGet()
            return false
        }

        // A capture stamp that goes backwards means the platform delivered out of order,
        // and the receiver's arithmetic is built on these being monotonic. Counted rather
        // than corrected, for the same reason GPS counts it: a silent fix is a measurement
        // nobody can trust.
        // Against the previous *event*, and updated on every event, not only on the ones
        // the gate kept. The baseline used to advance after the gate, which made a reversal
        // between two gated events invisible -- at 10 Hz, offers at 0, 50 ms, 40 ms counted
        // nothing, though the platform had plainly delivered out of order. What is being
        // detected is a property of the delivery, so the gate has no business in it.
        val previous = lastCaptureNs.getAndSet(reading.captureMonoNs)
        if (previous != Long.MIN_VALUE && reading.captureMonoNs < previous) {
            nonMonotonic.incrementAndGet()
        }

        if (!gate.accept(reading.captureMonoNs)) {
            gated.incrementAndGet()
            return false
        }
        accepted.incrementAndGet()

        // Pairing error, on the samples that actually went out rather than on everything
        // the sensor produced -- a gated sample's staleness costs nobody anything.
        gyroAgeTotalNs.addAndGet(reading.gyroAgeNs)
        gyroAgeMaxNs.getAndUpdate { maxOf(it, reading.gyroAgeNs) }
        if (reading.gyroAgeNs > gate.periodNanos) staleGyro.incrementAndGet()

        return if (sink(sample(reading))) {
            delivered.incrementAndGet()
            true
        } else {
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
        get() {
            val kept = accepted.get()
            return Stats(
                seen = seen.get(),
                accepted = kept,
                gated = gated.get(),
                refusedStopped = refusedStopped.get(),
                unpaired = unpaired.get(),
                delivered = delivered.get(),
                refusedBySink = refusedBySink.get(),
                nonMonotonicSamples = nonMonotonic.get(),
                staleGyroSamples = staleGyro.get(),
                gyroAgeMeanNs = if (kept == 0L) 0 else gyroAgeTotalNs.get() / kept,
                gyroAgeMaxNs = gyroAgeMaxNs.get(),
                rateHz = gate.hz,
            )
        }

    data class Stats(
        val seen: Long,
        val accepted: Long,
        val gated: Long,
        val refusedStopped: Long,
        /** Accelerometer events with no gyroscope reading to pair with yet. */
        val unpaired: Long,
        val delivered: Long,
        val refusedBySink: Long,
        val nonMonotonicSamples: Long,
        /** Samples whose gyro half was older than one commanded period. */
        val staleGyroSamples: Long,
        /**
         * The commanded rate when these statistics were read.
         *
         * `staleGyroSamples` counts against the period at the time of each sample, so the
         * number means nothing on its own once the rate has been re-commanded: three
         * samples with an identical 15 ms age at 50, 10 and 200 Hz give a count of one, and
         * nothing recorded which regime produced it. `gyroAgeMeanNs` mixes regimes the same
         * way. Carrying the rate does not un-mix them, but it stops the number being read
         * as if it had one meaning.
         */
        val rateHz: Double,
        val gyroAgeMeanNs: Long,
        val gyroAgeMaxNs: Long,
    ) {
        /** Every event the platform gave us is under exactly one heading. */
        val balances: Boolean get() = seen == accepted + gated + refusedStopped + unpaired

        /** And every accepted one either went out or was refused by the transport. */
        val acceptedBalances: Boolean get() = accepted == delivered + refusedBySink
    }

    companion object {
        fun sample(reading: ImuReading) = ImuSample(
            captureMonoNs = reading.captureMonoNs,
            ax = reading.ax,
            ay = reading.ay,
            az = reading.az,
            gx = reading.gx,
            gy = reading.gy,
            gz = reading.gz,
            accuracy = reading.accuracy,
        )
    }
}
