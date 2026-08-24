package com.dsrc.phone.sensors

import com.dsrc.phone.config.SensingConfig
import com.dsrc.transport.ImuSample
import java.util.concurrent.atomic.AtomicLong

/**
 * One accelerometer reading paired with the most recent gyroscope reading.
 *
 * The pairing is the approximation this whole modality rests on, so it carries its own
 * error term: the two sensors fire independently and at rates that drift against each
 * other, and [gyroAgeNs] is *how far apart* the two halves were stamped -- a magnitude, not
 * a signed staleness. The gyro can be stamped either side of the accelerometer, since both
 * register on one handler at the same commanded period, and `ImuPairing.outOfOrderPairings`
 * carries which. Zero would be a claim the two were simultaneous, which they never are.
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

    /**
     * The lock every counter is read and written under.
     *
     * Plain fields rather than atomics, because the property that matters is not that each
     * counter is individually consistent but that the *set* of them is: the one production
     * reader asserts an identity across eleven, and a live pipeline can always be
     * mid-`offer`. Eleven atomics give eleven consistent numbers that need not add up.
     */
    private val counters = Any()

    private var seen = 0L
    private var accepted = 0L
    private var gated = 0L
    private var refusedStopped = 0L
    private var unpaired = 0L
    private var refusedBySink = 0L
    private var delivered = 0L
    private var nonMonotonic = 0L
    private var staleGyro = 0L
    private var gyroAgeTotalNs = 0L
    private var gyroAgeMaxNs = 0L

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
    fun offerUnpaired() = synchronized(counters) {
        seen++
        if (!running) {
            refusedStopped++
            return@synchronized
        }
        unpaired++
    }

    /**
     * Offer one paired reading.
     *
     * The whole body is under the counter lock, the sink included. That is deliberate: an
     * offer that released the lock across the sink would let `stats()` see a sample counted
     * as accepted and not yet as delivered or refused, and the identity the production
     * reader asserts would be false in flight. At 50 Hz, and with a sink that enqueues
     * rather than blocks, the lock costs nothing worth measuring.
     */
    fun offer(reading: ImuReading): Boolean = synchronized(counters) {
        seen++
        if (!running) {
            refusedStopped++
            return@synchronized false
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
        // No sentinel test. `captureMonoNs < Long.MIN_VALUE` is false for every Long, so
        // the guard could not fire, and RateGate records the argument against reusing
        // Long.MIN_VALUE for "not set" in this domain. A guard nothing can reach is a claim
        // a reader has to disprove.
        val previous = lastCaptureNs.getAndSet(reading.captureMonoNs)
        if (reading.captureMonoNs < previous) {
            nonMonotonic++
        }

        if (!gate.accept(reading.captureMonoNs)) {
            gated++
            return@synchronized false
        }
        accepted++

        // Pairing error, on the samples that actually went out rather than on everything
        // the sensor produced -- a gated sample's staleness costs nobody anything.
        gyroAgeTotalNs += reading.gyroAgeNs
        gyroAgeMaxNs = maxOf(gyroAgeMaxNs, reading.gyroAgeNs)
        if (reading.gyroAgeNs > gate.periodNanos) staleGyro++

        return@synchronized if (sink(sample(reading))) {
            delivered++
            true
        } else {
            refusedBySink++
            false
        }
    }

    fun setRate(hz: Double) = gate.setRate(hz)

    val rateHz: Double get() = gate.hz

    fun stop() {
        running = false
    }

    val stats: Stats
        get() = synchronized(counters) {
            // Under the same lock the counters are written under, so the snapshot is exact
            // rather than merely well-ordered.
            //
            // The first attempt here was a read-order rule, copied from `Session.stats()`:
            // headings first, `seen` last, which makes `seen >= the parts` hold by
            // construction. That is sound and it is not enough, because the one production
            // reader asserts *equality* across eleven counters, and a live pipeline can
            // always be mid-`offer`. A lock is free at 50 Hz -- this is not a hot path in
            // any sense that matters -- and it turns an identity that was true most of the
            // time into one that is true.
            //
            // Neither this nor the ordering it replaces is pinned by a test. No
            // single-threaded test can tell them apart, and a sampling probe that catches a
            // regression most runs is a pin nobody can trust either way; that lesson cost a
            // whole round on the transport. Correct by construction, and said so here.
            val kept = accepted
            Stats(
                seen = seen,
                accepted = kept,
                gated = gated,
                refusedStopped = refusedStopped,
                unpaired = unpaired,
                delivered = delivered,
                refusedBySink = refusedBySink,
                nonMonotonicSamples = nonMonotonic,
                staleGyroSamples = staleGyro,
                gyroAgeMeanNs = if (kept == 0L) 0 else gyroAgeTotalNs / kept,
                gyroAgeMaxNs = gyroAgeMaxNs,
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
        /**
         * Samples whose two halves were stamped more than one commanded period apart.
         *
         * Either direction. It reads as "older than one period" and is not: since
         * `gyroAgeNs` became a magnitude this also counts a gyro stamped *after* the
         * accelerometer by more than a period, which is the opposite condition and equally
         * a pairing that has stopped being an approximation.
         */
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
