package com.dsrc.phone.sensors

/** What to do with one accelerometer event. */
sealed interface ImuOutcome {
    /** A paired sample, ready for the pipeline. */
    data class Paired(val reading: ImuReading) : ImuOutcome

    /** No gyroscope reading yet, so there is nothing honest to send. */
    data object Unpaired : ImuOutcome

    /** The sensor clock is not the app's clock. Nothing is emitted for the session. */
    data object WrongTimebase : ImuOutcome
}

/**
 * Everything [ImuSource] decides, with no Android in it.
 *
 * Round 1 of this task found the reason to separate them: every claim task 20 makes lives
 * in the listener callback, and a callback that only a device can invoke is a callback no
 * test reaches. Thirteen mutations to that callback survived 267 JVM tests, and four of
 * them applied together -- never registering the gyroscope, never unregistering, dropping
 * the unpaired branch and dropping the timebase gate -- left the instrumented suite at 53
 * passing while the `imu` channel transmitted nothing for a whole session. A pure verdict
 * function was pinned; its *use* was pinned by nothing, which is a distinction worth
 * carrying: extracting a function only helps if the caller is reachable too.
 *
 * So the pairing state, the timebase gate and the age arithmetic live here, and
 * [ImuSource] is the thin part that turns a `SensorEvent` into one of these calls.
 */
class ImuPairing(
    /** Widest plausible gap between a sensor capture and its delivery here. */
    private val maxDeliveryNs: Long = MAX_PLAUSIBLE_DELIVERY_NS,
) {

    @Volatile
    var timebase: ImuTimebase = ImuTimebase.UNKNOWN
        private set

    /** The measured delivery delta on the deciding event, for the log line. */
    @Volatile
    var timebaseOffsetNs: Long = 0
        private set

    /**
     * How far apart the two candidate clocks were when the decision was taken.
     *
     * This is the error a wrong guess would have cost, which is the number worth keeping:
     * on the accepting branch it says how much the decision could have been wrong by.
     */
    @Volatile
    var clockGapNs: Long = 0
        private set

    /**
     * Accelerometer events discarded because the timebase is wrong.
     *
     * Atomic like everything on [ImuPipeline], not a plain `Long`. These two are written on
     * the sensor thread and read from `onSensingDown` on the service thread, and neither
     * `unregisterListener` nor `quitSafely()` establishes a happens-before with the sensor
     * thread's writes -- so the teardown line could print a stale value. [onGyro]'s own
     * comment criticises a class that asserts two concurrency stories at once, and these
     * two were the second story.
     */
    val refusedWrongTimebase = java.util.concurrent.atomic.AtomicLong(0)

    /** Paired samples whose gyro half was stamped *after* the accelerometer's. */
    val outOfOrderPairings = java.util.concurrent.atomic.AtomicLong(0)

    private var hasGyro = false
    private var gyroNs = 0L
    private var gx = 0.0
    private var gy = 0.0
    private var gz = 0.0

    /**
     * A gyroscope reading, gated on the same timebase question as the accelerometer.
     *
     * The gate used to run on one stream only, so a device whose *gyroscope* sat on a
     * different base still produced paired samples: the sample's stamp is the
     * accelerometer's, so nothing looked wrong, and the mismatch surfaced only as a large
     * `gyroAgeNs`, which is counted and refuses nothing. The plan called this "taken from
     * one event", which understated it -- it was taken from one stream.
     *
     * The two sensors are registered together at one rate on one handler, so if their
     * stamps disagree about which clock they are on, the pairing is wrong even though the
     * stamp is right. Same question, same answer, both streams.
     */
    fun onGyro(captureNs: Long, x: Double, y: Double, z: Double, appNowNs: Long, monoNowNs: Long) {
        if (!checkTimebase(captureNs, appNowNs, monoNowNs)) {
            refusedWrongTimebase.incrementAndGet()
            return
        }
        // Written as one group, read as one group, on one thread. The previous shape made
        // the timestamp an AtomicLong and left the three axes plain, which asserted two
        // different concurrency stories at once: the atomic bought nothing on a single
        // handler, and would not have helped on two, since a reader could still take a
        // fresh timestamp with stale axes.
        gx = x
        gy = y
        gz = z
        gyroNs = captureNs
        hasGyro = true
    }

    fun onAccelerometer(
        captureNs: Long,
        x: Double,
        y: Double,
        z: Double,
        accuracy: Long?,
        appNowNs: Long,
        monoNowNs: Long,
    ): ImuOutcome {
        if (!checkTimebase(captureNs, appNowNs, monoNowNs)) {
            // Counted, because it was not. A wrong-clock session and a device with no IMU
            // produced identical statistics -- `seen` never moved on either -- so the two
            // were distinguishable only by a field in a log line.
            refusedWrongTimebase.incrementAndGet()
            return ImuOutcome.WrongTimebase
        }
        if (!hasGyro) return ImuOutcome.Unpaired

        val age = captureNs - gyroNs
        // Both sensors are registered on one handler at the same commanded period, so a
        // gyro sample captured *after* an accelerometer sample can be delivered before it
        // and the age comes out negative. Left signed it cancelled real error in the mean:
        // ages of +18 ms and -18 ms averaged to zero and reported a perfect pairing. The
        // magnitude is what the statistic is about, and the direction is worth its own
        // counter rather than being folded into the number it corrupts.
        if (age < 0) outOfOrderPairings.incrementAndGet()
        return ImuOutcome.Paired(
            ImuReading(
                captureMonoNs = captureNs,
                ax = x, ay = y, az = z,
                gx = gx, gy = gy, gz = gz,
                accuracy = accuracy,
                gyroAgeNs = if (age < 0) -age else age,
            )
        )
    }

    /**
     * Decide, once, whether the sensor clock is the clock the rest of this app uses.
     *
     * The first version of this compared the event stamp against `elapsedRealtime` and
     * accepted any small positive difference as delivery latency. Round 1 showed that
     * admits the exact bug it was written for. The headline vendor bug is
     * `SensorEvent.timestamp` on `System.nanoTime` (CLOCK_MONOTONIC) instead of
     * `elapsedRealtimeNanos` (CLOCK_BOOTTIME), and those two do not differ by an *epoch* --
     * they differ by the device's accumulated suspend time, which starts at zero on boot
     * and grows. A handset that had suspended for 1.5 s produced a difference of 1.5 s,
     * comfortably inside a two-second bound, and every sample after it carried a silent
     * constant 1.5 s error: about 21 m of road at 50 km/h.
     *
     * So both candidates are read at delivery and the event is attributed to whichever it
     * is closer to. The ambiguous case is exactly the harmless one: when the two clocks
     * have barely diverged, it does not matter which the sensor is using, because choosing
     * wrong costs at most that gap. That is recorded as [clockGapNs] rather than being
     * silently discarded, so an accepting verdict still says how wrong it could be.
     */
    private fun checkTimebase(eventNs: Long, appNowNs: Long, monoNowNs: Long): Boolean {
        when (timebase) {
            ImuTimebase.MATCHED -> return true
            ImuTimebase.MISMATCHED -> return false
            ImuTimebase.UNKNOWN -> Unit
        }
        timebaseOffsetNs = appNowNs - eventNs
        clockGapNs = appNowNs - monoNowNs
        timebase = verdictFor(
            deliveryDeltaNs = timebaseOffsetNs,
            clockGapNs = clockGapNs,
            maxDeliveryNs = maxDeliveryNs,
        )
        return timebase == ImuTimebase.MATCHED
    }

    companion object {
        /**
         * Two seconds. Above any real delivery latency, and the *only* thing this bound
         * now has to separate -- telling the two clocks apart is [clockGapNs]'s job.
         */
        const val MAX_PLAUSIBLE_DELIVERY_NS = 2_000_000_000L

        /**
         * How far the two clocks may drift apart before it matters which one a stamp is on.
         *
         * 50 ms, which at 50 km/h is about 0.7 m -- inside the positional error the fusion
         * already carries from GPS. Below this the attribution is not worth making; above
         * it, it is, and the arithmetic below makes it.
         */
        const val MAX_TOLERABLE_CLOCK_GAP_NS = 50_000_000L

        /**
         * Which clock an event stamp belongs to, as a pure function of two measurements.
         *
         * @param deliveryDeltaNs app clock at delivery, minus the event stamp.
         * @param clockGapNs app clock minus the monotonic clock: the accumulated suspend
         *   time, and the error a wrong attribution would cost.
         */
        fun verdictFor(
            deliveryDeltaNs: Long,
            clockGapNs: Long,
            maxDeliveryNs: Long = MAX_PLAUSIBLE_DELIVERY_NS,
        ): ImuTimebase {
            // Ahead of a clock read after it: not one clock, whatever the gap.
            if (deliveryDeltaNs < 0) return ImuTimebase.MISMATCHED

            // The two clocks have barely diverged, so the question is moot: whichever the
            // sensor uses, the stamp is right to within `clockGapNs`.
            //
            // Against its own tolerance, not the delivery bound. Reusing the two-second
            // delivery bound here re-admitted the whole bug: a 1.5-second gap came out
            // "moot" when 1.5 seconds is about 21 m of road at 50 km/h. What is being asked
            // is not "could this be delivery latency" but "is the error small enough that
            // choosing wrong does not matter", and those are different questions with
            // different answers.
            // Signed, and a magnitude guard was tried here and reverted, because it is
            // not the fix it looks like. Round 3 observed that a large *negative* gap reads
            // as "barely diverged" and asked whether that fails open. It does -- but taking
            // the magnitude changes nothing, and the algebra says why: with `clockGapNs`
            // negative, `fromMono = deliveryDeltaNs + |clockGapNs|` is always larger than
            // `fromApp = deliveryDeltaNs`, so the branch below reduces to exactly the test
            // above it. Both arms return the same verdict for every negative gap.
            //
            // What actually guards the transposition is that the single call site passes
            // both clocks by name. A magnitude here would have looked like protection and
            // supplied none, which is worse than the honest version.
            if (clockGapNs <= MAX_TOLERABLE_CLOCK_GAP_NS) {
                return if (deliveryDeltaNs <= maxDeliveryNs) {
                    ImuTimebase.MATCHED
                } else {
                    ImuTimebase.MISMATCHED
                }
            }

            // They have diverged enough to tell apart. An event on the app's own clock sits
            // near zero; one on the monotonic clock sits near the gap. Attribute it to the
            // nearer, and refuse if it is nearer the wrong one -- or near neither.
            val fromApp = deliveryDeltaNs
            val fromMono = if (deliveryDeltaNs > clockGapNs) deliveryDeltaNs - clockGapNs else clockGapNs - deliveryDeltaNs
            return if (fromApp <= maxDeliveryNs && fromApp < fromMono) {
                ImuTimebase.MATCHED
            } else {
                ImuTimebase.MISMATCHED
            }
        }
    }
}
