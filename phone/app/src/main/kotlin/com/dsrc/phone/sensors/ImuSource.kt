package com.dsrc.phone.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import com.dsrc.phone.config.SensingConfig

/** Whether the sensor clock and the app's clock are the same clock. */
enum class ImuTimebase {
    /** Not established yet: no event has arrived. */
    UNKNOWN,

    /** `SensorEvent.timestamp` agrees with `elapsedRealtime`, so stamps are comparable. */
    MATCHED,

    /** It does not, and nothing is captured. See [ImuSource]. */
    MISMATCHED,
}

/**
 * Accelerometer and gyroscope, paired into one sample per accelerometer event.
 *
 * Three things decide whether these samples are worth anything, and all three are about
 * time or pairing rather than about reading a sensor.
 *
 * **The clock.** Everything else in this app stamps on `SystemClock.elapsedRealtimeNanos`:
 * GPS fixes, the transport's enqueue stamp, the timebase exchange with the Jetson.
 * `SensorEvent.timestamp` is *documented* to be the same base, and on most devices it is,
 * but it is a long-standing vendor bug for it to be `System.nanoTime` or a boot-relative
 * count with a different epoch. A sample whose capture stamp is on a different timebase is
 * worse than a missing sample: it arrives looking valid and lands the Jetson's fusion at
 * the wrong instant. So the offset is *measured* on the first event rather than assumed,
 * and a mismatch stops capture instead of guessing a correction. Sensing carries on with
 * camera and GPS -- an IMU stream is not worth taking the phone down for, and it is
 * certainly not worth a stream of confidently wrong timestamps.
 *
 * **The pairing.** [ImuSample] carries both sensors in one message, and they arrive on
 * separate callbacks that drift against each other. The accelerometer drives and the gyro
 * is taken from its latest reading.
 *
 * Two things that earlier drafts of this comment got wrong, both worth correcting rather
 * than deleting. The accelerometer is not "the faster stream": both sensors are registered
 * at the same commanded period, and on the emulator both report the same 2-100 Hz range.
 * With equal rates and independent phase the gyro's age is roughly uniform over one period
 * and reaches a full period, which is exactly the threshold `staleGyroSamples` calls the
 * point where the pairing stops being an approximation. Driving from the accelerometer is
 * still the right choice -- the stamp has to be one of the two, and it should be the one
 * belonging to the reading that produced the sample -- but not for the reason given.
 *
 * And the age does *not* ride along with the sample. [ImuSample] carries `ax..gz` and
 * `accuracy`, and the wire contract is frozen; the pairing error is a phone-side statistic
 * on [ImuPipeline.Stats], which the peer never sees.
 *
 * **The rate.** Android treats a sampling period as a hint and delivers faster or slower
 * than asked. The commanded rate comes from [ImuPipeline]'s gate, as it does for camera
 * and GPS; what is requested here only sets how much the platform offers.
 */
class ImuSource(
    context: Context,
    private val config: SensingConfig,
    /**
     * The clock everything else in this app stamps on: GPS fixes, the transport's enqueue
     * stamp, the timebase exchange.
     *
     * **No default, deliberately.** Both clocks used to have one and the single
     * construction site supplied neither, so nothing anywhere chose them -- and replacing
     * this one's sibling default with `elapsedRealtimeNanos` made `clockGapNs` identically
     * zero, the attribution branch of `verdictFor` dead code, and the vendor bug this whole
     * class exists to detect undetectable, with both suites green.
     *
     * No test can catch that on an emulator: a machine that never suspends has the two
     * clocks a few hundred nanoseconds apart, which is the same distance two sequential
     * reads of *one* clock are apart. So the hazard is removed rather than pinned -- the
     * wiring is now a visible decision at the call site instead of a default nobody reads.
     */
    private val appClock: () -> Long,
    /**
     * The other candidate. Read at the same instant as [appClock] so the difference is the
     * device's accumulated suspend time -- which is what tells the two apart, and what a
     * wrong attribution would cost.
     */
    private val monoClock: () -> Long,
) {

    private val manager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    // Volatile because `stop()` reaches them from two threads: the main thread through
    // `onSensingDown`, and the sensor thread through `stopBecauseOfTimebase`. Benign today
    // -- the second unregister and quit are no-ops -- but this class made its other
    // cross-thread fields atomic for precisely this reason and then left these two out.
    @Volatile
    private var thread: HandlerThread? = null

    @Volatile
    private var listener: SensorEventListener? = null

    private val pairing = ImuPairing()

    val timebase: ImuTimebase get() = pairing.timebase
    val timebaseOffsetNs: Long get() = pairing.timebaseOffsetNs

    /**
     * How far the two candidate clocks were apart when the timebase was decided.
     *
     * Exposed because three places claimed an accepting verdict "still says how wrong it
     * could be" and none of them could print it: the only reader was the refusing branch,
     * so a session accepted with a 49 ms gap and one accepted with a 0.3 ms gap were
     * indistinguishable in every artefact the phone produced.
     */
    val clockGapNs: Long get() = pairing.clockGapNs

    /** Accelerometer events discarded because the sensor clock is not ours. */
    val refusedWrongTimebase: Long get() = pairing.refusedWrongTimebase.get()

    /**
     * Paired samples whose gyro half was stamped after the accelerometer's.
     *
     * The counter that carries the sign `gyroAgeNs` gave up when it became a magnitude.
     * Without a reader the mean was a mean of magnitudes with no way to tell a systematic
     * gyro lag from symmetric jitter -- the information was removed from one place and
     * landed in another nobody could reach.
     */
    val outOfOrderPairings: Long get() = pairing.outOfOrderPairings.get()


    /** Guards the teardown against the second event that arrives before it finishes. */
    @Volatile
    private var stoppedForTimebase = false

    fun start(onReading: (ImuReading) -> Unit, onUnpaired: () -> Unit) {
        val accelerometer = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        if (accelerometer == null || gyroscope == null) {
            // Not a failure of ours, and not something a retry fixes. Named rather than
            // silently producing nothing.
            Log.w(TAG, "no IMU: accelerometer=$accelerometer gyroscope=$gyroscope")
            return
        }

        // Its own thread, named so the service's thread census can see it. Three resources
        // were found unstarted in task 17 precisely because they had no named thread.
        val worker = HandlerThread(THREAD_NAME).also { it.start() }
        thread = worker
        val handler = Handler(worker.looper)

        val callback = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_GYROSCOPE -> {
                        pairing.onGyro(
                            captureNs = event.timestamp,
                            x = event.values[0].toDouble(),
                            y = event.values[1].toDouble(),
                            z = event.values[2].toDouble(),
                            appNowNs = appClock(),
                            monoNowNs = monoClock(),
                        )
                        if (pairing.timebase == ImuTimebase.MISMATCHED) stopBecauseOfTimebase()
                    }
                    Sensor.TYPE_ACCELEROMETER -> {
                        val outcome = pairing.onAccelerometer(
                            captureNs = event.timestamp,
                            x = event.values[0].toDouble(),
                            y = event.values[1].toDouble(),
                            z = event.values[2].toDouble(),
                            accuracy = event.accuracy.toLong(),
                            appNowNs = appClock(),
                            monoNowNs = monoClock(),
                        )
                        when (outcome) {
                            is ImuOutcome.Paired -> onReading(outcome.reading)
                            ImuOutcome.Unpaired -> onUnpaired()
                            ImuOutcome.WrongTimebase -> stopBecauseOfTimebase()
                        }
                    }
                }
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
                // Carried per sample from SensorEvent.accuracy instead, which is the
                // reading that corresponds to the stamp. Nothing to do here.
            }
        }
        listener = callback

        // maxReportLatencyUs = 0: no batching. A batch hands over a burst of events whose
        // timestamps are right but whose arrival is late, and the rate gate would then pass
        // the first of the burst and drop the rest -- turning a 50 Hz command into one
        // sample per batch. Latency is worth more here than the wakeup saving.
        this.accelerometer = accelerometer
        this.gyroscope = gyroscope
        this.handler = handler
        register(config.imuHz)
    }

    /**
     * Ask the platform for a new sampling period.
     *
     * The rate gate can only ever *lower* a rate: it drops samples the platform already
     * produced. So a command raising `imu_hz` above what was requested at start changed
     * nothing on the wire while the pipeline reported the new rate as in force -- commanded
     * 200 Hz, measured 50, reported 200. Nobody on either side could see the difference.
     *
     * Re-registering is not restarting capture in the sense task 22 forbids: the pipeline,
     * its counters and the session are untouched, and the listener object is the same one.
     * Only the period the platform is asked for changes.
     */
    @Synchronized
    fun setRate(hz: Double) {
        // Synchronized against `stop()` for the same reason GpsLocationSource is: a
        // `rate_cmd` is applied on the delivery thread while the stop runs on the main
        // thread, and a re-registration landing after `unregisterListener` holds both
        // sensors awake at the commanded rate for the life of the process, delivering into
        // a looper that has been quit. `stop()` nulls `listener` under this same monitor,
        // so the null check below is what refuses a late command; holding the monitor is
        // what makes it decisive rather than a narrowed window.
        val callback = listener ?: return
        val sensor = accelerometer ?: return
        val gyro = gyroscope ?: return
        val worker = handler ?: return
        manager.unregisterListener(callback)
        register(hz, callback, sensor, gyro, worker)
    }

    private fun register(
        hz: Double,
        callback: SensorEventListener? = listener,
        sensor: Sensor? = accelerometer,
        gyro: Sensor? = gyroscope,
        worker: Handler? = handler,
    ) {
        if (callback == null || sensor == null || gyro == null || worker == null) return
        val periodUs = (1_000_000.0 / hz).toInt().coerceAtLeast(1)
        manager.registerListener(callback, sensor, periodUs, 0, worker)
        manager.registerListener(callback, gyro, periodUs, 0, worker)
        requestedHz = hz
    }

    /** The period last asked of the platform, which bounds what the gate can pass. */
    @Volatile
    var requestedHz: Double = 0.0
        private set

    @Volatile
    private var accelerometer: Sensor? = null

    @Volatile
    private var gyroscope: Sensor? = null

    @Volatile
    private var handler: Handler? = null


    @Synchronized
    fun stop() {
        listener?.let { manager.unregisterListener(it) }
        listener = null
        thread?.quitSafely()
        thread = null
    }

    /**
     * Tear down because the sensor clock is not ours.
     *
     * Deliberately the whole teardown and not a flag. Leaving the listeners registered
     * meant two sensors held awake at the commanded rate for the life of the process,
     * delivering into a looper whose every message was discarded.
     */
    private fun stopBecauseOfTimebase() {
        if (stoppedForTimebase) return
        stoppedForTimebase = true
        Log.e(
            TAG,
            "sensor clock is not elapsedRealtime: delivery delta ${pairing.timebaseOffsetNs}ns " +
                "against a clock gap of ${pairing.clockGapNs}ns. Stopping IMU capture; a stamp " +
                "on the wrong timebase would land the Jetson's fusion at the wrong instant.",
        )
        stop()
    }

    companion object {
        const val THREAD_NAME = "dsrc-imu"



        private const val TAG = "ImuSource"
    }
}
