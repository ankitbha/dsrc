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
import java.util.concurrent.atomic.AtomicLong

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
 * separate callbacks that drift against each other. The accelerometer drives, because it
 * is the faster stream on every device here, so pairing against the latest gyro reading
 * keeps the gyro's age below one accelerometer period rather than the other way round. The
 * age rides along with the sample so the approximation is measurable rather than assumed.
 *
 * **The rate.** Android treats a sampling period as a hint and delivers faster or slower
 * than asked. The commanded rate comes from [ImuPipeline]'s gate, as it does for camera
 * and GPS; what is requested here only sets how much the platform offers.
 */
class ImuSource(
    context: Context,
    private val config: SensingConfig,
    /** Injectable so the timebase check can be driven from both sides in a test. */
    private val appClock: () -> Long = SystemClock::elapsedRealtimeNanos,
) {

    private val manager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private var thread: HandlerThread? = null
    private var listener: SensorEventListener? = null

    @Volatile
    var timebase: ImuTimebase = ImuTimebase.UNKNOWN
        private set

    /** The measured difference, for the log line that explains a refusal. */
    @Volatile
    var timebaseOffsetNs: Long = 0
        private set

    private val gyroNs = AtomicLong(Long.MIN_VALUE)
    private var gx = 0.0
    private var gy = 0.0
    private var gz = 0.0

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
                        gx = event.values[0].toDouble()
                        gy = event.values[1].toDouble()
                        gz = event.values[2].toDouble()
                        gyroNs.set(event.timestamp)
                    }
                    Sensor.TYPE_ACCELEROMETER -> {
                        if (!checkTimebase(event.timestamp)) return
                        val gyroAt = gyroNs.get()
                        if (gyroAt == Long.MIN_VALUE) {
                            // The accelerometer can fire before the first gyro event. There
                            // is no defensible filler: zeros are a reading, and the message
                            // has no null for those axes.
                            onUnpaired()
                            return
                        }
                        onReading(
                            ImuReading(
                                captureMonoNs = event.timestamp,
                                ax = event.values[0].toDouble(),
                                ay = event.values[1].toDouble(),
                                az = event.values[2].toDouble(),
                                gx = gx,
                                gy = gy,
                                gz = gz,
                                accuracy = event.accuracy.toLong(),
                                gyroAgeNs = event.timestamp - gyroAt,
                            )
                        )
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
        val periodUs = (1_000_000.0 / config.imuHz).toInt().coerceAtLeast(1)
        manager.registerListener(callback, accelerometer, periodUs, 0, handler)
        manager.registerListener(callback, gyroscope, periodUs, 0, handler)
    }

    /**
     * Compare the sensor clock against the app clock, once.
     *
     * The app clock is read *now*, at delivery, and the event was captured before it, so a
     * healthy difference is small and positive -- it is just delivery latency. What this is
     * looking for is a different *epoch*, which on a device that has been up any length of
     * time is seconds to hours, not milliseconds. Hence a deliberately generous bound: a
     * tight one would turn a slow first delivery into a false alarm, and the failure being
     * detected is not subtle.
     *
     * A negative difference is as damning as a large one: it means the sensor stamp is
     * ahead of a clock that was read afterwards.
     */
    private fun checkTimebase(eventNs: Long): Boolean {
        if (timebase == ImuTimebase.MISMATCHED) return false
        if (timebase == ImuTimebase.MATCHED) return true

        val delta = appClock() - eventNs
        timebaseOffsetNs = delta
        timebase = verdictFor(delta)
        if (timebase == ImuTimebase.MISMATCHED) {
            Log.e(
                TAG,
                "sensor clock is not elapsedRealtime: delivery delta ${delta}ns is outside " +
                    "[0, $MAX_PLAUSIBLE_DELIVERY_NS]. Not capturing IMU; a stamp on the " +
                    "wrong timebase would land the Jetson's fusion at the wrong instant.",
            )
        }
        return timebase == ImuTimebase.MATCHED
    }

    fun stop() {
        listener?.let { manager.unregisterListener(it) }
        listener = null
        thread?.quitSafely()
        thread = null
    }

    companion object {
        const val THREAD_NAME = "dsrc-imu"

        /**
         * The widest plausible gap between a sensor capture and its delivery here.
         *
         * Two seconds, which is far above any real delivery latency and far below the
         * epoch difference this is looking for.
         */
        const val MAX_PLAUSIBLE_DELIVERY_NS = 2_000_000_000L

        /**
         * The verdict for one measured delivery delta, as a pure function.
         *
         * Separated from the listener so the *policy* can be pinned on the JVM, which is
         * the only part a test can settle: whether a given handset really has the vendor
         * bug is not something the emulator can tell us -- its virtual sensors report the
         * same clock -- so what gets tested is what we do when the numbers say they differ.
         */
        fun verdictFor(deliveryDeltaNs: Long): ImuTimebase =
            if (deliveryDeltaNs in 0..MAX_PLAUSIBLE_DELIVERY_NS) {
                ImuTimebase.MATCHED
            } else {
                ImuTimebase.MISMATCHED
            }

        private const val TAG = "ImuSource"
    }
}
