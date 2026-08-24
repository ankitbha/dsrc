package com.dsrc.phone.sensors

import android.util.Log
import com.dsrc.transport.PhoneTelemetry

/**
 * What the phone managed, reported upstream so a throttling handset is visible rather than
 * merely quiet.
 *
 * The phone reports and the Jetson decides. That is not a stylistic preference — the spec's
 * "configuration flows one way" section says the phone "applies what arrives and reports
 * what it achieved; it originates no sensing decision of its own", and thermal pressure is
 * the case that most tempts a device to decide for itself. A phone that quietly halved its
 * own camera rate when it got warm would leave the Jetson comparing a model against inputs
 * it never asked for and cannot see it did not get.
 *
 * So "degrades rather than fails" means: keep running at whatever the platform will give,
 * report the shortfall honestly, and let the far side lower the commanded rate if it wants
 * to. `achieved` beside `rates` is exactly that shortfall.
 */
class TelemetryReporter(
    private val monoClock: () -> Long,
    private val sample: () -> Sample,
    private val sink: (PhoneTelemetry) -> Boolean,
) {
    /** One reading of everything the report needs, taken together. */
    data class Sample(
        val thermalStatus: String,
        val thermalHeadroom: Double?,
        /** Cumulative deliveries per modality, in `RateCommand.RATE_KEYS` order. */
        val delivered: Map<String, Long>,
        /** Cumulative drops per modality, in `PhoneTelemetry.DROP_KEYS` order. */
        val dropped: Map<String, Long>,
        val hereCalls: Long,
        val hereErrors: Long,
    )

    /**
     * Drive the reporter from its own thread until stopped.
     *
     * Its own thread and its own name, so the service's census sees it. The cadence is not
     * a commanded rate -- telemetry is the phone talking about itself, and the Jetson has
     * no knob for it -- so it is fixed here.
     */
    fun start() {
        val worker = Thread({
            while (running) {
                // Logged, not swallowed. The sibling loop in HerePipeline logs and this one
                // did not, which is how an unguarded API-30 call took the whole telemetry
                // stream down on Android 10 with nothing to say so -- the counters read the
                // same as they do one tick after come-up.
                runCatching { report() }.onFailure { Log.e(TAG, "telemetry tick failed", it) }
                if (!running) return@Thread
                try {
                    Thread.sleep(PERIOD_MS)
                } catch (e: InterruptedException) {
                    Thread.currentThread().interrupt()
                    return@Thread
                }
            }
        }, THREAD_NAME)
        worker.isDaemon = true
        thread = worker
        worker.start()
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
    }

    @Volatile
    private var running = true

    @Volatile
    private var thread: Thread? = null

    private val lock = Any()
    private var lastAtNs = 0L
    private var lastDelivered = mapOf<String, Long>()
    private var reports = 0L
    private var skipped = 0L

    /**
     * Build and send one report, or skip it.
     *
     * Achieved is a *rate*, so it needs two readings and the time between them. The first
     * call therefore establishes the baseline and sends nothing: a report built from one
     * reading would have to divide by the time since some arbitrary epoch, which is not a
     * rate anyone asked about.
     */
    fun report(): Boolean {
        val now = monoClock()
        val reading = sample()

        val previous = synchronized(lock) {
            val was = lastDelivered
            val wasAt = lastAtNs
            lastDelivered = reading.delivered
            lastAtNs = now
            if (was.isEmpty()) {
                skipped++
                null
            } else {
                was to wasAt
            }
        } ?: return false

        val (baseline, baselineAtNs) = previous
        val elapsedNs = now - baselineAtNs
        if (elapsedNs <= 0) {
            // Two readings in the same nanosecond, or a clock that went backwards. Dividing
            // would be a rate of infinity, which the wire refuses -- and refusing the frame
            // takes the thermal status down with it.
            synchronized(lock) { skipped++ }
            return false
        }

        val achieved = com.dsrc.transport.RateCommand.RATE_KEYS.associateWith { key ->
            val delta = (reading.delivered[key] ?: 0L) - (baseline[key] ?: 0L)
            // A counter that went backwards means the modality was restarted underneath us.
            // Reporting a negative rate would be a number nobody can act on.
            if (delta <= 0L) 0.0 else delta * 1e9 / elapsedNs
        }

        val sent = sink(
            PhoneTelemetry(
                captureMonoNs = now,
                thermalStatus = reading.thermalStatus,
                thermalHeadroom = reading.thermalHeadroom,
                achieved = achieved,
                dropped = PhoneTelemetry.DROP_KEYS.associateWith { reading.dropped[it] ?: 0L },
                hereCalls = reading.hereCalls,
                hereErrors = reading.hereErrors,
            )
        )
        synchronized(lock) { reports++ }
        return sent
    }

    val stats: Stats
        get() = synchronized(lock) { Stats(reports = reports, skipped = skipped) }

    data class Stats(val reports: Long, val skipped: Long)

    companion object {
        const val THREAD_NAME = "dsrc-telemetry"

        private const val TAG = "TelemetryReporter"

        /**
         * How often a report goes up.
         *
         * One second. Fast enough that a handset heating up is visible while the Jetson can
         * still do something about it, slow enough that the report is not itself a load --
         * and it doubles as the averaging window for `achieved`, which wants to be long
         * enough that a 0.2 Hz modality is not always reported as zero.
         */
        const val PERIOD_MS = 1_000L
    }
}
