package com.dsrc.phone.net

import android.util.Log

/**
 * Sends the time-sync pings the spec makes the phone responsible for.
 *
 * `Session.sendTimeSyncPing` has existed since task 15 and nothing outside the
 * tests ever called it, so a real drive sent none. The consequence is not a
 * failure anyone sees: the Jetson accumulates no samples, its offset estimate
 * never forms, and every camera and GPS stamp falls back to arrival-time proxy
 * for the whole session. Measured on a 60 s run with a real handset --
 * `samples_accepted: 0`, 287 frames, every tick `measured_arrival_proxy`. The run
 * produced advisories the whole time, because the proxy is wrong only by the link
 * segment and that is tens of milliseconds against a 2 s staleness threshold. It
 * is the shape of defect that survives every test that injects its own pings.
 *
 * **Cadence is the spec's, not a guess.** 4 Hz for the first ten seconds, 1 Hz
 * after: the far side needs five samples before its estimate opens the gate, so
 * the fast phase is what makes the first advisory alignable in seconds rather
 * than in a minute, and the steady rate is two ~200-byte frames a second against
 * a camera stream, about a tenth of a percent.
 */
class TimeSyncDriver(
    /**
     * One send, which is the whole dependency. Taking the `SessionHolder` itself
     * would have this class reach through an object it uses one method of, and
     * that object is final and needs a socket to construct -- so the narrow
     * function is both the smaller coupling and the one a test can supply.
     */
    private val sendPing: (Long) -> Boolean,
    private val monoClock: () -> Long = System::nanoTime,
    private val sleeper: (Long) -> Unit = { Thread.sleep(it) },
) {
    @Volatile
    private var thread: Thread? = null

    @Volatile
    private var stopped = false

    private var exchangeId = 0L

    /** Pings sent and refused, so a drive that never synced says so as a number. */
    @Volatile
    var sent: Long = 0
        private set

    @Volatile
    var refused: Long = 0
        private set

    fun start(): TimeSyncDriver {
        if (thread?.isAlive == true) return this
        stopped = false
        val worker = Thread({ loop() }, THREAD_NAME)
        worker.isDaemon = true
        thread = worker
        worker.start()
        return this
    }

    private fun loop() {
        try {
            pingUntilStopped()
        } catch (e: InterruptedException) {
            // `stop()` interrupts to cut a sleep short, so this is how the thread
            // is asked to finish rather than something going wrong. Uncaught, it
            // escaped the worker and the instrumented teardown test reported it as
            // a failure -- correctly, since an exception leaving a thread is
            // indistinguishable from a real one at that point. The flag is
            // restored so anything above can still see the interrupt.
            Thread.currentThread().interrupt()
        }
    }

    private fun pingUntilStopped() {
        val startedNs = monoClock()
        while (!stopped) {
            if (sendPing(++exchangeId)) sent++ else refused++
            // Measured from the driver's own start, not from the session's. A link
            // that drops and redials does not restart the fast phase: the far side
            // keeps its samples across a reconnect only if the session did, and
            // when it did not, its own gate is what refuses to convert -- so
            // re-flooding here would spend bandwidth on a decision made there.
            val elapsedNs = monoClock() - startedNs
            val periodMs = if (elapsedNs < FAST_PHASE_NS) FAST_PERIOD_MS else STEADY_PERIOD_MS
            sleeper(periodMs)
        }
    }

    fun stop() {
        stopped = true
        val worker = thread ?: return
        thread = null
        worker.interrupt()
        worker.join(JOIN_MS)
        Log.i(TAG, "time sync stopped: sent=$sent refused=$refused")
    }

    companion object {
        const val THREAD_NAME = "dsrc-timesync"

        /** `specs/transport_protocol.md`: sampling, first 10 s. */
        const val FAST_PHASE_NS = 10_000_000_000L
        const val FAST_PERIOD_MS = 250L

        /** `specs/transport_protocol.md`: sampling, thereafter. */
        const val STEADY_PERIOD_MS = 1_000L

        private const val JOIN_MS = 2_000L
        private const val TAG = "TimeSyncDriver"
    }
}
