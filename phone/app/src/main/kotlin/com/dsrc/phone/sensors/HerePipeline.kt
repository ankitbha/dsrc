package com.dsrc.phone.sensors

import com.dsrc.phone.config.SensingConfig
import com.dsrc.transport.HereQuery
import android.util.Log
import com.dsrc.transport.HereResponse

/**
 * The HERE path: call at the commanded rate, forward the reply untouched, count everything.
 *
 * Same shape as the other three modalities, reusing [RateGate], because "at the commanded
 * rate" has to mean one thing everywhere.
 *
 * The heading the others do not have is `unconfigured`: the gate opened and there was no
 * query to run. The phone makes no call until the Jetson tells it what to ask, because a
 * default query shape would be the phone originating a sensing decision — the one thing the
 * configuration surface exists to prevent. A drive that produces no `here` frames because
 * nobody configured one is a legible outcome; frames for a corridor nobody chose is not.
 */
class HerePipeline(
    config: SensingConfig,
    /** Null when no API key was configured, which disables the modality without failing. */
    private val client: HereClient?,
    private val monoClock: () -> Long,
    /** Where a reply goes. Returning false means the transport refused it. */
    private val sink: (HereResponse, ByteArray) -> Boolean,
) {
    private val gate = RateGate(config.hereHz)

    private val counters = Any()

    private var seen = 0L
    private var accepted = 0L
    private var gated = 0L
    private var refusedStopped = 0L
    private var unconfigured = 0L
    private var delivered = 0L
    private var refusedBySink = 0L
    private var calls = 0L
    private var errors = 0L

    @Volatile
    private var running = true

    @Volatile
    private var query: HereQuery? = null

    val isStopped: Boolean get() = !running

    /** Apply a commanded query shape. Null leaves the current one alone. */
    fun setQuery(next: HereQuery?) {
        if (next != null) query = next
    }

    fun setRate(hz: Double) = gate.setRate(hz)

    /**
     * Offer a tick. Returns true when a reply went out.
     *
     * The call happens outside the counter lock: it is a network round trip of up to ten
     * seconds, and holding a lock across it would block the teardown log behind a stalled
     * HTTP read. The counters that bracket it are taken under the lock either side, so the
     * accounting stays exact even though the call is not atomic with it.
     */
    fun tick(): Boolean {
        val target = synchronized(counters) {
            seen++
            // Configuration before the gate, deliberately. The other way round, a tick with
            // nothing to ask consumes the rate slot -- so the first tick after a query
            // finally arrives is gated, and at 0.2 Hz that is five seconds of silence for
            // no reason. Nothing to ask is not the same as asking too often.
            when {
                !running -> {
                    refusedStopped++
                    null
                }
                // No key, or no query: both mean there is nothing to ask, and a drive with
                // no HERE traffic is worth distinguishing from one where the calls failed.
                client == null || query == null -> {
                    unconfigured++
                    null
                }
                !gate.accept(monoClock()) -> {
                    gated++
                    null
                }
                else -> {
                    accepted++
                    query
                }
            }
        } ?: return false

        val call = client!!.fetch(target)
        val response = HereResponse(
            captureMonoNs = call.requestMonoNs,
            requestUrl = call.requestUrl,
            status = call.status.toLong(),
            contentType = call.contentType,
            // What the Jetson asked to have recorded, echoed. The phone cannot derive a
            // position from `in`, which is an opaque HERE expression that may be a corridor
            // of a hundred points, and zeros would be three fields claiming to be a query
            // location that is not one.
            queryLat = target.lat,
            queryLon = target.lon,
            queryRadiusM = target.radiusM,
            requestMonoNs = call.requestMonoNs,
            responseMonoNs = call.responseMonoNs,
        )

        val sent = sink(response, call.body)
        synchronized(counters) {
            calls++
            if (call.status !in 200..299) errors++
            if (sent) delivered++ else refusedBySink++
        }
        return sent
    }

    /**
     * Drive the pipeline from its own thread until stopped.
     *
     * Its own thread because a call can block for ten seconds and everything else here runs
     * on a looper that must not. Named so the service's thread census can see it — the
     * census is how three resources were caught unstarted in task 17, and how the IMU
     * source's registration is caught now.
     *
     * The tick interval is not the rate: the gate decides that. This only has to offer
     * often enough that the gate is never the thing waiting.
     */
    fun start() {
        val worker = Thread({
            while (running) {
                runCatching { tick() }.onFailure { Log.e(TAG, "here tick failed; continuing", it) }
                if (!running) return@Thread
                try {
                    Thread.sleep(TICK_MS)
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
    private var thread: Thread? = null

    val stats: Stats
        get() = synchronized(counters) {
            Stats(
                seen = seen,
                accepted = accepted,
                gated = gated,
                refusedStopped = refusedStopped,
                unconfigured = unconfigured,
                delivered = delivered,
                refusedBySink = refusedBySink,
                calls = calls,
                errors = errors,
                rateHz = gate.hz,
                configured = client != null && query != null,
            )
        }

    companion object {
        const val THREAD_NAME = "dsrc-here"

        /**
         * How often the loop offers a tick.
         *
         * Not the rate. The gate decides the rate; this only has to be short enough that
         * the gate is never the thing waiting, and long enough not to spin. At the default
         * 0.2 Hz the gate accepts one tick in twenty-five.
         */
        const val TICK_MS = 200L

        private const val TAG = "HerePipeline"
    }

    data class Stats(
        val seen: Long,
        val accepted: Long,
        val gated: Long,
        val refusedStopped: Long,
        /** Ticks that found no commanded query. */
        val unconfigured: Long,
        val delivered: Long,
        val refusedBySink: Long,
        val calls: Long,
        /** Calls that came back non-2xx, including the zero that means no response. */
        val errors: Long,
        val rateHz: Double,
        val configured: Boolean,
    ) {
        /** Every tick is under exactly one heading. */
        val balances: Boolean get() = seen == accepted + gated + refusedStopped + unconfigured

        /** And every accepted tick made a call that either went out or was refused. */
        val acceptedBalances: Boolean get() = accepted == delivered + refusedBySink
    }
}
