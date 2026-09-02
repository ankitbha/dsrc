package com.dsrc.phone.sensors

import android.os.PowerManager
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Counts thermal-status transitions as they happen, independent of the 1 Hz poll
 * `TelemetryReporter` samples on.
 *
 * [PowerManager.OnThermalStatusChangedListener] fires the instant the platform's status
 * changes, which a once-a-second poll can miss entirely between two samples, or timestamp to
 * no better than the second it happened to land on. Registering it is the one behaviour
 * change in this task -- everything this class does with what it observes is additive
 * record-keeping. `TelemetryReporter.Sample.thermalStatus` keeps reading
 * `PowerManager.currentThermalStatus` on its own poll, so nothing this class sees can move a
 * commanded rate; see `TelemetryReporterTest`'s direction test for the boundary this holds.
 *
 * `register`/`unregister` are injected rather than a raw `PowerManager`, the same shape
 * `TelemetryReporter`'s `sink` already takes: a JVM test can then count registrations without
 * a real platform object, which `PowerManager` cannot be constructed as in a unit test.
 */
class ThermalStatusWatcher(
    private val register: (Executor, PowerManager.OnThermalStatusChangedListener) -> Unit,
    private val unregister: (PowerManager.OnThermalStatusChangedListener) -> Unit,
    private val monoClock: () -> Long,
    private val executor: Executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, THREAD_NAME).apply { isDaemon = true }
    },
) {
    /** One transition: the two statuses it was between, and when this side observed it. */
    data class Transition(val fromStatus: String, val toStatus: String, val atMonoNs: Long)

    private val lock = Any()
    private var lastRawStatus: Int? = null
    private var changes = 0L
    private var last: Transition? = null
    private var registered = false

    private val listener = PowerManager.OnThermalStatusChangedListener { status -> onStatusChanged(status) }

    /**
     * The callback body, exposed for a test to drive directly -- registering a real listener
     * needs an executor and a platform object a JVM test does not have, and the logic worth
     * pinning is what happens to a status integer, not how it arrives.
     *
     * The first call after (re)registration names no transition: the platform delivers the
     * current status immediately on registering, and a "transition" from nothing to the
     * status already in effect is not one.
     */
    internal fun onStatusChanged(status: Int) {
        synchronized(lock) {
            val from = lastRawStatus
            lastRawStatus = status
            if (from == null || from == status) return
            changes++
            last = Transition(
                fromStatus = ThermalReader.statusName(from),
                toStatus = ThermalReader.statusName(status),
                atMonoNs = monoClock(),
            )
        }
    }

    fun start() {
        register(executor, listener)
        registered = true
    }

    /**
     * Unregisters the listener it registered, and shuts down the default executor's
     * background thread. Idempotent: a second call does nothing further. An injected
     * executor -- a test's inline one, or a caller's own pool -- is left running; only the
     * single-thread pool this class creates for itself by default is this class's to shut
     * down, the same `encodeExecutor?.shutdown()` convention `SensingService` uses for its
     * own executor field.
     */
    fun stop() {
        if (registered) {
            unregister(listener)
            registered = false
        }
        (executor as? ExecutorService)?.shutdown()
    }

    val changesCount: Long get() = synchronized(lock) { changes }
    val lastTransition: Transition? get() = synchronized(lock) { last }

    companion object {
        const val THREAD_NAME = "dsrc-thermal-watcher"
    }
}
