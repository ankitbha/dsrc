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
    executor: Executor? = null,
) {
    /** One transition: the two statuses it was between, and when this side observed it. */
    data class Transition(val fromStatus: String, val toStatus: String, val atMonoNs: Long)

    /** [changesCount] and [lastTransition], read together under one lock acquisition. */
    data class Snapshot(val changesCount: Long, val lastTransition: Transition?)

    // `null` unless this instance created its own executor, which happens exactly when the
    // caller passed none: that is the only executor this class shuts down on `stop()`. A
    // caller-supplied one is used the same way but never touched at teardown, whether or not
    // it happens to be an `ExecutorService`.
    private val ownExecutor: ExecutorService? = if (executor == null) {
        Executors.newSingleThreadExecutor { runnable -> Thread(runnable, THREAD_NAME).apply { isDaemon = true } }
    } else {
        null
    }
    private val activeExecutor: Executor = executor ?: ownExecutor!!

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
        register(activeExecutor, listener)
        registered = true
    }

    /**
     * Unregisters the listener it registered, and shuts down this class's own executor's
     * background thread if it created one. Idempotent: a second call does nothing further.
     * Runs regardless of whether `start()` was ever called, the same as the unregister guard
     * above is independent of it. A caller-supplied executor is left running -- a shared pool
     * handed in by the caller is the caller's to shut down, not this class's -- the same
     * `encodeExecutor?.shutdown()` convention `SensingService` uses for its own executor
     * field, which it also creates and owns outright.
     */
    fun stop() {
        if (registered) {
            unregister(listener)
            registered = false
        }
        ownExecutor?.shutdown()
    }

    val changesCount: Long get() = synchronized(lock) { changes }
    val lastTransition: Transition? get() = synchronized(lock) { last }

    /**
     * [changesCount] and [lastTransition] read under one lock acquisition, for a caller that
     * needs both to describe the same observed state. Reading the two properties above
     * separately can interleave with [onStatusChanged] between the calls -- on the very first
     * transition of a service run, that window can land with the count already at 1 while
     * [lastTransition] is still null, which a caller building one record from both would
     * otherwise report as a rise with no transition to name.
     */
    fun snapshot(): Snapshot = synchronized(lock) { Snapshot(changes, last) }

    companion object {
        const val THREAD_NAME = "dsrc-thermal-watcher"
    }
}
