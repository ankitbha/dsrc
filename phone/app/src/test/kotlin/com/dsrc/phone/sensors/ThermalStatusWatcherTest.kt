package com.dsrc.phone.sensors

import android.os.PowerManager
import java.util.concurrent.Executor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class ThermalStatusWatcherTest {

    /** Runs whatever is submitted immediately, on the calling thread -- a test has no
     * reason to wait on a real executor to observe what a callback did. */
    private val inlineExecutor = Executor { it.run() }

    private fun watcher(
        onRegister: (Executor, PowerManager.OnThermalStatusChangedListener) -> Unit = { _, _ -> },
        onUnregister: (PowerManager.OnThermalStatusChangedListener) -> Unit = {},
        monoClock: () -> Long = { 0L },
    ) = ThermalStatusWatcher(
        register = onRegister, unregister = onUnregister, monoClock = monoClock, executor = inlineExecutor,
    )

    @Test
    fun `before any callback there is no transition`() {
        val w = watcher()
        assertEquals(0L, w.changesCount)
        assertNull(w.lastTransition)
    }

    @Test
    fun `the first callback after registration names no transition`() {
        // The platform delivers the current status immediately on registering; a
        // transition from nothing to what is already in effect is not one.
        val w = watcher()
        w.onStatusChanged(PowerManager.THERMAL_STATUS_NONE)
        assertEquals(0L, w.changesCount)
        assertNull(w.lastTransition)
    }

    @Test
    fun `a transition increments the count and sets the three fields`() {
        var now = 1_000L
        val w = watcher(monoClock = { now })
        w.onStatusChanged(PowerManager.THERMAL_STATUS_NONE)
        now = 2_000L
        w.onStatusChanged(PowerManager.THERMAL_STATUS_LIGHT)

        assertEquals(1L, w.changesCount)
        val transition = w.lastTransition!!
        assertEquals("nominal", transition.fromStatus)
        assertEquals("light", transition.toStatus)
        assertEquals(2_000L, transition.atMonoNs)
    }

    @Test
    fun `two transitions inside one report period are counted, and the loss is visible`() {
        // Pins D9's stated bound: the count rises by 2 while only the most recent
        // transition is carried in `lastTransition` -- the gap is visible in the count,
        // not silently lost.
        val w = watcher()
        w.onStatusChanged(PowerManager.THERMAL_STATUS_NONE)
        w.onStatusChanged(PowerManager.THERMAL_STATUS_LIGHT)
        w.onStatusChanged(PowerManager.THERMAL_STATUS_SEVERE)

        assertEquals(2L, w.changesCount)
        val transition = w.lastTransition!!
        assertEquals("light", transition.fromStatus)
        assertEquals("severe", transition.toStatus)
    }

    @Test
    fun `repeating the same status is not a transition`() {
        val w = watcher()
        w.onStatusChanged(PowerManager.THERMAL_STATUS_NONE)
        w.onStatusChanged(PowerManager.THERMAL_STATUS_NONE)
        assertEquals(0L, w.changesCount)
    }

    @Test
    fun `start registers exactly the listener that stop unregisters`() {
        var registered: PowerManager.OnThermalStatusChangedListener? = null
        var unregistered: PowerManager.OnThermalStatusChangedListener? = null
        val w = watcher(
            onRegister = { _, listener -> registered = listener },
            onUnregister = { listener -> unregistered = listener },
        )

        w.start()
        assertTrue(registered != null)
        assertNull(unregistered)

        w.stop()
        assertSame("nulling the field and releasing the registration are different things", registered, unregistered)
    }

    @Test
    fun `stop is idempotent -- a second call does not unregister again`() {
        var unregisterCalls = 0
        val w = watcher(onUnregister = { unregisterCalls++ })
        w.start()
        w.stop()
        w.stop()
        assertEquals(1, unregisterCalls)
    }

    @Test
    fun `stop before start does not unregister anything`() {
        var unregisterCalls = 0
        val w = watcher(onUnregister = { unregisterCalls++ })
        w.stop()
        assertEquals(0, unregisterCalls)
    }

    @Test
    fun `stop shuts down the executor this class created for itself`() {
        // The default executor -- built when no `executor` argument is passed at all -- is a
        // single-thread pool with its own dedicated thread; `resourcesHeldAfterTeardown`
        // counts the `thermalWatcher` field, not that thread, so nothing else in
        // `SensingService`'s census can see it leaked. Captured through `onRegister` rather
        // than a constructor argument, since exercising the real default means not passing
        // one.
        var capturedExecutor: Executor? = null
        val w = ThermalStatusWatcher(
            register = { executor, _ -> capturedExecutor = executor },
            unregister = {},
            monoClock = { 0L },
        )
        w.start()
        w.stop()
        val pool = capturedExecutor as java.util.concurrent.ExecutorService
        assertTrue(pool.isShutdown)
    }

    @Test
    fun `stop shuts down its own default executor even when start was never called`() {
        // The unregister guard above is conditioned on `registered`; the executor shutdown is
        // not -- a caller that never started this watcher still gets its executor's thread
        // cleaned up.
        var capturedExecutor: Executor? = null
        val w = ThermalStatusWatcher(
            register = { executor, _ -> capturedExecutor = executor },
            unregister = {},
            monoClock = { 0L },
        )
        w.stop()
        w.start()  // only to observe which executor `register` would have received
        val pool = capturedExecutor as java.util.concurrent.ExecutorService
        assertTrue(pool.isShutdown)
    }

    @Test
    fun `stop leaves a caller-supplied executor running`() {
        // The other half of the ownership decision: a pool the caller handed in is the
        // caller's to shut down, not this class's, whether or not it happens to be an
        // `ExecutorService` -- shutting down a shared pool out from under whoever else uses it
        // would be a surprise this class has no business causing.
        val pool = java.util.concurrent.Executors.newSingleThreadExecutor()
        val w = ThermalStatusWatcher(
            register = { _, _ -> }, unregister = {}, monoClock = { 0L }, executor = pool,
        )
        w.start()
        w.stop()
        assertFalse(pool.isShutdown)
        pool.shutdown()
    }

    @Test
    fun `snapshot reports the count and the last transition together`() {
        var now = 1_000L
        val w = watcher(monoClock = { now })
        w.onStatusChanged(PowerManager.THERMAL_STATUS_NONE)
        now = 2_000L
        w.onStatusChanged(PowerManager.THERMAL_STATUS_LIGHT)

        val snapshot = w.snapshot()
        assertEquals(1L, snapshot.changesCount)
        assertEquals("nominal", snapshot.lastTransition?.fromStatus)
        assertEquals("light", snapshot.lastTransition?.toStatus)
    }

    @Test
    fun `snapshot before any transition is a zero count and no transition`() {
        val w = watcher()
        val snapshot = w.snapshot()
        assertEquals(0L, snapshot.changesCount)
        assertNull(snapshot.lastTransition)
    }
}
