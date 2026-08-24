package com.dsrc.phone

import org.junit.Assert.assertEquals
import org.junit.Test

class SensingStatusTest {

    @Test
    fun `a new listener is told the current state immediately`() {
        // A UI attaching after sensing already started must not sit blank until the
        // next change, which on a running session could be minutes.
        val status = SensingStatus(SensingState.RUNNING)
        val seen = mutableListOf<SensingState>()
        status.addListener { seen.add(it) }
        assertEquals(listOf(SensingState.RUNNING), seen)
    }

    @Test
    fun `a change reaches every listener`() {
        val status = SensingStatus()
        val a = mutableListOf<SensingState>()
        val b = mutableListOf<SensingState>()
        status.addListener { a.add(it) }
        status.addListener { b.add(it) }
        status.set(SensingState.STARTING)
        assertEquals(listOf(SensingState.IDLE, SensingState.STARTING), a)
        assertEquals(listOf(SensingState.IDLE, SensingState.STARTING), b)
    }

    @Test
    fun `setting the same state again notifies nobody`() {
        // Otherwise a service re-emitting its state on every intent would redraw the
        // UI continuously.
        val status = SensingStatus()
        val seen = mutableListOf<SensingState>()
        status.addListener { seen.add(it) }
        status.set(SensingState.IDLE)
        status.set(SensingState.IDLE)
        assertEquals(listOf(SensingState.IDLE), seen)
    }

    @Test
    fun `a removed listener stops hearing`() {
        val status = SensingStatus()
        val seen = mutableListOf<SensingState>()
        val listener = SensingStatus.Listener { seen.add(it) }
        status.addListener(listener)
        status.removeListener(listener)
        status.set(SensingState.RUNNING)
        assertEquals(listOf(SensingState.IDLE), seen)
    }

    @Test
    fun `removing a listener that was never added is harmless`() {
        val status = SensingStatus()
        status.removeListener { }
        assertEquals(0, status.listenerCount)
    }

    @Test
    fun `attach and detach leave no listener behind`() {
        // The Activity adds in onStart and removes in onStop; a leak here would hold a
        // destroyed Activity for the life of the service.
        val status = SensingStatus()
        repeat(5) {
            val listener = SensingStatus.Listener { }
            status.addListener(listener)
            status.removeListener(listener)
        }
        assertEquals(0, status.listenerCount)
    }

    @Test
    fun `state is readable without a listener`() {
        val status = SensingStatus()
        status.set(SensingState.STOPPED_ERROR)
        assertEquals(SensingState.STOPPED_ERROR, status.state)
    }

    @Test
    fun `a published state becomes visible to another thread`() {
        // Losing @Volatile survived: every existing test reads the field on the thread that
        // wrote it. The service publishes from a binder thread and the UI reads from the
        // main one, so a stale read shows a stopped session as RUNNING with an inert Stop
        // button -- the exact failure onDestroy's publish exists to prevent.
        repeat(200) {
            val status = SensingStatus()
            status.set(SensingState.IDLE)
            val seen = java.util.concurrent.atomic.AtomicReference<SensingState?>(null)
            val reader = Thread {
                val deadline = System.nanoTime() + 200_000_000L
                while (System.nanoTime() < deadline) {
                    if (status.state == SensingState.RUNNING) {
                        seen.set(SensingState.RUNNING)
                        return@Thread
                    }
                    Thread.onSpinWait()
                }
            }
            reader.start()
            status.set(SensingState.RUNNING)
            reader.join(2_000)
            assertEquals("the reader never saw the published state", SensingState.RUNNING, seen.get())
        }
    }

}
