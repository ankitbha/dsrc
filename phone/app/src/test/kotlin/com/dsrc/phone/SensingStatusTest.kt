package com.dsrc.phone

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
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


    @Test
    fun `a listener that throws does not starve the listeners behind it`() {
        // Insertion order, and the delivery loop had no guard: the first listener to throw
        // ended the loop, so every listener added after it kept a stale state -- and for a
        // terminal state there is no next change to correct it.
        val status = SensingStatus()
        val before = mutableListOf<SensingState>()
        val after = mutableListOf<SensingState>()
        status.addListener { before.add(it) }
        // Throws on the change only, not on the attach. A listener that threw on both
        // made the failure count 2 and the number stop meaning "one delivery failed".
        status.addListener { if (it == SensingState.STARTING) throw RuntimeException("a bug in some UI") }
        status.addListener { after.add(it) }

        status.set(SensingState.STARTING)

        assertEquals(listOf(SensingState.IDLE, SensingState.STARTING), before)
        assertEquals(
            "the listener behind the throwing one was never told",
            listOf(SensingState.IDLE, SensingState.STARTING),
            after,
        )
        assertEquals(1L, status.listenerFailures.get().toLong())
        assertTrue(status.lastListenerFailure?.contains("a bug in some UI") == true)
    }

    @Test
    fun `a listener that throws on being attached does not escape addListener`() {
        // The other delivery site. A new listener is handed the current state immediately,
        // and that call was unguarded too -- so registering a faulty listener threw at
        // whoever registered it, which in the Activity's case is onStart.
        val status = SensingStatus()
        status.addListener { throw RuntimeException("throws on attach") }
        assertEquals(1L, status.listenerFailures.get().toLong())
        assertEquals(
            "it stays registered; it is not the holder's job to judge",
            1L,
            status.listenerCount.toLong(),
        )
    }

    @Test
    fun `a throwing listener is not reported as a state change failure`() {
        // The severe half. `set` is called from the service's own handle(), inside
        // react(STARTING)'s try and after come-up has already succeeded, so an escaping
        // listener exception was caught as a *start failure* and offered as Failed while the
        // machine was RUNNING -- which the machine accepts, with no teardown behind it. The
        // state must still advance and the exception must not come back out.
        val status = SensingStatus()
        status.addListener { throw RuntimeException("boom") }
        status.set(SensingState.RUNNING)
        assertEquals("the state must advance regardless", SensingState.RUNNING, status.state)
    }


    @Test
    fun `a re-entrant set does not strand the listeners behind it on a superseded state`() {
        // `set` used to deliver the value it was called with. A listener that calls `set`
        // re-entrantly -- which is what a reactor attached to this holder does -- runs the
        // inner delivery to completion; the outer loop then resumes and hands the *older*
        // value to every listener behind it, and for a terminal state that is the last
        // thing they ever hear. The measured shape was ui=[IDLE, RUNNING, STARTING] with
        // status.state already RUNNING: the display left one transition in the past, with
        // nothing further coming, which is the disagreement this class exists to prevent.
        val status = SensingStatus()
        val ui = mutableListOf<SensingState>()
        var reentered = false
        status.addListener {
            if (it == SensingState.STARTING && !reentered) {
                reentered = true
                status.set(SensingState.RUNNING)
            }
        }
        status.addListener { ui.add(it) }

        status.set(SensingState.STARTING)

        assertEquals(SensingState.RUNNING, status.state)
        assertEquals(
            "the listener behind the re-entrant one was left on a superseded state",
            SensingState.RUNNING,
            ui.last(),
        )
    }

}
