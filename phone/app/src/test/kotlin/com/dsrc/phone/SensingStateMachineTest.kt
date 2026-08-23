package com.dsrc.phone

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SensingStateMachineTest {

    private fun machine(at: SensingState = SensingState.IDLE) = SensingStateMachine(at)

    private fun accepted(t: Transition): Transition.Accepted {
        assertTrue("expected an accepted transition, got $t", t is Transition.Accepted)
        return t as Transition.Accepted
    }

    // -- the happy path ------------------------------------------------------

    @Test
    fun `start moves idle to starting`() {
        val m = machine()
        assertEquals(SensingState.STARTING, accepted(m.offer(SensingEvent.Start)).to)
        assertEquals(SensingState.STARTING, m.state)
    }

    @Test
    fun `started moves starting to running`() {
        val m = machine(SensingState.STARTING)
        assertEquals(SensingState.RUNNING, accepted(m.offer(SensingEvent.Started)).to)
    }

    @Test
    fun `a full cycle returns to idle and ignores nothing`() {
        val m = machine()
        m.offer(SensingEvent.Start)
        m.offer(SensingEvent.Started)
        m.offer(SensingEvent.Stop)
        m.offer(SensingEvent.Stopped)
        assertEquals(SensingState.IDLE, m.state)
        assertEquals(0, m.ignoredEvents)
    }

    @Test
    fun `sensing can be restarted after a clean stop`() {
        val m = machine()
        repeat(3) {
            m.offer(SensingEvent.Start)
            m.offer(SensingEvent.Started)
            assertEquals(SensingState.RUNNING, m.state)
            m.offer(SensingEvent.Stop)
            m.offer(SensingEvent.Stopped)
            assertEquals(SensingState.IDLE, m.state)
        }
        assertEquals(0, m.ignoredEvents)
    }

    // -- idempotence ---------------------------------------------------------

    @Test
    fun `starting twice does not start twice and is not an error`() {
        val m = machine()
        m.offer(SensingEvent.Start)
        val second = m.offer(SensingEvent.Start)
        assertTrue(second is Transition.Ignored)
        assertEquals(SensingState.STARTING, m.state)
        assertEquals(1, m.ignoredEvents)
    }

    @Test
    fun `start while running is ignored`() {
        val m = machine(SensingState.RUNNING)
        assertTrue(m.offer(SensingEvent.Start) is Transition.Ignored)
        assertEquals(SensingState.RUNNING, m.state)
    }

    @Test
    fun `stop while idle is a no-op rather than a crash`() {
        val m = machine()
        assertTrue(m.offer(SensingEvent.Stop) is Transition.Ignored)
        assertEquals(SensingState.IDLE, m.state)
        assertEquals(1, m.ignoredEvents)
    }

    @Test
    fun `stopping twice is counted once and does not double-stop`() {
        val m = machine(SensingState.RUNNING)
        m.offer(SensingEvent.Stop)
        assertTrue(m.offer(SensingEvent.Stop) is Transition.Ignored)
        assertEquals(SensingState.STOPPING, m.state)
    }

    @Test
    fun `a start racing a shutdown is dropped, not queued`() {
        // Queueing it would start a session the user had just asked to end.
        val m = machine(SensingState.STOPPING)
        assertTrue(m.offer(SensingEvent.Start) is Transition.Ignored)
        assertEquals(SensingState.STOPPING, m.state)
    }

    @Test
    fun `an out-of-order completion is ignored`() {
        val m = machine(SensingState.RUNNING)
        assertTrue(m.offer(SensingEvent.Stopped) is Transition.Ignored)
        assertEquals(SensingState.RUNNING, m.state)

        val starting = machine(SensingState.IDLE)
        assertTrue(starting.offer(SensingEvent.Started) is Transition.Ignored)
        assertEquals(SensingState.IDLE, starting.state)
    }

    // -- stop from everywhere ------------------------------------------------

    @Test
    fun `stop from any active state reaches idle`() {
        for (from in listOf(SensingState.STARTING, SensingState.RUNNING)) {
            val m = machine(from)
            m.offer(SensingEvent.Stop)
            m.offer(SensingEvent.Stopped)
            assertEquals("from $from", SensingState.IDLE, m.state)
        }
    }

    // -- revocation ----------------------------------------------------------

    @Test
    fun `a revoked permission stops sensing under its own name`() {
        // Distinct from a failure because the remedy is different: grant, not retry.
        for (from in listOf(SensingState.STARTING, SensingState.RUNNING)) {
            val m = machine(from)
            assertEquals(
                SensingState.STOPPED_PERMISSION_REVOKED,
                accepted(m.offer(SensingEvent.PermissionRevoked)).to,
            )
        }
    }

    @Test
    fun `a revoke while idle is ignored`() {
        val m = machine()
        assertTrue(m.offer(SensingEvent.PermissionRevoked) is Transition.Ignored)
        assertEquals(SensingState.IDLE, m.state)
    }

    @Test
    fun `sensing can restart after a revoke is resolved`() {
        val m = machine(SensingState.RUNNING)
        m.offer(SensingEvent.PermissionRevoked)
        assertEquals(SensingState.STARTING, accepted(m.offer(SensingEvent.Start)).to)
    }

    // -- failure -------------------------------------------------------------

    @Test
    fun `a failure records its reason`() {
        val m = machine(SensingState.STARTING)
        assertEquals(SensingState.STOPPED_ERROR, accepted(m.offer(SensingEvent.Failed("camera busy"))).to)
        assertEquals("camera busy", m.lastFailure)
    }

    @Test
    fun `a failure during shutdown is still a failure`() {
        // Teardown can fail too, and silently landing in IDLE would hide it.
        val m = machine(SensingState.STOPPING)
        assertEquals(SensingState.STOPPED_ERROR, accepted(m.offer(SensingEvent.Failed("release"))).to)
    }

    @Test
    fun `a failure while already stopped is ignored and does not overwrite the reason`() {
        val m = machine(SensingState.RUNNING)
        m.offer(SensingEvent.Failed("first"))
        assertTrue(m.offer(SensingEvent.Failed("second")) is Transition.Ignored)
        assertEquals("first", m.lastFailure)
    }

    @Test
    fun `an idle machine has no failure recorded`() {
        val m = machine()
        m.offer(SensingEvent.Stop)
        assertNull(m.lastFailure)
    }

    @Test
    fun `sensing can restart after a failure`() {
        val m = machine(SensingState.RUNNING)
        m.offer(SensingEvent.Failed("boom"))
        assertEquals(SensingState.STARTING, accepted(m.offer(SensingEvent.Start)).to)
    }

    // -- invariants ----------------------------------------------------------

    // -- residency ------------------------------------------------------------

    @Test
    fun `requiresService covers exactly the states that need the service alive`() {
        // STOPPING is included and isActive is not enough: teardown still needs the
        // service. Everything else must let it go.
        val expected = mapOf(
            SensingState.IDLE to false,
            SensingState.STARTING to true,
            SensingState.RUNNING to true,
            SensingState.STOPPING to true,
            SensingState.STOPPED_PERMISSION_REVOKED to false,
            SensingState.STOPPED_ERROR to false,
        )
        for (state in SensingState.entries) {
            assertEquals(state.name, expected.getValue(state), state.requiresService)
        }
    }

    @Test
    fun `an ignored event leaves a state that does not require the service`() {
        // The zombie-service case: an intent the machine ignores still creates the
        // service, so if the resulting state claimed residency nothing would ever
        // stop it. Every ignorable event arriving at rest must leave the machine in a
        // state that releases.
        val atRest = listOf(
            SensingState.IDLE,
            SensingState.STOPPED_PERMISSION_REVOKED,
            SensingState.STOPPED_ERROR,
        )
        val events = listOf(
            SensingEvent.Stop,
            SensingEvent.Stopped,
            SensingEvent.Started,
            SensingEvent.PermissionRevoked,
            SensingEvent.Failed("x"),
        )
        for (state in atRest) {
            for (event in events) {
                val m = machine(state)
                assertTrue("$state + $event should be ignored", m.offer(event) is Transition.Ignored)
                assertFalse("$state + $event must not require the service", m.state.requiresService)
            }
        }
    }

    @Test
    fun `requiresService is true whenever isActive is`() {
        for (state in SensingState.entries) {
            if (state.isActive) assertTrue(state.name, state.requiresService)
        }
    }

    @Test
    fun `isActive is exactly starting and running`() {
        for (state in SensingState.entries) {
            val expected = state == SensingState.STARTING || state == SensingState.RUNNING
            assertEquals(state.name, expected, state.isActive)
        }
    }

    @Test
    fun `no event from any state ever produces an undefined result`() {
        // Exhaustive sweep: every state crossed with every event either transitions or
        // is ignored, and the state afterwards is always a declared one.
        val events = listOf(
            SensingEvent.Start,
            SensingEvent.Started,
            SensingEvent.Stop,
            SensingEvent.Stopped,
            SensingEvent.PermissionRevoked,
            SensingEvent.Failed("x"),
        )
        for (state in SensingState.entries) {
            for (event in events) {
                val m = machine(state)
                val transition = m.offer(event)
                assertTrue(SensingState.entries.contains(m.state))
                when (transition) {
                    is Transition.Ignored -> assertEquals(
                        "$state + $event was ignored but the state moved", state, m.state,
                    )
                    is Transition.Accepted -> {
                        assertEquals(state, transition.from)
                        assertEquals(m.state, transition.to)
                    }
                }
            }
        }
    }

    @Test
    fun `an ignored event never changes the state and is always counted`() {
        val events = listOf(
            SensingEvent.Start, SensingEvent.Started, SensingEvent.Stop,
            SensingEvent.Stopped, SensingEvent.PermissionRevoked, SensingEvent.Failed("x"),
        )
        for (state in SensingState.entries) {
            for (event in events) {
                val m = machine(state)
                val before = m.ignoredEvents
                if (m.offer(event) is Transition.Ignored) {
                    assertEquals(state, m.state)
                    assertEquals(before + 1, m.ignoredEvents)
                }
            }
        }
    }
}
