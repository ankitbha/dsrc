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
    fun `returning to idle clears the recorded failure`() {
        // A reason that outlives its state reads as a live error to anything that shows
        // it when non-null, which is what a task-23 UI would do.
        val m = machine(SensingState.RUNNING)
        m.offer(SensingEvent.Failed("camera busy"))
        assertEquals("camera busy", m.lastFailure)
        m.offer(SensingEvent.Stop)
        assertEquals(SensingState.IDLE, m.state)
        assertNull("a cleared state must not carry a stale reason", m.lastFailure)
    }

    @Test
    fun `a full stop cycle clears the recorded failure`() {
        val m = machine(SensingState.RUNNING)
        m.offer(SensingEvent.Failed("boom"))
        m.offer(SensingEvent.Start)
        m.offer(SensingEvent.Started)
        assertEquals("a failure survives until the machine returns to IDLE", "boom", m.lastFailure)
        m.offer(SensingEvent.Stop)
        m.offer(SensingEvent.Stopped)
        assertNull(m.lastFailure)
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
    fun `an ignored event never leaves the service needing to stay resident`() {
        // The zombie-service invariant. Any intent creates the service, including one
        // the machine ignores, so if an ignored event could leave a state that claims
        // residency nothing would ever stop it.
        //
        // Which events are ignored is deliberately not hardcoded here -- that belongs
        // to the transition-table test. This asserts the consequence, so it keeps
        // holding when the table changes.
        val events = listOf(
            SensingEvent.Start,
            SensingEvent.Started,
            SensingEvent.Stop,
            SensingEvent.Stopped,
            SensingEvent.PermissionRevoked,
            SensingEvent.Failed("x"),
        )
        var ignoredSeen = 0
        for (state in SensingState.entries.filterNot { it.requiresService }) {
            for (event in events) {
                val m = machine(state)
                if (m.offer(event) is Transition.Ignored) {
                    ignoredSeen++
                    assertFalse("$state + $event must not require the service", m.state.requiresService)
                }
            }
        }
        // Guard against the assertion never running: if a table change made every
        // event acceptable everywhere, the loop above would pass while checking nothing.
        assertTrue("no ignored event was exercised", ignoredSeen > 0)
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

    // -- the whole transition table, pinned explicitly ------------------------

    @Test
    fun `every state and event pair produces exactly the intended result`() {
        // The table is written out rather than derived, so a change to the production
        // `when` has to be reflected here deliberately. Two sweeps over structural
        // invariants used to stand in for this and pinned almost none of it: they
        // asserted that an ignored event does not move the state and that Accepted.from
        // matches, both of which hold for any table at all.
        //
        // null means "ignored".
        val I = SensingState.IDLE
        val ST = SensingState.STARTING
        val R = SensingState.RUNNING
        val SP = SensingState.STOPPING
        val REV = SensingState.STOPPED_PERMISSION_REVOKED
        val ERR = SensingState.STOPPED_ERROR

        val table: Map<Pair<SensingState, String>, SensingState?> = mapOf(
            (I to "Start") to ST,
            (I to "Started") to null,
            (I to "Stop") to null,
            (I to "Stopped") to null,
            (I to "PermissionRevoked") to null,
            (I to "Failed") to null,

            (ST to "Start") to null,
            (ST to "Started") to R,
            (ST to "Stop") to SP,
            (ST to "Stopped") to null,
            (ST to "PermissionRevoked") to REV,
            (ST to "Failed") to ERR,

            (R to "Start") to null,
            (R to "Started") to null,
            (R to "Stop") to SP,
            (R to "Stopped") to null,
            (R to "PermissionRevoked") to REV,
            (R to "Failed") to ERR,

            (SP to "Start") to null,
            (SP to "Started") to null,
            (SP to "Stop") to null,
            (SP to "Stopped") to I,
            (SP to "PermissionRevoked") to null,
            (SP to "Failed") to ERR,

            (REV to "Start") to ST,
            (REV to "Started") to null,
            (REV to "Stop") to I,
            (REV to "Stopped") to null,
            (REV to "PermissionRevoked") to null,
            (REV to "Failed") to null,

            (ERR to "Start") to ST,
            (ERR to "Started") to null,
            (ERR to "Stop") to I,
            (ERR to "Stopped") to null,
            (ERR to "PermissionRevoked") to null,
            (ERR to "Failed") to null,
        )

        val events = mapOf(
            "Start" to SensingEvent.Start,
            "Started" to SensingEvent.Started,
            "Stop" to SensingEvent.Stop,
            "Stopped" to SensingEvent.Stopped,
            "PermissionRevoked" to SensingEvent.PermissionRevoked,
            "Failed" to SensingEvent.Failed("probe"),
        )

        // The table must cover the whole cross product, or a missing row would quietly
        // exempt a pair from being checked at all.
        assertEquals(SensingState.entries.size * events.size, table.size)

        for (state in SensingState.entries) {
            for ((name, event) in events) {
                val expected = table.getValue(state to name)
                val m = machine(state)
                val transition = m.offer(event)
                if (expected == null) {
                    assertTrue("$state + $name should be ignored, got $transition", transition is Transition.Ignored)
                    assertEquals("$state + $name must not move", state, m.state)
                } else {
                    assertTrue("$state + $name should be accepted, got $transition", transition is Transition.Accepted)
                    assertEquals("$state + $name", expected, m.state)
                }
            }
        }
    }

    @Test
    fun `ignoredEvents counts every ignored event, not just the first`() {
        // Asserting only 0 or 1 left an `ignoredEvents = 1` mutant alive, so the
        // "counted so a duplicate stays visible" claim was not pinned as a count.
        val m = machine()
        repeat(5) { m.offer(SensingEvent.Stop) }
        assertEquals(5, m.ignoredEvents)

        m.offer(SensingEvent.Start)
        assertEquals(5, m.ignoredEvents)
        repeat(3) { m.offer(SensingEvent.Start) }
        assertEquals(8, m.ignoredEvents)
    }

    @Test
    fun `a revoke does not overwrite a recorded failure`() {
        // Under a `from != IDLE` guard a revoke arriving in STOPPED_ERROR replaced it,
        // discarding why the app actually stopped.
        val m = machine(SensingState.RUNNING)
        m.offer(SensingEvent.Failed("camera busy"))
        assertEquals(SensingState.STOPPED_ERROR, m.state)
        assertTrue(m.offer(SensingEvent.PermissionRevoked) is Transition.Ignored)
        assertEquals(SensingState.STOPPED_ERROR, m.state)
        assertEquals("camera busy", m.lastFailure)
    }
}
