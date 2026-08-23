package com.dsrc.phone.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RateGateTest {

    private val ms = 1_000_000L

    @Test
    fun `the first frame is always accepted`() {
        assertTrue(RateGate(10.0).accept(0L))
        assertTrue(RateGate(10.0).accept(123_456_789L))
    }

    @Test
    fun `a frame inside the period is rejected`() {
        val gate = RateGate(10.0)
        assertTrue(gate.accept(0))
        assertFalse(gate.accept(50 * ms))
        assertFalse(gate.accept(99 * ms))
    }

    @Test
    fun `a frame at the period boundary is accepted`() {
        val gate = RateGate(10.0)
        assertTrue(gate.accept(0))
        assertTrue(gate.accept(100 * ms))
    }

    @Test
    fun `a source that is an exact multiple hits the target rate exactly`() {
        // 30 Hz source, 10 Hz target: every third frame, no drift over a minute.
        val gate = RateGate(10.0)
        var accepted = 0
        val sourcePeriod = 1_000_000_000L / 30
        for (i in 0 until 1800) {
            if (gate.accept(i * sourcePeriod)) accepted++
        }
        assertEquals("60 s at 10 Hz", 600, accepted)
    }

    @Test
    fun `a source that is not a multiple stays within one frame of the target`() {
        // The undershoot case: scheduling each slot from *now* rather than from the
        // previous slot gives 7.5 Hz here instead of 10.
        val gate = RateGate(10.0)
        var accepted = 0
        val sourcePeriod = 1_000_000_000L / 24
        val seconds = 10
        for (i in 0 until 24 * seconds) {
            if (gate.accept(i * sourcePeriod)) accepted++
        }
        assertTrue("expected ~${10 * seconds}, got $accepted", accepted in (10 * seconds - 1)..(10 * seconds + 1))
    }

    @Test
    fun `a stall does not produce a catch-up burst`() {
        // The deficit case, and the one that matters on a throttling phone: a gate that
        // advances slots unconditionally owes 50 frames after a 5 s stall at 10 Hz and
        // pays them back as fast as the camera can deliver.
        val gate = RateGate(10.0)
        assertTrue(gate.accept(0))

        val afterStall = 5_000 * ms
        assertTrue("the first frame after a stall is due", gate.accept(afterStall))

        assertFalse("no burst", gate.accept(afterStall + 1 * ms))
        assertFalse("no burst", gate.accept(afterStall + 50 * ms))
        assertFalse("no burst", gate.accept(afterStall + 99 * ms))
        assertTrue("back to a normal cadence", gate.accept(afterStall + 100 * ms))
    }

    @Test
    fun `a stall costs exactly one period, not the whole backlog`() {
        val gate = RateGate(20.0)
        assertTrue(gate.accept(0))
        val stalled = 3_000 * ms
        assertTrue(gate.accept(stalled))
        var accepted = 0
        // Feed one hour of frames at 100 Hz; a burst would show as far more than the
        // 20 Hz the gate was asked for.
        for (i in 1..1000) {
            if (gate.accept(stalled + i * 10 * ms)) accepted++
        }
        assertTrue("expected ~200 over 10 s at 20 Hz, got $accepted", accepted in 199..201)
    }

    @Test
    fun `a rate change takes effect on the next decision`() {
        val gate = RateGate(1.0)
        assertTrue(gate.accept(0))
        assertFalse(gate.accept(100 * ms))
        gate.setRate(10.0)
        assertTrue("the new period is 100 ms", gate.accept(101 * ms))
        assertEquals(10.0, gate.hz, 0.0)
    }

    @Test
    fun `a rate change does not re-arm the gate`() {
        // Otherwise repeated commands would drive the camera faster than any rate
        // actually asked for.
        val gate = RateGate(1.0)
        assertTrue(gate.accept(0))
        repeat(50) { gate.setRate(1.0) }
        assertFalse("still inside the period", gate.accept(500 * ms))
    }

    @Test
    fun `a large rate increase takes effect without waiting out the old period`() {
        // 0.2 Hz to 10 Hz: leaving the old slot alone would stall for the remaining
        // five seconds, which is indistinguishable from ignoring the command.
        val gate = RateGate(0.2)
        assertTrue(gate.accept(0))
        gate.setRate(10.0)
        assertFalse("still not earlier than the new period", gate.accept(99 * ms))
        assertTrue("one new period after the last frame", gate.accept(100 * ms))
    }

    @Test
    fun `a rate decrease is honoured immediately`() {
        val gate = RateGate(10.0)
        assertTrue(gate.accept(0))
        gate.setRate(1.0)
        assertFalse(gate.accept(100 * ms))
        assertFalse(gate.accept(999 * ms))
        assertTrue(gate.accept(1_000 * ms))
    }

    @Test
    fun `a rate change never emits sooner than the new rate allows`() {
        // The property that makes re-anchoring safe, swept over pairs of rates.
        for (from in listOf(0.5, 1.0, 5.0, 30.0)) {
            for (to in listOf(0.5, 1.0, 5.0, 30.0)) {
                val gate = RateGate(from)
                assertTrue(gate.accept(0))
                gate.setRate(to)
                val newPeriod = RateGate.periodNsFor(to)
                assertFalse(
                    "$from -> $to emitted early",
                    gate.accept(newPeriod - 1),
                )
                assertTrue("$from -> $to never emitted", gate.accept(newPeriod))
            }
        }
    }

    @Test
    fun `the period matches the rate`() {
        assertEquals(1_000_000_000L, RateGate.periodNsFor(1.0))
        assertEquals(100_000_000L, RateGate.periodNsFor(10.0))
        assertEquals(20_000_000L, RateGate.periodNsFor(50.0))
        assertEquals(1_000_000L, RateGate.periodNsFor(1000.0))
    }

    @Test
    fun `the period never rounds down to zero`() {
        // A zero period would accept every frame regardless of rate.
        assertTrue(RateGate.periodNsFor(1000.0) >= 1)
        assertTrue(RateGate.periodNsFor(999.9) >= 1)
    }

    @Test
    fun `a rate outside the wire's range is refused`() {
        for (bad in listOf(0.0, -1.0, 1000.1, Double.NaN, Double.POSITIVE_INFINITY)) {
            val threw = runCatching { RateGate.periodNsFor(bad) }.isFailure
            assertTrue("rate $bad should be refused", threw)
        }
    }

    @Test
    fun `the wire's boundary rates are accepted`() {
        RateGate.periodNsFor(1000.0)
        RateGate.periodNsFor(0.001)
    }

    @Test
    fun `a non-monotonic timestamp does not unlock the gate`() {
        // A clock that goes backwards must not be read as a period having elapsed.
        val gate = RateGate(10.0)
        assertTrue(gate.accept(1_000 * ms))
        assertFalse(gate.accept(500 * ms))
        assertFalse(gate.accept(0))
    }
}
