package com.dsrc.phone.net

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * The pings the spec makes the phone responsible for, and nothing sent for a year.
 *
 * `Session.sendTimeSyncPing` existed and only tests called it, so a real drive
 * produced zero samples on the Jetson and every stamp took the arrival proxy.
 * Measured on a handset: `samples_accepted: 0` across 287 frames.
 */
class TimeSyncDriverTest {

    /** A holder that records what it was asked to send, without a socket. */
    private class Recording(private val accept: Boolean = true) {
        val exchanges = mutableListOf<Long>()
        fun send(id: Long): Boolean {
            synchronized(exchanges) { exchanges.add(id) }
            return accept
        }
        fun count() = synchronized(exchanges) { exchanges.size }
        fun ids() = synchronized(exchanges) { exchanges.toList() }
    }

    /**
     * Drives the loop deterministically: the sleeper records the period asked for
     * and releases the test, so cadence is asserted without waiting real seconds.
     */
    private fun driver(
        recording: Recording,
        clockNs: () -> Long,
        periods: MutableList<Long>,
        stopAfter: Int,
        done: CountDownLatch,
    ): TimeSyncDriver {
        return TimeSyncDriver(
            sendPing = recording::send,
            monoClock = clockNs,
            sleeper = { period ->
                periods.add(period)
                if (periods.size >= stopAfter) done.countDown()
            },
        )
    }

    @Test
    fun `the first ten seconds are sampled at four hertz, then one`() {
        // The spec's table, not a guess: the far side needs five samples before its
        // gate opens, so the fast phase is what makes the first advisory alignable
        // in seconds rather than in a minute.
        assertEquals(250L, TimeSyncDriver.FAST_PERIOD_MS)
        assertEquals(1_000L, TimeSyncDriver.STEADY_PERIOD_MS)
        assertEquals(10_000_000_000L, TimeSyncDriver.FAST_PHASE_NS)
    }

    @Test
    fun `pings are sent, numbered from one, and never repeat an exchange id`() {
        // A repeated id makes two exchanges indistinguishable to the far side,
        // which matches an answer to the wrong send and produces an offset built
        // from two unrelated instants.
        val recording = Recording()
        val periods = mutableListOf<Long>()
        val done = CountDownLatch(1)
        var now = 0L
        val d = driver(recording, { now += 250_000_000L; now }, periods, stopAfter = 6, done)

        d.start()
        assertTrue(done.await(5, TimeUnit.SECONDS))
        d.stop()

        val ids = recording.ids()
        assertTrue("no pings were sent at all", ids.size >= 5)
        assertEquals("ids must start at 1", 1L, ids.first())
        assertEquals("ids repeated", ids.size, ids.toSet().size)
        assertEquals("ids must be consecutive", (1L..ids.size).toList(), ids)
    }

    @Test
    fun `the cadence drops to the steady rate after the fast phase`() {
        val recording = Recording()
        val periods = mutableListOf<Long>()
        val done = CountDownLatch(1)
        // The first read is `startedNs`, so the reads after it are the iterations:
        // 1 s and 2 s are inside the fast phase, 20 s is past it.
        val stamps = listOf(0L, 1_000_000_000L, 2_000_000_000L, 20_000_000_000L)
        var i = 0
        val d = driver(recording, { stamps[minOf(i++, stamps.size - 1)] }, periods, stopAfter = 3, done)

        d.start()
        assertTrue(done.await(5, TimeUnit.SECONDS))
        d.stop()

        assertEquals(TimeSyncDriver.FAST_PERIOD_MS, periods[0])
        assertEquals("still fast inside the first ten seconds", TimeSyncDriver.FAST_PERIOD_MS, periods[1])
        assertEquals("should have dropped to the steady rate", TimeSyncDriver.STEADY_PERIOD_MS, periods[2])
    }

    @Test
    fun `a refused ping is counted rather than lost`() {
        // A drive that never synced has to be visible as a number. Silently
        // dropping refusals is how "the Jetson converted nothing" became invisible
        // for a whole device test.
        val recording = Recording(accept = false)
        val periods = mutableListOf<Long>()
        val done = CountDownLatch(1)
        var now = 0L
        val d = driver(recording, { now += 250_000_000L; now }, periods, stopAfter = 4, done)

        d.start()
        assertTrue(done.await(5, TimeUnit.SECONDS))
        d.stop()

        assertTrue("refusals were not counted", d.refused >= 3)
        assertEquals("a refused ping must not count as sent", 0L, d.sent)
    }

    @Test
    fun `starting twice does not add a second pinger`() {
        val recording = Recording()
        val periods = mutableListOf<Long>()
        val done = CountDownLatch(1)
        var now = 0L
        val d = driver(recording, { now += 250_000_000L; now }, periods, stopAfter = 4, done)

        d.start()
        d.start()
        assertTrue(done.await(5, TimeUnit.SECONDS))
        d.stop()

        // Two pingers on one session would double the rate and interleave two id
        // sequences, so the far side would see repeats from a counter that is not
        // itself synchronised.
        val ids = recording.ids()
        assertEquals("a second pinger duplicated ids", ids.size, ids.toSet().size)
    }

    @Test
    fun `stopping mid-sleep finishes quietly rather than throwing out of the thread`() {
        // `stop()` interrupts to cut the sleep short. Uncaught, the
        // InterruptedException escaped the worker and the instrumented teardown
        // test failed on it -- an exception leaving a thread is indistinguishable
        // from a real fault at that point. Driven with the real sleeper, because
        // the injected one in the other tests never blocks and so never interrupts.
        val recording = Recording()
        val escaped = mutableListOf<Throwable>()
        val d = TimeSyncDriver(sendPing = recording::send)

        d.start()
        Thread.getAllStackTraces().keys
            .firstOrNull { it.name == TimeSyncDriver.THREAD_NAME }
            ?.setUncaughtExceptionHandler { _, e -> synchronized(escaped) { escaped.add(e) } }
        Thread.sleep(60)
        d.stop()

        assertTrue("an exception escaped the driver thread: $escaped",
                   synchronized(escaped) { escaped.isEmpty() })
    }
}
