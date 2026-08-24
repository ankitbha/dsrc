package com.dsrc.phone.ui

import com.dsrc.transport.AdvisoryMessage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AdvisoryHolderTest {

    private val second = 1_000_000_000L

    private fun advisory(recSpeed: Double = 13.4) = AdvisoryMessage(
        captureMonoNs = 1,
        recSpeedMps = recSpeed,
        recSpeedDisplay = 30.0,
        currentSpeedDisplay = 28.0,
        units = "mph",
        headwayTargetS = 2.0,
        laneText = "keep",
        mergeText = "",
        trafficText = "moderate",
        confidence = 0.87,
        confidenceLabel = "high",
        action = mapOf(
            "lane_preference" to "keep",
            "merge_mode" to "normal",
            "desired_speed_bin" to "nominal",
            "desired_headway_bin" to "normal",
        ),
    )

    private fun running(maxAgeNs: Long = AdvisoryHolder.MAX_AGE_NS) =
        AdvisoryHolder(maxAgeNs).also { it.start() }

    @Test
    fun `an advisory arriving after the session stopped is refused`() {
        // Stopping does not join the delivery thread, so a frame already off the inbound
        // queue is delivered to completion -- and a thread descheduled between the dequeue
        // and this call resumes after teardown has cleared the holder and refills it. The
        // driver has pressed Stop and is looking at a full recommendation, for up to the
        // whole expiry. Clearing cannot fix that on its own: the clear has already
        // happened, so the holder has to refuse.
        val holder = running()
        holder.clear()
        holder.accept(advisory(), nowNs = 0)

        assertNull("a stopped session put an advisory on the display", holder.current(nowNs = 0))
        assertEquals(1, holder.stats.afterStop)
        assertEquals("and it is not counted as one that was shown", 0, holder.stats.received)
    }

    @Test
    fun `starting again takes advisories once more`() {
        // Refusing after a stop must not refuse for ever -- a second drive in the same
        // process would show nothing at all.
        val holder = running()
        holder.clear()
        holder.accept(advisory(), nowNs = 0)
        holder.start()
        holder.accept(advisory(recSpeed = 22.0), nowNs = second)

        assertEquals(22.0, holder.current(nowNs = second)!!.recSpeedMps, 1e-9)
    }

    @Test
    fun `a steady stream never blinks off`() {
        // The arrival stamp is refreshed on every advisory, not only when the panel was
        // dark. Taken only on the first one -- "the timer starts when the panel lights up"
        // -- a healthy 4 Hz stream would blank every three seconds while advisories were
        // arriving the whole time.
        val holder = running()
        var now = 0L
        repeat(40) {
            holder.accept(advisory(), nowNs = now)
            now += second / 4
            assertNotNull("the panel went dark at ${now / 1_000_000} ms", holder.current(now))
        }
    }

    @Test
    fun `an advisory is shown while it is current`() {
        val holder = running()
        holder.accept(advisory(), nowNs = 0)

        assertNotNull(holder.current(nowNs = second))
        assertTrue(holder.stats.showing)
    }

    @Test
    fun `an advisory older than the limit is not shown`() {
        // The whole point. The transport keeps only the newest -- `advisory` is latest_wins
        // at depth one and the spec's reason is "a stale advisory is useless" -- but that
        // governs the queue. Once the Jetson stops sending, or the link drops, nothing
        // arrives to displace what is on the screen, and a recommendation about road the
        // driver has already covered is worse than a blank panel because it looks current.
        val holder = running()
        holder.accept(advisory(), nowNs = 0)

        assertNull(holder.current(nowNs = AdvisoryHolder.MAX_AGE_NS + 1))
        assertEquals(1, holder.stats.expired)
    }

    @Test
    fun `the boundary is landed on, not stepped over`() {
        // Exactly at the limit is still current: "older than" is strict, and an advisory
        // that arrived exactly three seconds ago has not yet aged past three seconds.
        val holder = running()
        holder.accept(advisory(), nowNs = 0)
        assertNotNull(holder.current(nowNs = AdvisoryHolder.MAX_AGE_NS))
        assertNull(holder.current(nowNs = AdvisoryHolder.MAX_AGE_NS + 1))
    }

    @Test
    fun `the limit is the value it is documented as`() {
        // Asserted against a literal, not against the constant. A test that derives its
        // inputs from the value under test moves with it -- which is how a sibling constant
        // in this repo went unpinned through two validation rounds.
        assertEquals(3_000_000_000L, AdvisoryHolder.MAX_AGE_NS)
    }

    @Test
    fun `a newer advisory replaces one that is still current`() {
        // latest_wins, on this side too. Keeping the older one because it had not expired
        // would show a recommendation the Jetson has already superseded.
        val holder = running()
        holder.accept(advisory(recSpeed = 10.0), nowNs = 0)
        holder.accept(advisory(recSpeed = 20.0), nowNs = second)

        assertEquals(20.0, holder.current(nowNs = second)!!.recSpeedMps, 1e-9)
        assertEquals(2, holder.stats.received)
    }

    @Test
    fun `a fresh advisory after an expiry is shown again`() {
        // The expiry clears what it holds, so the next arrival has to be able to refill it.
        // Clearing in a way that also blocked later arrivals would leave the panel dark for
        // the rest of the drive after one gap in the stream.
        val holder = running()
        holder.accept(advisory(), nowNs = 0)
        assertNull(holder.current(nowNs = 10 * second))

        holder.accept(advisory(recSpeed = 22.0), nowNs = 10 * second)
        assertEquals(22.0, holder.current(nowNs = 10 * second)!!.recSpeedMps, 1e-9)
    }

    @Test
    fun `the age is measured from arrival, not from the sender's capture stamp`() {
        // `t_capture_mono_ns` is the Jetson's clock, and relating it to ours takes the
        // timebase exchange. A panel that went blank because a clock estimate wandered
        // would be a fault invented by its own safety check. Two advisories with identical
        // capture stamps, arriving a long way apart, must age from when they arrived.
        val holder = running()
        holder.accept(advisory(), nowNs = 0)
        assertNull(holder.current(nowNs = 10 * second))

        holder.accept(advisory(), nowNs = 100 * second)
        assertNotNull(
            "the same capture stamp, freshly arrived, is current",
            holder.current(nowNs = 100 * second),
        )
    }

    @Test
    fun `clearing takes the advisory off the screen`() {
        // A driver who stopped the session is not being advised.
        val holder = running()
        holder.accept(advisory(), nowNs = 0)
        holder.clear()

        assertNull(holder.current(nowNs = 0))
        assertTrue(!holder.stats.showing)
    }

    @Test
    fun `expiry is counted once, not on every look`() {
        val holder = running()
        holder.accept(advisory(), nowNs = 0)
        repeat(5) { holder.current(nowNs = 10 * second) }

        assertEquals(1, holder.stats.expired)
    }
}
