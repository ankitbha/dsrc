package com.dsrc.transport

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class QueuesTest {

    private fun queues() = OutboundQueues()

    private fun OutboundQueues.put(channel: String, marker: Int = 0) =
        enqueue(channel, mapOf("k" to JsonValue.Num(marker.toLong())), ByteArray(0), monoNs = 0, wallNs = 0)

    private fun Outbound.marker() = (extensions.getValue("k") as JsonValue.Num).value

    // -- sequence numbers ----------------------------------------------------

    @Test
    fun `sequence numbers start at zero and count per channel`() {
        val q = queues()
        assertEquals(0, q.put(Channels.GPS).sequence)
        assertEquals(1, q.put(Channels.GPS).sequence)
        assertEquals(0, q.put(Channels.IMU).sequence, "each channel counts separately")
    }

    @Test
    fun `a sequence number is assigned before the overflow decision`() {
        // This is what makes a gap the peer's evidence of a drop. Assigning at send time
        // would renumber the survivors and hide every drop.
        val q = queues()
        val depth = Channels.policy(Channels.GPS).depth
        val assigned = (0..depth).map { q.put(Channels.GPS, it).sequence }
        assertEquals((0L..depth.toLong()).toList(), assigned)

        val delivered = mutableListOf<Long>()
        while (true) delivered.add((q.poll() ?: break).sequence)
        assertEquals(depth, delivered.size, "the oldest was dropped")
        assertEquals(1L, delivered.first(), "so the peer sees a gap at 0")
    }

    @Test
    fun `the hello reserves control sequence zero so ordinary traffic starts at one`() {
        // A peer that restarted control at 0 would duplicate the hello's number, and the
        // gap rule cannot see it -- it only fires on a sequence greater than expected --
        // so the divergence would be silent and permanent.
        val q = queues()
        assertEquals(0, q.reserveHelloSequence())
        assertEquals(1, q.put(Channels.CONTROL).sequence)
    }

    @Test
    fun `reserving the hello does not disturb other channels`() {
        val q = queues()
        q.reserveHelloSequence()
        assertEquals(0, q.put(Channels.GPS).sequence)
    }

    // -- overflow ------------------------------------------------------------

    @Test
    fun `a reliable channel drops the oldest and counts it`() {
        val q = queues()
        val depth = Channels.policy(Channels.HERE).depth
        repeat(depth) { q.put(Channels.HERE, it) }
        assertEquals(0, q.counters().getValue(Channels.HERE).dropped)

        val result = q.put(Channels.HERE, depth)
        assertEquals(0L, result.displaced?.marker(), "the oldest was displaced")
        assertEquals(1, q.counters().getValue(Channels.HERE).dropped)
    }

    @Test
    fun `a reliable channel keeps the newest when it overflows`() {
        val q = queues()
        val depth = Channels.policy(Channels.HERE).depth
        repeat(depth + 3) { q.put(Channels.HERE, it) }
        val markers = mutableListOf<Long>()
        while (true) markers.add((q.poll() ?: break).marker())
        assertEquals(depth, markers.size)
        assertEquals((3L until (depth + 3).toLong()).toList(), markers, "the three oldest went")
    }

    @Test
    fun `a latest-wins channel holds one and replaces it`() {
        val q = queues()
        q.put(Channels.CAMERA, 1)
        val result = q.put(Channels.CAMERA, 2)
        assertEquals(1L, result.displaced?.marker())
        assertEquals(2L, q.poll()?.marker(), "the newest survives")
        assertNull(q.poll())
    }

    @Test
    fun `every latest-wins replacement is counted`() {
        val q = queues()
        repeat(5) { q.put(Channels.CAMERA, it) }
        assertEquals(4, q.counters().getValue(Channels.CAMERA).dropped)
        assertEquals(5, q.counters().getValue(Channels.CAMERA).enqueued)
    }

    @Test
    fun `nothing is dropped while a queue has room`() {
        val q = queues()
        repeat(Channels.policy(Channels.IMU).depth) { q.put(Channels.IMU, it) }
        assertEquals(0, q.counters().getValue(Channels.IMU).dropped)
    }

    // -- priority ------------------------------------------------------------

    @Test
    fun `high drains before normal drains before bulk`() {
        val q = queues()
        q.put(Channels.CAMERA)          // bulk
        q.put(Channels.GPS)             // normal
        q.put(Channels.CONTROL)         // high
        assertEquals(Channels.CONTROL, q.poll()?.channel)
        assertEquals(Channels.GPS, q.poll()?.channel)
        assertEquals(Channels.CAMERA, q.poll()?.channel)
    }

    @Test
    fun `a saturated normal tier does not starve high`() {
        val q = queues()
        repeat(64) { q.put(Channels.GPS, it) }
        q.put(Channels.ADVISORY)
        assertEquals(Channels.ADVISORY, q.poll()?.channel, "high goes first regardless of depth")
    }

    @Test
    fun `bulk can be starved, which is accepted`() {
        // The high and normal tiers here carry heartbeats, commands and small records and
        // cannot saturate a link that is carrying camera frames at all, so strict
        // priority is the right trade rather than a hazard.
        val q = queues()
        q.put(Channels.CAMERA)
        repeat(10) { q.put(Channels.GPS, it) }
        repeat(10) { assertEquals(Channels.GPS, q.poll()?.channel) }
        assertEquals(Channels.CAMERA, q.poll()?.channel)
    }

    @Test
    fun `channels at one priority take turns`() {
        // Otherwise the first channel in the table would drain completely before its
        // equals were served at all.
        val q = queues()
        repeat(3) { q.put(Channels.GPS, it) }
        repeat(3) { q.put(Channels.IMU, it) }
        repeat(3) { q.put(Channels.TELEMETRY, it) }
        val order = (1..9).map { q.poll()!!.channel }
        val firstThree = order.take(3).toSet()
        assertEquals(
            3,
            firstThree.size,
            "each normal channel should be served once before any is served twice",
        )
    }

    @Test
    fun `the round-robin cursor advances past the channel just served`() {
        val q = queues()
        repeat(2) { q.put(Channels.GPS, it) }
        repeat(2) { q.put(Channels.IMU, it) }
        val first = q.poll()!!.channel
        val second = q.poll()!!.channel
        assertTrue(first != second, "served $first twice in a row")
    }

    @Test
    fun `an empty tier is skipped rather than blocking a lower one`() {
        val q = queues()
        q.put(Channels.CAMERA)
        assertEquals(Channels.CAMERA, q.poll()?.channel)
    }

    // -- accounting ----------------------------------------------------------

    @Test
    fun `polling an empty set returns null`() {
        assertNull(queues().poll())
    }

    @Test
    fun `the counters balance for every channel`() {
        val q = queues()
        repeat(100) { q.put(Channels.GPS, it) }
        repeat(50) { q.put(Channels.CAMERA, it) }
        repeat(30) { q.poll() }
        for ((channel, counters) in q.counters()) {
            assertTrue(
                counters.enqueued == counters.dropped + counters.sent + counters.pending,
                "$channel: $counters",
            )
        }
    }

    @Test
    fun `pending counts what the writer has not taken`() {
        val q = queues()
        repeat(5) { q.put(Channels.GPS, it) }
        assertEquals(5, q.pending())
        q.poll()
        assertEquals(4, q.pending())
    }

    @Test
    fun `an unknown channel is refused rather than queued somewhere`() {
        val error = runCatching { queues().put("nonsense") }.exceptionOrNull()
        assertTrue(error is FramingError, "expected FramingError, got $error")
    }

    @Test
    fun `concurrent enqueues assign distinct sequence numbers`() {
        // The sensors run on their own threads; two of them sharing a sequence number
        // would make the peer's gap arithmetic meaningless.
        //
        // This test used to assert only that 1,000 messages were counted, which is
        // exactly what its name does not say: pinning every IMU message to sequence 0
        // left it green. The sequences themselves are collected now.
        val q = queues()
        val assigned = java.util.concurrent.ConcurrentLinkedQueue<Long>()
        val threads = (1..4).map {
            Thread {
                repeat(250) {
                    assigned.add(q.enqueue(Channels.IMU, emptyMap(), ByteArray(0), 0, 0).sequence)
                }
            }
        }
        threads.forEach { it.start() }
        threads.forEach { it.join() }

        assertEquals(1000, assigned.size)
        assertEquals(1000, assigned.toSet().size, "a sequence number was issued twice")
        // Contiguous from zero, not merely distinct: a gap is the peer's evidence that
        // the sender dropped something, so an unexplained one here would be a lie.
        assertEquals((0L until 1000L).toSet(), assigned.toSet())

        val counters = q.counters().getValue(Channels.IMU)
        assertEquals(1000, counters.enqueued)
        // Checked against the real queue depth. Comparing it against the other counters
        // is a tautology, because `pending` is computed from them.
        assertEquals(q.depth(Channels.IMU), counters.pending, "derived pending disagrees with the queue")
    }

    @Test
    fun `a drawn sequence number leaves the derived backlog equal to the real one`() {
        // The two paths that draw a sequence without queueing -- the hello and every
        // keepalive -- must move `sent` as well as `enqueued`, or the derived backlog
        // grows by one per keepalive: about 600 after a ten-minute drive, on a channel
        // whose queue is empty. Both mutations survived the whole suite before this.
        val q = queues()
        q.reserveHelloSequence()
        assertEquals(q.depth(Channels.CONTROL), q.counters().getValue(Channels.CONTROL).pending)

        repeat(50) { q.nextSequenceFor(Channels.CONTROL) }
        assertEquals(q.depth(Channels.CONTROL), q.counters().getValue(Channels.CONTROL).pending)
        assertEquals(0, q.counters().getValue(Channels.CONTROL).pending)

        // And with real traffic mixed in, so the identity is not just 0 == 0.
        q.put(Channels.CONTROL)
        q.put(Channels.CONTROL)
        q.nextSequenceFor(Channels.CONTROL)
        assertEquals(2, q.depth(Channels.CONTROL))
        assertEquals(q.depth(Channels.CONTROL), q.counters().getValue(Channels.CONTROL).pending)
    }
}
