package com.dsrc.phone.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameBufferTest {

    private fun frame(id: Long) = CapturedFrame(
        frameId = id, width = 2, height = 2, format = "jpeg", quality = 85,
        captureMonoNs = id * 1_000_000, jpeg = byteArrayOf(id.toByte()),
    )

    @Test
    fun `an empty buffer drains to null rather than blocking`() {
        assertNull(FrameBuffer().drain())
    }

    @Test
    fun `a frame offered into an empty buffer displaces nothing`() {
        assertNull(FrameBuffer().offer(frame(1)))
    }

    @Test
    fun `drain returns what was offered`() {
        val buffer = FrameBuffer()
        buffer.offer(frame(7))
        assertEquals(frame(7), buffer.drain())
    }

    @Test
    fun `drain empties the buffer`() {
        val buffer = FrameBuffer()
        buffer.offer(frame(1))
        buffer.drain()
        assertNull(buffer.drain())
    }

    @Test
    fun `a newer frame replaces an unsent one and the old one is returned`() {
        val buffer = FrameBuffer()
        buffer.offer(frame(1))
        assertEquals(frame(1), buffer.offer(frame(2)))
        assertEquals("the newest frame is the one kept", frame(2), buffer.drain())
    }

    @Test
    fun `a replacement counts exactly one drop`() {
        val buffer = FrameBuffer()
        buffer.offer(frame(1))
        buffer.offer(frame(2))
        assertEquals(1, buffer.stats.dropped)
        buffer.offer(frame(3))
        assertEquals(2, buffer.stats.dropped)
    }

    @Test
    fun `a drained frame is not a dropped frame`() {
        val buffer = FrameBuffer()
        buffer.offer(frame(1))
        buffer.drain()
        buffer.offer(frame(2))
        assertEquals("draining then offering displaces nothing", 0, buffer.stats.dropped)
    }

    @Test
    fun `the counters balance over a lossy run`() {
        val buffer = FrameBuffer()
        repeat(100) { i ->
            buffer.offer(frame(i.toLong()))
            if (i % 3 == 0) buffer.drain()
        }
        val stats = buffer.stats
        assertEquals(100, stats.accepted)
        assertTrue("accepted must equal dropped + drained + held: $stats", stats.balances)
    }

    @Test
    fun `closing discards rather than drops, and the discard is counted`() {
        // A drop is a frame a newer one displaced -- the channel's ordinary loss. A
        // discard is a frame abandoned because sensing ended. Keeping them apart stops
        // shutdown inflating the in-flight loss; counting the discard at all is what
        // keeps the balance identity true, which an earlier version broke after every
        // stop, leaving those frames counted nowhere.
        val buffer = FrameBuffer()
        buffer.offer(frame(1))
        buffer.close()
        assertEquals("not a drop", 0, buffer.stats.dropped)
        assertEquals("but counted", 1, buffer.stats.discarded)
        assertTrue("the identity must hold after a stop: ${buffer.stats}", buffer.stats.balances)
        assertNull(buffer.drain())
    }

    @Test
    fun `closing an empty buffer discards nothing`() {
        val buffer = FrameBuffer()
        buffer.close()
        assertEquals(0, buffer.stats.discarded)
        assertTrue(buffer.stats.balances)
    }

    @Test
    fun `a frame offered after closing is refused and counted`() {
        // Compressing a 720p frame takes tens of milliseconds, so an encode that began
        // before a stop finishes after it. A late arrival must not sit in the buffer
        // waiting for a session that has ended.
        val buffer = FrameBuffer()
        buffer.close()
        assertNull(buffer.offer(frame(9)))
        assertNull("nothing may be held after closing", buffer.drain())
        val stats = buffer.stats
        assertEquals(1, stats.accepted)
        assertEquals(1, stats.discarded)
        assertEquals(0, stats.dropped)
        assertTrue("the identity must hold: $stats", stats.balances)
    }

    @Test
    fun `closing twice discards once`() {
        val buffer = FrameBuffer()
        buffer.offer(frame(1))
        buffer.close()
        buffer.close()
        assertEquals(1, buffer.stats.discarded)
    }

    @Test
    fun `counters never decrease`() {
        val buffer = FrameBuffer()
        var last = FrameBuffer.Stats(0, 0, 0, 0, false)
        repeat(50) { i ->
            buffer.offer(frame(i.toLong()))
            if (i % 2 == 0) buffer.drain()
            val now = buffer.stats
            assertTrue(now.accepted >= last.accepted)
            assertTrue(now.dropped >= last.dropped)
            assertTrue(now.drained >= last.drained)
            last = now
        }
    }

    @Test
    fun `concurrent offers and drains keep the counters balanced`() {
        // The buffer is written from the encoder thread and read by whatever drains it,
        // so the accounting has to hold under contention rather than only in sequence.
        val buffer = FrameBuffer()
        val producer = Thread { repeat(2_000) { buffer.offer(frame(it.toLong())) } }
        val consumer = Thread { repeat(2_000) { buffer.drain() } }
        producer.start(); consumer.start()
        producer.join(); consumer.join()
        val stats = buffer.stats
        assertEquals(2_000, stats.accepted)
        assertTrue("counters must balance under contention: $stats", stats.balances)
    }
}
