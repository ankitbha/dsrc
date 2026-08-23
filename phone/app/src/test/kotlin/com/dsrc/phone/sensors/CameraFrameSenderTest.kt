package com.dsrc.phone.sensors

import com.dsrc.transport.CameraFrameMessage
import com.dsrc.transport.Channels
import com.dsrc.transport.JsonValue
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicLong

class CameraFrameSenderTest {

    private class Sent(
        val channel: String,
        val extensions: Map<String, JsonValue>,
        val payload: ByteArray,
    )

    private fun frame(id: Long, jpeg: ByteArray = byteArrayOf(1, 2, 3)) = CapturedFrame(
        frameId = id,
        width = 1280,
        height = 720,
        format = "jpeg",
        quality = 85,
        captureMonoNs = 5_000 + id,
        jpeg = jpeg,
    )

    private fun sender(
        accept: Boolean = true,
        sink: ConcurrentLinkedQueue<Sent> = ConcurrentLinkedQueue(),
    ) = sink to CameraFrameSender(
        drain = { null },
        send = { channel, extensions, payload ->
            sink.add(Sent(channel, extensions, payload))
            accept
        },
    )

    @Test
    fun `a dispatched frame carries every field the wire contract names`() {
        val (sink, sender) = sender()
        assertTrue(sender.dispatch(frame(7)))

        val sent = sink.single()
        assertEquals(Channels.CAMERA, sent.channel)
        assertArrayEquals(byteArrayOf(1, 2, 3), sent.payload)

        // Read back through the decoder rather than asserting on the map, so the test
        // fails if a field is spelled differently from what a receiver reads. Comparing
        // key by key against a literal map would agree with a typo shared by both.
        val decoded = CameraFrameMessage.fromWire(sent.extensions, sent.payload)
        assertEquals(7, decoded.frameId)
        assertEquals(1280, decoded.width)
        assertEquals(720, decoded.height)
        assertEquals("jpeg", decoded.format)
        assertEquals(85L, decoded.quality)
        assertEquals(5_007, decoded.captureMonoNs)
    }

    @Test
    fun `a refused frame is counted apart from a sent one`() {
        val (_, sender) = sender(accept = false)
        assertFalse(sender.dispatch(frame(1)))
        assertTrue(sender.dispatch(frame(2)).not())

        val stats = sender.stats
        assertEquals(2, stats.drained)
        assertEquals(0, stats.sent)
        assertEquals(2, stats.refused)
        assertTrue(stats.balances)
    }

    @Test
    fun `the balance identity is a real function of three counted fields`() {
        val (_, sender) = sender()
        sender.dispatch(frame(1))
        assertTrue(sender.stats.balances)
        // Not a tautology: a stats value whose terms disagree must report false, which is
        // what task 18's derived `gated` term could not do.
        assertFalse(CameraFrameSender.Stats(drained = 5, sent = 1, refused = 1).balances)
        assertTrue(CameraFrameSender.Stats(drained = 2, sent = 1, refused = 1).balances)
    }

    @Test
    fun `the loop drains everything available before it sleeps`() {
        // Three frames waiting, then nothing. A sender that slept after each one would
        // cap the send rate at 1/pollMs whatever rate was commanded -- at 50 Hz, legal on
        // the wire, that more than halves it.
        val waiting = ConcurrentLinkedQueue(listOf(frame(1), frame(2), frame(3)))
        val sleeps = AtomicLong(0)
        val idled = java.util.concurrent.CountDownLatch(1)

        // Each send records how many sleeps had happened *before* it. Counting sleeps at
        // the end instead measured a race: the loop keeps spinning while the test reads
        // the counter, so the number depended on scheduling and said nothing about order.
        val sleepsBeforeEachSend = ConcurrentLinkedQueue<Long>()

        val sender = CameraFrameSender(
            drain = { waiting.poll() },
            send = { _, _, _ -> sleepsBeforeEachSend.add(sleeps.get()); true },
            pollMs = 1,
            sleeper = {
                sleeps.incrementAndGet()
                idled.countDown()
                // Blocks, so an idle loop is not a spin and the counts stay legible.
                Thread.sleep(20)
            },
        )
        sender.start()

        assertTrue("never went idle", idled.await(5, java.util.concurrent.TimeUnit.SECONDS))
        sender.stop()

        assertEquals("all three moved", listOf(3), listOf(sleepsBeforeEachSend.size))
        assertEquals(
            "every frame went out before any sleep",
            listOf(0L, 0L, 0L),
            sleepsBeforeEachSend.toList(),
        )
        assertEquals(3, sender.stats.sent)
        assertTrue(sender.stats.balances)
    }

    @Test
    fun `stop ends the loop`() {
        val sleeps = AtomicLong(0)
        val sender = CameraFrameSender(
            drain = { null },
            send = { _, _, _ -> true },
            pollMs = 1,
            sleeper = { sleeps.incrementAndGet(); Thread.sleep(1) },
        )
        sender.start()
        while (sleeps.get() < 2) Thread.sleep(1)
        sender.stop()
        Thread.sleep(50)
        val after = sleeps.get()
        Thread.sleep(50)
        assertEquals("the loop kept polling after stop", after, sleeps.get())
    }

    private fun assertArrayEquals(expected: ByteArray, actual: ByteArray) {
        org.junit.Assert.assertArrayEquals(expected, actual)
    }
}
