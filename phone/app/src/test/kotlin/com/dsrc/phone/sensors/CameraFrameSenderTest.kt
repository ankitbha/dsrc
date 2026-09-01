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
        val wantsWireStamp: Boolean,
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
        send = { channel, extensions, payload, wantsWireStamp ->
            sink.add(Sent(channel, extensions, payload, wantsWireStamp))
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
    fun `every dispatch asks for a wire stamp`() {
        // The busiest channel, and the one task 33's transport stage needs: the network
        // hop cannot be measured at all without the departure stamp this asks for.
        val (sink, sender) = sender()
        sender.dispatch(frame(1))

        assertTrue(sink.single().wantsWireStamp)
    }

    @Test
    fun `the encode timestamps ride the wire when the frame carries them`() {
        val (sink, sender) = sender()
        val withEncodeStamps = CapturedFrame(
            frameId = 9, width = 1280, height = 720, format = "jpeg", quality = 85,
            captureMonoNs = 5_009, jpeg = byteArrayOf(1),
            encodeStartMonoNs = 5_020, encodeDoneMonoNs = 5_035,
        )
        sender.dispatch(withEncodeStamps)

        val decoded = CameraFrameMessage.fromWire(sink.single().extensions, sink.single().payload)
        assertEquals(5_020L, decoded.encodeStartMonoNs)
        assertEquals(5_035L, decoded.encodeDoneMonoNs)
    }

    @Test
    fun `a frame with no encode timestamps omits them rather than sending null`() {
        val (sink, sender) = sender()
        sender.dispatch(frame(2))

        val extensions = sink.single().extensions
        assertFalse(extensions.containsKey(CameraFrameMessage.KEY_ENCODE_START))
        assertFalse(extensions.containsKey(CameraFrameMessage.KEY_ENCODE_DONE))
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
            send = { _, _, _, _ -> sleepsBeforeEachSend.add(sleeps.get()); true },
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
            send = { _, _, _, _ -> true },
            pollMs = 1,
            sleeper = { sleeps.incrementAndGet(); Thread.sleep(1) },
        )
        sender.start()
        while (sleeps.get() < 2) Thread.sleep(1)

        // Read at the moment of the stop, not 50 ms later, and that is the finding. Taking
        // the baseline after a sleep hides every sleep that happened inside the window --
        // so deleting the loop's own `if (stopped.get()) break` left the test green, because
        // the one extra poll it allows lands in exactly that gap. The counter is not the
        // subject; what happens between `stop()` returning and the loop noticing is.
        val atStop = sleeps.get()
        sender.stop()
        Thread.sleep(100)

        // At most one more: the poll already in flight when stop() was called may finish.
        // A second means the loop went round again, which is what the flag is for. The
        // upper bound is the assertion -- equality would be a race, since the in-flight
        // poll may or may not have started.
        val extra = sleeps.get() - atStop
        assertTrue(
            "the loop went round $extra more times after stop; at most one poll may be in flight",
            extra <= 1,
        )
    }

    private fun assertArrayEquals(expected: ByteArray, actual: ByteArray) {
        org.junit.Assert.assertArrayEquals(expected, actual)
    }

    @Test
    fun `stop does not wait out a poll interval`() {
        // The flag and the interrupt are individually redundant -- deleting either leaves
        // the loop terminating -- so neither was pinned. They are not equivalent though:
        // without the interrupt, stop() waits for the current sleep to finish, and with a
        // long poll interval that is the difference between prompt teardown and a stall.
        // Pinned as promptness rather than as either line.
        val interrupted = java.util.concurrent.CountDownLatch(1)
        val sleeping = java.util.concurrent.CountDownLatch(1)
        val sender = CameraFrameSender(
            drain = { null },
            send = { _, _, _, _ -> true },
            pollMs = 30_000,
            sleeper = { millis ->
                sleeping.countDown()
                try {
                    Thread.sleep(millis)
                } catch (e: InterruptedException) {
                    interrupted.countDown()
                    throw e
                }
            },
        )
        sender.start()
        assertTrue("never reached the sleep", sleeping.await(5, java.util.concurrent.TimeUnit.SECONDS))

        val started = System.nanoTime()
        sender.stop()
        assertTrue(
            "stop() left the loop waiting out a 30 s poll",
            interrupted.await(5, java.util.concurrent.TimeUnit.SECONDS),
        )
        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        assertTrue("stop took $elapsedMs ms", elapsedMs < 2_000)
    }


    @Test
    fun `starting twice is refused`() {
        // Item 10 claimed both double-start guards were pinned; only SessionHolder's was.
        // A second start here spawns a second dsrc-camera-send thread and overwrites the
        // field, so stop() interrupts only the newer one and the first polls the buffer
        // forever -- stealing frames from the live sender.
        val sender = CameraFrameSender(drain = { null }, send = { _, _, _, _ -> true }, pollMs = 5)
        sender.start()
        try {
            val second = runCatching { sender.start() }
            assertTrue("a second start was allowed", second.isFailure)
            assertTrue(
                "wrong failure: ${second.exceptionOrNull()}",
                second.exceptionOrNull() is IllegalStateException,
            )
        } finally {
            sender.stop()
        }
    }

}
