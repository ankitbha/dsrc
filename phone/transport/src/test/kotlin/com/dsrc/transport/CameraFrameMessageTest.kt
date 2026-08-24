package com.dsrc.transport

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The camera message, and the sender rule now reaching the channel that carries most of
 * the traffic.
 */
class CameraFrameMessageTest {

    private fun message(
        captureMonoNs: Long = 5_000,
        frameId: Long = 7,
        width: Long = 1280,
        height: Long = 720,
        format: String = "jpeg",
        quality: Long? = 85,
    ) = CameraFrameMessage(captureMonoNs, frameId, width, height, format, quality)

    private val jpeg = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0x00, 0xFF.toByte(), 0xD9.toByte())

    @Test
    fun `a round trip preserves every field`() {
        val original = message()
        val decoded = CameraFrameMessage.fromWire(original.toExtensions(), jpeg)
        assertEquals(original, decoded)
    }

    @Test
    fun `the key names match the Python message exactly`() {
        // Read off deployment/jetson/transport/messages.py. A cross-language contract
        // with a renamed key fails as a missing field on the far side, at run time, on
        // the busiest channel.
        assertEquals(
            setOf("t_capture_mono_ns", "frame_id", "width", "height", "format", "quality"),
            message().toExtensions().keys,
        )
    }

    @Test
    fun `a format with no quality setting carries null`() {
        val decoded = CameraFrameMessage.fromWire(message(quality = null).toExtensions(), jpeg)
        assertNull(decoded.quality)
    }

    @Test
    fun `a missing field is refused by name`() {
        for (key in listOf("frame_id", "width", "height", "format", "quality", "t_capture_mono_ns")) {
            val stripped = message().toExtensions() - key
            val error = runCatching { CameraFrameMessage.fromWire(stripped, jpeg) }
                .exceptionOrNull()
            assertTrue("$key was optional", error is MessageError)
            assertEquals(
                "$key gave the wrong reason",
                RefusalReason.MISSING_FIELD,
                (error as MessageError).reason,
            )
        }
    }

    @Test
    fun `the four rules this decoder used to invent are gone`() {
        // A zero dimension, a quality outside 1..100, an empty format and a negative
        // frame_id were all refused here and are all accepted by Python, with nothing in
        // the spec's message or refusal tables to justify either answer. A unilateral
        // receiver rule refuses what the peer legitimately sends, and two implementations
        // disagreeing about whether a record is acceptable is worse than either answer.
        // DifferentialTest holds the reconciled verdicts; this states the intent locally.
        //
        // A bad *setting* still dies where settings enter: SensingConfig refuses a quality
        // outside 1..100 and a zero or odd dimension on construction.
        CameraFrameMessage.fromWire(message(width = 0).toExtensions(), jpeg)
        CameraFrameMessage.fromWire(message(quality = 0).toExtensions(), jpeg)
        CameraFrameMessage.fromWire(message(quality = 101).toExtensions(), jpeg)
        CameraFrameMessage.fromWire(message(format = "").toExtensions(), jpeg)
        CameraFrameMessage.fromWire(message(frameId = -1).toExtensions(), jpeg)
    }

    @Test
    fun `the camera channel is no longer exempt from the sender rule`() {
        // It was, and it was the one channel where an unchecked field would travel
        // thousands of times before anyone looked.
        assertTrue(MessageValidation.ALL_CHANNELS_HAVE_A_DECODER)

        // A missing field, since the range rules this decoder used to invent are gone.
        val error = runCatching {
            MessageValidation.check(Channels.CAMERA, message().toExtensions() - "frame_id", jpeg)
        }.exceptionOrNull()
        assertTrue("an invalid frame passed outbound validation", error is MessageError)
        assertEquals(RefusalReason.MISSING_FIELD, (error as MessageError).reason)
    }

    @Test
    fun `a valid frame passes outbound validation with its payload`() {
        MessageValidation.check(Channels.CAMERA, message().toExtensions(), jpeg)
    }

    @Test
    fun `an empty payload is accepted, because a lost image is worth seeing`() {
        // Refusing here would hide an encoder that produced nothing behind a transport
        // refusal, which is the wrong place to learn about it.
        val decoded = CameraFrameMessage.fromWire(message().toExtensions(), ByteArray(0))
        assertEquals(7, decoded.frameId)
    }
}
