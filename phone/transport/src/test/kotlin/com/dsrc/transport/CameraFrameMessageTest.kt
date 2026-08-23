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
    fun `a zero dimension is refused`() {
        for (bad in listOf(message(width = 0), message(height = 0))) {
            val error = runCatching { CameraFrameMessage.fromWire(bad.toExtensions(), jpeg) }
                .exceptionOrNull()
            assertTrue("expected a MessageError, got $error", error is MessageError)
            assertEquals(RefusalReason.OUT_OF_RANGE, (error as MessageError).reason)
        }
    }

    @Test
    fun `a negative dimension is refused as a count, not silently accepted`() {
        val error = runCatching {
            CameraFrameMessage.fromWire(message(width = -1280).toExtensions(), jpeg)
        }.exceptionOrNull()
        // The exception *type* is asserted because runCatching cannot tell a named
        // refusal from an unrelated throw, which has hidden a defect here before.
        assertTrue("expected a MessageError, got $error", error is MessageError)
        assertEquals(RefusalReason.OUT_OF_RANGE, (error as MessageError).reason)
    }

    @Test
    fun `a quality outside 1 to 100 is refused`() {
        for (bad in listOf(0L, 101L, -5L)) {
            val error = runCatching {
                CameraFrameMessage.fromWire(message(quality = bad).toExtensions(), jpeg)
            }.exceptionOrNull()
            assertTrue("quality $bad was accepted", error is MessageError)
        }
    }

    @Test
    fun `an empty format is refused`() {
        val error = runCatching {
            CameraFrameMessage.fromWire(message(format = "").toExtensions(), jpeg)
        }.exceptionOrNull()
        assertTrue("expected a MessageError, got $error", error is MessageError)
        assertEquals(RefusalReason.UNKNOWN_VALUE, (error as MessageError).reason)
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
    fun `the camera channel is no longer exempt from the sender rule`() {
        // It was, and it was the one channel where an unchecked field would travel
        // thousands of times before anyone looked.
        assertTrue(Channels.CAMERA !in OutboundValidation.WITHOUT_A_TYPED_DECODER)

        val error = runCatching {
            OutboundValidation.check(Channels.CAMERA, message(width = 0).toExtensions(), jpeg)
        }.exceptionOrNull()
        assertTrue("an invalid frame passed outbound validation", error is MessageError)
    }

    @Test
    fun `a valid frame passes outbound validation with its payload`() {
        OutboundValidation.check(Channels.CAMERA, message().toExtensions(), jpeg)
    }

    @Test
    fun `an empty payload is accepted, because a lost image is worth seeing`() {
        // Refusing here would hide an encoder that produced nothing behind a transport
        // refusal, which is the wrong place to learn about it.
        val decoded = CameraFrameMessage.fromWire(message().toExtensions(), ByteArray(0))
        assertEquals(7, decoded.frameId)
    }
}
