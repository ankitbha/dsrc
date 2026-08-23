package com.dsrc.phone.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CapturedFrameTest {

    private fun frame(
        width: Int = 4,
        height: Int = 4,
        quality: Int? = 85,
        jpeg: ByteArray = byteArrayOf(1, 2, 3),
    ) = CapturedFrame(1, width, height, "jpeg", quality, 1_000, jpeg)

    @Test
    fun `a non-positive dimension is refused`() {
        // Nothing constructed an invalid frame, so both guards were unpinned.
        assertTrue(runCatching { frame(width = 0) }.isFailure)
        assertTrue(runCatching { frame(height = -1) }.isFailure)
    }

    @Test
    fun `quality outside 1 to 100 is refused`() {
        assertTrue(runCatching { frame(quality = 0) }.isFailure)
        assertTrue(runCatching { frame(quality = 101) }.isFailure)
    }

    @Test
    fun `a null quality is allowed, since the wire field is nullable`() {
        assertEquals(null, frame(quality = null).quality)
    }

    @Test
    fun `equality compares the jpeg bytes, not the array reference`() {
        // A data class with a ByteArray compares references by default, which makes two
        // identical frames unequal and silently weakens every test that compares frames.
        assertEquals(frame(jpeg = byteArrayOf(1, 2, 3)), frame(jpeg = byteArrayOf(1, 2, 3)))
        assertNotEquals(frame(jpeg = byteArrayOf(1, 2, 3)), frame(jpeg = byteArrayOf(9)))
    }

    @Test
    fun `hashCode agrees with equals on identical bytes`() {
        assertEquals(
            frame(jpeg = byteArrayOf(4, 5)).hashCode(),
            frame(jpeg = byteArrayOf(4, 5)).hashCode(),
        )
    }
}
