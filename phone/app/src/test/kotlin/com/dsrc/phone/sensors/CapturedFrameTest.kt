package com.dsrc.phone.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CapturedFrameTest {

    private fun frame(
        frameId: Long = 1,
        width: Int = 4,
        height: Int = 4,
        quality: Int? = 85,
        captureMonoNs: Long = 1_000,
        jpeg: ByteArray = byteArrayOf(1, 2, 3),
    ) = CapturedFrame(frameId, width, height, "jpeg", quality, captureMonoNs, jpeg)

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

    @Test
    fun `hashCode distinguishes frames that differ`() {
        // The agreement test above compares two hashCodes of *equal* frames, which
        // `hashCode() = 0` satisfies perfectly -- and nothing else pinned the method, so a
        // constant hash was a passing implementation. A constant hash is not a correctness
        // bug on its own; it turns any map or set of frames into a linked list, which for a
        // buffer of encoded frames is the difference between a lookup and a scan.
        //
        // Distinctness is not a contract hashCode owes in general, so this asserts it only
        // where the fields genuinely differ and a collision would mean the field is not
        // being read at all.
        val hashes = listOf(
            frame(jpeg = byteArrayOf(4, 5)),
            frame(jpeg = byteArrayOf(4, 6)),
            frame(frameId = 99),
            frame(width = 640),
            frame(height = 480),
            frame(quality = 50),
            frame(captureMonoNs = 7),
        ).map { it.hashCode() }
        assertEquals(
            "every field must reach the hash: ${hashes.size - hashes.toSet().size} collided",
            hashes.size,
            hashes.toSet().size,
        )
    }
}
