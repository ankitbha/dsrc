package com.dsrc.phone.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YuvPackerTest {

    /** A plane whose bytes encode their own (row, col) so mis-striding is visible. */
    private fun plane(rows: Int, cols: Int, stride: Int, pixelStride: Int = 1, base: Int = 0): ByteArray {
        val out = ByteArray((rows - 1) * stride + (cols - 1) * pixelStride + 1)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                out[r * stride + c * pixelStride] = (base + r * 16 + c).toByte()
            }
        }
        return out
    }

    @Test
    fun `output size is the nv21 size for the dimensions`() {
        val out = YuvPacker.toNv21(
            y = plane(4, 4, 4), u = plane(2, 2, 2), v = plane(2, 2, 2),
            width = 4, height = 4, yRowStride = 4, uvRowStride = 2, uvPixelStride = 1,
        )
        assertEquals(4 * 4 + 2 * 2 * 2, out.size)
    }

    @Test
    fun `luma padding is stripped`() {
        // Row stride wider than the image is the normal case, not an edge case: ignoring
        // it shears the picture diagonally, and the result still decodes.
        val out = YuvPacker.toNv21(
            y = plane(2, 2, stride = 8), u = plane(1, 1, 1), v = plane(1, 1, 1),
            width = 2, height = 2, yRowStride = 8, uvRowStride = 1, uvPixelStride = 1,
        )
        assertEquals(0x00.toByte(), out[0])
        assertEquals(0x01.toByte(), out[1])
        assertEquals("second row must start at byte 2, not byte 8", 0x10.toByte(), out[2])
        assertEquals(0x11.toByte(), out[3])
    }

    @Test
    fun `chroma is interleaved v then u`() {
        // NV21 is V,U. Swapping them leaves a perfect image with the colour axes
        // inverted -- skin blue, sky orange -- and no error anywhere.
        val out = YuvPacker.toNv21(
            y = plane(2, 2, 2), u = plane(1, 1, 1, base = 0x40), v = plane(1, 1, 1, base = 0x50),
            width = 2, height = 2, yRowStride = 2, uvRowStride = 1, uvPixelStride = 1,
        )
        assertEquals("V comes first", 0x50.toByte(), out[4])
        assertEquals("then U", 0x40.toByte(), out[5])
    }

    @Test
    fun `an interleaved chroma plane is read at its pixel stride`() {
        // pixelStride 2 is what a semi-planar camera reports; reading it as 1 halves
        // the chroma resolution and tints alternate columns.
        val out = YuvPacker.toNv21(
            y = plane(4, 4, 4),
            u = plane(2, 2, stride = 8, pixelStride = 2, base = 0x40),
            v = plane(2, 2, stride = 8, pixelStride = 2, base = 0x50),
            width = 4, height = 4, yRowStride = 4, uvRowStride = 8, uvPixelStride = 2,
        )
        val chroma = out.copyOfRange(16, out.size)
        assertEquals(0x50.toByte(), chroma[0])
        assertEquals(0x40.toByte(), chroma[1])
        assertEquals("second chroma sample of row 0", 0x51.toByte(), chroma[2])
        assertEquals(0x41.toByte(), chroma[3])
    }

    @Test
    fun `every luma byte reaches the output exactly once`() {
        val w = 8; val h = 6
        val out = YuvPacker.toNv21(
            y = plane(h, w, stride = w + 5), u = plane(h / 2, w / 2, w / 2), v = plane(h / 2, w / 2, w / 2),
            width = w, height = h, yRowStride = w + 5, uvRowStride = w / 2, uvPixelStride = 1,
        )
        for (r in 0 until h) {
            for (c in 0 until w) {
                assertEquals("luma ($r,$c)", (r * 16 + c).toByte(), out[r * w + c])
            }
        }
    }

    @Test
    fun `odd dimensions are refused`() {
        // 4:2:0 subsamples by two, so an odd dimension has no defined last chroma row.
        for (dims in listOf(3 to 4, 4 to 3, 5 to 5)) {
            val threw = runCatching {
                YuvPacker.toNv21(
                    y = ByteArray(64), u = ByteArray(64), v = ByteArray(64),
                    width = dims.first, height = dims.second,
                    yRowStride = 8, uvRowStride = 4, uvPixelStride = 1,
                )
            }.isFailure
            assertTrue("${dims.first}x${dims.second} should be refused", threw)
        }
    }

    @Test
    fun `a stride narrower than the width is refused`() {
        val threw = runCatching {
            YuvPacker.toNv21(
                y = ByteArray(64), u = ByteArray(16), v = ByteArray(16),
                width = 8, height = 8, yRowStride = 4, uvRowStride = 4, uvPixelStride = 1,
            )
        }.isFailure
        assertTrue(threw)
    }

    @Test
    fun `a short plane is refused rather than producing a truncated image`() {
        for (case in listOf("y", "u", "v")) {
            val threw = runCatching {
                YuvPacker.toNv21(
                    y = if (case == "y") ByteArray(4) else ByteArray(64),
                    u = if (case == "u") ByteArray(1) else ByteArray(16),
                    v = if (case == "v") ByteArray(1) else ByteArray(16),
                    width = 8, height = 8, yRowStride = 8, uvRowStride = 4, uvPixelStride = 1,
                )
            }.isFailure
            assertTrue("a short $case plane should be refused", threw)
        }
    }

    @Test
    fun `an unsupported pixel stride is refused`() {
        for (bad in listOf(0, 3, -1)) {
            val threw = runCatching {
                YuvPacker.toNv21(
                    y = ByteArray(64), u = ByteArray(64), v = ByteArray(64),
                    width = 8, height = 8, yRowStride = 8, uvRowStride = 8, uvPixelStride = bad,
                )
            }.isFailure
            assertTrue("pixel stride $bad should be refused", threw)
        }
    }

    @Test
    fun `chroma padding is stripped on every row, not just the first`() {
        // The row-stride bug that no test caught: reading chroma at
        // `row * chromaWidth * pixelStride` instead of `row * uvRowStride` is identical
        // on row 0 and wrong on every row after it. At real 720p semi-planar geometry
        // (uvRowStride 1536, chromaWidth x pixelStride 1280) that corrupts the whole
        // frame below the first chroma row, and the emulator cannot catch it because its
        // virtual camera reports rowStride == width.
        val w = 8; val h = 8
        val uvStride = 16          // wider than chromaWidth (4), i.e. padded
        val out = YuvPacker.toNv21(
            y = plane(h, w, stride = w),
            u = plane(h / 2, w / 2, stride = uvStride, base = 0x40),
            v = plane(h / 2, w / 2, stride = uvStride, base = 0x50),
            width = w, height = h, yRowStride = w, uvRowStride = uvStride, uvPixelStride = 1,
        )
        val chroma = out.copyOfRange(w * h, out.size)
        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                val at = (row * (w / 2) + col) * 2
                assertEquals("V at row $row col $col", (0x50 + row * 16 + col).toByte(), chroma[at])
                assertEquals("U at row $row col $col", (0x40 + row * 16 + col).toByte(), chroma[at + 1])
            }
        }
    }

    @Test
    fun `a chroma row stride too narrow for the samples is refused`() {
        // Nothing tested this bound at all.
        val threw = runCatching {
            YuvPacker.toNv21(
                y = ByteArray(64), u = ByteArray(64), v = ByteArray(64),
                width = 8, height = 8,
                yRowStride = 8, uvRowStride = 3, uvPixelStride = 2,   // needs 4*2 = 8
            )
        }.isFailure
        assertTrue(threw)
    }

    @Test
    fun `a plane one byte short is refused`() {
        // The existing short-plane test uses ByteArray(1) against a need of 16, which is
        // far too coarse to catch an off-by-one in the size formula.
        val w = 8; val h = 8
        val uvNeeded = (h / 2 - 1) * (w / 2) + (w / 2 - 1) + 1   // = 16 at stride 4, pixelStride 1
        YuvPacker.toNv21(
            y = ByteArray(w * h), u = ByteArray(uvNeeded), v = ByteArray(uvNeeded),
            width = w, height = h, yRowStride = w, uvRowStride = w / 2, uvPixelStride = 1,
        )
        for (which in listOf("y", "u", "v")) {
            val error = runCatching {
                YuvPacker.toNv21(
                    y = ByteArray(if (which == "y") w * h - 1 else w * h),
                    u = ByteArray(if (which == "u") uvNeeded - 1 else uvNeeded),
                    v = ByteArray(if (which == "v") uvNeeded - 1 else uvNeeded),
                    width = w, height = h, yRowStride = w, uvRowStride = w / 2, uvPixelStride = 1,
                )
            }.exceptionOrNull()
            // The type matters: isFailure alone cannot tell a named refusal from an
            // ArrayIndexOutOfBounds thrown later by the copy, so dropping the bound
            // check looked like it changed nothing.
            assertTrue(
                "$which one byte short must be refused by name, got $error",
                error is IllegalArgumentException,
            )
            assertTrue(
                "the message should name the plane: ${error?.message}",
                error?.message?.contains("$which plane") == true,
            )
        }
    }

    @Test
    fun `a short plane is refused by name, not by an index error`() {
        // runCatching{}.isFailure cannot tell a named IllegalArgumentException from an
        // IndexOutOfBoundsException out of copyInto, so deleting the bound check looked
        // like it changed nothing.
        val error = runCatching {
            YuvPacker.toNv21(
                y = ByteArray(4), u = ByteArray(16), v = ByteArray(16),
                width = 8, height = 8, yRowStride = 8, uvRowStride = 4, uvPixelStride = 1,
            )
        }.exceptionOrNull()
        assertTrue("expected IllegalArgumentException, got $error", error is IllegalArgumentException)
        assertTrue("the message should name the plane: ${error?.message}", error?.message?.contains("y plane") == true)
    }

    @Test
    fun `a realistic 720p geometry packs to the right size`() {
        val w = 1280; val h = 720
        val out = YuvPacker.toNv21(
            y = ByteArray(1536 * h), u = ByteArray(1536 * h / 2), v = ByteArray(1536 * h / 2),
            width = w, height = h, yRowStride = 1536, uvRowStride = 1536, uvPixelStride = 2,
        )
        assertEquals(w * h * 3 / 2, out.size)
    }
}
