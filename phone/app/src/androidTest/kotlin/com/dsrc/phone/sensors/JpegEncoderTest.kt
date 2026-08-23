package com.dsrc.phone.sensors

import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * [YuvPacker] and [JpegEncoder] together, against a known colour.
 *
 * The packer's unit tests prove its output is what its author intended; they cannot
 * prove that intent matches what the platform encoder expects. NV21 is V-then-U, and
 * getting the pair backwards produces a perfect image with the colour axes inverted --
 * skin blue, sky orange -- which no dimension check, marker check or decodability check
 * can see. A synthetic frame of one known colour can.
 *
 * Instrumented rather than JVM because the encoder is a platform API.
 */
@RunWith(AndroidJUnit4::class)
class JpegEncoderTest {

    private val width = 16
    private val height = 16

    /** BT.601 for pure red: the values a camera would report for a red scene. */
    private val redY = 76
    private val redU = 85
    private val redV = 255

    private fun planarFrame(y: Int, u: Int, v: Int): ByteArray {
        val chromaW = width / 2
        val chromaH = height / 2
        return YuvPacker.toNv21(
            y = ByteArray(width * height) { y.toByte() },
            u = ByteArray(chromaW * chromaH) { u.toByte() },
            v = ByteArray(chromaW * chromaH) { v.toByte() },
            width = width,
            height = height,
            yRowStride = width,
            uvRowStride = chromaW,
            uvPixelStride = 1,
        )
    }

    private fun decodeCentre(jpeg: ByteArray): Int {
        val bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
        assertNotNull("the JPEG must decode", bitmap)
        return bitmap!!.getPixel(width / 2, height / 2)
    }

    @Test
    fun aRedFramePacksAndEncodesAsRed() {
        // The whole chain. If YuvPacker wrote U before V, this decodes blue.
        val jpeg = JpegEncoder.compress(planarFrame(redY, redU, redV), width, height, 90)
        val pixel = decodeCentre(jpeg)
        val r = Color.red(pixel)
        val g = Color.green(pixel)
        val b = Color.blue(pixel)
        assertTrue("expected red, got r=$r g=$g b=$b", r > 180 && g < 80 && b < 80)
        assertTrue("red must dominate blue, got r=$r b=$b", r > b + 100)
    }

    @Test
    fun swappingTheChromaPlanesIsVisible() {
        // Proof the test above is not vacuous: feeding the planes the other way round
        // must produce a visibly different colour, so a real swap inside YuvPacker
        // would be caught rather than looking identical.
        val correct = decodeCentre(JpegEncoder.compress(planarFrame(redY, redU, redV), width, height, 90))
        val swapped = decodeCentre(JpegEncoder.compress(planarFrame(redY, redV, redU), width, height, 90))
        assertTrue(
            "a chroma swap must change the decoded colour: correct=${Integer.toHexString(correct)} " +
                "swapped=${Integer.toHexString(swapped)}",
            Color.red(correct) > Color.red(swapped) + 100,
        )
        assertTrue("and it should read blue instead", Color.blue(swapped) > Color.blue(correct) + 100)
    }

    @Test
    fun aGreyFrameEncodesAsGrey() {
        // Neutral chroma is 128; a frame that decodes grey confirms the luma plane is
        // being read at the right offset, independently of the chroma order.
        val jpeg = JpegEncoder.compress(planarFrame(128, 128, 128), width, height, 90)
        val pixel = decodeCentre(jpeg)
        val r = Color.red(pixel)
        val g = Color.green(pixel)
        val b = Color.blue(pixel)
        assertTrue("expected grey, got r=$r g=$g b=$b", maxOf(r, g, b) - minOf(r, g, b) < 24)
        assertTrue("and mid-grey, got r=$r", r in 100..160)
    }

    @Test
    fun theEncodedSizeGrowsWithQuality() {
        val frame = planarFrame(redY, redU, redV)
        val low = JpegEncoder.compress(frame, width, height, 20).size
        val high = JpegEncoder.compress(frame, width, height, 95).size
        assertTrue("quality 95 ($high B) should exceed quality 20 ($low B)", high > low)
    }

    @Test
    fun aShortBufferIsRefused() {
        // A short buffer produces a JPEG that decodes to a green-bottomed image rather
        // than an error, so it has to be refused before the encoder sees it.
        val error = runCatching {
            JpegEncoder.compress(ByteArray(width * height), width, height, 85)
        }.exceptionOrNull()
        assertTrue("expected IllegalArgumentException, got $error", error is IllegalArgumentException)
    }

    @Test
    fun qualityOutsideTheRangeIsRefused() {
        val frame = planarFrame(redY, redU, redV)
        for (bad in listOf(0, 101, -1)) {
            assertTrue(
                "quality $bad should be refused",
                runCatching { JpegEncoder.compress(frame, width, height, bad) }.isFailure,
            )
        }
    }

    @Test
    fun theOutputIsAWholeJpegFile() {
        val jpeg = JpegEncoder.compress(planarFrame(redY, redU, redV), width, height, 85)
        assertEquals(0xFF.toByte(), jpeg[0])
        assertEquals(0xD8.toByte(), jpeg[1])
        assertEquals(0xFF.toByte(), jpeg[jpeg.size - 2])
        assertEquals(0xD9.toByte(), jpeg[jpeg.size - 1])
    }
}
