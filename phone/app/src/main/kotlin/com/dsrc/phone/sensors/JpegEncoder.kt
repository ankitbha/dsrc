package com.dsrc.phone.sensors

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream

/**
 * NV21 to JPEG, via the platform encoder.
 *
 * Thin on purpose: the part worth testing is [YuvPacker], which is where stride and
 * chroma-order mistakes live and which needs no Android. This is the one call that
 * does, so it is kept small enough to read in one go.
 */
object JpegEncoder {

    fun compress(nv21: ByteArray, width: Int, height: Int, quality: Int): ByteArray {
        require(quality in 1..100) { "quality $quality outside 1..100" }
        val expected = width * height * 3 / 2
        require(nv21.size >= expected) {
            // A short buffer produces a JPEG that decodes to a green-bottomed image
            // rather than an error, so it is refused here instead.
            "nv21 buffer is ${nv21.size}, need $expected for ${width}x$height"
        }
        val out = ByteArrayOutputStream(expected / 8)
        val image = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val ok = image.compressToJpeg(Rect(0, 0, width, height), quality, out)
        check(ok) { "compressToJpeg refused a ${width}x$height frame at quality $quality" }
        return out.toByteArray()
    }
}
