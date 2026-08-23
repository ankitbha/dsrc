package com.dsrc.phone.sensors

/**
 * Packs `YUV_420_888` planes into the NV21 layout the platform JPEG encoder wants.
 *
 * Pure, and separated from the compression itself because this is where the bugs
 * are. Camera planes carry a row stride that is usually wider than the image and a
 * pixel stride that is 1 or 2 depending on whether the chroma planes are
 * interleaved. Ignoring either produces a picture that is skewed, green, or
 * diagonally sheared -- all of which decode without error, so nothing complains.
 */
object YuvPacker {

    /**
     * @param y luma bytes, `yRowStride` per row
     * @param u/v chroma bytes, `uvRowStride` per row and `uvPixelStride` between samples
     * @return NV21: the full luma plane followed by interleaved V,U pairs
     */
    fun toNv21(
        y: ByteArray,
        u: ByteArray,
        v: ByteArray,
        width: Int,
        height: Int,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
    ): ByteArray {
        require(width > 0 && height > 0) { "image is ${width}x$height" }
        require(width % 2 == 0 && height % 2 == 0) {
            // 4:2:0 chroma is subsampled by two in both axes, so an odd dimension has
            // no well-defined last row or column of chroma.
            "4:2:0 needs even dimensions, got ${width}x$height"
        }
        require(yRowStride >= width) { "y row stride $yRowStride is narrower than width $width" }
        require(uvPixelStride == 1 || uvPixelStride == 2) { "uv pixel stride $uvPixelStride" }
        val chromaWidth = width / 2
        val chromaHeight = height / 2
        require(uvRowStride >= chromaWidth * uvPixelStride) {
            "uv row stride $uvRowStride cannot hold $chromaWidth samples at stride $uvPixelStride"
        }
        require(y.size >= (height - 1) * yRowStride + width) { "y plane too small: ${y.size}" }
        val uvNeeded = (chromaHeight - 1) * uvRowStride + (chromaWidth - 1) * uvPixelStride + 1
        require(u.size >= uvNeeded) { "u plane too small: ${u.size}, need $uvNeeded" }
        require(v.size >= uvNeeded) { "v plane too small: ${v.size}, need $uvNeeded" }

        val out = ByteArray(width * height + chromaWidth * chromaHeight * 2)
        var o = 0
        for (row in 0 until height) {
            val from = row * yRowStride
            y.copyInto(out, o, from, from + width)
            o += width
        }
        // NV21 is V then U. Getting this pair backwards swaps the colour axes: skin
        // goes blue and sky goes orange, and the image is otherwise perfect.
        for (row in 0 until chromaHeight) {
            var uv = row * uvRowStride
            for (col in 0 until chromaWidth) {
                out[o++] = v[uv]
                out[o++] = u[uv]
                uv += uvPixelStride
            }
        }
        return out
    }
}
