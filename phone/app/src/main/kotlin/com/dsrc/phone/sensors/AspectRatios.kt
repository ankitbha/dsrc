package com.dsrc.phone.sensors

/**
 * Which of CameraX's two supported aspect ratios a requested size is asking for.
 *
 * Needed because `ResolutionSelector` defaults to 4:3 and silently excludes every
 * candidate of the other ratio *before* the resolution rule runs. A 16:9 request on a
 * device that offers 16:9 exactly then comes back as the nearest 4:3 size, which looks
 * like the device not supporting it.
 *
 * Pure arithmetic, kept out of the Android adapter so it can be tested without one.
 */
object AspectRatios {

    /** 4:3, matching `androidx.camera.core.AspectRatio.RATIO_4_3`. */
    const val RATIO_4_3 = 0

    /** 16:9, matching `androidx.camera.core.AspectRatio.RATIO_16_9`. */
    const val RATIO_16_9 = 1

    private const val FOUR_THIRDS = 4.0 / 3.0
    private const val SIXTEEN_NINTHS = 16.0 / 9.0

    /**
     * The closer of the two ratios to `width x height`, by log distance.
     *
     * Log distance rather than a plain difference so the choice does not depend on
     * whether the size is given in landscape or portrait: 1280x720 and 720x1280 are the
     * same ratio and must not resolve differently.
     */
    fun nearest(width: Int, height: Int): Int {
        require(width > 0 && height > 0) { "size is ${width}x$height" }
        val ratio = maxOf(width, height).toDouble() / minOf(width, height).toDouble()
        val to43 = kotlin.math.abs(kotlin.math.ln(ratio / FOUR_THIRDS))
        val to169 = kotlin.math.abs(kotlin.math.ln(ratio / SIXTEEN_NINTHS))
        return if (to169 < to43) RATIO_16_9 else RATIO_4_3
    }
}
