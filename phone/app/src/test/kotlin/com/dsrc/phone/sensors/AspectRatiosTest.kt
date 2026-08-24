package com.dsrc.phone.sensors

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AspectRatiosTest {

    @Test
    fun `sixteen by nine sizes resolve to sixteen by nine`() {
        // The case that was silently broken: ResolutionSelector defaults to 4:3 and
        // excludes 16:9 candidates before any resolution rule runs, so asking a device
        // that offers 1280x720 exactly returned 640x480.
        for (size in listOf(1280 to 720, 1920 to 1080, 3840 to 2160, 854 to 480)) {
            assertEquals(
                "${size.first}x${size.second}",
                AspectRatios.RATIO_16_9,
                AspectRatios.nearest(size.first, size.second),
            )
        }
    }

    @Test
    fun `four by three sizes resolve to four by three`() {
        for (size in listOf(640 to 480, 1024 to 768, 1856 to 1392, 320 to 240)) {
            assertEquals(
                "${size.first}x${size.second}",
                AspectRatios.RATIO_4_3,
                AspectRatios.nearest(size.first, size.second),
            )
        }
    }

    @Test
    fun `orientation does not change the answer`() {
        // 1280x720 and 720x1280 are the same ratio and must not resolve differently.
        for (size in listOf(1280 to 720, 640 to 480, 1920 to 1080)) {
            assertEquals(
                "${size.first}x${size.second} vs its transpose",
                AspectRatios.nearest(size.first, size.second),
                AspectRatios.nearest(size.second, size.first),
            )
        }
    }

    @Test
    fun `a square is closer to four by three`() {
        assertEquals(AspectRatios.RATIO_4_3, AspectRatios.nearest(512, 512))
    }

    @Test
    fun `an extreme ratio picks the wider of the two`() {
        assertEquals(AspectRatios.RATIO_16_9, AspectRatios.nearest(3000, 1000))
    }

    @Test
    fun `the crossover sits at the geometric midpoint, not the arithmetic one`() {
        // The margin has to be narrower than the thing it discriminates. It was 0.02, and
        // the whole gap between the two candidate crossovers is 0.0159 -- geometric
        // 1.539601 against arithmetic 1.555556 -- so the upper probe at midpoint + 0.02
        // landed above *both* and the test passed whether the code used log distance or a
        // plain difference. It was measuring that a crossover exists, not where.
        val geometric = kotlin.math.sqrt((4.0 / 3.0) * (16.0 / 9.0))
        val arithmetic = ((4.0 / 3.0) + (16.0 / 9.0)) / 2.0
        val gap = arithmetic - geometric
        assertTrue("the two crossovers have moved: gap $gap", gap in 0.010..0.020)

        val height = 1000
        val margin = gap / 4.0
        assertEquals(
            AspectRatios.RATIO_4_3,
            AspectRatios.nearest(((geometric - margin) * height).toInt(), height),
        )
        assertEquals(
            AspectRatios.RATIO_16_9,
            AspectRatios.nearest(((geometric + margin) * height).toInt(), height),
        )
        // And the discriminating case: strictly between the two crossovers, log distance
        // says 16:9 and a plain difference says 4:3.
        val between = (geometric + arithmetic) / 2.0
        assertEquals(
            "a ratio between the geometric and arithmetic crossovers must resolve by log distance",
            AspectRatios.RATIO_16_9,
            AspectRatios.nearest((between * height).toInt(), height),
        )
    }

    @Test
    fun `a non-positive dimension is refused`() {
        assertTrue(runCatching { AspectRatios.nearest(0, 100) }.isFailure)
        assertTrue(runCatching { AspectRatios.nearest(100, -1) }.isFailure)
    }

    @Test
    fun `the constants match the CameraX values they stand for`() {
        // Hardcoded rather than imported so this module stays testable without Android;
        // if AspectRatio ever renumbers, the adapter's when-branch is where it shows.
        assertEquals(0, AspectRatios.RATIO_4_3)
        assertEquals(1, AspectRatios.RATIO_16_9)
    }
}
