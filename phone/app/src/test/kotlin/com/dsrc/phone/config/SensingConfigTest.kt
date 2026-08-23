package com.dsrc.phone.config

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SensingConfigTest {

    @Test
    fun `the defaults are valid`() {
        val config = SensingConfig()
        assertEquals(5.0, config.cameraHz, 0.0)
        assertEquals(1280, config.cameraWidth)
        assertEquals(85, config.jpegQuality)
    }

    @Test
    fun `a zero rate is refused on every channel`() {
        // A zero is read as a period, so the field that meant "10 Hz" says "never" --
        // the case the protocol's sender rule names.
        assertTrue(runCatching { SensingConfig(cameraHz = 0.0) }.isFailure)
        assertTrue(runCatching { SensingConfig(gpsHz = 0.0) }.isFailure)
        assertTrue(runCatching { SensingConfig(imuHz = 0.0) }.isFailure)
        assertTrue(runCatching { SensingConfig(hereHz = 0.0) }.isFailure)
    }

    @Test
    fun `a rate above the wire's ceiling is refused`() {
        assertTrue(runCatching { SensingConfig(imuHz = 1000.1) }.isFailure)
        SensingConfig(imuHz = 1000.0)
    }

    @Test
    fun `a non-finite rate is refused`() {
        for (bad in listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY)) {
            assertTrue("$bad should be refused", runCatching { SensingConfig(cameraHz = bad) }.isFailure)
        }
    }

    @Test
    fun `odd camera dimensions are refused`() {
        // They would fail later inside the YUV packer, where the reason is much less
        // obvious than at the setting that caused it.
        assertTrue(runCatching { SensingConfig(cameraWidth = 1281) }.isFailure)
        assertTrue(runCatching { SensingConfig(cameraHeight = 721) }.isFailure)
    }

    @Test
    fun `a non-positive dimension is refused`() {
        assertTrue(runCatching { SensingConfig(cameraWidth = 0) }.isFailure)
        assertTrue(runCatching { SensingConfig(cameraHeight = -2) }.isFailure)
    }

    @Test
    fun `quality outside 1 to 100 is refused`() {
        assertTrue(runCatching { SensingConfig(jpegQuality = 0) }.isFailure)
        assertTrue(runCatching { SensingConfig(jpegQuality = 101) }.isFailure)
        SensingConfig(jpegQuality = 1)
        SensingConfig(jpegQuality = 100)
    }

    @Test
    fun `every default rate is within the range the wire accepts`() {
        val config = SensingConfig()
        for (hz in listOf(config.cameraHz, config.gpsHz, config.imuHz, config.hereHz)) {
            assertTrue("$hz must be in (0, 1000]", hz > 0.0 && hz <= SensingConfig.MAX_HZ)
        }
    }
}
