package com.dsrc.phone.sensors

import com.dsrc.phone.sensors.GpsLocationSource.Companion.normaliseBearing
import com.dsrc.phone.sensors.GpsLocationSource.Companion.periodMs
import com.dsrc.phone.sensors.GpsLocationSource.Companion.reading
import com.dsrc.phone.sensors.GpsLocationSource.Companion.utcNanos
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The field mapping, which is the half of the adapter that can be wrong.
 *
 * Nothing here touches `Location`: its getters throw against the unit-test `android.jar`
 * stubs, which is exactly why the mapping takes a [PlatformFix] of primitives instead.
 */
class GpsLocationSourceTest {

    private fun fix(
        latitude: Double = 40.7128,
        longitude: Double = -74.0060,
        fixMonoNs: Long = 1_000_000_000,
        receiptMonoNs: Long = 1_050_000_000,
        hasSpeed: Boolean = true,
        speedMps: Float = 13.4f,
        hasBearing: Boolean = true,
        bearingDeg: Float = 91.5f,
        hasAltitude: Boolean = true,
        altitudeM: Double = 12.5,
        utcEpochMs: Long = 1_770_000_000_000,
        satellitesUsedInFix: Int = 9,
    ) = PlatformFix(
        fixMonoNs = fixMonoNs,
        receiptMonoNs = receiptMonoNs,
        latitude = latitude,
        longitude = longitude,
        hasSpeed = hasSpeed,
        speedMps = speedMps,
        hasBearing = hasBearing,
        bearingDeg = bearingDeg,
        hasAltitude = hasAltitude,
        altitudeM = altitudeM,
        utcEpochMs = utcEpochMs,
        satellitesUsedInFix = satellitesUsedInFix,
    )

    @Test
    fun `a complete fix maps every field`() {
        val result = reading(fix())
        val record = result.record

        assertTrue(record.valid)
        assertEquals(40.7128, record.latitude!!, 1e-9)
        assertEquals(-74.0060, record.longitude!!, 1e-9)
        assertEquals(13.4, record.speedMps!!, 1e-6)
        assertEquals(91.5, record.headingDeg!!, 1e-4)
        assertEquals(12.5, record.altitudeM!!, 1e-9)
        assertEquals(9, record.satellites)
        // fix_quality 1, not 0: zero is the wire's "no fix", so a valid record must not
        // carry it.
        assertEquals(1, record.fixQuality)
        assertEquals(1_770_000_000_000_000_000L, record.utcEpochNs)
        assertEquals(1_000_000_000, record.captureMonoNs)
    }

    @Test
    fun `hdop is null because no Android API reports it`() {
        // Not an oversight, and specifically not Location.accuracy: that is a metre
        // radius and HDOP is dimensionless satellite geometry, with no conversion
        // between them. A derived number here would be invented data on the wire.
        assertNull(reading(fix()).record.hdop)
    }

    @Test
    fun `an absent field is null rather than zero`() {
        val record = reading(
            fix(hasSpeed = false, hasBearing = false, hasAltitude = false)
        ).record
        // Zero would be a claim: stopped, facing north, at sea level. Many devices report
        // no speed or bearing while stationary, so zero here would be the common case.
        assertNull(record.speedMps)
        assertNull(record.headingDeg)
        assertNull(record.altitudeM)
        assertTrue("an absent speed does not invalidate the position", record.valid)
    }

    @Test
    fun `both clocks come off the same base so the latency is real`() {
        val result = reading(fix(fixMonoNs = 1_000_000_000, receiptMonoNs = 1_250_000_000))
        assertEquals(250_000_000, result.deliveryLatencyNs)
        assertEquals(1_000_000_000, result.fixMonoNs)
    }

    @Test
    fun `a receipt stamp before the fix is clamped, not reported as negative`() {
        // Both stamps are elapsedRealtime, so this should be impossible; if it happens
        // the fix stamp is the trustworthy one, and a negative latency in the log would
        // be read as a clock bug rather than as the platform misreporting.
        val result = reading(fix(fixMonoNs = 5_000, receiptMonoNs = 4_000))
        assertEquals(5_000, result.receiptMonoNs)
        assertEquals(0, result.deliveryLatencyNs)
    }

    @Test
    fun `a bearing outside the documented range is folded, not sent`() {
        assertEquals(0.0, normaliseBearing(0f), 1e-9)
        assertEquals(45.0, normaliseBearing(45f), 1e-4)
        // 360 has been seen from real devices and is refused by our own outbound
        // validation, so it would be counted as a bug on this side.
        assertEquals(0.0, normaliseBearing(360f), 1e-9)
        assertEquals(1.0, normaliseBearing(361f), 1e-4)
        assertEquals(350.0, normaliseBearing(-10f), 1e-4)
        assertEquals(0.0, normaliseBearing(Float.NaN), 1e-9)
        assertEquals(0.0, normaliseBearing(Float.POSITIVE_INFINITY), 1e-9)
    }

    @Test
    fun `a wall clock with no time fix is null rather than 1970`() {
        assertNull("no time fix yet", utcNanos(0))
        assertNull("pre-2000 is not a GPS clock", utcNanos(1_000))
        assertNull(utcNanos(-1))
        assertEquals(1_770_000_000_000_000_000L, utcNanos(1_770_000_000_000))
        // Above this the nanosecond conversion overflows, which would put a negative
        // timestamp on the wire beside a valid position.
        assertNull(utcNanos(Long.MAX_VALUE))
        assertNull(utcNanos(Long.MAX_VALUE / 1_000_000L + 1))
    }

    @Test
    fun `a negative satellite count is clamped to zero`() {
        // num_sats is a required count and the decoder refuses a negative one, so an
        // absurd platform value would become an outbound refusal rather than a record.
        assertEquals(0, reading(fix(satellitesUsedInFix = -3)).record.satellites)
    }

    @Test
    fun `the request period is the reciprocal of the commanded rate`() {
        assertEquals(1_000, periodMs(1.0))
        assertEquals(200, periodMs(5.0))
        assertEquals(100, periodMs(10.0))
        assertEquals(1, periodMs(1_000.0))
        // The wire's floor. A period this long is the point: it means "almost never".
        assertTrue(periodMs(1e-9) > 0)
    }

    @Test
    fun `a bearing is normalised on the way through reading, not only in the helper`() {
        // normaliseBearing was tested directly and never through reading(), so its only
        // call site could be replaced with a plain toDouble() and the suite stayed green.
        // A device reporting 360 or a negative bearing would then put an out-of-range
        // heading on the wire -- the thing the docstring says our own outbound validation
        // would refuse.
        assertEquals(0.0, reading(fix(bearingDeg = 360f)).record.headingDeg!!, 1e-9)
        assertEquals(350.0, reading(fix(bearingDeg = -10f)).record.headingDeg!!, 1e-4)
        assertEquals(1.0, reading(fix(bearingDeg = 361f)).record.headingDeg!!, 1e-4)
        assertEquals(0.0, reading(fix(bearingDeg = Float.NaN)).record.headingDeg!!, 1e-9)
    }

    @Test
    fun `the wall clock conversion is applied on the way through reading`() {
        // Same shape: utcNanos was pinned directly and its call site was not.
        assertNull(reading(fix(utcEpochMs = 0)).record.utcEpochNs)
        assertEquals(1_770_000_000_000_000_000L, reading(fix(utcEpochMs = 1_770_000_000_000)).record.utcEpochNs)
    }

}
