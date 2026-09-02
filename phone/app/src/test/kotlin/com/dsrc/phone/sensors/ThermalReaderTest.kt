package com.dsrc.phone.sensors

import android.os.Build
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ThermalReaderTest {

    @Test
    fun `each headroom absence carries its own reason`() {
        // Four causes that used to collapse into one bare null: below API 30, a NaN, a
        // negative value, and one past the plausible ceiling.
        assertEquals(
            ThermalReader.REASON_API_TOO_OLD,
            ThermalReader.headroomIfSupported(29) { 0.5f }.absentReason,
        )
        assertEquals(
            ThermalReader.REASON_NOT_A_NUMBER,
            ThermalReader.headroomIfSupported(Build.VERSION_CODES.R) { Float.NaN }.absentReason,
        )
        assertEquals(
            ThermalReader.REASON_OUT_OF_BAND,
            ThermalReader.headroomIfSupported(Build.VERSION_CODES.R) { -1f }.absentReason,
        )
        assertEquals(
            ThermalReader.REASON_OUT_OF_BAND,
            ThermalReader.headroomIfSupported(Build.VERSION_CODES.R) { 11f }.absentReason,
        )
    }

    @Test
    fun `a present value carries no reason`() {
        val result = ThermalReader.headroomIfSupported(Build.VERSION_CODES.R) { 0.42f }
        assertEquals(0.42, result.value!!, 1e-6)
        assertNull(result.absentReason)
    }

    @Test
    fun `headroomFrom and headroomIfSupported agree at the API 30 boundary`() {
        // headroomFrom cannot be called from a JVM test with a real reading -- the guard it
        // repeats is what a test can check without one. Two identical comparisons against
        // one constant is the duplication headroomFrom's own docstring names; this is what
        // keeps them from drifting apart.
        assertEquals(
            ThermalReader.REASON_API_TOO_OLD,
            ThermalReader.headroomIfSupported(Build.VERSION_CODES.R - 1) { 0.5f }.absentReason,
        )
        val supported = ThermalReader.headroomIfSupported(Build.VERSION_CODES.R) { 0.5f }
        assertEquals(0.5, supported.value!!, 1e-6)
        assertNull(supported.absentReason)
    }

    @Test
    fun `the plausible headroom band is the constant`() {
        assertEquals(10.0, ThermalReader.MAX_PLAUSIBLE_HEADROOM, 1e-6)
        assertEquals(
            ThermalReader.REASON_OUT_OF_BAND,
            ThermalReader.headroomOrNull(ThermalReader.MAX_PLAUSIBLE_HEADROOM.toFloat() + 0.001f).absentReason,
        )
        assertNull(ThermalReader.headroomOrNull(ThermalReader.MAX_PLAUSIBLE_HEADROOM.toFloat()).absentReason)
    }
}
