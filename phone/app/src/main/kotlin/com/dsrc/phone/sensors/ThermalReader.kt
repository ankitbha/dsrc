package com.dsrc.phone.sensors

import android.os.PowerManager

/** How hot the phone says it is, in the wire's vocabulary. */
object ThermalReader {

    /**
     * `PowerManager`'s status constants, named for the wire.
     *
     * The integers are an Android enum and the wire wants a string; mapping here keeps the
     * translation in one place rather than at every reader. An unrecognised value becomes
     * `unknown` rather than a number stringified, because a receiver keying on these should
     * not have to guess whether "7" is a status or a bug.
     */
    fun statusName(status: Int): String = when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "nominal"
        PowerManager.THERMAL_STATUS_LIGHT -> "light"
        PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "severe"
        PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
        else -> "unknown"
    }

    /**
     * Headroom, or null when the platform will not say.
     *
     * `getThermalHeadroom` returns `NaN` when it has no estimate — too soon after boot, too
     * soon after the last call, or unsupported on the device. `thermal_headroom` is nullable
     * on the wire precisely so that has somewhere to go, and the alternative is worse than
     * useless: canonical JSON refuses to encode a NaN on both sides, so a NaN here would not
     * produce a wrong number, it would fail the whole telemetry frame and take the thermal
     * status down with it — the phone would go quiet about being hot at the moment it was
     * hottest.
     *
     * A negative or absurd value goes the same way. The API documents `[0, 1]` with values
     * above 1 meaning throttling, and anything outside a generous band is the platform
     * misbehaving rather than a reading.
     */
    fun headroomOrNull(raw: Float): Double? {
        val value = raw.toDouble()
        if (!value.isFinite()) return null
        if (value < 0.0 || value > MAX_PLAUSIBLE_HEADROOM) return null
        return value
    }

    /**
     * The widest headroom worth reporting.
     *
     * The API's scale is `[0, 1]` with above-1 meaning throttled; ten is far past anything
     * meaningful and well clear of a device that reports a little over 1 under load.
     */
    const val MAX_PLAUSIBLE_HEADROOM = 10.0
}
