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
     * Headroom, or a named reason there is none.
     *
     * `getThermalHeadroom` returns `NaN` when it has no estimate — too soon after boot, too
     * soon after the last call, or unsupported on the device. `thermal_headroom` is nullable
     * on the wire precisely so that has somewhere to go, and the alternative is worse than
     * useless: canonical JSON refuses to encode a NaN on both sides, so a NaN here would not
     * produce a wrong number, it would fail the whole telemetry frame and take the thermal
     * status down with it — the phone would go quiet about being hot at the moment it was
     * hottest. [Headroom.absentReason] is what turns "the platform never answers on this
     * device" from prose in a docstring into a measured count.
     *
     * A negative or absurd value goes the same way. The API documents `[0, 1]` with values
     * above 1 meaning throttling, and anything outside a generous band is the platform
     * misbehaving rather than a reading.
     */
    data class Headroom(val value: Double?, val absentReason: String?)

    const val REASON_API_TOO_OLD = "api_too_old"
    const val REASON_NOT_A_NUMBER = "not_a_number"
    const val REASON_OUT_OF_BAND = "out_of_band"

    /**
     * Read the headroom from the platform, or the reason there is none.
     *
     * `getThermalHeadroom` is API 30 and this app's `minSdk` is 29 — which is a guess about
     * the handset, not a device we have. Called unguarded it raises `NoSuchMethodError` on
     * Android 10, inside a lambda whose caller wraps it in `runCatching`, so the whole
     * telemetry stream went silent for the entire drive: no thermal status, no achieved
     * rates, no drops, and nothing logged. The status is API 29 and would have been fine on
     * its own, which is what makes losing it to the headroom call the wrong trade.
     */
    fun headroomFrom(power: PowerManager): Headroom {
        // The guard written where lint can read it. Expressed through `headroomIfSupported`
        // instead, lint sees only a call behind a lambda and an `sdkInt` parameter it cannot
        // trace, and reports the unguarded-call error anyway -- which would mean either
        // suppressing the one check that caught this defect, or losing it. The two are kept
        // in step by a test that asserts they agree at the boundary.
        if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.R) {
            return Headroom(null, REASON_API_TOO_OLD)
        }
        return headroomOrNull(power.getThermalHeadroom(FORECAST_SECONDS))
    }

    /**
     * The guard itself, with the platform behind a lambda.
     *
     * Split out so a test can assert the call is *not made* below API 30 — passing a
     * `PowerManager` would not do it, because on a JVM the android.jar stub throws for its
     * own reasons and the test would pass whether the guard existed or not.
     *
     * [headroomFrom] repeats the predicate rather than calling this, because lint cannot
     * trace a version check through a lambda and would report the platform call as
     * unguarded. The duplication is two identical comparisons against one constant, and a
     * test asserts they agree at the boundary so they cannot drift apart.
     */
    internal fun headroomIfSupported(sdkInt: Int, read: () -> Float): Headroom {
        if (sdkInt < android.os.Build.VERSION_CODES.R) return Headroom(null, REASON_API_TOO_OLD)
        return headroomOrNull(read())
    }

    /**
     * How far ahead the headroom call is asked to forecast.
     *
     * Zero: the reading wanted is now, not a prediction. A forecast would make the number
     * the Jetson sees a guess about a guess.
     */
    const val FORECAST_SECONDS = 0

    fun headroomOrNull(raw: Float): Headroom {
        val value = raw.toDouble()
        if (!value.isFinite()) return Headroom(null, REASON_NOT_A_NUMBER)
        if (value < 0.0 || value > MAX_PLAUSIBLE_HEADROOM) return Headroom(null, REASON_OUT_OF_BAND)
        return Headroom(value, null)
    }

    /**
     * The widest headroom worth reporting.
     *
     * The API's scale is `[0, 1]` with above-1 meaning throttled; ten is far past anything
     * meaningful and well clear of a device that reports a little over 1 under load.
     */
    const val MAX_PLAUSIBLE_HEADROOM = 10.0
}
