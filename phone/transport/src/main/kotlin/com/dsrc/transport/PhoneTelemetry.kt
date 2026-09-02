package com.dsrc.transport

/**
 * The phone reporting on itself: thermal state, achieved rates, what it dropped.
 *
 * `achieved` answers a different question from `rates` on a command -- what the phone
 * managed, not what it was asked for -- and the pair is how a throttling handset becomes
 * visible from the Jetson rather than merely quiet.
 */
data class PhoneTelemetry(
    val captureMonoNs: Long,
    val thermalStatus: String,
    val thermalHeadroom: Double?,
    val achieved: Map<String, Double>,
    val dropped: Map<String, Long>,
    val hereCalls: Long,
    val hereErrors: Long,
    /**
     * An absolute temperature, for handsets that will not compute headroom.
     *
     * `thermal_headroom` is skin temperature over the throttling threshold, so a device
     * that publishes no threshold returns `NaN` forever and the Jetson is left with only
     * the six-step status -- which does not move until the handset is already in trouble.
     * This is the fallback: a number the far side can trend.
     *
     * Null wherever the kernel's zones are unreadable, which is a per-device SELinux
     * decision rather than anything this side controls, and meaningless without
     * [skinTempZone] -- zone names do not mean what they look like, so the same number
     * from two handsets can be two different sensors.
     */
    val skinTempC: Double? = null,
    /** Which kernel zone [skinTempC] came from. Vendor-named, so it identifies the sensor. */
    val skinTempZone: String? = null,
    /**
     * Why [thermalHeadroom] is null, when it is: one of `api_too_old`, `not_a_number` (the
     * platform's own catch-all for "too soon after boot, too soon after the last call, or
     * unsupported"), or `out_of_band`. Null exactly when [thermalHeadroom] has a value -- a
     * value needs no excuse.
     */
    val thermalHeadroomAbsent: String? = null,
    /** As [thermalHeadroomAbsent], for [skinTempC]: `no_zones_listed`, `no_preferred_zone`,
     * `unreadable`, or `implausible`. */
    val skinTempAbsent: String? = null,
    /**
     * Transitions the phone's thermal-status listener has counted this service run,
     * independent of [thermalStatus]'s own 1 Hz poll. Sent by every phone build that knows
     * this field exists, even when it is zero -- an older build omits it, which is the one
     * way it is absent here.
     */
    val thermalStatusChanges: Long = 0,
    /** The most recent transition's endpoints and when the phone observed it, on the
     * phone's own clock. All three null before the first transition. */
    val thermalChangeFrom: String? = null,
    val thermalChangeTo: String? = null,
    val thermalChangeAtMonoNs: Long? = null,
) {
    fun toExtensions(): Map<String, JsonValue> = buildMap {
        putAll(base())
        skinTempC?.let { put("skin_temp_c", JsonValue.Real(it)) }
        skinTempZone?.let { put("skin_temp_zone", JsonValue.Text(it)) }
        thermalHeadroomAbsent?.let { put("thermal_headroom_absent", JsonValue.Text(it)) }
        skinTempAbsent?.let { put("skin_temp_absent", JsonValue.Text(it)) }
        put("thermal_status_changes", JsonValue.Num(thermalStatusChanges))
        thermalChangeFrom?.let { put("thermal_change_from", JsonValue.Text(it)) }
        thermalChangeTo?.let { put("thermal_change_to", JsonValue.Text(it)) }
        thermalChangeAtMonoNs?.let { put("thermal_change_at_mono_ns", JsonValue.Num(it)) }
    }

    private fun base(): Map<String, JsonValue> = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
        "thermal_status" to JsonValue.Text(thermalStatus),
        "thermal_headroom" to Fields.toWire(thermalHeadroom),
        "achieved" to Fields.objectOf(RateCommand.RATE_KEYS.associateWith { achieved.getValue(it) }),
        "dropped" to Fields.countsOf(DROP_KEYS.associateWith { dropped.getValue(it) }),
        "here_calls" to JsonValue.Num(hereCalls),
        "here_errors" to JsonValue.Num(hereErrors),
    )

    companion object {
        val DROP_KEYS = listOf("camera", "gps", "imu", "here")

        fun fromWire(extensions: Map<String, JsonValue>, payload: ByteArray): PhoneTelemetry {
            Fields.checkNoPayload(payload, Channels.TELEMETRY)
            return PhoneTelemetry(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                thermalStatus = Fields.requireString(extensions, "thermal_status"),
                thermalHeadroom = Fields.checkFinite(
                    "thermal_headroom",
                    Fields.optionalNumber(extensions, "thermal_headroom"),
                ),
                // Absent-tolerant, not merely nullable: a phone built before these existed
                // does not write them, and requiring them would refuse all of its telemetry.
                skinTempC = Fields.checkFinite(
                    "skin_temp_c",
                    Fields.absentableNumber(extensions, "skin_temp_c"),
                ),
                skinTempZone = Fields.absentableString(extensions, "skin_temp_zone"),
                thermalHeadroomAbsent = Fields.absentableString(extensions, "thermal_headroom_absent"),
                skinTempAbsent = Fields.absentableString(extensions, "skin_temp_absent"),
                thermalStatusChanges = Fields.absentableInt(extensions, "thermal_status_changes") ?: 0,
                thermalChangeFrom = Fields.absentableString(extensions, "thermal_change_from"),
                thermalChangeTo = Fields.absentableString(extensions, "thermal_change_to"),
                thermalChangeAtMonoNs = Fields.absentableInt(extensions, "thermal_change_at_mono_ns"),
                achieved = Fields.requireNestedNumbers(extensions, "achieved", RateCommand.RATE_KEYS),
                dropped = Fields.requireNestedCounts(extensions, "dropped", DROP_KEYS),
                hereCalls = Fields.requireCount(extensions, "here_calls"),
                hereErrors = Fields.requireCount(extensions, "here_errors"),
            )
        }
    }
}
