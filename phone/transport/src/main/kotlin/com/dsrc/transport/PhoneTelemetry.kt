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
) {
    fun toExtensions(): Map<String, JsonValue> = mapOf(
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
            Fields.checkReserved(extensions)
            return PhoneTelemetry(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                thermalStatus = Fields.requireString(extensions, "thermal_status"),
                thermalHeadroom = Fields.checkFinite(
                    "thermal_headroom",
                    Fields.optionalNumber(extensions, "thermal_headroom"),
                ),
                achieved = Fields.requireNestedNumbers(extensions, "achieved", RateCommand.RATE_KEYS),
                dropped = Fields.requireNestedCounts(extensions, "dropped", DROP_KEYS),
                hereCalls = Fields.requireCount(extensions, "here_calls"),
                hereErrors = Fields.requireCount(extensions, "here_errors"),
            )
        }
    }
}
