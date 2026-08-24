package com.dsrc.transport

/**
 * The Jetson's command: what rate each modality should run at.
 *
 * The phone's whole configuration surface today, and inbound only -- see "Configuration
 * flows one way" in `specs/transport_protocol.md`.
 *
 * `rates` is where the sender rule earns its keep. The spec uses a zero here as its
 * worked example: "it is read as a period, so the field that should have said '10 Hz'
 * instead says 'never', and the failure surfaces on the far side of the link." A phone
 * that accepted it would stop sensing and look healthy doing it.
 */
data class RateCommand(
    val captureMonoNs: Long,
    val rates: Map<String, Double>,
    val trigger: String,
    val shadow: Boolean,
) {
    fun toExtensions(): Map<String, JsonValue> = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
        "rates" to Fields.objectOf(RATE_KEYS.associateWith { rates.getValue(it) }),
        "trigger" to JsonValue.Text(trigger),
        "shadow" to JsonValue.Bool(shadow),
    )

    companion object {
        val RATE_KEYS = listOf("camera_hz", "gps_hz", "imu_hz", "here_hz")

        /** The wire's ceiling. Above it a value is a bug rather than a request. */
        const val MAX_RATE_HZ = 1000.0

        fun fromWire(extensions: Map<String, JsonValue>, payload: ByteArray): RateCommand {
            Fields.checkNoPayload(payload, Channels.RATE_CMD)
            Fields.checkReserved(extensions)
            val rates = Fields.requireNestedNumbers(extensions, "rates", RATE_KEYS)
            for ((key, value) in rates) {
                if (!(value > 0.0 && value <= MAX_RATE_HZ)) {
                    throw MessageError(
                        RefusalReason.OUT_OF_RANGE,
                        "rates.$key is $value, outside (0, $MAX_RATE_HZ]",
                    )
                }
            }
            return RateCommand(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                rates = rates,
                trigger = Fields.requireString(extensions, "trigger"),
                shadow = Fields.requireBool(extensions, "shadow"),
            )
        }
    }
}
