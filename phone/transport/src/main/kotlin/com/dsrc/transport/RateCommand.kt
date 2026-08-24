package com.dsrc.transport

/**
 * The Jetson's command: what rate each modality should run at.
 *
 * The phone's whole configuration surface today, and inbound only -- see "Configuration
 * flows one way" in `specs/transport_protocol.md`.
 *
 * `here` is optional, and optional from the first version that reads it. The spec spells
 * out why: an unknown header field is ignored, so an old receiver tolerates a new sender,
 * but a new receiver that `require`s a field refuses an old sender's command outright. The
 * breakage is one-directional, so a required field would make widening this a coordinated
 * flag day across Kotlin, Python and the golden vectors. Absent means "no change", which is
 * distinct from present-and-empty.
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
    /** The HERE query shape, or null when this command does not change it. */
    val here: HereQuery? = null,
) {
    fun toExtensions(): Map<String, JsonValue> = buildMap {
        put(Fields.CAPTURE_KEY, JsonValue.Num(captureMonoNs))
        put("rates", Fields.objectOf(RATE_KEYS.associateWith { rates.getValue(it) }))
        put("trigger", JsonValue.Text(trigger))
        put("shadow", JsonValue.Bool(shadow))
        here?.let { put("here", it.toJson()) }
    }

    companion object {
        val RATE_KEYS = listOf("camera_hz", "gps_hz", "imu_hz", "here_hz")

        /** The wire's ceiling. Above it a value is a bug rather than a request. */
        const val MAX_RATE_HZ = 1000.0

        fun fromWire(extensions: Map<String, JsonValue>, payload: ByteArray): RateCommand {
            Fields.checkNoPayload(payload, Channels.RATE_CMD)
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
                here = HereQuery.fromWire(extensions["here"]),
            )
        }
    }
}
