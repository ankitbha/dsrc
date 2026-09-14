package com.dsrc.transport

/**
 * What the driver is shown: a recommended speed.
 *
 * Inbound on the phone. The controller is a deterministic argmax over three speed
 * fractions, so there is no headway head, no lane or merge head, and no calibrated
 * confidence to carry.
 */
data class AdvisoryMessage(
    val captureMonoNs: Long,
    val recSpeedMps: Double,
    val recSpeedDisplay: Double,
    val currentSpeedDisplay: Double,
    val units: String,
    val trafficText: String,
) {
    fun toExtensions(): Map<String, JsonValue> = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
        "rec_speed_mps" to Fields.toWire(recSpeedMps),
        "rec_speed_display" to Fields.toWire(recSpeedDisplay),
        "current_speed_display" to Fields.toWire(currentSpeedDisplay),
        "units" to JsonValue.Text(units),
        "traffic_text" to JsonValue.Text(trafficText),
    )

    companion object {
        val DISPLAY_UNITS = setOf("mph", "kmh", "mps")

        fun fromWire(extensions: Map<String, JsonValue>, payload: ByteArray): AdvisoryMessage {
            Fields.checkNoPayload(payload, Channels.ADVISORY)

            val units = Fields.requireString(extensions, "units")
            if (units !in DISPLAY_UNITS) {
                throw MessageError(RefusalReason.UNKNOWN_VALUE, "units '$units' not one of $DISPLAY_UNITS")
            }

            return AdvisoryMessage(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                recSpeedMps = Fields.checkFinite("rec_speed_mps", Fields.requireNumber(extensions, "rec_speed_mps"))!!,
                recSpeedDisplay = Fields.checkFinite(
                    "rec_speed_display",
                    Fields.requireNumber(extensions, "rec_speed_display"),
                )!!,
                currentSpeedDisplay = Fields.checkFinite(
                    "current_speed_display",
                    Fields.requireNumber(extensions, "current_speed_display"),
                )!!,
                units = units,
                trafficText = Fields.requireString(extensions, "traffic_text"),
            )
        }
    }
}
