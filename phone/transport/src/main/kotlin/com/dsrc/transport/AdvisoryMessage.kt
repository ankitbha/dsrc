package com.dsrc.transport

/**
 * What the driver is shown, and the policy decision behind it.
 *
 * Inbound on the phone. `action` carries the four v2 heads from
 * `specs/action_schema.md`, and unlike the additive rate objects its heads are a **closed**
 * set: an unknown head is a policy this build cannot honour, so accepting it would mean
 * displaying an advisory whose reasoning is partly unread.
 */
data class AdvisoryMessage(
    val captureMonoNs: Long,
    val recSpeedMps: Double,
    val recSpeedDisplay: Double,
    val currentSpeedDisplay: Double,
    val units: String,
    val headwayTargetS: Double,
    val laneText: String,
    val mergeText: String,
    val trafficText: String,
    val confidence: Double,
    val confidenceLabel: String,
    val action: Map<String, String>,
) {
    fun toExtensions(): Map<String, JsonValue> = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
        "rec_speed_mps" to Fields.toWire(recSpeedMps),
        "rec_speed_display" to Fields.toWire(recSpeedDisplay),
        "current_speed_display" to Fields.toWire(currentSpeedDisplay),
        "units" to JsonValue.Text(units),
        "headway_target_s" to Fields.toWire(headwayTargetS),
        "lane_text" to JsonValue.Text(laneText),
        "merge_text" to JsonValue.Text(mergeText),
        "traffic_text" to JsonValue.Text(trafficText),
        "confidence" to Fields.toWire(confidence),
        "confidence_label" to JsonValue.Text(confidenceLabel),
        "action" to Fields.stringsOf(ACTION_HEADS.associateWith { action.getValue(it) }),
    )

    companion object {
        val ACTION_HEADS = listOf(
            "desired_speed_bin",
            "desired_headway_bin",
            "lane_preference",
            "merge_mode",
        )
        val ACTION_VALUES = mapOf(
            "desired_speed_bin" to setOf("slow", "nominal", "fast"),
            "desired_headway_bin" to setOf("normal", "larger", "largest"),
            "lane_preference" to setOf("keep", "prefer_left_if_safe", "prefer_right_if_safe"),
            "merge_mode" to setOf("normal", "create_gap", "hold_lane"),
        )
        val DISPLAY_UNITS = setOf("mph", "kmh", "mps")

        fun fromWire(extensions: Map<String, JsonValue>, payload: ByteArray): AdvisoryMessage {
            Fields.checkNoPayload(payload, Channels.ADVISORY)

            val units = Fields.requireString(extensions, "units")
            if (units !in DISPLAY_UNITS) {
                throw MessageError(RefusalReason.UNKNOWN_VALUE, "units '$units' not one of $DISPLAY_UNITS")
            }

            val nested = Fields.requireObject(extensions, "action")
            val missing = ACTION_HEADS.filter { it !in nested }
            if (missing.isNotEmpty()) {
                throw MessageError(RefusalReason.MISSING_FIELD, "action missing ${missing.joinToString(", ")}")
            }
            val unexpected = nested.keys.filter { it !in ACTION_HEADS }
            if (unexpected.isNotEmpty()) {
                throw MessageError(
                    RefusalReason.UNKNOWN_VALUE,
                    "action has unexpected ${unexpected.sorted().joinToString(", ")}",
                )
            }
            val action = ACTION_HEADS.associateWith { head ->
                // unknown_value even for the wrong JSON type. The heads are a closed set and
                // the spec gives that its own row; an integer here is outside the set as
                // surely as a misspelled string is, and Python reports it that way. Filing
                // it as wrong_type was correct about the type and wrong about the cause the
                // reader needs.
                val value = (nested.getValue(head) as? JsonValue.Text)?.value
                    ?: throw MessageError(
                        RefusalReason.UNKNOWN_VALUE,
                        "action.$head is ${nested.getValue(head)}, not one of " +
                            "${ACTION_VALUES.getValue(head)}",
                    )
                if (value !in ACTION_VALUES.getValue(head)) {
                    // A value outside a closed set, exactly like `units`, which the spec
                    // gives an adjacent refusal row.
                    throw MessageError(
                        RefusalReason.UNKNOWN_VALUE,
                        "action.$head is '$value', not one of ${ACTION_VALUES.getValue(head)}",
                    )
                }
                value
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
                headwayTargetS = Fields.checkFinite(
                    "headway_target_s",
                    Fields.requireNumber(extensions, "headway_target_s"),
                )!!,
                laneText = Fields.requireString(extensions, "lane_text"),
                mergeText = Fields.requireString(extensions, "merge_text"),
                trafficText = Fields.requireString(extensions, "traffic_text"),
                confidence = Fields.checkFinite("confidence", Fields.requireNumber(extensions, "confidence"))!!,
                confidenceLabel = Fields.requireString(extensions, "confidence_label"),
                action = action,
            )
        }
    }
}
