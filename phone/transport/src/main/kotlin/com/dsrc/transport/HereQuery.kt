package com.dsrc.transport

/**
 * The shape of the HERE traffic query, chosen by the Jetson.
 *
 * The phone composes a URL from this and reads nothing else in it. The spec is explicit
 * that "the HERE query shape and location-referencing mode" is set upstream, because the
 * state that would justify choosing is all on the Jetson — a phone picking its own corridor
 * is a phone originating a sensing decision.
 *
 * [locationRef] is HERE's `locationReferencing` parameter and [in] is its `in` parameter,
 * both passed through verbatim. Neither is parsed here: validating HERE's own query grammar
 * on the phone would mean tracking their API, and getting it wrong would refuse a query the
 * Jetson meant.
 */
data class HereQuery(
    /** HERE's `in` parameter — a corridor, a circle, or whatever else it accepts. */
    val `in`: String,
    /** HERE's `locationReferencing` parameter. */
    val locationRef: String,
) {
    fun toJson(): JsonValue = JsonValue.Obj(
        mapOf(
            "in" to JsonValue.Text(`in`),
            "location_ref" to JsonValue.Text(locationRef),
        )
    )

    companion object {
        /**
         * Decode the optional `here` object.
         *
         * Absent is null — "this command does not change the query" — which is why the
         * field can be added without a flag day. Present but malformed is a refusal: a
         * command that names a query and gets it wrong is not the same as one that says
         * nothing, and silently ignoring it would leave the phone querying yesterday's
         * corridor while the Jetson believed it had moved.
         */
        fun fromWire(value: JsonValue?): HereQuery? {
            if (value == null || value is JsonValue.Null) return null
            if (value !is JsonValue.Obj) {
                throw MessageError(RefusalReason.WRONG_TYPE, "'here' is not an object")
            }
            return HereQuery(
                `in` = requireText(value.entries, "in"),
                locationRef = requireText(value.entries, "location_ref"),
            )
        }

        private fun requireText(entries: Map<String, JsonValue>, key: String): String {
            val value = entries[key]
                ?: throw MessageError(RefusalReason.MISSING_FIELD, "'here.$key' is missing")
            if (value is JsonValue.Null) {
                throw MessageError(RefusalReason.NULL_NOT_ALLOWED, "'here.$key' is null")
            }
            if (value !is JsonValue.Text) {
                throw MessageError(RefusalReason.WRONG_TYPE, "'here.$key' is not a string")
            }
            if (value.value.isBlank()) {
                throw MessageError(RefusalReason.OUT_OF_RANGE, "'here.$key' is blank")
            }
            return value.value
        }
    }
}
