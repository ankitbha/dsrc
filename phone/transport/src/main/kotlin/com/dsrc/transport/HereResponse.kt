package com.dsrc.transport

/**
 * One HERE traffic reply, with the request that produced it and both clocks around it.
 *
 * The body rides in the payload untouched. `t_request_mono_ns` and `t_response_mono_ns`
 * bracket the call, so a receiver can tell a slow road from a slow API without guessing.
 */
data class HereResponse(
    val captureMonoNs: Long,
    val requestUrl: String,
    val status: Long,
    val contentType: String?,
    val queryLat: Double,
    val queryLon: Double,
    val queryRadiusM: Double,
    val requestMonoNs: Long,
    val responseMonoNs: Long,
) {
    fun toExtensions(): Map<String, JsonValue> = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
        "request_url" to JsonValue.Text(requestUrl),
        "status" to JsonValue.Num(status),
        "content_type" to Fields.toWire(contentType),
        "query_lat" to Fields.toWire(queryLat),
        "query_lon" to Fields.toWire(queryLon),
        "query_radius_m" to Fields.toWire(queryRadiusM),
        "t_request_mono_ns" to JsonValue.Num(requestMonoNs),
        "t_response_mono_ns" to JsonValue.Num(responseMonoNs),
    )

    companion object {
        fun fromWire(extensions: Map<String, JsonValue>, payload: ByteArray): HereResponse {
            Fields.checkReserved(extensions)
            // The payload is the response body and is not validated: an empty one is a
            // failed call worth seeing rather than one the transport hides.
            return HereResponse(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                requestUrl = Fields.requireString(extensions, "request_url"),
                status = Fields.requireInt(extensions, "status"),
                contentType = Fields.optionalString(extensions, "content_type"),
                queryLat = Fields.checkFinite("query_lat", Fields.requireNumber(extensions, "query_lat"))!!,
                queryLon = Fields.checkFinite("query_lon", Fields.requireNumber(extensions, "query_lon"))!!,
                queryRadiusM = Fields.checkFinite(
                    "query_radius_m",
                    Fields.requireNumber(extensions, "query_radius_m"),
                )!!,
                requestMonoNs = Fields.requireInt(extensions, "t_request_mono_ns"),
                responseMonoNs = Fields.requireInt(extensions, "t_response_mono_ns"),
            )
        }
    }
}
