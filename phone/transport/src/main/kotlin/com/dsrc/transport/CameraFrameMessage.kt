package com.dsrc.transport

/**
 * One JPEG frame, header fields only -- the image itself rides in the payload.
 *
 * Mirrors `CameraFrame` in `deployment/jetson/transport/messages.py` field for field.
 * Its absence was a hole in the sender rule on the busiest channel: `camera` was listed
 * as having no typed decoder, so a frame's own fields were never checked outbound, and a
 * negative width or a missing `frame_id` would have travelled and come back as the peer's
 * drop counter.
 */
data class CameraFrameMessage(
    val captureMonoNs: Long,
    val frameId: Long,
    val width: Long,
    val height: Long,
    val format: String,
    /** Present-and-null for a format that has no quality setting. */
    val quality: Long?,
    /**
     * When the phone's own encoder started and finished turning the packed pixels into
     * the JPEG payload, on the same clock as [captureMonoNs] and the header's own
     * `t_mono_ns` -- so every phone-side duration between capture and the wire is a
     * plain subtraction, exact, with no timebase involved.
     *
     * Absent-tolerant rather than merely nullable: added to a channel that already
     * ships, so a build from before task 33 does not write them at all, and requiring
     * them would refuse every frame from that build.
     */
    val encodeStartMonoNs: Long? = null,
    val encodeDoneMonoNs: Long? = null,
) {
    fun toExtensions(): Map<String, JsonValue> {
        val base = mapOf(
            Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
            KEY_FRAME_ID to JsonValue.Num(frameId),
            KEY_WIDTH to JsonValue.Num(width),
            KEY_HEIGHT to JsonValue.Num(height),
            KEY_FORMAT to JsonValue.Text(format),
            KEY_QUALITY to Fields.toWire(quality),
        )
        val encode = buildMap<String, JsonValue> {
            if (encodeStartMonoNs != null) put(KEY_ENCODE_START, JsonValue.Num(encodeStartMonoNs))
            if (encodeDoneMonoNs != null) put(KEY_ENCODE_DONE, JsonValue.Num(encodeDoneMonoNs))
        }
        return base + encode
    }

    companion object {
        const val KEY_FRAME_ID = "frame_id"
        const val KEY_WIDTH = "width"
        const val KEY_HEIGHT = "height"
        const val KEY_FORMAT = "format"
        const val KEY_QUALITY = "quality"
        const val KEY_ENCODE_START = "t_encode_start_mono_ns"
        const val KEY_ENCODE_DONE = "t_encode_done_mono_ns"

        /**
         * @param payload the JPEG, which is not a header field and is deliberately not
         *   validated: a zero-length image is a defect worth seeing on the wire rather
         *   than one the transport hides by refusing the message. The parameter exists so
         *   every channel's decoder has the same shape and the payload rule is stated
         *   somewhere rather than being absent by omission.
         */
        fun fromWire(extensions: Map<String, JsonValue>, @Suppress("UNUSED_PARAMETER") payload: ByteArray): CameraFrameMessage {

            // No range rules here, deliberately. Earlier versions refused a zero dimension,
            // a quality outside 1..100, an empty format and a negative frame_id -- none of
            // which the spec's message table or refusal table mentions, and all of which
            // Python accepts. A unilateral receiver rule refuses what the peer legitimately
            // sends, which is the dangerous direction of a cross-language disagreement, and
            // the two implementations disagreeing about whether a record is acceptable is
            // worse than either answer.
            //
            // A bad *setting* still dies where settings enter: SensingConfig refuses a
            // quality outside 1..100 and a zero or odd dimension on construction. That is
            // the right place for a rule this contract does not carry.
            val width = Fields.requireInt(extensions, KEY_WIDTH)
            val height = Fields.requireInt(extensions, KEY_HEIGHT)
            val quality = Fields.optionalInt(extensions, KEY_QUALITY)
            val format = Fields.requireString(extensions, KEY_FORMAT)

            return CameraFrameMessage(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                frameId = Fields.requireInt(extensions, KEY_FRAME_ID),
                width = width,
                height = height,
                format = format,
                quality = quality,
                encodeStartMonoNs = Fields.absentableInt(extensions, KEY_ENCODE_START),
                encodeDoneMonoNs = Fields.absentableInt(extensions, KEY_ENCODE_DONE),
            )
        }
    }
}
