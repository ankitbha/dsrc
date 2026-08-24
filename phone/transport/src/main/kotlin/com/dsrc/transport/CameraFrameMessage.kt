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
) {
    fun toExtensions(): Map<String, JsonValue> = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
        KEY_FRAME_ID to JsonValue.Num(frameId),
        KEY_WIDTH to JsonValue.Num(width),
        KEY_HEIGHT to JsonValue.Num(height),
        KEY_FORMAT to JsonValue.Text(format),
        KEY_QUALITY to Fields.toWire(quality),
    )

    companion object {
        const val KEY_FRAME_ID = "frame_id"
        const val KEY_WIDTH = "width"
        const val KEY_HEIGHT = "height"
        const val KEY_FORMAT = "format"
        const val KEY_QUALITY = "quality"

        /**
         * @param payload the JPEG, which is not a header field and is deliberately not
         *   validated: a zero-length image is a defect worth seeing on the wire rather
         *   than one the transport hides by refusing the message. The parameter exists so
         *   every channel's decoder has the same shape and the payload rule is stated
         *   somewhere rather than being absent by omission.
         */
        fun fromWire(extensions: Map<String, JsonValue>, @Suppress("UNUSED_PARAMETER") payload: ByteArray): CameraFrameMessage {

            val width = Fields.requireCount(extensions, KEY_WIDTH)
            val height = Fields.requireCount(extensions, KEY_HEIGHT)
            if (width == 0L || height == 0L) {
                throw MessageError(RefusalReason.OUT_OF_RANGE, "frame is ${width}x$height")
            }

            val quality = Fields.optionalInt(extensions, KEY_QUALITY)
            if (quality != null && quality !in 1..100) {
                throw MessageError(RefusalReason.OUT_OF_RANGE, "quality is $quality, outside 1..100")
            }

            val format = Fields.requireString(extensions, KEY_FORMAT)
            if (format.isEmpty()) {
                throw MessageError(RefusalReason.UNKNOWN_VALUE, "format is empty")
            }

            return CameraFrameMessage(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                frameId = Fields.requireCount(extensions, KEY_FRAME_ID),
                width = width,
                height = height,
                format = format,
                quality = quality,
            )
        }
    }
}
