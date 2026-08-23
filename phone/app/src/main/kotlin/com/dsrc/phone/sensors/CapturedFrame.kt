package com.dsrc.phone.sensors

/**
 * One encoded frame, in the shape the `camera` channel will carry.
 *
 * Field names match `specs/transport_golden_frames.json`'s `message_camera` case so
 * the hand-off in task 19 is a rename of nothing.
 */
data class CapturedFrame(
    val frameId: Long,
    val width: Int,
    val height: Int,
    val format: String,
    /** JPEG quality 1..100, or null when the encoder does not report one. */
    val quality: Int?,
    /**
     * When the frame came into being on this device, from `elapsedRealtimeNanos`.
     *
     * The same clock as the transport header's enqueue stamp, which is what makes
     * `t_mono_ns - t_capture_mono_ns` a valid subtraction. Taken in the analyzer
     * callback rather than at the shutter, so it carries the pipeline's own latency --
     * a known bias, and the more accurate sensor timestamp is on a different clock.
     */
    val captureMonoNs: Long,
    val jpeg: ByteArray,
) {
    init {
        require(width > 0 && height > 0) { "frame is ${width}x$height" }
        require(quality == null || quality in 1..100) { "quality $quality outside 1..100" }
    }

    // Generated equals on a data class with a ByteArray compares references, which
    // makes two identical frames unequal and is a silent trap in tests.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CapturedFrame) return false
        return frameId == other.frameId &&
            width == other.width &&
            height == other.height &&
            format == other.format &&
            quality == other.quality &&
            captureMonoNs == other.captureMonoNs &&
            jpeg.contentEquals(other.jpeg)
    }

    override fun hashCode(): Int {
        var result = frameId.hashCode()
        result = 31 * result + width
        result = 31 * result + height
        result = 31 * result + format.hashCode()
        result = 31 * result + (quality ?: 0)
        result = 31 * result + captureMonoNs.hashCode()
        result = 31 * result + jpeg.contentHashCode()
        return result
    }
}
