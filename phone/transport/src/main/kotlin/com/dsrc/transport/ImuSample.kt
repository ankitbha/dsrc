package com.dsrc.transport

/**
 * One IMU sample: three accelerometer axes, three gyroscope axes.
 *
 * Mirrors `ImuSample` in `deployment/jetson/transport/messages.py`. `accuracy` is the
 * platform's own confidence code and is nullable, because not every device reports one.
 */
data class ImuSample(
    val captureMonoNs: Long,
    val ax: Double,
    val ay: Double,
    val az: Double,
    val gx: Double,
    val gy: Double,
    val gz: Double,
    val accuracy: Long?,
) {
    fun toExtensions(): Map<String, JsonValue> = mapOf(
        Fields.CAPTURE_KEY to JsonValue.Num(captureMonoNs),
        "ax" to Fields.toWire(ax),
        "ay" to Fields.toWire(ay),
        "az" to Fields.toWire(az),
        "gx" to Fields.toWire(gx),
        "gy" to Fields.toWire(gy),
        "gz" to Fields.toWire(gz),
        "accuracy" to Fields.toWire(accuracy),
    )

    companion object {
        val AXES = listOf("ax", "ay", "az", "gx", "gy", "gz")

        fun fromWire(extensions: Map<String, JsonValue>, payload: ByteArray): ImuSample {
            Fields.checkNoPayload(payload, Channels.IMU)
            val axes = AXES.associateWith {
                Fields.checkFinite(it, Fields.requireNumber(extensions, it))!!
            }
            return ImuSample(
                captureMonoNs = Fields.requireInt(extensions, Fields.CAPTURE_KEY),
                ax = axes.getValue("ax"),
                ay = axes.getValue("ay"),
                az = axes.getValue("az"),
                gx = axes.getValue("gx"),
                gy = axes.getValue("gy"),
                gz = axes.getValue("gz"),
                accuracy = Fields.optionalInt(extensions, "accuracy"),
            )
        }
    }
}
