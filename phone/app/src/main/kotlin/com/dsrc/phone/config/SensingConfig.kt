package com.dsrc.phone.config

/**
 * Every sensing setting, in one place.
 *
 * The Jetson owns all of these -- see the "Configuration flows one way" section of
 * `specs/transport_protocol.md`. It cannot yet *say* so for anything but a rate,
 * because `rate_cmd` carries only the four frequencies, so this class is the
 * stand-in: shaped like the object the widened downlink will carry, so wiring it up
 * later is a substitution rather than a restructure.
 *
 * Validated on construction. An invalid setting has to be refused where it enters,
 * not where it is used -- a zero rate is read as a period and means "never", which
 * would surface as a camera that silently produces nothing.
 */
data class SensingConfig(
    val cameraHz: Double = 5.0,
    val gpsHz: Double = 1.0,
    val imuHz: Double = 50.0,
    val hereHz: Double = 0.2,
    val cameraWidth: Int = 1280,
    val cameraHeight: Int = 720,
    val jpegQuality: Int = 85,
) {
    init {
        for ((name, hz) in listOf("cameraHz" to cameraHz, "gpsHz" to gpsHz, "imuHz" to imuHz, "hereHz" to hereHz)) {
            require(hz.isFinite() && hz > 0.0 && hz <= MAX_HZ) { "$name is $hz, outside (0, $MAX_HZ] Hz" }
        }
        require(cameraWidth > 0 && cameraHeight > 0) { "camera size is ${cameraWidth}x$cameraHeight" }
        require(cameraWidth % 2 == 0 && cameraHeight % 2 == 0) {
            "4:2:0 chroma needs even dimensions, got ${cameraWidth}x$cameraHeight"
        }
        require(jpegQuality in 1..100) { "jpegQuality is $jpegQuality, outside 1..100" }
    }

    companion object {
        /** The wire's ceiling for a rate. */
        const val MAX_HZ = 1000.0
    }
}
