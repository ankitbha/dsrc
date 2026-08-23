package com.dsrc.phone.sensors

/**
 * Something that produces camera frames into a [CameraPipeline].
 *
 * An interface so the pipeline can be driven on a laptop. The real adapter needs a
 * camera; the capture *policy* -- rate, ids, counters, drop accounting -- does not,
 * and that is where the bugs are.
 */
interface CameraSource {
    fun start(pipeline: CameraPipeline)
    fun stop()
}
