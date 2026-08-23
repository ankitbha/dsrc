package com.dsrc.phone.sensors

import com.dsrc.transport.GpsRecord

/**
 * One location update as the phone saw it, with both clocks the task asks for.
 *
 * "Fix time and receipt time" are different instants and neither substitutes for the
 * other: [fixMonoNs] is when the fix was made, and [receiptMonoNs] is when we were handed
 * it. The difference is the location stack's own latency, which on a cold start or under
 * load is seconds rather than milliseconds.
 *
 * Only the fix time reaches the wire, as `t_capture_mono_ns` — the frozen contract has no
 * field for receipt, so it goes to the local log instead. Inventing one would be a
 * coordinated Python/Kotlin/golden-vector change (see plan_task19 O1).
 */
data class GpsReading(
    val record: GpsRecord,
    /** When the fix was made, on the same monotonic clock as everything else. */
    val fixMonoNs: Long,
    /** When this process was handed it. Always at or after [fixMonoNs]. */
    val receiptMonoNs: Long,
) {
    /** How long the location stack took to deliver the fix. */
    val deliveryLatencyNs: Long get() = receiptMonoNs - fixMonoNs
}

/**
 * Something that produces GPS readings.
 *
 * An interface for the same reason as the camera: the capture *policy* — rate limiting,
 * counting, the no-fix path — is pure logic and is where the bugs are, while the platform
 * adapter is thin.
 */
interface GpsSource {
    fun start(sink: (GpsReading) -> Unit)
    fun stop()
}

/** A source a test drives directly. */
class FakeGpsSource : GpsSource {
    private var sink: ((GpsReading) -> Unit)? = null

    var started = false
        private set

    var startCount = 0
        private set

    override fun start(sink: (GpsReading) -> Unit) {
        this.sink = sink
        started = true
        startCount++
    }

    override fun stop() {
        sink = null
        started = false
    }

    /** Push a reading, as the platform would. Returns whether anything was listening. */
    fun emit(reading: GpsReading): Boolean {
        val target = sink ?: return false
        target(reading)
        return true
    }
}
