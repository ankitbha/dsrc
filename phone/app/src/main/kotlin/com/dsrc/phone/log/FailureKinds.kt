package com.dsrc.phone.log

/**
 * Every failure this phone can report into its own session log, independent of
 * the link.
 *
 * Each of these reaches, before this file existed, exactly one place: a logcat
 * line at teardown. None of them can go over the wire -- the phone's own log is
 * how they are recorded, because the failure that matters most is the link
 * being down, and that is precisely the condition none of these can be sent
 * during.
 *
 * Closed, and checked against the Python registry's phone-device rows by
 * `InteropTest`: a kind added on one side and not the other fails that test
 * rather than drifting silently.
 */
object FailureKinds {
    const val LINK_DIAL_FAILED = "link.dial_failed"
    const val LINK_SESSION_ENDED = "link.session_ended"
    const val IMU_NO_HARDWARE = "imu.no_hardware"
    const val IMU_TIMEBASE_MISMATCHED = "imu.timebase_mismatched"
    const val HERE_UNCONFIGURED = "here.unconfigured"
    const val SERVICE_COME_UP_FAILED = "service.come_up_failed"
    const val SERVICE_PERMISSION_REVOKED = "service.permission_revoked"
    const val SERVICE_TEARDOWN_FAILED = "service.teardown_failed"
    const val SERVICE_RESOURCES_HELD = "service.resources_held"
    const val LOG_SELF = "log.self"

    val ALL: Set<String> = setOf(
        LINK_DIAL_FAILED, LINK_SESSION_ENDED, IMU_NO_HARDWARE, IMU_TIMEBASE_MISMATCHED,
        HERE_UNCONFIGURED, SERVICE_COME_UP_FAILED, SERVICE_PERMISSION_REVOKED,
        SERVICE_TEARDOWN_FAILED, SERVICE_RESOURCES_HELD, LOG_SELF,
    )
}
