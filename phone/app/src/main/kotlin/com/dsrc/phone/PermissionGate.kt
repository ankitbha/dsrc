package com.dsrc.phone

/**
 * Which required permissions are missing right now.
 *
 * Split out from the service so the check itself is testable: the service consults it
 * immediately before going to the foreground, because a permission can be revoked
 * between the Activity's check and the service actually starting, and `startForeground`
 * with a camera type and no camera permission is killed by the platform.
 */
object PermissionGate {

    fun missing(required: List<String>, isGranted: (String) -> Boolean): List<String> =
        required.filterNot(isGranted).sorted()
}
