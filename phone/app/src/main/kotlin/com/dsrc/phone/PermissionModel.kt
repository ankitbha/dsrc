package com.dsrc.phone

/**
 * What the app knows about one runtime permission.
 *
 * `NEVER_ASKED` and `DENIED_PERMANENTLY` exist as separate states because Android
 * cannot tell them apart on its own: `shouldShowRequestPermissionRationale` returns
 * false for both. Only the app's own record of having asked separates them, and
 * getting it wrong means either nagging a user who said no twice, or sending a
 * first-time user to Settings for a permission they were never offered.
 */
enum class PermissionState {
    GRANTED,
    NEVER_ASKED,
    DENIED_CAN_ASK,
    DENIED_PERMANENTLY,
}

/** What the app should do next about permissions. */
sealed interface PermissionAction {
    /** Ask the system, no explanation needed yet. */
    data class Request(val permissions: List<String>) : PermissionAction

    /** Explain first: these were denied once and can still be asked again. */
    data class Rationale(val permissions: List<String>) : PermissionAction

    /** Asking is now a no-op; only Settings can grant these. */
    data class OpenSettings(val permissions: List<String>) : PermissionAction

    /** Everything required is granted. */
    data object Proceed : PermissionAction
}

/**
 * The permission flow as pure logic, so its awkward cases are testable off-device.
 */
object PermissionModel {

    const val CAMERA = "android.permission.CAMERA"
    const val FINE_LOCATION = "android.permission.ACCESS_FINE_LOCATION"
    const val POST_NOTIFICATIONS = "android.permission.POST_NOTIFICATIONS"

    /** API level at which notifications became a runtime permission. */
    const val SDK_POST_NOTIFICATIONS = 33

    /**
     * The permissions sensing cannot start without, for a given platform level.
     *
     * Notifications are required rather than optional: the service runs in the
     * foreground, and a foreground service the user cannot see is the thing the
     * platform is trying to prevent.
     */
    fun required(sdkInt: Int): List<String> = buildList {
        add(CAMERA)
        add(FINE_LOCATION)
        if (sdkInt >= SDK_POST_NOTIFICATIONS) add(POST_NOTIFICATIONS)
    }

    /**
     * Classify one permission from the two platform signals plus our own record.
     *
     * `shouldShowRationale` is only meaningful when the permission is not granted;
     * the platform's value for a granted permission is unspecified, so it is ignored
     * rather than trusted.
     */
    fun classify(granted: Boolean, shouldShowRationale: Boolean, hasAsked: Boolean): PermissionState =
        when {
            granted -> PermissionState.GRANTED
            shouldShowRationale -> PermissionState.DENIED_CAN_ASK
            hasAsked -> PermissionState.DENIED_PERMANENTLY
            else -> PermissionState.NEVER_ASKED
        }

    /**
     * The next action, given every required permission's state.
     *
     * Order matters. A permanent denial outranks a rationale because asking again
     * does nothing at all, so a screen that offered "allow" would be a dead end.
     */
    fun next(states: Map<String, PermissionState>): PermissionAction {
        val permanent = states.filterValues { it == PermissionState.DENIED_PERMANENTLY }.keys.sorted()
        if (permanent.isNotEmpty()) return PermissionAction.OpenSettings(permanent)

        val rationale = states.filterValues { it == PermissionState.DENIED_CAN_ASK }.keys.sorted()
        if (rationale.isNotEmpty()) return PermissionAction.Rationale(rationale)

        val unasked = states.filterValues { it == PermissionState.NEVER_ASKED }.keys.sorted()
        if (unasked.isNotEmpty()) return PermissionAction.Request(unasked)

        return PermissionAction.Proceed
    }
}
