package com.dsrc.phone

/** Where the sensing lifecycle is. One service owns all four modalities. */
enum class SensingState {
    IDLE,
    STARTING,
    RUNNING,
    STOPPING,
    STOPPED_PERMISSION_REVOKED,
    STOPPED_ERROR,
    ;

    /** Whether sensing is meant to be producing data right now. */
    val isActive: Boolean get() = this == STARTING || this == RUNNING
}

sealed interface SensingEvent {
    data object Start : SensingEvent
    data object Started : SensingEvent
    data object Stop : SensingEvent
    data object Stopped : SensingEvent
    data object PermissionRevoked : SensingEvent
    data class Failed(val reason: String) : SensingEvent
}

/**
 * The result of offering an event to the machine.
 *
 * An event that does not apply is [Ignored] rather than an exception. A service is
 * driven by intents it does not control -- a redelivered start, a stop racing a
 * crash -- and crashing on an unexpected one would turn a harmless duplicate into a
 * lost session. Ignored events are counted so a duplicate stays visible.
 */
sealed interface Transition {
    data class Accepted(val from: SensingState, val to: SensingState) : Transition
    data class Ignored(val state: SensingState, val event: SensingEvent) : Transition
}

/**
 * Sensing lifecycle as pure logic.
 *
 * Deliberately holds no Android type, so every awkward path -- a revoke during
 * startup, a stop arriving twice, a restart after failure -- is a unit test rather
 * than something only reproducible on a handset.
 */
class SensingStateMachine(initial: SensingState = SensingState.IDLE) {

    var state: SensingState = initial
        private set

    var ignoredEvents: Int = 0
        private set

    /** Why the machine last landed in [SensingState.STOPPED_ERROR], if it did. */
    var lastFailure: String? = null
        private set

    fun offer(event: SensingEvent): Transition {
        val from = state
        val to = nextState(from, event)
        if (to == null) {
            ignoredEvents++
            return Transition.Ignored(from, event)
        }
        if (event is SensingEvent.Failed) lastFailure = event.reason
        state = to
        return Transition.Accepted(from, to)
    }

    private fun nextState(from: SensingState, event: SensingEvent): SensingState? = when (event) {
        // Start is idempotent while already coming up or up: a redelivered intent
        // must not open a second session.
        SensingEvent.Start -> when (from) {
            SensingState.IDLE,
            SensingState.STOPPED_PERMISSION_REVOKED,
            SensingState.STOPPED_ERROR,
            -> SensingState.STARTING
            SensingState.STARTING, SensingState.RUNNING -> null
            // A start racing a shutdown is dropped rather than queued; the caller
            // sees the state and can ask again once it reaches IDLE.
            SensingState.STOPPING -> null
        }

        SensingEvent.Started -> if (from == SensingState.STARTING) SensingState.RUNNING else null

        // Stop is accepted from anything that is up or coming up, and is a no-op
        // everywhere else.
        SensingEvent.Stop -> if (from.isActive) SensingState.STOPPING else null

        SensingEvent.Stopped -> if (from == SensingState.STOPPING) SensingState.IDLE else null

        // A revoke matters whenever sensing is up; it is not an error, and it is
        // distinct from one, because the remedy is a permission grant.
        SensingEvent.PermissionRevoked ->
            if (from.isActive) SensingState.STOPPED_PERMISSION_REVOKED else null

        // A failure can arrive while stopping too -- shutdown itself can fail -- so
        // it is accepted from any state that is not already terminal.
        is SensingEvent.Failed -> when (from) {
            SensingState.STARTING, SensingState.RUNNING, SensingState.STOPPING -> SensingState.STOPPED_ERROR
            SensingState.IDLE,
            SensingState.STOPPED_PERMISSION_REVOKED,
            SensingState.STOPPED_ERROR,
            -> null
        }
    }
}
