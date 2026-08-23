package com.dsrc.phone

import java.util.concurrent.CopyOnWriteArrayList

/**
 * The one place the current sensing state is published.
 *
 * The service owns the lifecycle and the UI only reads it. Sharing a holder rather
 * than letting the Activity drive the service keeps them from disagreeing across a
 * rotation, when the Activity is destroyed and rebuilt while sensing carries on.
 *
 * A new listener is handed the current state immediately, so a UI attaching after
 * sensing already started does not sit blank until the next change.
 */
class SensingStatus(initial: SensingState = SensingState.IDLE) {

    fun interface Listener {
        fun onState(state: SensingState)
    }

    @Volatile
    var state: SensingState = initial
        private set

    private val listeners = CopyOnWriteArrayList<Listener>()

    fun set(next: SensingState) {
        if (next == state) return
        state = next
        listeners.forEach { it.onState(next) }
    }

    fun addListener(listener: Listener) {
        listeners.add(listener)
        listener.onState(state)
    }

    fun removeListener(listener: Listener) {
        listeners.remove(listener)
    }

    val listenerCount: Int get() = listeners.size

    companion object {
        /** Shared between the service and whatever is displaying it. */
        val shared = SensingStatus()
    }
}
