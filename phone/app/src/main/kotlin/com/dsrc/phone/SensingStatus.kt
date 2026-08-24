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
        listeners.forEach { notify(it, next) }
    }

    fun addListener(listener: Listener) {
        listeners.add(listener)
        notify(listener, state)
    }

    /**
     * Deliver to one listener, and keep its failure to itself.
     *
     * Unguarded, this had two effects, both worse than the bug it was carrying. The
     * obvious one: listeners run in insertion order, so the first that threw starved
     * every later one, and a UI attached after it stayed on a stale state until the next
     * change -- which for a terminal state never comes.
     *
     * The other is why this is transport-shaped rather than a nicety. `set` is called from
     * the service's own `handle`, inside `react(STARTING)`'s try, *after* come-up has
     * already succeeded. So a throwing listener was caught as a start failure and offered
     * as `Failed` while the machine was RUNNING -- an arm the machine accepts, with no
     * teardown behind it. A bug in a UI callback became a sensing lifecycle event that
     * orphaned a whole set of workers. A listener's exception is the listener's problem.
     */
    private fun notify(listener: Listener, state: SensingState) {
        try {
            listener.onState(state)
        } catch (t: Throwable) {
            listenerFailures.incrementAndGet()
            lastListenerFailure = "${t.javaClass.name}: ${t.message}"
        }
    }

    /** Listener callbacks that threw. Non-zero means some UI is not being told. */
    val listenerFailures = java.util.concurrent.atomic.AtomicInteger(0)

    @Volatile
    var lastListenerFailure: String? = null
        private set

    fun removeListener(listener: Listener) {
        listeners.remove(listener)
    }

    val listenerCount: Int get() = listeners.size

    companion object {
        /** Shared between the service and whatever is displaying it. */
        val shared = SensingStatus()
    }
}
