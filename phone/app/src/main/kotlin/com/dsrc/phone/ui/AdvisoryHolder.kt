package com.dsrc.phone.ui

import com.dsrc.transport.AdvisoryMessage

/**
 * The advisory the driver is being shown, and when it stops counting.
 *
 * The transport already keeps only the newest — `advisory` is `latest_wins` at depth one,
 * and the spec's reason is written in the table: "a stale advisory is useless". But that
 * only governs what is queued. Once the Jetson stops sending, or the link drops, the last
 * advisory sits on the screen indefinitely with nothing to displace it, and a recommendation
 * about road the driver has already passed is worse than a blank panel: it looks current.
 *
 * So the holder expires on *arrival* time, measured locally. Not on `t_capture_mono_ns`,
 * which is the Jetson's clock — relating the two takes the timebase exchange, and a display
 * that goes blank because a clock estimate wandered would be a fault invented by its own
 * safety check. What matters here is how long since anything was heard, and that is a local
 * question.
 */
class AdvisoryHolder(
    private val maxAgeNs: Long = MAX_AGE_NS,
) {
    private val lock = Any()

    private var latest: AdvisoryMessage? = null
    private var arrivedAtNs = 0L
    private var received = 0L
    private var expired = 0L
    private var afterStop = 0L

    /**
     * Whether advisories are being taken at all.
     *
     * Stopping does not join the delivery thread -- `SessionHolder.stop()` closes the
     * session and interrupts, and `Session.finish()` joins nothing -- so a frame already
     * off the inbound queue is delivered to completion. A delivery thread descheduled
     * between the dequeue and `accept` therefore resumes *after* teardown has cleared the
     * holder and refills it, putting a full recommendation in front of a driver who has
     * pressed Stop, for up to the whole expiry. Clearing cannot fix that on its own,
     * because the clear has already happened; the holder has to refuse.
     */
    private var accepting = true

    /** Take a newly arrived advisory. Replaces any previous one, however recent. */
    fun accept(advisory: AdvisoryMessage, nowNs: Long) = synchronized(lock) {
        if (!accepting) {
            // A late delivery from a session that has ended. Counted, because "the panel
            // went blank" and "an advisory arrived after the stop and was refused" are
            // different facts about a drive.
            afterStop++
            return@synchronized
        }
        latest = advisory
        arrivedAtNs = nowNs
        received++
    }

    /** Begin taking advisories. Called when sensing comes up. */
    fun start() = synchronized(lock) {
        accepting = true
        latest = null
    }

    /**
     * The advisory to show, or null when there is nothing current.
     *
     * Counting an expiry here rather than at a timer means the count moves only when
     * someone actually asks — which is the only moment it could have been shown.
     */
    fun current(nowNs: Long): AdvisoryMessage? = synchronized(lock) {
        val held = latest ?: return null
        if (nowNs - arrivedAtNs > maxAgeNs) {
            latest = null
            expired++
            return null
        }
        return held
    }

    /**
     * Drop whatever is held.
     *
     * Called when sensing stops. A driver who stopped the session is not being advised,
     * and leaving the last recommendation up would say otherwise.
     */
    fun clear() = synchronized(lock) {
        accepting = false
        latest = null
    }

    val stats: Stats
        get() = synchronized(lock) {
            Stats(
                received = received,
                expired = expired,
                showing = latest != null,
                afterStop = afterStop,
            )
        }

    data class Stats(
        val received: Long,
        val expired: Long,
        val showing: Boolean,
        /** Advisories that arrived after the session ended, and were refused. */
        val afterStop: Long,
    )

    companion object {
        /**
         * How long an advisory stays current.
         *
         * Three seconds, which at 50 km/h is about 40 m of road. Past that the
         * recommendation is about ground the driver has covered, and the panel should say
         * nothing rather than something that was true a moment ago. The spec does not fix
         * this — it fixes only that the queue holds one — so it is a choice, recorded here.
         */
        const val MAX_AGE_NS = 3_000_000_000L
    }
}
