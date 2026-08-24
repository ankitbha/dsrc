package com.dsrc.phone.config

import com.dsrc.transport.RateCommand

/**
 * Applies a `rate_cmd` to the running modalities, without restarting capture.
 *
 * The phone originates no sensing decision of its own — it applies what arrives and
 * reports what it achieved. This is the "applies what arrives" half.
 *
 * Nothing here validates the command. `RateCommand.fromWire` has already refused a rate
 * outside `(0, 1000]`, a malformed `here` object and every missing field, and the transport
 * refused the frame before this was called. A zero rate in particular never reaches here,
 * which matters: it would be applied as a period, so a field that should have said "10 Hz"
 * would say "never", and the phone would stop sensing and look healthy doing it.
 */
class ConfigApplier(
    private val targets: Targets,
) {
    /** What a command can reach. Each modality applies its own rate; none is restarted. */
    interface Targets {
        fun setCameraRate(hz: Double)
        fun setGpsRate(hz: Double)
        fun setImuRate(hz: Double)
        fun setHereRate(hz: Double)
        fun setHereQuery(query: com.dsrc.transport.HereQuery?)
    }

    private val lock = Any()

    private var applied = 0L
    private var shadowed = 0L

    @Volatile
    private var lastTrigger: String? = null

    @Volatile
    private var current: RateCommand? = null

    /**
     * The query in force, which is not the last command's.
     *
     * A command that omits `here` means "no change", so deriving this from the last command
     * reported no query configured while HERE was still querying the one before it.
     */
    @Volatile
    private var currentQuery: com.dsrc.transport.HereQuery? = null

    /**
     * Apply one command, or record it without applying.
     *
     * `shadow` is the Jetson asking what *would* happen: the spec defines it as whether the
     * command "was gated for real or only recorded". A shadow command that changed a rate
     * would make the comparison it exists for meaningless, so it changes nothing at all —
     * not the rates, not the query, and not [current], which is what the phone is actually
     * running.
     */
    fun apply(command: RateCommand) {
        synchronized(lock) {
            lastTrigger = command.trigger
            if (command.shadow) {
                shadowed++
                return
            }
            applied++
            current = command
            command.here?.let { currentQuery = it }
        }

        // Outside the lock: these reach into the pipelines, and holding a lock across a
        // modality's own synchronisation is how lock cycles get built. Ordering between two
        // commands is preserved by the delivery thread, which is single-threaded.
        with(targets) {
            setCameraRate(command.rates.getValue("camera_hz"))
            setGpsRate(command.rates.getValue("gps_hz"))
            setImuRate(command.rates.getValue("imu_hz"))
            setHereRate(command.rates.getValue("here_hz"))
            // Null means "this command does not change the query", which is what makes the
            // field optional. Passing it through rather than skipping the call keeps the
            // decision in one place.
            setHereQuery(command.here)
        }
    }

    val stats: Stats
        get() = synchronized(lock) {
            Stats(
                applied = applied,
                shadowed = shadowed,
                lastTrigger = lastTrigger,
                currentRates = current?.rates ?: emptyMap(),
                hereConfigured = currentQuery != null,
            )
        }

    data class Stats(
        val applied: Long,
        /** Commands recorded but deliberately not acted on. */
        val shadowed: Long,
        val lastTrigger: String?,
        /** The rates in force, empty until a real command has arrived. */
        val currentRates: Map<String, Double>,
        val hereConfigured: Boolean,
    )
}
