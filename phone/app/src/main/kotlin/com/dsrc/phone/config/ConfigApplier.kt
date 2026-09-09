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

    /** One HERE query a minute while shadowing, whatever the command asked for.
     *
     *  Constant on purpose: in shadow the controller's rate is a recorded decision
     *  rather than an instruction, so honouring it would make the fetch cadence
     *  depend on a policy that is not in force. A fixed rate gives every shadow
     *  drive the same, comparable feed sampling. */
    private val SHADOW_HERE_HZ = 1.0 / 60.0

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
        val shadow: Boolean
        synchronized(lock) {
            lastTrigger = command.trigger
            shadow = command.shadow
            if (shadow) {
                shadowed++
            } else {
                applied++
                current = command
                command.here?.let { currentQuery = it }
            }
        }

        if (shadow) {
            // HERE is the one exception to "shadow applies nothing", decided on
            // 2026-09-08. Issuing a query IS an action -- it spends cellular data and
            // paid quota -- so the COMMANDED rate is not honoured. But a shadow drive
            // that fetches nothing cannot answer the two questions section I asks of
            // every drive, HERE-reported speed against experienced speed and feed lag,
            // and before this every shadow run in the project recorded here_calls 0.
            //
            // So a fixed, slow rate is applied and nothing else is. The camera, GPS and
            // IMU rates stay unapplied, which is what makes the drive a shadow drive:
            // the trajectory is unchanged and the controller's decisions are still only
            // recorded. The Jetson's own record is untouched -- these commands are still
            // marked shadow, and `shadowed` still counts them.
            //
            // Outside the lock for the same reason the live path is: holding one across
            // a pipeline's own synchronisation is how lock cycles get built.
            with(targets) {
                setHereRate(SHADOW_HERE_HZ)
                setHereQuery(command.here)
            }
            return
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
