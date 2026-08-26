package com.dsrc.phone.sensors

import android.util.Log
import java.io.File

/**
 * A temperature straight from the kernel, for handsets that will not compute headroom.
 *
 * `getThermalHeadroom` is a *normalised* number: the skin temperature over the threshold at
 * which the device throttles. A phone that publishes no thresholds has no denominator, so
 * the platform returns `NaN` forever and [ThermalReader] reports null — correctly, but the
 * Jetson is then left with only the six-step status, which does not move until the handset
 * is already in trouble. The moto g power this was written against is such a device: its
 * thermal HAL is connected and reporting, and `getThermalHeadroom` still never answers.
 *
 * The zones underneath the HAL are ordinary files, and an app can read them. So this is the
 * fallback: an absolute temperature the far side can trend, where the normalised one is
 * unavailable.
 *
 * **It is best effort and cannot be anything else.** `/sys/class/thermal` is not a public
 * API. Whether an app may read it is an SELinux decision that varies by vendor and platform
 * version, the zones are named by the vendor, and nothing guarantees any particular one
 * exists. Every failure here is a null, never an exception and never a log line per read.
 *
 * **The zone name travels with the reading, and that is not decoration.** Zone names do not
 * mean what they look like. On the device this was built against the HAL's `skin` sensor
 * matched `xo_therm` to within 0.007 °C, while `quiet_therm` -- the name Qualcomm platforms
 * conventionally use for skin -- read 1.2 °C lower and is a different sensor. A bare
 * temperature would therefore mean a different thing on every handset, and comparing two of
 * them would be meaningless. Reported with its zone, the number can be interpreted, and a
 * per-device mapping can be built later from drives rather than guessed at now.
 */
class ThermalZones(private val root: File = File(DEFAULT_ROOT)) {

    /** A temperature and the sensor it came from. Neither is useful without the other. */
    data class Reading(val celsius: Double, val zone: String)

    private var resolved: Zone? = null
    private var searched = false

    private data class Zone(val temperature: File, val name: String)

    /**
     * The current temperature, or null if this device will not give one.
     *
     * The zone is resolved once and then reused. Rescanning would mean stat-ing sixty-odd
     * directories every second to answer a question whose answer cannot change, and a
     * device that has no usable zone would pay that cost forever to keep learning it has
     * none -- so a failed search is remembered too.
     */
    fun read(): Reading? {
        val zone = resolve() ?: return null
        val raw = readTrimmed(zone.temperature) ?: return null
        val celsius = celsiusOf(raw) ?: return null
        return Reading(celsius, zone.name)
    }

    private fun resolve(): Zone? {
        if (searched) return resolved
        searched = true
        resolved = search()
        Log.i(TAG, "thermal zone: ${resolved?.name ?: "none readable"}")
        return resolved
    }

    private fun search(): Zone? {
        val directories = try {
            root.listFiles { file -> file.name.startsWith("thermal_zone") }
        } catch (e: SecurityException) {
            null
        } ?: return null

        val byName = mutableMapOf<String, Zone>()
        for (directory in directories) {
            val name = readTrimmed(File(directory, "type")) ?: continue
            val temperature = File(directory, "temp")
            // Kept only if it reads a plausible value *now*. A zone whose `type` is readable
            // and whose `temp` is not would otherwise be resolved once and then return null
            // for the whole drive, with the search already marked done and no way back.
            if (readTrimmed(temperature)?.let(::celsiusOf) == null) continue
            byName.putIfAbsent(name, Zone(temperature, name))
        }
        for (candidate in PREFERRED) {
            byName[candidate]?.let { return it }
        }
        return null
    }

    private fun readTrimmed(file: File): String? = try {
        file.readText().trim().ifEmpty { null }
    } catch (e: Exception) {
        // Includes the SELinux denial, which is the expected outcome on some handsets and
        // not worth a line per read.
        null
    }

    companion object {
        const val DEFAULT_ROOT = "/sys/class/thermal"
        private const val TAG = "ThermalZones"

        /**
         * Zone types to prefer, best first.
         *
         * These are the names Qualcomm and MediaTek platforms give to sensors placed near
         * the case rather than on a die, which is what a thermal budget is actually about:
         * a CPU core reads far hotter and swings far faster than the phone gets.
         *
         * `xo_therm` sits above `quiet_therm` on measurement, against the convention: on the
         * moto g power the HAL's own `skin` sensor matched `xo_therm` to 0.007 °C while
         * `quiet_therm` read 1.2 °C cooler, so the conventional name is a different sensor
         * there. That is one handset, which is exactly why the order is a preference among
         * names rather than a claim about which is *the* skin sensor, and why the zone that
         * won is reported with every reading.
         */
        val PREFERRED = listOf(
            "skin",
            "xo_therm",
            "xo-therm-adc",
            "quiet_therm",
            "msm_therm",
            "sdm-therm",
            "ap_therm",
            "battery",
        )

        /**
         * The coldest and hottest readings worth believing.
         *
         * Zones report things that are not temperatures at all: on the device this was
         * written against, `soc` reads a flat `100.0` and `ibat` reads `-351`, neither of
         * which is degrees of anything. A reading outside a band no handset can occupy is
         * that kind of value, and reporting it would put a number on the wire that the far
         * side has no way to recognise as meaningless.
         */
        const val MIN_PLAUSIBLE_C = -40.0
        const val MAX_PLAUSIBLE_C = 125.0

        /**
         * Interpret one `temp` file's contents.
         *
         * The kernel's thermal sysfs is documented as millidegrees Celsius and that is what
         * every zone on the test device reports, but the convention is not universally
         * honoured and a handful of drivers report whole degrees. Magnitude separates them:
         * no phone is 1000 °C, and no phone is 0.04 °C, so a value at or past a thousand is
         * millidegrees and anything smaller is already degrees.
         */
        fun celsiusOf(raw: String): Double? {
            val number = raw.toLongOrNull() ?: return null
            val celsius = if (kotlin.math.abs(number) >= 1000L) number / 1000.0 else number.toDouble()
            if (celsius < MIN_PLAUSIBLE_C || celsius > MAX_PLAUSIBLE_C) return null
            return celsius
        }
    }
}
