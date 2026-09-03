package com.dsrc.phone.sensors

import android.net.ConnectivityManager
import android.net.NetworkCapabilities

/**
 * Which network the phone's own traffic is currently going over, in the wire's vocabulary.
 *
 * The only thing on this handset that leaves the phone by IP is the HERE query: the sensor
 * link to the Jetson is a USB tunnel in the car, and loopback on the handset when it is
 * `adb reverse`. So this field answers one question — what carried HERE — and it is a
 * property of the rig rather than of the system under test.
 *
 * It exists because that property is no longer constant. A handset with its own SIM has one
 * answer for the life of the drive and nothing needs recording. A handset tethered to
 * another phone has at least two, they differ between drives, and they can change inside
 * one drive when the tether drops and the handset falls back or goes to nothing at all. A
 * HERE call that fails then arrives as `status = 0`, which is the same value a DNS failure,
 * a lost signal and an unreachable HERE all produce — so without this field the cause is
 * not recoverable from the record afterwards, only guessed at.
 *
 * Reported every telemetry period rather than once at start, because "which network" is not
 * a fact about the session; it is a fact about the moment, and a mid-drive switch is exactly
 * the event worth seeing.
 */
object NetworkReader {

    const val CELLULAR = "cellular"
    const val WIFI = "wifi"
    const val ETHERNET = "ethernet"
    const val BLUETOOTH = "bluetooth"
    const val VPN = "vpn"
    const val WIFI_AWARE = "wifi_aware"
    const val LOWPAN = "lowpan"

    /**
     * Platform constant to wire name, in the order a multi-transport reading lists them.
     *
     * All seven exist at this app's `minSdk` of 29 and all seven are valid arguments to
     * `hasTransport` there, so none of them needs an API guard. The later additions —
     * `TRANSPORT_USB` at API 31 and beyond — are deliberately absent: `hasTransport` rejects
     * a transport its own platform version does not know, so probing for one would raise on
     * the devices that matter. A network carried by a transport not on this list is reported
     * as [REASON_NO_KNOWN_TRANSPORT] rather than silently as no network.
     */
    val TRANSPORTS: List<Pair<Int, String>> = listOf(
        NetworkCapabilities.TRANSPORT_CELLULAR to CELLULAR,
        NetworkCapabilities.TRANSPORT_WIFI to WIFI,
        NetworkCapabilities.TRANSPORT_ETHERNET to ETHERNET,
        NetworkCapabilities.TRANSPORT_BLUETOOTH to BLUETOOTH,
        NetworkCapabilities.TRANSPORT_VPN to VPN,
        NetworkCapabilities.TRANSPORT_WIFI_AWARE to WIFI_AWARE,
        NetworkCapabilities.TRANSPORT_LOWPAN to LOWPAN,
    )

    /**
     * Joins the transports of one network into a single value.
     *
     * A network can genuinely have more than one — a VPN network reports `vpn` and, on the
     * platform versions that populate it, the transport underneath it as well. Reporting
     * only the first would answer the question wrongly whenever the tunnel is what is
     * listed first, and reporting a list would put a second type into a string field.
     */
    const val SEPARATOR = "+"

    /** No `ConnectivityManager`: the service could not obtain the system service at all. */
    const val REASON_NO_MANAGER = "no_manager"

    /**
     * The platform says there is no active network. A measured negative, and the one
     * absence reason here that is a fact about the phone rather than about the reading:
     * the handset has no route off itself, so a HERE call now will fail.
     */
    const val REASON_NO_ACTIVE_NETWORK = "no_active_network"

    /**
     * There was an active network and its capabilities read back null, which is the race
     * between the two calls: the network went away in between. Distinct from
     * [REASON_NO_ACTIVE_NETWORK] because that one is an answer and this one is its absence.
     */
    const val REASON_NO_CAPABILITIES = "no_capabilities"

    /** Capabilities that name no transport this build knows. Not the same as no network. */
    const val REASON_NO_KNOWN_TRANSPORT = "no_known_transport"

    /** `ACCESS_NETWORK_STATE` not held. Named apart, because it is a build defect and the
     * others are not: it would be true for every sample of every drive. */
    const val REASON_PERMISSION_DENIED = "permission_denied"

    /** Anything else the platform threw. */
    const val REASON_ACCESSOR_RAISED = "accessor_raised"

    /** Every reason a reading can be absent. The closed set a receiver may key on. */
    val ABSENCE_REASONS: List<String> = listOf(
        REASON_NO_MANAGER,
        REASON_NO_ACTIVE_NETWORK,
        REASON_NO_CAPABILITIES,
        REASON_NO_KNOWN_TRANSPORT,
        REASON_PERMISSION_DENIED,
        REASON_ACCESSOR_RAISED,
    )

    /**
     * A transport list, or the named reason there is none. Never both, and never neither —
     * a value needs no excuse, and an absence with no reason is the bare null this whole
     * field exists to avoid.
     */
    data class Reading(val value: String?, val absentReason: String?)

    /**
     * The half that holds the rule, separated from the platform call so it can be tested
     * without a device. Takes the probe rather than the capabilities object, because
     * `NetworkCapabilities` cannot be constructed in a JVM unit test.
     */
    fun readingFor(hasTransport: (Int) -> Boolean): Reading {
        val present = TRANSPORTS.filter { (constant, _) -> hasTransport(constant) }
        return if (present.isEmpty()) {
            Reading(null, REASON_NO_KNOWN_TRANSPORT)
        } else {
            Reading(present.joinToString(SEPARATOR) { it.second }, null)
        }
    }

    /**
     * Read a network source, or name why the reading did not happen.
     *
     * Nothing here throws. This is called from the telemetry loop, whose sibling failure is
     * on record: one unguarded platform call took the entire telemetry stream down for a
     * whole drive — thermal status, achieved rates and drops with it — because the caller's
     * `runCatching` swallowed it a level too high to tell anyone. A network reading is worth
     * strictly less than the frame it rides on, so it never costs the frame.
     */
    fun <N> readingFrom(
        activeNetwork: () -> N?,
        transportProbe: (N) -> ((Int) -> Boolean)?,
    ): Reading {
        return try {
            val active = activeNetwork() ?: return Reading(null, REASON_NO_ACTIVE_NETWORK)
            val probe = transportProbe(active) ?: return Reading(null, REASON_NO_CAPABILITIES)
            readingFor(probe)
        } catch (e: SecurityException) {
            // Caught before the general case below, which it would otherwise fall into: a
            // SecurityException is a RuntimeException, and collapsing the two would hide a
            // missing permission -- true of every sample of every drive -- among failures
            // that are transient by nature.
            Reading(null, REASON_PERMISSION_DENIED)
        } catch (e: RuntimeException) {
            Reading(null, REASON_ACCESSOR_RAISED)
        }
    }

    /**
     * The platform call. A thin adapter over [readingFrom] and nothing else.
     *
     * The two accessors are passed as functions rather than called here because
     * `ConnectivityManager` and `Network` are final platform classes that a JVM unit test
     * cannot construct or stand in for, and no mocking library is on the test classpath. Put
     * the branches behind this seam and every one of them -- including both throwing paths,
     * which are the ones that matter -- is reachable from a test on this machine. Left
     * inline they would be defended only by the argument that they cannot be provoked.
     */
    fun from(connectivity: ConnectivityManager?): Reading {
        if (connectivity == null) return Reading(null, REASON_NO_MANAGER)
        return readingFrom(
            activeNetwork = { connectivity.activeNetwork },
            transportProbe = { network ->
                connectivity.getNetworkCapabilities(network)?.let { it::hasTransport }
            },
        )
    }
}
