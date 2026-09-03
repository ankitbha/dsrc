package com.dsrc.phone.sensors

import android.net.NetworkCapabilities
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NetworkReaderTest {

    /** A probe that answers true for exactly the transports given. */
    private fun probeFor(vararg present: Int): (Int) -> Boolean = { it in present.toSet() }

    /** A source that hands back one network and probes it however the test says. */
    private fun reading(
        active: String? = "network",
        probe: ((Int) -> Boolean)? = probeFor(NetworkCapabilities.TRANSPORT_WIFI),
    ) = NetworkReader.readingFrom(activeNetwork = { active }, transportProbe = { probe })

    @Test
    fun `a tethered handset reads as wifi`() {
        // The configuration this field was added for: the handset's route to the internet
        // is another phone's hotspot, so the transport is WiFi and not the cellular radio
        // a SIM would have used.
        val result = reading(probe = probeFor(NetworkCapabilities.TRANSPORT_WIFI))
        assertEquals(NetworkReader.WIFI, result.value)
        assertNull(result.absentReason)
    }

    @Test
    fun `a handset on its own radio reads as cellular`() {
        val result = reading(probe = probeFor(NetworkCapabilities.TRANSPORT_CELLULAR))
        assertEquals(NetworkReader.CELLULAR, result.value)
        assertNull(result.absentReason)
    }

    @Test
    fun `a network with two transports names both, in the declared order`() {
        // A VPN network reports `vpn` and, where the platform populates it, the transport
        // underneath. Reporting only the first would answer the question wrongly whenever
        // the tunnel is what came first -- and the order has to be the declared one, not
        // the probe's, or the same configuration reads as two different values.
        val result = reading(
            probe = probeFor(
                NetworkCapabilities.TRANSPORT_VPN,
                NetworkCapabilities.TRANSPORT_WIFI,
            ),
        )
        assertEquals("wifi+vpn", result.value)
        assertNull(result.absentReason)
    }

    @Test
    fun `every declared transport has a distinct name and survives a round trip`() {
        val names = NetworkReader.TRANSPORTS.map { it.second }
        assertEquals(names.size, names.toSet().size)
        NetworkReader.TRANSPORTS.forEach { (constant, name) ->
            assertEquals(name, reading(probe = probeFor(constant)).value)
        }
    }

    @Test
    fun `no active network is a measured negative, not a failed reading`() {
        // The platform answering "there is no route" is a fact about the handset: a HERE
        // call made now will fail. It is kept apart from every reason below, all of which
        // mean the reading did not happen.
        val result = NetworkReader.readingFrom<String>(
            activeNetwork = { null },
            transportProbe = { probeFor() },
        )
        assertEquals(NetworkReader.REASON_NO_ACTIVE_NETWORK, result.absentReason)
        assertNull(result.value)
    }

    @Test
    fun `a network whose capabilities read back null is not the same as no network`() {
        // The race between the two calls: there was an active network and it went away
        // before its capabilities could be read.
        val result = reading(probe = null)
        assertEquals(NetworkReader.REASON_NO_CAPABILITIES, result.absentReason)
        assertNull(result.value)
    }

    @Test
    fun `capabilities naming no transport this build knows are reported as such`() {
        // Not as "no network": there is one, and it is carried by something with no name
        // here. Collapsing the two would report a handset that has a route as one that
        // does not.
        val result = reading(probe = probeFor())
        assertEquals(NetworkReader.REASON_NO_KNOWN_TRANSPORT, result.absentReason)
        assertNull(result.value)
    }

    @Test
    fun `a missing permission is named apart from a transient failure`() {
        // SecurityException is a RuntimeException, so the order of the two catches is the
        // whole behaviour: caught in the wrong order this reads as `accessor_raised`, and
        // a build defect true of every sample of every drive hides among failures that
        // come and go.
        val result = NetworkReader.readingFrom<String>(
            activeNetwork = { throw SecurityException("ACCESS_NETWORK_STATE not held") },
            transportProbe = { probeFor() },
        )
        assertEquals(NetworkReader.REASON_PERMISSION_DENIED, result.absentReason)
        assertNull(result.value)
    }

    @Test
    fun `any other platform throw is caught and named`() {
        val result = NetworkReader.readingFrom<String>(
            activeNetwork = { throw IllegalStateException("binder died") },
            transportProbe = { probeFor() },
        )
        assertEquals(NetworkReader.REASON_ACCESSOR_RAISED, result.absentReason)
        assertNull(result.value)
    }

    @Test
    fun `a throw from the second accessor is caught too`() {
        // The first accessor is not the only one that can fail, and a try that covered
        // only the first call would let this one out into the telemetry loop.
        val result = NetworkReader.readingFrom<String>(
            activeNetwork = { "network" },
            transportProbe = { throw IllegalStateException("binder died late") },
        )
        assertEquals(NetworkReader.REASON_ACCESSOR_RAISED, result.absentReason)
        assertNull(result.value)
    }

    @Test
    fun `no ConnectivityManager is its own reason`() {
        val result = NetworkReader.from(null)
        assertEquals(NetworkReader.REASON_NO_MANAGER, result.absentReason)
        assertNull(result.value)
    }

    @Test
    fun `a reading carries a value or a reason and never both or neither`() {
        // The property the whole field rests on. A bare null says "no number"; this pair
        // exists so the record says which of several causes produced it.
        val cases = listOf(
            reading(probe = probeFor(NetworkCapabilities.TRANSPORT_WIFI)),
            reading(probe = probeFor()),
            reading(probe = null),
            NetworkReader.readingFrom<String>({ null }, { probeFor() }),
            NetworkReader.readingFrom<String>({ throw SecurityException() }, { probeFor() }),
            NetworkReader.readingFrom<String>({ throw IllegalStateException() }, { probeFor() }),
            NetworkReader.from(null),
        )
        cases.forEach { result ->
            assertTrue(
                "value=${result.value} reason=${result.absentReason}",
                (result.value == null) != (result.absentReason == null),
            )
        }
    }

    @Test
    fun `every reason a reading can carry is in the declared set`() {
        // The set a receiver keys on. A reason invented at a call site and left out of
        // here would decode as an unknown string on the far side, which is the closed
        // vocabulary failing open.
        val produced = listOf(
            reading(probe = probeFor()),
            reading(probe = null),
            NetworkReader.readingFrom<String>({ null }, { probeFor() }),
            NetworkReader.readingFrom<String>({ throw SecurityException() }, { probeFor() }),
            NetworkReader.readingFrom<String>({ throw IllegalStateException() }, { probeFor() }),
            NetworkReader.from(null),
        ).mapNotNull { it.absentReason }
        assertEquals(NetworkReader.ABSENCE_REASONS.toSet(), produced.toSet())
        produced.forEach { assertTrue(it, it in NetworkReader.ABSENCE_REASONS) }
        assertNotNull(NetworkReader.SEPARATOR)
    }
}
