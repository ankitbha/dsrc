package com.dsrc.phone.sensors

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.file.Files

class ThermalZonesTest {

    private lateinit var root: File

    @Before
    fun makeRoot() {
        root = Files.createTempDirectory("thermal").toFile()
    }

    @After
    fun removeRoot() {
        root.deleteRecursively()
    }

    /** One zone directory, as the kernel lays it out. A null field is left absent. */
    private fun zone(index: Int, type: String?, temp: String?) {
        val directory = File(root, "thermal_zone$index").also { it.mkdirs() }
        type?.let { File(directory, "type").writeText("$it\n") }
        temp?.let { File(directory, "temp").writeText("$it\n") }
    }

    @Test
    fun `a millidegree reading becomes degrees`() {
        zone(0, "quiet_therm", "28926")
        val reading = ThermalZones(root).read()

        assertEquals(28.926, reading!!.celsius, 1e-9)
        assertEquals("quiet_therm", reading.zone)
    }

    @Test
    fun `a driver reporting whole degrees is read as degrees`() {
        // The kernel documents millidegrees and every zone on the test handset obliges,
        // but the convention is not universally honoured. 31 is 31 C, not 0.031 C.
        zone(0, "quiet_therm", "31")

        assertEquals(31.0, ThermalZones(root).read()!!.celsius, 1e-9)
    }

    @Test
    fun `the preference order decides which zone is used, not the directory order`() {
        // Laid out so directory order and preference order disagree: `battery` is zone 0
        // and last preferred, `skin` is zone 2 and first. Reading them in the order the
        // filesystem hands them back would take the battery, which is not the phone.
        zone(0, "battery", "27300")
        zone(1, "msm_therm", "30306")
        zone(2, "skin", "28578")

        val reading = ThermalZones(root).read()
        assertEquals("skin", reading!!.zone)
        assertEquals(28.578, reading.celsius, 1e-9)
    }

    @Test
    fun `xo_therm is preferred over quiet_therm, which is the measured order not the conventional one`() {
        // Both present, as they are on the moto g power. `quiet_therm` is the name
        // Qualcomm platforms conventionally use for skin, so the conventional choice is
        // the wrong one here: the HAL's own `skin` sensor matched `xo_therm` to 0.007 C
        // while `quiet_therm` read 1.2 C cooler and is a different sensor. The values
        // below are the ones actually measured on that handset.
        zone(0, "quiet_therm", "28926")
        zone(1, "xo_therm", "30112")

        val reading = ThermalZones(root).read()
        assertEquals("xo_therm", reading!!.zone)
        assertEquals(30.112, reading.celsius, 1e-9)
    }

    @Test
    fun `a zone that is not a temperature at all is refused`() {
        // Both taken from the moto g power: `soc` reads a flat 100.0 and `ibat` reads
        // -351, neither of which is degrees of anything. Reported, they would put a
        // number on the wire that the far side cannot recognise as meaningless.
        assertNull(ThermalZones.celsiusOf("100000000"))
        assertNull(ThermalZones.celsiusOf("-2000000"))
        assertNull(ThermalZones.celsiusOf("not a number"))
        assertNull(ThermalZones.celsiusOf(""))
    }

    @Test
    fun `the plausible band is the one the constants name`() {
        // Pinned against the constants from the outside, so widening the band silently is
        // a test failure rather than a behaviour change nobody sees. Derived arithmetic
        // would agree with any band at all.
        assertNotNull(ThermalZones.celsiusOf("-40000"))
        assertNotNull(ThermalZones.celsiusOf("125000"))
        assertNull(ThermalZones.celsiusOf("-40001"))
        assertNull(ThermalZones.celsiusOf("125001"))
        assertEquals(-40.0, ThermalZones.MIN_PLAUSIBLE_C, 1e-9)
        assertEquals(125.0, ThermalZones.MAX_PLAUSIBLE_C, 1e-9)
    }

    @Test
    fun `a missing root is null, not an exception`() {
        // The expected outcome on a handset whose SELinux policy hides the directory. A
        // throw here would come out of the telemetry sample and take the whole report with
        // it, which is exactly the failure the headroom NaN guard exists to prevent.
        assertNull(ThermalZones(File(root, "nothing-here")).read())
    }

    @Test
    fun `a zone with no readable temperature is not chosen`() {
        // A `type` that reads and a `temp` that does not. Choosing it would resolve the
        // search to a zone that returns null for the rest of the drive, with the search
        // already marked done and no way to reconsider.
        zone(0, "skin", null)
        zone(1, "quiet_therm", "28926")

        val reading = ThermalZones(root).read()
        assertEquals("quiet_therm", reading!!.zone)
    }

    @Test
    fun `a zone whose only reading is implausible is not chosen`() {
        // Same trap one step in: `soc` is preferred by nothing, but a zone named like a
        // candidate that only ever reads nonsense must not win the search either.
        zone(0, "skin", "100000000")
        zone(1, "quiet_therm", "28926")

        assertEquals("quiet_therm", ThermalZones(root).read()!!.zone)
    }

    @Test
    fun `no recognised zone is null rather than an arbitrary one`() {
        // A temperature from a die nobody named is not a thermal budget. Guessing would
        // put a number on the wire whose meaning no consumer could establish.
        zone(0, "cpuss-0-usr", "33000")
        zone(1, "gpu-usr", "30700")

        assertNull(ThermalZones(root).read())
    }

    @Test
    fun `the temperature is re-read on every call`() {
        // The zone is cached; the reading must not be. Trending is the entire purpose, and
        // a cached first sample would report a cold phone for the whole drive.
        zone(0, "skin", "28000")
        val zones = ThermalZones(root)
        assertEquals(28.0, zones.read()!!.celsius, 1e-9)

        File(File(root, "thermal_zone0"), "temp").writeText("41000\n")
        assertEquals(41.0, zones.read()!!.celsius, 1e-9)
    }

    @Test
    fun `a device with no zones is not searched again on every report`() {
        // A failed search is remembered, so a handset that will never have a zone does not
        // pay a directory scan every second forever to keep learning the same thing.
        // Pinned by its consequence: a zone appearing after the first read is not picked
        // up, which is only true if the search really did not run again.
        val zones = ThermalZones(root)
        assertNull(zones.read())

        zone(0, "skin", "28000")
        assertNull("the zone search ran a second time", zones.read())
    }

    @Test
    fun `every preferred name is one a reading can actually come from`() {
        // Each candidate resolves on its own, so a typo in the list is a failure here
        // rather than a name that silently never matches on any handset.
        for (name in ThermalZones.PREFERRED) {
            val fresh = Files.createTempDirectory("thermal-one").toFile()
            File(fresh, "thermal_zone0").also { it.mkdirs() }.let {
                File(it, "type").writeText(name)
                File(it, "temp").writeText("30000")
            }
            assertEquals(name, ThermalZones(fresh).read()?.zone)
            fresh.deleteRecursively()
        }
        assertTrue(ThermalZones.PREFERRED.isNotEmpty())
    }
}
