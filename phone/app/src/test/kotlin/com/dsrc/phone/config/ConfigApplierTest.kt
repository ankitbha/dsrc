package com.dsrc.phone.config

import com.dsrc.transport.HereQuery
import com.dsrc.transport.RateCommand
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ConfigApplierTest {

    /** Records what each modality was told, so a command reaching the wrong one is visible. */
    private class Recorder : ConfigApplier.Targets {
        val camera = mutableListOf<Double>()
        val gps = mutableListOf<Double>()
        val imu = mutableListOf<Double>()
        val here = mutableListOf<Double>()
        val queries = mutableListOf<HereQuery?>()

        override fun setCameraRate(hz: Double) { camera.add(hz) }
        override fun setGpsRate(hz: Double) { gps.add(hz) }
        override fun setImuRate(hz: Double) { imu.add(hz) }
        override fun setHereRate(hz: Double) { here.add(hz) }
        override fun setHereQuery(query: HereQuery?) { queries.add(query) }
    }

    private val query = HereQuery(
        `in` = "corridor:40.7,-74.0;40.8,-74.1;r=200",
        locationRef = "shape",
        lat = 40.7128,
        lon = -74.0060,
        radiusM = 9_000.0,
    )

    private fun command(
        camera: Double = 5.0,
        gps: Double = 1.0,
        imu: Double = 50.0,
        here: Double = 0.2,
        shadow: Boolean = false,
        query: HereQuery? = null,
        trigger: String = "thermal",
    ) = RateCommand(
        captureMonoNs = 1,
        rates = mapOf("camera_hz" to camera, "gps_hz" to gps, "imu_hz" to imu, "here_hz" to here),
        trigger = trigger,
        shadow = shadow,
        here = query,
    )

    @Test
    fun `each rate reaches its own modality`() {
        // Four distinct values, so a command routed to the wrong modality is visible. With
        // them equal, every permutation of the four calls would pass.
        val recorder = Recorder()
        ConfigApplier(recorder).apply(command(camera = 2.0, gps = 3.0, imu = 4.0, here = 5.0))

        assertEquals(listOf(2.0), recorder.camera)
        assertEquals(listOf(3.0), recorder.gps)
        assertEquals(listOf(4.0), recorder.imu)
        assertEquals(listOf(5.0), recorder.here)
    }

    @Test
    fun `a shadow command changes nothing at all`() {
        // The spec defines shadow as whether the command "was gated for real or only
        // recorded". A shadow command that moved a rate would make the comparison it exists
        // for meaningless -- the Jetson would be measuring a phone that had already acted.
        val recorder = Recorder()
        val applier = ConfigApplier(recorder)
        applier.apply(command(camera = 9.0, shadow = true, query = query))

        assertTrue("no rate may be applied: ${recorder.camera}", recorder.camera.isEmpty())
        assertTrue(recorder.gps.isEmpty())
        assertTrue(recorder.imu.isEmpty())
        assertTrue(recorder.here.isEmpty())
        assertTrue("not even the query", recorder.queries.isEmpty())

        val stats = applier.stats
        assertEquals(1, stats.shadowed)
        assertEquals(0, stats.applied)
        assertTrue("nor may it become what the phone is running", stats.currentRates.isEmpty())
        assertFalse(stats.hereConfigured)
    }

    @Test
    fun `a real command after a shadow one still applies`() {
        // The shadow branch returns early; returning from the wrong place would swallow
        // every command after the first shadow.
        val recorder = Recorder()
        val applier = ConfigApplier(recorder)
        applier.apply(command(camera = 9.0, shadow = true))
        applier.apply(command(camera = 7.0))

        assertEquals(listOf(7.0), recorder.camera)
        assertEquals(1, applier.stats.applied)
        assertEquals(1, applier.stats.shadowed)
    }

    @Test
    fun `the query is passed through, including the null that means no change`() {
        // Null is "this command does not change the query", which is what makes the field
        // optional. The applier passes it and HerePipeline decides, so the decision lives
        // in one place rather than two.
        val recorder = Recorder()
        val applier = ConfigApplier(recorder)
        applier.apply(command(query = query))
        applier.apply(command(query = null))

        assertEquals(listOf(query, null), recorder.queries)
        assertTrue(
            "a later command that omits the query means no change, so the query is still " +
                "configured -- the previous assertion had a right operand that was always true",
            applier.stats.hereConfigured,
        )
    }

    @Test
    fun `the rates in force are the last real command's, not a shadow one's`() {
        val applier = ConfigApplier(Recorder())
        applier.apply(command(camera = 7.0))
        applier.apply(command(camera = 9.0, shadow = true))

        assertEquals(7.0, applier.stats.currentRates.getValue("camera_hz"), 1e-9)
    }

    @Test
    fun `the trigger is recorded for both kinds`() {
        // The Jetson asks "what would you have done"; knowing which command prompted it is
        // the point of recording a shadow at all.
        //
        // Both kinds, as the name says. This applied only a shadow command, so moving the
        // write inside the shadow branch left the whole suite green while a live drive
        // recorded no trigger at all -- or kept whatever the last shadow command left.
        val applier = ConfigApplier(Recorder())

        applier.apply(command(shadow = true, trigger = "advisory_bin_boundary"))
        assertEquals("advisory_bin_boundary", applier.stats.lastTrigger)

        applier.apply(command(shadow = false, trigger = "event_from_free_tier"))
        assertEquals("event_from_free_tier", applier.stats.lastTrigger)
    }

    @Test
    fun `the counters separate what was applied from what was only recorded`() {
        // The Jetson's shadow_mode credits this side with "counting what it shadowed", and
        // task 35 scores from these drives. A pure-shadow drive and a fully live one have
        // to be distinguishable from what this side reports.
        val applier = ConfigApplier(Recorder())
        repeat(3) { applier.apply(command(shadow = true, trigger = "idle")) }
        repeat(2) { applier.apply(command(shadow = false, trigger = "idle")) }

        assertEquals(3L, applier.stats.shadowed)
        assertEquals(2L, applier.stats.applied)
    }

    @Test
    fun `applying twice does not restart anything, it just sets the rate again`() {
        // "Without restarting capture" is the task. Asserted by what the modality is told:
        // two rate changes, no other call.
        val recorder = Recorder()
        val applier = ConfigApplier(recorder)
        applier.apply(command(camera = 2.0))
        applier.apply(command(camera = 8.0))

        assertEquals(listOf(2.0, 8.0), recorder.camera)
        assertEquals(2, applier.stats.applied)
    }
}
