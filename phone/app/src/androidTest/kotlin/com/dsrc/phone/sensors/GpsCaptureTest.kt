package com.dsrc.phone.sensors

import android.Manifest
import android.content.Context
import android.location.Criteria
import android.location.Location
import android.location.LocationManager
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.dsrc.phone.MainActivity
import com.dsrc.phone.config.SensingConfig
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * The platform adapter against a real [LocationManager], fed by a test provider.
 *
 * The unit tests cover the field mapping, where the arithmetic errors are. What they
 * cannot see is whether the adapter is *attached to anything*: the provider name, the
 * requested interval, and the two registration overloads -- one takes a `Looper` and one a
 * `Handler`, and choosing wrongly compiles, runs, logs a cheerful line and delivers
 * nothing, which is indistinguishable from a phone with no satellite lock.
 *
 * Two platform behaviours had to be defeated to get a fix through, and both are worth
 * knowing outside this file.
 *
 * `setTestProviderLocation` fails **silently**: `LocationManagerService` returns without
 * throwing when its appop check fails, so a rejected push is indistinguishable from a
 * delivered one. The grant also has to be waited for -- `executeShellCommand` is
 * asynchronous, and closing the descriptor without draining it lets the next call run
 * before the grant lands.
 *
 * And a location request from an app that is not in the foreground is marked
 * `(inactive)` and receives nothing. The provider's own event log read
 * `received location[1]` while the listener got none: the fix arrived and was delivered to
 * no one. Setting `location_background_throttle_interval_ms` to zero does *not* fix it --
 * that was tried and removed once it proved to change nothing. What the platform wants is
 * the foreground, which is what the service's `foregroundServiceType="location"` buys in
 * production, and what [inForeground] stands in for here. Granting
 * `ACCESS_BACKGROUND_LOCATION` would also work and is the wrong trade: it would hand the
 * app a capability production neither needs nor wants, to satisfy a test.
 */
@RunWith(AndroidJUnit4::class)
class GpsCaptureTest {

    @get:Rule
    val permissions: GrantPermissionRule = GrantPermissionRule.grant(
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION,
    )

    private lateinit var context: Context
    private lateinit var manager: LocationManager
    private var source: GpsLocationSource? = null
    private var providerAdded = false
    private var foreground: ActivityScenario<MainActivity>? = null

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        manager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

        // Mock locations are gated on an appop rather than a manifest permission, so it is
        // granted the way developer options would. Drained, because the command is
        // asynchronous and the grant has to have landed before addTestProvider runs.
        shell("appops set ${context.packageName} android:mock_location allow")

        manager.addTestProvider(
            LocationManager.GPS_PROVIDER,
            false, false, false, false,
            true, true, true,
            Criteria.POWER_LOW,
            Criteria.ACCURACY_FINE,
        )
        manager.setTestProviderEnabled(LocationManager.GPS_PROVIDER, true)
        providerAdded = true
    }

    /**
     * Put the app in the foreground before asking for location.
     *
     * Android 12 marks a location registration `(inactive)` unless the app is in the
     * foreground or holds `ACCESS_BACKGROUND_LOCATION`, and an inactive registration gets
     * nothing -- the provider's event log said `received location[1]` while the listener
     * saw none. In production the foreground *service* is what earns delivery, which is
     * why the service declares `foregroundServiceType="location"`. Adding the background
     * permission to the manifest to satisfy a test would grant the app a capability
     * production does not need and does not want.
     */
    private fun inForeground() {
        foreground = ActivityScenario.launch(MainActivity::class.java)
    }

    @After
    fun tearDown() {
        source?.stop()
        foreground?.close()
        if (providerAdded) {
            runCatching { manager.removeTestProvider(LocationManager.GPS_PROVIDER) }
        }
        // Device-wide. Left granted, every later test on this emulator could mock
        // locations, including ones that assume they cannot.
        runCatching { shell("appops set ${context.packageName} android:mock_location default") }
    }

    @Test
    fun a_mock_fix_arrives_as_a_reading_with_both_clocks() {
        inForeground()
        val received = LinkedBlockingQueue<GpsReading>()
        val adapter = GpsLocationSource(context, SensingConfig(gpsHz = 5.0))
        source = adapter
        adapter.start { received.add(it) }

        val before = SystemClock.elapsedRealtimeNanos()
        push(latitude = 40.7128, longitude = -74.0060, speed = 13.4f, bearing = 91.5f, altitude = 12.5)

        val reading = received.poll(10, TimeUnit.SECONDS)
        assertNotNull("no fix was delivered: the adapter is attached to nothing", reading)
        reading!!

        assertTrue(reading.record.valid)
        assertEquals(40.7128, reading.record.latitude!!, 1e-6)
        assertEquals(-74.0060, reading.record.longitude!!, 1e-6)
        assertEquals(13.4, reading.record.speedMps!!, 1e-4)
        assertEquals(91.5, reading.record.headingDeg!!, 1e-3)
        assertEquals(12.5, reading.record.altitudeM!!, 1e-6)
        assertEquals("fix_quality 0 is the wire's no-fix", 1, reading.record.fixQuality)
        assertNull("no Android API reports HDOP", reading.record.hdop)

        // The clocks are the half of the task the wire cannot carry. Both are
        // elapsedRealtime, so the receipt sits at or after the fix and the difference is a
        // real latency rather than a latency plus an unknown offset between two bases.
        assertTrue(
            "receipt ${reading.receiptMonoNs} precedes fix ${reading.fixMonoNs}",
            reading.receiptMonoNs >= reading.fixMonoNs,
        )
        assertTrue("receipt precedes the push", reading.receiptMonoNs >= before)
        assertTrue(
            "delivery latency ${reading.deliveryLatencyNs} ns is not plausible",
            reading.deliveryLatencyNs in 0..10_000_000_000L,
        )
    }

    @Test
    fun a_fix_with_no_speed_or_bearing_carries_nulls_not_zeroes() {
        inForeground()
        val received = LinkedBlockingQueue<GpsReading>()
        val adapter = GpsLocationSource(context, SensingConfig(gpsHz = 5.0))
        source = adapter
        adapter.start { received.add(it) }

        // The common case on a stationary vehicle, and the one where zero would be a
        // false claim rather than a missing value.
        push(latitude = 1.5, longitude = 2.5, speed = null, bearing = null, altitude = null)

        val reading = received.poll(10, TimeUnit.SECONDS)
        assertNotNull("no fix was delivered", reading)
        assertNull(reading!!.record.speedMps)
        assertNull(reading.record.headingDeg)
        assertNull(reading.record.altitudeM)
        assertTrue("an absent speed does not invalidate the position", reading.record.valid)
    }

    @Test
    fun stopping_ends_delivery_and_deregisters() {
        inForeground()
        val received = LinkedBlockingQueue<GpsReading>()
        val adapter = GpsLocationSource(context, SensingConfig(gpsHz = 5.0))
        source = adapter
        adapter.start { received.add(it) }
        push(latitude = 3.0, longitude = 4.0)
        assertNotNull("the first fix never arrived", received.poll(10, TimeUnit.SECONDS))
        assertTrue("never registered", liveRequests().contains(context.packageName))

        adapter.stop()
        source = null

        push(latitude = 5.0, longitude = 6.0)
        assertNull("a fix arrived after stop()", received.poll(2, TimeUnit.SECONDS))

        // Polled with a deadline: removeUpdates travels to the service over a binder, so a
        // single dump read immediately afterwards can still list the request, and that
        // would read as a leak. A leak is the request never going away.
        val deadline = System.nanoTime() + 5_000_000_000L
        var live = liveRequests()
        while (System.nanoTime() < deadline && live.contains(context.packageName)) {
            Thread.sleep(100)
            live = liveRequests()
        }
        // A leaked registration keeps the GNSS engine powered after the service tears
        // down, which in a car is a battery drain nothing reports.
        assertFalse("the request outlived stop() by over 5 s:\n$live", live.contains(context.packageName))
    }

    @Test
    fun the_adapter_asks_the_platform_for_the_commanded_rate() {
        val adapter = GpsLocationSource(context, SensingConfig(gpsHz = 5.0))
        source = adapter
        adapter.start { }

        // Read from the platform's own record, not from our fields: asserting that we
        // stored a listener would pass whether or not the framework accepted it.
        val live = liveRequests()
        assertTrue("no live request for ${context.packageName}:\n$live", live.contains(context.packageName))
        assertTrue("not on the gps provider:\n$live", live.contains("gps"))
        // 200 ms is 5 Hz -- the commanded rate reaching the platform rather than a default.
        assertTrue("not the commanded 5 Hz:\n$live", live.contains("200ms"))
    }

    /**
     * Live location registrations, one per line.
     *
     * `dumpsys location` also keeps an event log that records every past
     * `+registration ... -> Request[...]`, so matching the package name anywhere in the
     * dump would match registrations that have already been removed -- a leak test that
     * can never pass. Event-log lines carry a leading timestamp; live ones do not.
     */
    private fun liveRequests(): String {
        val timestamped = Regex("""^\s*\d\d-\d\d \d\d:\d\d""")
        return shell("dumpsys location").lines()
            .filter { it.contains("Request[") && !timestamped.containsMatchIn(it) }
            .joinToString("\n") { line -> "gps ".takeIf { "@+" in line }.orEmpty() + line.trim() }
    }

    private fun push(
        latitude: Double,
        longitude: Double,
        speed: Float? = null,
        bearing: Float? = null,
        altitude: Double? = null,
    ) {
        val location = Location(LocationManager.GPS_PROVIDER).apply {
            this.latitude = latitude
            this.longitude = longitude
            this.accuracy = 5f
            this.time = System.currentTimeMillis()
            this.elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos()
            speed?.let { this.speed = it }
            bearing?.let { this.bearing = it }
            altitude?.let { this.altitude = it }
        }
        manager.setTestProviderLocation(LocationManager.GPS_PROVIDER, location)
    }

    private fun shell(command: String): String {
        val descriptor = InstrumentationRegistry.getInstrumentation().uiAutomation
            .executeShellCommand(command)
        return ParcelFileDescriptor.AutoCloseInputStream(descriptor).use {
            it.readBytes().toString(Charsets.UTF_8)
        }
    }
}
