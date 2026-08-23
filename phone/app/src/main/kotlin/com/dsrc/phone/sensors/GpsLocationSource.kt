package com.dsrc.phone.sensors

import android.annotation.SuppressLint
import android.content.Context
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import com.dsrc.phone.config.SensingConfig
import java.util.concurrent.atomic.AtomicInteger

/**
 * A platform fix reduced to primitives, so the field mapping can be unit-tested.
 *
 * `Location` is an Android class whose getters throw against the unit-test `android.jar`
 * stubs, so a mapping that read one directly could only be exercised on a device. The
 * mapping is where the bugs are -- unit confusions, a bearing that arrives as 360, a
 * bogus wall clock -- and the extraction either side of it is three lines with nothing to
 * get wrong.
 */
data class PlatformFix(
    /** When the fix was made, on `elapsedRealtime`. */
    val fixMonoNs: Long,
    /** When this process was handed it, on the same clock. */
    val receiptMonoNs: Long,
    val latitude: Double,
    val longitude: Double,
    val hasSpeed: Boolean = false,
    val speedMps: Float = 0f,
    val hasBearing: Boolean = false,
    val bearingDeg: Float = 0f,
    val hasAltitude: Boolean = false,
    val altitudeM: Double = 0.0,
    /** GPS wall time in milliseconds, or 0 when the provider gave none. */
    val utcEpochMs: Long = 0,
    /** Satellites used in the fix, from the last GNSS status. */
    val satellitesUsedInFix: Int = 0,
)

/**
 * GPS from the platform, using [LocationManager] rather than the fused provider.
 *
 * Deviates from the plan's D9, for two reasons that only became visible against the
 * frozen wire contract:
 *
 * `num_sats` is a required, non-nullable field. `FusedLocationProviderClient` has no
 * satellite count -- it is not in `Location`, and `GnssStatus` callbacks are a
 * `LocationManager` facility -- so every fused fix would go out as a valid fix carrying
 * `num_sats: 0`, which a receiver reads as a contradiction rather than as "unknown". The
 * field has no null to mean unknown, so the only way to fill it honestly is the API that
 * reports it.
 *
 * And fused positions are blended across GNSS, wifi and cell, and interpolated. This
 * phone exists to record what the road did, and a smoothed track is the wrong ground
 * truth to compare a traffic model against.
 *
 * `hdop` stays null. No Android API exposes it, and deriving one from `Location.accuracy`
 * would be inventing a number -- accuracy is a metre radius, HDOP is dimensionless
 * satellite geometry, and there is no conversion between them.
 */
class GpsLocationSource(
    context: Context,
    private val config: SensingConfig,
) : GpsSource {

    private val manager =
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    /**
     * Last satellite count, updated out of band.
     *
     * GNSS status and location updates are separate callbacks that do not arrive
     * together, so the count attached to a fix is the most recent one known rather than
     * one measured at the moment of that fix. Off by at most one status interval, and the
     * alternative -- omitting it -- is a required field left at zero.
     */
    private val satellitesInFix = AtomicInteger(0)

    private var sink: ((GpsReading) -> Unit)? = null
    private var thread: HandlerThread? = null
    private var listener: LocationListener? = null
    private var gnssCallback: GnssStatus.Callback? = null

    @SuppressLint("MissingPermission")
    override fun start(sink: (GpsReading) -> Unit) {
        this.sink = sink

        // Its own thread: these callbacks land on the looper that registered them, and
        // the main thread is the UI's. A fix that has to wait behind a layout pass gets a
        // receipt stamp that measures our scheduling rather than the location stack's.
        val worker = HandlerThread("dsrc-gps").also { it.start() }
        thread = worker
        val looper = worker.looper
        // The GNSS status callback takes a Handler and location updates take a Looper;
        // they are the same thread either way.
        val handler = Handler(looper)

        val status = object : GnssStatus.Callback() {
            override fun onSatelliteStatusChanged(status: GnssStatus) {
                var used = 0
                for (index in 0 until status.satelliteCount) {
                    if (status.usedInFix(index)) used++
                }
                // Used in fix, not visible: `num_sats` follows the GGA convention, and
                // visible-but-unused satellites contribute nothing to the position.
                satellitesInFix.set(used)
            }
        }
        gnssCallback = status
        manager.registerGnssStatusCallback(status, handler)

        val updates = LocationListener { location -> deliver(location) }
        listener = updates
        manager.requestLocationUpdates(
            LocationManager.GPS_PROVIDER,
            periodMs(config.gpsHz),
            // No distance filter: a stationary vehicle at a light still needs fixes, and
            // a filter here would silently become a second rate limit alongside the gate.
            0f,
            updates,
            looper,
        )
        Log.i(TAG, "GPS updates requested at ${config.gpsHz} Hz (${periodMs(config.gpsHz)} ms)")
    }

    override fun stop() {
        listener?.let { manager.removeUpdates(it) }
        gnssCallback?.let { manager.unregisterGnssStatusCallback(it) }
        listener = null
        gnssCallback = null
        sink = null
        // quitSafely, so a fix already on the queue is delivered rather than dropped
        // without being counted anywhere.
        thread?.quitSafely()
        thread = null
    }

    private fun deliver(location: Location) {
        val target = sink ?: return
        target(reading(extract(location, satellitesInFix.get())))
    }

    private fun extract(location: Location, satellites: Int) = PlatformFix(
        fixMonoNs = location.elapsedRealtimeNanos,
        receiptMonoNs = SystemClock.elapsedRealtimeNanos(),
        latitude = location.latitude,
        longitude = location.longitude,
        hasSpeed = location.hasSpeed(),
        speedMps = location.speed,
        hasBearing = location.hasBearing(),
        bearingDeg = location.bearing,
        hasAltitude = location.hasAltitude(),
        altitudeM = location.altitude,
        utcEpochMs = location.time,
        satellitesUsedInFix = satellites,
    )

    companion object {
        private const val TAG = "GpsLocationSource"

        /** Below this, `Location.time` is not a plausible GPS wall clock. */
        private const val EARLIEST_PLAUSIBLE_UTC_MS = 946_684_800_000L  // 2000-01-01Z

        fun periodMs(hz: Double): Long = (1_000.0 / hz).toLong().coerceAtLeast(0L)

        /**
         * Turn a platform fix into a wire record.
         *
         * Every conversion here is a place the units can be wrong, so each one is stated
         * rather than assumed: speed is already m/s, bearing is already degrees, altitude
         * is metres above the WGS84 ellipsoid, and `Location.time` is milliseconds where
         * the wire wants nanoseconds.
         */
        fun reading(fix: PlatformFix): GpsReading {
            val record = GpsPipeline.record(
                fixMonoNs = fix.fixMonoNs,
                latitude = fix.latitude,
                longitude = fix.longitude,
                // hasSpeed() false means the provider had none, which is present-and-null
                // on the wire. Zero would be a claim that the vehicle is stopped.
                speedMps = if (fix.hasSpeed) fix.speedMps.toDouble() else null,
                headingDeg = if (fix.hasBearing) normaliseBearing(fix.bearingDeg) else null,
                satellites = fix.satellitesUsedInFix.coerceAtLeast(0).toLong(),
                hdop = null,
                altitudeM = if (fix.hasAltitude) fix.altitudeM else null,
                utcEpochNs = utcNanos(fix.utcEpochMs),
            )
            return GpsReading(
                record = record,
                fixMonoNs = fix.fixMonoNs,
                // Both stamps come off elapsedRealtime, so unlike the camera's pair the
                // difference here is a real latency rather than a latency plus an unknown
                // offset between two clock bases.
                receiptMonoNs = maxOf(fix.receiptMonoNs, fix.fixMonoNs),
            )
        }

        /**
         * Fold a bearing into `[0, 360)`.
         *
         * Documented as being in that range already; devices have been seen reporting
         * exactly 360, and a negative value is the same bearing spelled the other way.
         * Sending either would be refused by our own outbound validation and counted as
         * a bug on this side, which is worse than normalising a value whose meaning is
         * unambiguous.
         */
        fun normaliseBearing(degrees: Float): Double {
            if (!degrees.isFinite()) return 0.0
            val folded = degrees.toDouble() % 360.0
            return if (folded < 0) folded + 360.0 else folded
        }

        /**
         * GPS wall time in nanoseconds, or null when the provider gave nothing usable.
         *
         * A zero or pre-2000 stamp means the provider has no time fix yet, which is
         * common in the first seconds of a cold start. Multiplying it up would put a
         * 1970 timestamp on the wire next to a valid position.
         */
        fun utcNanos(millis: Long): Long? {
            if (millis < EARLIEST_PLAUSIBLE_UTC_MS) return null
            if (millis > Long.MAX_VALUE / 1_000_000L) return null
            return millis * 1_000_000L
        }
    }
}
