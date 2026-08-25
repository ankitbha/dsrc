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

    // Volatile because `stop()` writes them on the main thread while `setRate` reads them
    // on the transport's delivery thread, with no other happens-before edge between the two.
    @Volatile
    private var sink: ((GpsReading) -> Unit)? = null

    @Volatile
    private var thread: HandlerThread? = null

    @Volatile
    private var listener: LocationListener? = null

    @Volatile
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
        this.looper = looper
        request(config.gpsHz)
    }

    /**
     * Ask the provider for a new interval.
     *
     * The rate gate can only ever *lower* a rate -- it drops fixes the provider already
     * delivered -- so a command raising `gps_hz` above what was requested at start changed
     * nothing while the pipeline reported the new rate as in force.
     *
     * `requestLocationUpdates` with the same listener replaces the previous request rather
     * than adding a second one, so this is not restarting capture: the pipeline, its
     * counters and the session are untouched.
     *
     * **Not pinned by a test.** The IMU half of this is, by counting frames the peer
     * receives after a raise the gate cannot serve. The GPS half is not: the emulator's
     * provider serves a static position, so asking it for a shorter interval does not make
     * it deliver faster and no measurement on this machine can tell the fix from its
     * absence. Same code path and same reasoning as the IMU; said here rather than left to
     * a green suite to imply.
     */
    /**
     * Ask the provider for a new interval.
     *
     * Synchronized against [stop], and that is not decoration. A `rate_cmd` is applied on
     * the transport's delivery thread while `stop()` runs on the main thread, and the guard
     * used to be a plain read of two fields followed by a call. If the stop interleaved
     * between them, the re-request landed *after* `removeUpdates` — and since the service
     * nulls its reference in the same breath, nothing was left that could ever remove the
     * updates again. The location indicator stays lit for the life of the process, no data
     * flows because `sink` is null, and no counter moves: it is invisible from inside.
     *
     * The fields it reads are also `@Volatile` now, for the same reason `ImuSource` made
     * its own so and said as much: this class has the identical two-thread situation and
     * was the one that did not.
     */
    @Synchronized
    @SuppressLint("MissingPermission")
    fun setRate(hz: Double) {
        if (listener != null && looper != null) request(hz)
    }

    @SuppressLint("MissingPermission")
    private fun request(hz: Double) {
        val updates = listener ?: return
        val target = looper ?: return
        manager.requestLocationUpdates(
            LocationManager.GPS_PROVIDER,
            periodMs(hz),
            // No distance filter: a stationary vehicle at a light still needs fixes, and
            // a filter here would silently become a second rate limit alongside the gate.
            0f,
            updates,
            target,
        )
        requestedHz = hz
        Log.i(TAG, "GPS updates requested at $hz Hz (${periodMs(hz)} ms)")
    }

    /** The interval last asked of the provider, which bounds what the gate can pass. */
    @Volatile
    var requestedHz: Double = 0.0
        private set

    @Volatile
    private var looper: android.os.Looper? = null

    @Synchronized
    override fun stop() {
        // A rate command already waiting on this monitor gets in only after `listener` is
        // null below, so it re-requests nothing.
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
                // Counted, like the receipt clamp above it and for the same reason. A
                // negative platform count was silently rewritten to 0, which on the wire is
                // indistinguishable from "no satellites used" -- on a record carrying
                // valid = true and fix_quality = 1. A correction nobody can see is a
                // measurement nobody can trust.
                satellites = if (fix.satellitesUsedInFix < 0) {
                    clampedSatellites.incrementAndGet()
                    0L
                } else {
                    fix.satellitesUsedInFix.toLong()
                },
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
                //
                // The clamp is counted, and it was not. Its counterpart on the other stamp
                // -- a fix arriving with a *fix* time older than the previous one -- is
                // counted as `nonMonotonicFixes`, but a receipt stamp older than the fix it
                // belongs to was silently corrected. That made two of the three clock
                // assertions in the on-device test tautologies: `receiptMonoNs >=
                // fixMonoNs` and a non-negative latency are true for every input once the
                // clamp is applied, so only the 10-second upper bound could fail, in a test
                // whose subject is both clocks. In a task whose deliverable is logging fix
                // time and receipt time, a corrected stamp is exactly the thing worth
                // knowing about.
                receiptMonoNs = if (fix.receiptMonoNs < fix.fixMonoNs) {
                    clampedReceipts.incrementAndGet()
                    fix.fixMonoNs
                } else {
                    fix.receiptMonoNs
                },
            )
        }

        /**
         * Fixes whose receipt stamp arrived older than the fix stamp, and was clamped.
         *
         * Non-zero means the provider's two clocks disagree, which is worth seeing rather
         * than corrected in silence. Beside `reading`, which is where the clamp happens
         * and which is a companion function because the tests drive it directly.
         */
        val clampedReceipts = java.util.concurrent.atomic.AtomicLong(0)

        /** Fixes whose satellite count arrived negative, and was rewritten to zero. */
        val clampedSatellites = java.util.concurrent.atomic.AtomicLong(0)

        /**
         * Fold a bearing into `[0, 360)`, or null if there is no bearing to fold.
         *
         * Documented as being in that range already; devices have been seen reporting
         * exactly 360, and a negative value is the same bearing spelled the other way.
         * Folding those is right because their meaning is unambiguous: 361 is 1, and -10
         * is 350.
         *
         * The reason this used to give for folding was wrong, and worth correcting rather
         * than deleting: it said an out-of-range bearing "would be refused by our own
         * outbound validation". It would not. Neither implementation range-checks
         * `heading_deg` -- Kotlin only calls `checkFinite`, and Python accepts 360, -10,
         * 720 and 1e9 alike, all verified. The folding stands on its own; it was not
         * standing on that.
         *
         * A non-finite bearing returns **null**, not zero. Zero is a legitimate heading,
         * so returning it for "no idea" makes an unknown indistinguishable from a claim of
         * due north -- and `heading_deg` is nullable precisely so it does not have to be.
         * The spec is explicit that an unknown value is present-and-null, never a sentinel.
         */
        fun normaliseBearing(degrees: Float): Double? {
            if (!degrees.isFinite()) return null
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
