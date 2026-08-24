package com.dsrc.phone

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import androidx.lifecycle.LifecycleService
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import com.dsrc.phone.config.LinkConfig
import com.dsrc.phone.config.SensingConfig
import com.dsrc.phone.net.SessionHolder
import com.dsrc.phone.sensors.CameraPipeline
import com.dsrc.phone.sensors.CameraXSource
import com.dsrc.phone.sensors.CameraFrameSender
import com.dsrc.phone.sensors.GpsLocationSource
import com.dsrc.phone.sensors.GpsPipeline
import com.dsrc.phone.sensors.GpsReading
import com.dsrc.phone.sensors.GpsSource
import com.dsrc.transport.Channels
import com.dsrc.transport.Frame
import java.time.Instant
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Foreground service owning the whole sensing lifecycle.
 *
 * One service for all four modalities, so there is a single answer to "is sensing
 * running" -- four services would mean four lifecycles to keep in agreement, and a
 * partial failure with no name for it.
 *
 * One link, too, for the same reason: camera and GPS share a single session, so the
 * peer sees one connection whose per-channel counters add up, rather than two whose
 * relative timing it would have to reconstruct.
 */
class SensingService : LifecycleService() {

    private var machine = SensingStateMachine()
    private val status = SensingStatus.shared

    private var lastStartId = 0

    private var pipeline: CameraPipeline? = null
    private var cameraSource: CameraXSource? = null
    private var encodeExecutor: ExecutorService? = null
    private var link: SessionHolder? = null
    private var frameSender: CameraFrameSender? = null
    private var gpsPipeline: GpsPipeline? = null
    private var gpsSource: GpsSource? = null

    override fun onBind(intent: Intent): IBinder? {
        // LifecycleService dispatches lifecycle events from its overrides, so every one
        // of them has to call through even where the result is discarded.
        super.onBind(intent)
        return null
    }

    override fun onCreate() {
        super.onCreate()
        // The machine is per-instance while SensingStatus is process-global, so a new
        // instance and the published state have to be reconciled. Which way depends on
        // what was published:
        //
        //  - an active state is provably stale, because a brand-new instance means
        //    nothing is capturing; publish the truth instead;
        //  - a terminal stopped state is a fact the previous instance recorded, and
        //    overwriting it would erase the only place the failure was visible. Adopt
        //    it, so Stop can still clear it -- a Stop intent always arrives at a new
        //    instance, and a machine that started at IDLE would ignore it.
        val published = status.state
        if (published.requiresService) {
            Log.w(TAG, "published state $published cannot be true for a new instance")
            status.set(SensingState.IDLE)
        } else {
            machine = SensingStateMachine(published)
            Log.i(TAG, "created; adopted published state $published")
        }
    }

    override fun onDestroy() {
        // Reached without passing through the machine when the platform tears the
        // service down -- task removed, or a stop that skipped our path. Sensing is
        // definitively not running once the service is gone, and leaving the UI on
        // RUNNING would show a live session with an inert Stop button.
        Log.i(TAG, "destroying at ${machine.state}")
        if (machine.state.requiresService) {
            Log.w(TAG, "destroyed while ${machine.state}; publishing IDLE")
            status.set(SensingState.IDLE)
        }
        // Reached without passing through the machine on any platform teardown -- task
        // removed, low memory, an external stop -- and those paths never call
        // onSensingDown, so the camera and encoder threads outlived the service. The
        // camera itself is released by the service's own lifecycle unbinding CameraX,
        // which is why leaked threads were the only symptom and nothing noticed.
        onSensingDown()
        super.onDestroy()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        lastStartId = startId
        when (intent?.action) {
            ACTION_START -> handle(SensingEvent.Start)
            ACTION_STOP -> handle(SensingEvent.Stop)
            else -> Log.w(TAG, "ignoring intent with action ${intent?.action}")
        }

        // Every arriving intent creates this service, whether or not the machine acts
        // on it -- a duplicate stop and an unknown action both do. react() only stops
        // the service on a transition, so without this an ignored intent leaves it
        // resident and never in the foreground.
        if (!machine.state.requiresService) release()

        // The app decides when sensing runs. START_NOT_STICKY keeps the platform from
        // resurrecting a capture session nobody asked for after a low-memory kill.
        return START_NOT_STICKY
    }

    private fun handle(event: SensingEvent) {
        when (val transition = machine.offer(event)) {
            is Transition.Ignored -> {
                // Not an error: a redelivered start or a duplicate stop. Counted in the
                // machine so it stays visible rather than vanishing.
                Log.i(TAG, "ignored $event in ${transition.state}")
                return
            }
            is Transition.Accepted -> {
                Log.i(TAG, "${transition.from} -> ${transition.to} on $event")
                status.set(transition.to)
                react(transition.to)
            }
        }
    }

    private fun react(state: SensingState) {
        when (state) {
            SensingState.STARTING -> {
                try {
                    // The foreground call comes first, unconditionally. We were started
                    // with startForegroundService(), which is a promise to the platform
                    // to call startForeground() within a few seconds; returning before
                    // that -- as a permission refusal used to -- earns a
                    // ForegroundServiceDidNotStartInTimeException and kills the process.
                    // Refusing after entering the foreground costs a notification that
                    // is removed a moment later by release().
                    enterForegroundOrOverride()

                    val missing = missingPermissions()
                    if (missing.isNotEmpty()) {
                        // Revoked between the Activity's check and now. Not a failure:
                        // the remedy is a grant, not a retry.
                        Log.w(TAG, "cannot start, missing $missing")
                        handle(SensingEvent.PermissionRevoked)
                        return
                    }

                    onSensingUp()
                    handle(SensingEvent.Started)
                } catch (t: Throwable) {
                    // Startup failure has to land in the machine, not just the log, or
                    // the UI would show STARTING for the rest of the drive.
                    Log.e(TAG, "sensing failed to start", t)
                    handle(SensingEvent.Failed(t.toString()))
                }
            }
            SensingState.STOPPING -> {
                try {
                    onSensingDown()
                    handle(SensingEvent.Stopped)
                } catch (t: Throwable) {
                    // The machine models a failure during shutdown; until now nothing
                    // could produce one, and the exception escaped to onStartCommand
                    // and killed the process instead.
                    Log.e(TAG, "sensing failed to stop cleanly", t)
                    handle(SensingEvent.Failed(t.toString()))
                }
            }
            SensingState.IDLE,
            SensingState.STOPPED_ERROR,
            SensingState.STOPPED_PERMISSION_REVOKED,
            -> release()
            SensingState.RUNNING -> Unit
        }
    }

    /**
     * Required permissions that are not granted right now.
     *
     * Routed through an overridable hook because the refusal branch is otherwise
     * unreachable under test: the instrumentation grant rule satisfies every
     * requirement, so `missing` was always empty and the whole gate never ran.
     * Revoking a live permission instead would restart the process mid-test.
     */
    private fun missingPermissions(): List<String> =
        permissionOverride?.invoke()
            ?: PermissionGate.missing(PermissionModel.required(Build.VERSION.SDK_INT)) {
                checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED
            }

    private fun onSensingUp() {
        // No re-entrancy release here, and the reason is worth stating because an earlier
        // version had one.
        //
        // Fields cannot be live on entry. `react()` reaches this only on a transition into
        // STARTING, and every route into STARTING has already cleared them: a previous
        // come-up that threw released in `allocateAndStart`'s own catch below; a previous
        // run that succeeded is RUNNING, where the machine ignores Start; and STOPPING has
        // been through `onSensingDown`. So a conditional release at the top could never
        // fire, and it was unpinnable for that reason rather than for a hard-to-hit race --
        // which is what I first recorded, wrongly.
        //
        // A guard nothing can reach is the shape this codebase keeps producing, so it is
        // gone and the invariant is written down instead. The catch below is the
        // load-bearing half and it *is* reachable.
        try {
            allocateAndStart()
        } catch (t: Throwable) {
            // Whatever was published before the throw is released here rather than left to
            // a teardown that may not come.
            Log.e(TAG, "sensing failed to come up; releasing what was allocated", t)
            onSensingDown()
            throw t
        }
    }

    private fun allocateAndStart() {
        val config = SensingConfig()

        // The link first, so both modalities have somewhere to send to before either can
        // produce anything. It is up before it is connected -- send() refuses until the
        // handshake completes, counted where it happens.
        val holder = SessionHolder(
            config = LinkConfig(),
            deviceId = deviceId(),
            monoClock = SystemClock::elapsedRealtimeNanos,
            wallClock = ::wallClockNanos,
            onFrame = ::onInboundFrame,
        )
        link = holder
        holder.start()

        // One thread, because two would let frames finish compressing out of order and
        // make the monotonic frame_id a lie.
        val encoder = Executors.newSingleThreadExecutor()
        val pipe = CameraPipeline(config, encoder)
        val source = CameraXSource(this, this, config)

        // Published before start(), so a throw out of start still leaves the executor
        // and pipeline reachable for onSensingDown to release. Assigning afterwards
        // orphaned the encoder thread with nothing holding a reference to it.
        pipeline = pipe
        cameraSource = source
        encodeExecutor = encoder

        val sender = CameraFrameSender(
            drain = pipe::drain,
            send = { channel, extensions, payload -> holder.send(channel, extensions, payload) },
        )
        frameSender = sender
        sender.start()

        val gps = GpsPipeline(config) { reading ->
            holder.send(Channels.GPS, reading.record.toExtensions())
        }
        gpsPipeline = gps
        val locations = GpsLocationSource(this, config)
        gpsSource = locations

        source.start(pipe)
        locations.start { reading ->
            recordReceipt(reading)
            gps.offer(reading)
        }
        Log.i(
            TAG,
            "capture starting: camera ${config.cameraHz} Hz, gps ${config.gpsHz} Hz, " +
                "link ${LinkConfig().host}:${LinkConfig().port}",
        )
        // Last, so a test can fail a start with every resource already published -- which is
        // the only shape in which the leak this guards against can happen. Unreachable in
        // production for the same reason the other two seams are: nothing here throws on a
        // healthy device.
        comeUpFailureOverride?.invoke()
    }

    /**
     * Both GPS clocks, which is the half of the task the wire cannot carry.
     *
     * `t_capture_mono_ns` takes the fix time; the frozen contract has no field for
     * receipt, so it goes here. Logcat is the interim destination -- task 25 gives it a
     * file -- and the pair is logged together rather than separately because the
     * difference is the number worth having: it is the location stack's own delivery
     * latency, and unlike the camera's two stamps both of these come off
     * `elapsedRealtime`, so subtracting them is legitimate.
     */
    private fun recordReceipt(reading: GpsReading) {
        Log.i(
            TAG,
            "gps fix=${reading.fixMonoNs} recv=${reading.receiptMonoNs} " +
                "latency_ms=${reading.deliveryLatencyNs / 1_000_000} valid=${reading.record.valid} " +
                "sats=${reading.record.satellites}",
        )
    }

    /**
     * Inbound traffic. Routing it is tasks 22 and 23; being counted is this task's job.
     *
     * Logged rather than dropped silently, so an advisory or a rate command arriving
     * before its handler exists shows up as an unhandled channel instead of as nothing at
     * all -- which is how a downlink that was never wired up looks identical to a Jetson
     * that never sent anything.
     */
    private fun onInboundFrame(frame: Frame) {
        Log.i(TAG, "inbound ${frame.channel} seq=${frame.sequence} (no handler yet)")
    }

    /**
     * A device identity for the hello.
     *
     * `ANDROID_ID` rather than `Build.MODEL`: the fleet is two identical handsets, so the
     * model name would give both the same id and the Jetson's logs could not tell one
     * drive from another. It needs no permission and is stable for this app on this
     * device.
     */
    private fun deviceId(): String {
        val id = Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)
        return if (id.isNullOrBlank()) "unknown-${Build.MODEL}" else id
    }

    /** Wall time in nanoseconds, matching `time.time_ns()` on the Jetson. */
    private fun wallClockNanos(): Long {
        val now = Instant.now()
        return now.epochSecond * 1_000_000_000L + now.nano
    }

    /**
     * Idempotent: called from the machine's teardown and again from onDestroy.
     *
     * Every release is independent and the fields are cleared in a `finally`, and that is
     * the whole point rather than defensive habit. Previously the nine calls ran in
     * sequence and the seven fields were nulled *last*, so a throw part-way through left
     * all of them set -- and `react(STOPPING)` catches exactly that and offers `Failed`,
     * which the machine turns into `STOPPED_ERROR`, from which `Start` goes to `STARTING`
     * and straight back into `onSensingUp` with a live CameraX binding, a live encoder
     * executor and a `SessionHolder` whose four threads and socket are still up. The new
     * holder then dials again and the Jetson sees a second session displacing the first.
     *
     * I had argued that route was unreachable and removed a guard on the strength of it.
     * The argument covered only the paths where teardown *succeeds*; the failure path is
     * the one the state machine explicitly models, sixty lines above. Making each release
     * independent closes it at the cause, so the invariant "no field is live on entry to
     * onSensingUp" holds because nothing can leave one set, not because a guard catches it.
     *
     * This method therefore does not throw, which has two consequences worth stating.
     * `onDestroy` calls it with no catch of its own, so a throw here was process death on
     * any platform teardown -- and a seam that escaped it did exactly that, killing the
     * instrumentation run with "Unable to stop service". And `react(STOPPING)`'s catch,
     * plus the machine's `Failed`-from-`STOPPING` arm, are now unreachable *through this
     * call*. They stay because that branch may grow a second call, but they are dead today
     * and saying so is better than leaving a reader to assume otherwise.
     */
    private fun onSensingDown() {
        // Order matters, and it is the reverse of construction: stop the producers first
        // so nothing new is offered, then the pipelines so anything queued drops out and
        // is counted, then the threads that were draining them, and the link last so a
        // frame already in flight still has somewhere to go.
        try {
            // First, and deliberately: a seam at the end simulates nothing, because no
            // release follows it and so none can be skipped. A test built on a trailing
            // seam passed whether or not the guard it meant to pin was there.
            release("test seam") { teardownFailureOverride?.invoke() }
            release("camera source") { cameraSource?.stop() }
            release("gps source") { gpsSource?.stop() }
            release("camera stats") { pipeline?.let { Log.i(TAG, "camera stats ${it.stats}") } }
            release("gps stats") { gpsPipeline?.let { Log.i(TAG, "gps stats ${it.stats}") } }
            release("camera pipeline") { pipeline?.stop() }
            release("gps pipeline") { gpsPipeline?.stop() }
            release("frame sender") { frameSender?.stop() }
            release("encoder") { encodeExecutor?.shutdown() }
            release("link stats") { link?.let { Log.i(TAG, "link stats ${it.stats()}") } }
            release("link") { link?.stop() }
        } finally {
            cameraSource = null
            gpsSource = null
            pipeline = null
            gpsPipeline = null
            frameSender = null
            encodeExecutor = null
            link = null
        }
    }

    /**
     * Run one release step, recording a failure rather than abandoning the rest.
     *
     * One resource refusing to stop must not leave the other six running: they are
     * independent, and the previous arrangement made them share a fate for no reason
     * beyond statement order.
     */
    private fun release(what: String, step: () -> Unit) {
        try {
            step()
        } catch (t: Throwable) {
            teardownFailures.incrementAndGet()
            Log.e(TAG, "releasing $what failed; continuing with the rest", t)
        }
    }

    /** The frame source, while sensing is up. */
    val frames: CameraPipeline? get() = pipeline

    /** The link, while sensing is up, for a test or a status reader. */
    val session: SessionHolder? get() = link

    /**
     * Entering the foreground, with a seam so its failure can be tested.
     *
     * The failure is the interesting case and it is unreachable otherwise: on a healthy
     * emulator `startForeground` always succeeds, so the catch that turns a refusal into
     * a recorded failure would never run under test.
     */
    private fun enterForegroundOrOverride() {
        val override = enterForegroundOverride
        if (override != null) override() else enterForeground()
    }

    private fun enterForeground() {
        createChannel()
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    /** Leave the foreground and let the service go. Safe to call more than once. */
    private fun release() {
        stopForegroundCompat()
        // The startId overload, so a start that arrived after this one was queued is
        // not discarded along with the stop.
        stopSelf(lastStartId)
    }

    /**
     * Leave the foreground explicitly.
     *
     * Deliberately not covered by a test: destroying the service also drops it out of
     * the foreground, so deleting this line passes the whole suite.
     *
     * The reason it cannot be pinned is not that the race is hard to build -- it is
     * that the outcome is unreachable as a lasting state. `release()` is only ever
     * called from `onStartCommand`'s call tree, so any later intent that makes
     * `stopSelf(lastStartId)` decline is itself an `onStartCommand` ending in either a
     * resident state or another `release()`. That guarantee is what makes this line
     * redundant today, and it breaks the moment a capture task calls `release()` from a
     * sensor callback or a coroutine -- at which point this becomes load-bearing.
     *
     * No version branch: STOP_FOREGROUND_REMOVE exists from API 24 and minSdk is 29.
     */
    private fun stopForegroundCompat() {
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    private fun createChannel() {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel_name),
            // Low: the notification is a requirement of running in the foreground,
            // not something to interrupt a driver with.
            NotificationManager.IMPORTANCE_LOW,
        )
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification =
        Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .build()

    companion object {
        private const val TAG = "SensingService"
        const val ACTION_START = "com.dsrc.phone.action.START"
        const val ACTION_STOP = "com.dsrc.phone.action.STOP"
        private const val CHANNEL_ID = "sensing"

        /**
         * Test seam for the permission gate. Null in production, where the real check
         * runs. Set by an instrumented test to drive the refusal path.
         */
        @Volatile
        internal var permissionOverride: (() -> List<String>)? = null

        /**
         * Test seam for the foreground transition. Null in production. Set by an
         * instrumented test to make it fail.
         */
        @Volatile
        internal var enterForegroundOverride: (() -> Unit)? = null

        /**
         * Test seam for a failure during teardown.
         *
         * Null in production. The route it opens -- a throw in `onSensingDown`, which
         * `react(STOPPING)` turns into `STOPPED_ERROR`, from which a `Start` re-enters
         * `onSensingUp` -- is the one I wrongly argued was unreachable, and nothing on a
         * healthy device throws there.
         */
        @Volatile
        internal var teardownFailureOverride: (() -> Unit)? = null

        /**
         * Test seam for a failure *after* everything is allocated.
         *
         * Null in production. The leak it exposes needs a throw with all seven fields
         * published, and nothing on a healthy device throws there.
         */
        @Volatile
        internal var comeUpFailureOverride: (() -> Unit)? = null

        /**
         * Release steps that threw, across every instance.
         *
         * On the companion rather than the instance because teardown is the last thing an
         * instance does: by the time a test can ask, there is nothing to ask. A test that
         * uses the seam above reads this to confirm the seam actually fired -- without it
         * the test passes just as well against a seam that was never wired up, which is
         * the shape of an assertion that proves nothing.
         */
        internal val teardownFailures = java.util.concurrent.atomic.AtomicInteger(0)
        private const val NOTIFICATION_ID = 1

        /**
         * Start sensing.
         *
         * Deliberately `startService`, not `startForegroundService`. The latter is a
         * promise to the platform to call `startForeground()` within moments, and
         * failing to keep it kills the whole process -- the ActivityManager throws
         * `ForegroundServiceDidNotStartInTimeException` the instant the service is
         * brought down with the promise outstanding, which is 1 ms after a failed
         * start, not after any timeout. No try/catch inside the service can survive
         * that, because the teardown itself is the trigger.
         *
         * The promise only exists to permit a background start, and the only caller is
         * a visible Activity, so there is nothing to gain by making it. Without it, a
         * `startForeground()` that throws -- a missing type permission on API 34+, a
         * start attributed to the background on API 31+ -- becomes a recorded failure
         * the driver can see instead of a dead process showing nothing.
         *
         * A future background trigger would have to make the promise, and would then
         * need a fallback that satisfies it before giving up.
         */
        fun start(context: Context) = context.startService(
            Intent(context, SensingService::class.java).setAction(ACTION_START)
        )

        fun stop(context: Context) = context.startService(
            Intent(context, SensingService::class.java).setAction(ACTION_STOP)
        )
    }
}
