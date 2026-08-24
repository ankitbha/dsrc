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
                    // After come-up and inside the same try, which is the whole point.
                    // `handle` publishes the state, so anything raised from here on is
                    // caught below and offered as `Failed` while the machine is already
                    // RUNNING -- an arm the machine accepts, and until round 4 an arm with
                    // no teardown behind it.
                    //
                    handle(SensingEvent.Started)
                    // *After* handle, and that is the entire point of the seam. Until
                    // handle publishes the state the machine is still STARTING, so a throw
                    // above this line takes the STARTING + Failed arm -- the ordinary
                    // come-up-failure route, which already has a dozen ways to happen and
                    // was never what this is for. Round 5 proved the difference by making
                    // RUNNING + Failed unreachable in the machine: all 53 instrumented
                    // tests still passed, because nothing was walking it.
                    //
                    // A seam, because with listener failures contained there is nothing
                    // left in this window that throws, and the arm is still reachable by
                    // construction: the next watchdog or bind escalation to offer Failed
                    // from RUNNING arrives exactly here.
                    startedFailureOverride?.invoke()
                } catch (t: Throwable) {
                    // Startup failure has to land in the machine, not just the log, or
                    // the UI would show STARTING for the rest of the drive.
                    Log.e(TAG, "sensing failed to start", t)
                    handle(SensingEvent.Failed(t.toString()))
                }
            }
            SensingState.STOPPING -> {
                // No teardown here. It used to sit above this line, and deleting it left
                // all 53 instrumented tests passing -- correctly, because every route
                // through STOPPING reaches handle(Stopped) -> IDLE -> react(IDLE), where
                // teardown now lives. A release nothing can skip is the same shape as a
                // guard nothing can reach, which is the thing this file keeps getting
                // wrong in the other direction.
                try {
                    handle(SensingEvent.Stopped)
                } catch (t: Throwable) {
                    // Recorded here, not offered to the machine. Round 4 showed why: the
                    // only way into this catch is through handle's own subtree, and
                    // `machine.offer` advances the state *before* anything below it runs --
                    // so by the time we arrive the machine is already IDLE, where `Failed`
                    // is ignored. The previous shape offered it anyway and left the failure
                    // in a log line: no STOPPED_ERROR, no lastFailure, nothing a caller
                    // could see. The claim that this catch was dead was wrong; what is
                    // actually dead is the machine's Failed-from-STOPPING arm, and only
                    // because every reachable door has the machine past STOPPING.
                    shutdownFailures.incrementAndGet()
                    lastShutdownFailure = "${t.javaClass.name}: ${t.message}"
                    Log.e(TAG, "sensing failed to stop cleanly", t)
                }
            }
            SensingState.IDLE,
            SensingState.STOPPED_ERROR,
            SensingState.STOPPED_PERMISSION_REVOKED,
            -> {
                // Teardown belongs to *entering a stopped state*, not to the one transition
                // that happens to pass through STOPPING. Round 4 found the hole by the door
                // I had not enumerated: `react(STARTING)`'s try encloses `handle(Started)`
                // too, so anything raised while publishing RUNNING is caught as a start
                // failure and offered as `Failed` from RUNNING -- an arm the machine accepts
                // -- and that route reaches STOPPED_ERROR without going near onSensingDown.
                // Every worker stayed live, and the Start that STOPPED_ERROR accepts then
                // built a second set on top.
                //
                // Putting it here closes the route for whatever offers `Failed` next -- a
                // link watchdog, a camera-bind escalation -- rather than for the one trigger
                // that exposed it. onSensingDown is idempotent and does not throw, so
                // arriving here from STOPPING, where it has already run, costs nothing.
                onSensingDown()
                release()
            }
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
     * instrumentation run with "Unable to stop service". It is now called outside
     * `react(STOPPING)`'s try, since leaving it inside implied a failure mode it does not
     * have.
     *
     * The second consequence I got wrong. I wrote that `react(STOPPING)`'s catch was dead
     * along with the machine's `Failed`-from-`STOPPING` arm; round 4 showed the catch is
     * reachable through the *other* statement in that try, and that when it fired the
     * failure was recorded nowhere. The arm is dead, but for a different reason: every
     * reachable door has the machine past `STOPPING` by the time the catch runs.
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
            release("camera pipeline") { pipeline?.stop() }
            release("gps pipeline") { gpsPipeline?.stop() }
            release("frame sender") { frameSender?.stop() }
            release("encoder") { encodeExecutor?.shutdown() }
            // Stats *after* the stops, and round 5 is why. `abandoned`, `refusedStopped`
            // and the buffer's `discarded` all require `running` to be false or the
            // executor to be down, so logging here read every one of them as zero on every
            // call -- and this log line is the only production reader of these stats. The
            // frames still queued at teardown were counted as `inFlight`, so the teardown
            // log read as an encoder backlog: exactly the misreading `abandoned` was added
            // to remove, by a recorder placed where it could not record.
            release("camera stats") {
                pipeline?.let {
                    if (!it.isStopped) statsReadBeforeStop.incrementAndGet()
                    Log.i(TAG, "camera stats ${it.stats}")
                }
            }
            release("gps stats") {
                gpsPipeline?.let {
                    if (!it.isStopped) statsReadBeforeStop.incrementAndGet()
                    // The two silent corrections are logged beside the pipeline's own
                    // counters. Without this they had no production reader at all: a
                    // corrected stamp and a rewritten satellite count were visible only to
                    // a unit test, in a task whose deliverable is logging both clocks.
                    Log.i(
                        TAG,
                        "gps stats ${it.stats} " +
                            "clampedReceipts=${GpsLocationSource.clampedReceipts.get()} " +
                            "clampedSatellites=${GpsLocationSource.clampedSatellites.get()}",
                    )
                }
            }
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
            resourcesHeldAfterTeardown = listOfNotNull(
                cameraSource, gpsSource, pipeline, gpsPipeline, frameSender, encodeExecutor, link,
            ).size
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
         * Test seam for a failure *after* come-up, while the machine is RUNNING.
         *
         * Null in production. It stands in for the next thing that legitimately raises
         * there -- the machine pins `RUNNING + Failed -> STOPPED_ERROR`, so the route
         * exists whether or not anything currently walks it.
         */
        @Volatile
        internal var startedFailureOverride: (() -> Unit)? = null

        /**
         * How many resource fields were still set when teardown finished.
         *
         * Round 4 pointed out that deleting all seven assignments above left every teardown
         * test green, and the reason it did is worth stating: once each release is
         * independently guarded, a stale field is behaviourally inert -- the object behind
         * it is already stopped, and the next come-up overwrites it. What a stale field
         * actually costs is memory, and it is not small: a retained CameraPipeline holds its
         * whole ring buffer of encoded frames, plus an encoder executor and a socket, for
         * the life of the process.
         *
         * So the property is asserted directly rather than through a consequence it does
         * not have. This does sit next to what it measures, which means one edit could
         * remove both -- said plainly rather than left for a reader to notice.
         */
        @Volatile
        internal var resourcesHeldAfterTeardown: Int = -1

        /**
         * Failures raised while publishing the stop, which the machine cannot represent.
         *
         * There is no reachable producer, and that is a stronger statement than the one
         * this said before ("a platform call I could not make throw"). With `onSensingDown`
         * non-throwing and listener failures contained, the whole subtree of
         * `handle(Stopped)` is `machine.offer`, which is pure, plus `Log`, `stopForeground`
         * and `stopSelf`. So this is not an unpinned recorder -- it is a recorder waiting
         * for a producer that does not exist yet, kept because the subtree will grow and a
         * failure there must not be a log line nobody reads.
         */
        internal val shutdownFailures = java.util.concurrent.atomic.AtomicInteger(0)

        @Volatile
        internal var lastShutdownFailure: String? = null

        /**
         * Stats lines read while their pipeline was still running.
         *
         * The ordering is the finding, not the logging: `abandoned`, `refusedStopped` and
         * the buffer's `discarded` can only move once the pipeline is stopped, and this log
         * line is the only production reader of either pipeline's stats. Reading first made
         * all three structurally zero on every call, while the frames still queued counted
         * as `inFlight` -- so the teardown log read as an encoder backlog, which is the
         * misreading `abandoned` was added to remove.
         *
         * A plain reordering leaves nothing behind for a test to see, so the ordering
         * counts itself.
         */
        internal val statsReadBeforeStop = java.util.concurrent.atomic.AtomicInteger(0)

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
