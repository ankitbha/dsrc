package com.dsrc.phone

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Foreground service owning the whole sensing lifecycle.
 *
 * One service for all four modalities, so there is a single answer to "is sensing
 * running" -- four services would mean four lifecycles to keep in agreement, and a
 * partial failure with no name for it.
 *
 * Task 17 stands this up empty: it starts, holds the foreground notification, and
 * stops. Capture arrives in tasks 18-21 and hangs off [onSensingUp] / [onSensingDown].
 */
class SensingService : Service() {

    private var machine = SensingStateMachine()
    private val status = SensingStatus.shared

    private var lastStartId = 0

    override fun onBind(intent: Intent?): IBinder? = null

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
        super.onDestroy()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
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

    /** Capture starts here from task 18 on. */
    private fun onSensingUp() = Unit

    /** And is torn down here. */
    private fun onSensingDown() = Unit

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
