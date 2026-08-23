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

    private val machine = SensingStateMachine()
    private val status = SensingStatus.shared

    private var lastStartId = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        // The machine is per-instance while SensingStatus is process-global, so a new
        // instance must publish its own state or the UI keeps showing whatever the
        // previous instance last said. Any intent creates an instance, so this runs
        // before the first event is handled.
        status.set(machine.state)
    }

    override fun onDestroy() {
        // Reached without passing through the machine when the platform tears the
        // service down -- task removed, or a stop that skipped our path. Sensing is
        // definitively not running once the service is gone, and leaving the UI on
        // RUNNING would show a live session with an inert Stop button.
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
                val missing = PermissionGate.missing(
                    PermissionModel.required(Build.VERSION.SDK_INT),
                ) { checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED }
                if (missing.isNotEmpty()) {
                    // Revoked between the Activity's check and now. Not a failure --
                    // the remedy is a grant, not a retry -- and going to the foreground
                    // with a camera type and no camera permission gets the service
                    // killed by the platform rather than started.
                    Log.w(TAG, "cannot start, missing $missing")
                    handle(SensingEvent.PermissionRevoked)
                    return
                }
                try {
                    enterForeground()
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
                onSensingDown()
                handle(SensingEvent.Stopped)
            }
            SensingState.IDLE,
            SensingState.STOPPED_ERROR,
            SensingState.STOPPED_PERMISSION_REVOKED,
            -> release()
            SensingState.RUNNING -> Unit
        }
    }

    /** Capture starts here from task 18 on. */
    private fun onSensingUp() = Unit

    /** And is torn down here. */
    private fun onSensingDown() = Unit

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

    private fun stopForegroundCompat() {
        // No version branch: STOP_FOREGROUND_REMOVE exists from API 24 and minSdk is 29.
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
        private const val NOTIFICATION_ID = 1

        fun start(context: Context) = context.startForegroundService(
            Intent(context, SensingService::class.java).setAction(ACTION_START)
        )

        fun stop(context: Context) = context.startService(
            Intent(context, SensingService::class.java).setAction(ACTION_STOP)
        )
    }
}
