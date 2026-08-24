package com.dsrc.phone

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat

/**
 * Start/stop control and a state readout.
 *
 * Deliberately thin: it reads [SensingStatus] and sends intents. The driver-facing
 * UI is task 23 and will attach to the same holder, so nothing here needs to move
 * when it arrives.
 */
class MainActivity : ComponentActivity() {

    private lateinit var stateLabel: TextView
    private lateinit var speedLabel: TextView
    private lateinit var adviceLabel: TextView
    private lateinit var confidenceLabel: TextView

    /**
     * Redraws the advisory panel.
     *
     * On its own tick rather than only on arrival, because the panel has to go blank when
     * *nothing* arrives -- that is the whole of the staleness rule, and an event-driven
     * redraw can only ever react to an event that happened.
     */
    private val advisoryTick = object : Runnable {
        override fun run() {
            refreshAdvisory()
            advisoryHandler.postDelayed(this, ADVISORY_TICK_MS)
        }
    }

    private val advisoryHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private lateinit var asked: AskedPermissions

    /** Set when the user pressed Start but permissions had to be requested first. */
    private var startRequested = false

    private val listener = SensingStatus.Listener {
        runOnUiThread { refresh() }
    }

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            // Only refusals are recorded, and a grant clears any earlier refusal.
            // Recording grants too would make a later revoke look permanent forever,
            // since the record never shrank.
            val split = PermissionResult.split(result)
            asked.markRefused(split.refused)
            asked.clearRefused(split.granted)
            // Continue the Start the user actually asked for. Without this, granting
            // returns to an idle screen and the user has to press Start again with no
            // indication that anything happened.
            if (startRequested && PermissionModel.next(currentStates()) is PermissionAction.Proceed) {
                startRequested = false
                SensingService.start(this)
            }
            refresh()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        asked = AskedPermissions(getSharedPreferences(PREFS, Context.MODE_PRIVATE))
        // The pending-result dispatch happens on the *new* instance after a rotation or
        // a process death mid-dialog, so a plain field would be false exactly when it
        // was needed.
        startRequested = savedInstanceState?.getBoolean(KEY_START_REQUESTED, false) ?: false

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            layoutParams = ViewGroup.LayoutParams(MATCH, MATCH)
        }
        stateLabel = TextView(this).apply { textSize = 20f }
        root.addView(stateLabel)

        // The advisory panel. Every string on it is the Jetson's: `rec_speed_display` and
        // `current_speed_display` arrive already converted into `units`, and the three text
        // fields arrive as text. The phone formats nothing, because a phone that rounded
        // 30.4 to 30 while the Jetson meant 30.4 would be showing a recommendation nobody
        // made.
        // Tagged so a test can address the advisory panel specifically. Walking every
        // TextView picks up the state line and both buttons, which are TextViews too, and a
        // test that asserts "nothing is displayed" would then be asserting the screen is
        // empty rather than that the panel is.
        speedLabel = TextView(this).apply { textSize = 44f; tag = TAG_SPEED }
        adviceLabel = TextView(this).apply { textSize = 18f; tag = TAG_ADVICE }
        confidenceLabel = TextView(this).apply { textSize = 16f; tag = TAG_CONFIDENCE }
        root.addView(speedLabel)
        root.addView(adviceLabel)
        root.addView(confidenceLabel)
        root.addView(Button(this).apply {
            text = "Start sensing"
            setOnClickListener { onStartClicked() }
        })
        root.addView(Button(this).apply {
            text = "Stop sensing"
            setOnClickListener { onStopClicked() }
        })
        setContentView(root)
    }

    override fun onStart() {
        super.onStart()
        SensingStatus.shared.addListener(listener)
        refresh()
        // Synchronously, before the resume traversal draws. Posting the tick is not enough:
        // the traversal goes through Choreographer's async message, which jumps ahead of an
        // ordinary posted callback, so the window was painted from whatever text the labels
        // still held -- the advisory that was on screen when the driver looked away, however
        // many minutes ago. Measured at two consecutive frames spanning ~29 ms showing an
        // advisory that had expired three and a half seconds earlier.
        //
        // With `blankAdvisory()` in `onStop` this is no longer what keeps a *stale* advisory
        // off the screen, and no test can separate the two: deleting this line alone leaves
        // the suite green. What it still does is put a *current* advisory up on the frame
        // the driver returns to, rather than a blank panel for up to one tick. Kept for
        // that, and unpinned, which is said here rather than left for a green suite to
        // imply.
        refreshAdvisory()
        advisoryHandler.post(advisoryTick)
    }

    override fun onStop() {
        SensingStatus.shared.removeListener(listener)
        advisoryHandler.removeCallbacks(advisoryTick)
        // And blanked on the way out, so there is nothing stale to paint even if something
        // draws before onStart runs. Belt and braces on a driver-facing surface: the two
        // together mean no path exists on which an old recommendation reaches a pixel.
        blankAdvisory()
        super.onStop()
    }

    /**
     * Show the current advisory, or nothing at all.
     *
     * "Nothing at all" is a real state and it is the safe one: an advisory that has expired
     * is about road the driver has covered, and leaving it up would say otherwise.
     */
    private fun blankAdvisory() {
        speedLabel.text = ""
        adviceLabel.text = ""
        confidenceLabel.text = ""
    }

    private fun refreshAdvisory() {
        val advisory = SensingService.advisories.current(android.os.SystemClock.elapsedRealtimeNanos())
        if (advisory == null) {
            blankAdvisory()
            return
        }
        speedLabel.text = "${advisory.recSpeedDisplay} ${advisory.units}"
        adviceLabel.text = listOf(advisory.laneText, advisory.mergeText, advisory.trafficText)
            .filter { it.isNotBlank() }
            .joinToString("  \u00b7  ")
        confidenceLabel.text = advisory.confidenceLabel
    }

    private fun onStartClicked() {
        startRequested = true
        when (val action = PermissionModel.next(currentStates())) {
            is PermissionAction.Proceed -> {
                startRequested = false
                SensingService.start(this)
            }
            is PermissionAction.Request -> requestPermissions.launch(action.permissions.toTypedArray())
            // Rationale and a fresh request are the same gesture here; a real
            // explanation screen is UI work that belongs with task 23.
            is PermissionAction.Rationale -> requestPermissions.launch(action.permissions.toTypedArray())
            is PermissionAction.OpenSettings -> openAppSettings()
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putBoolean(KEY_START_REQUESTED, startRequested)
    }

    private fun onStopClicked() {
        // A stop cancels a start the user is midway through granting for, or the
        // permission dialog's result would start sensing they just asked to end.
        startRequested = false
        SensingService.stop(this)
    }

    private fun refresh() {
        stateLabel.text = buildString {
            append(getString(R.string.app_name))
            append(": ")
            append(SensingStatus.shared.state.name)
            val action = PermissionModel.next(currentStates())
            if (action !is PermissionAction.Proceed) {
                append("\n(permissions: ")
                append(action::class.simpleName)
                append(")")
            }
        }
    }

    private fun currentStates(): Map<String, PermissionState> {
        // A permission granted in Settings never reaches the dialog callback, so the
        // refusal record has to be reconciled against reality wherever it is read.
        // Without this, granting outside the app leaves the record set, and the next
        // revoke reads as permanently denied for the rest of the install.
        val granted = PermissionModel.required(Build.VERSION.SDK_INT).filter {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
        asked.clearRefused(granted)
        return statesFor(granted)
    }

    private fun statesFor(granted: List<String>): Map<String, PermissionState> =
        PermissionModel.required(Build.VERSION.SDK_INT).associateWith { permission ->
            PermissionModel.classify(
                granted = permission in granted,
                shouldShowRationale = shouldShowRequestPermissionRationale(permission),
                hasAsked = asked.hasAsked(permission),
            )
        }

    private fun openAppSettings() {
        startActivity(
            Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                .setData(Uri.fromParts("package", packageName, null))
        )
    }

    companion object {
        /** View tags for the three advisory labels, so a test can address just those. */
        const val TAG_SPEED = "advisory-speed"
        const val TAG_ADVICE = "advisory-advice"
        const val TAG_CONFIDENCE = "advisory-confidence"

        /**
         * How often the advisory panel is redrawn.
         *
         * Well under the three-second expiry, so a stale advisory leaves the screen within
         * a tick of becoming stale rather than lingering until the next arrival.
         */
        const val ADVISORY_TICK_MS = 250L

        const val PREFS = "permissions"
        const val KEY_START_REQUESTED = "startRequested"
        const val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
    }
}
