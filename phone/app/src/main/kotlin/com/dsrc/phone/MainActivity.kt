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
            asked.markRefused(result.filterValues { !it }.keys)
            asked.clearRefused(result.filterValues { it }.keys)
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

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            layoutParams = ViewGroup.LayoutParams(MATCH, MATCH)
        }
        stateLabel = TextView(this).apply { textSize = 20f }
        root.addView(stateLabel)
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
    }

    override fun onStop() {
        SensingStatus.shared.removeListener(listener)
        super.onStop()
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

    private fun currentStates(): Map<String, PermissionState> =
        PermissionModel.required(Build.VERSION.SDK_INT).associateWith { permission ->
            PermissionModel.classify(
                granted = ContextCompat.checkSelfPermission(this, permission) ==
                    PackageManager.PERMISSION_GRANTED,
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

    private companion object {
        const val PREFS = "permissions"
        const val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
    }
}
