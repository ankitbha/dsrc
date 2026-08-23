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

    private val listener = SensingStatus.Listener { state ->
        runOnUiThread { stateLabel.text = getString(R.string.app_name) + ": " + state.name }
    }

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            // Record every permission we actually asked about. Without this the app
            // cannot tell "never asked" from "denied for good" -- the platform reports
            // both the same way.
            asked.markAsked(result.keys)
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
            setOnClickListener { SensingService.stop(this@MainActivity) }
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
        when (val action = PermissionModel.next(currentStates())) {
            is PermissionAction.Proceed -> SensingService.start(this)
            is PermissionAction.Request -> requestPermissions.launch(action.permissions.toTypedArray())
            // Rationale and a fresh request are the same gesture here; a real
            // explanation screen is UI work that belongs with task 23.
            is PermissionAction.Rationale -> requestPermissions.launch(action.permissions.toTypedArray())
            is PermissionAction.OpenSettings -> openAppSettings()
        }
    }

    private fun refresh() {
        stateLabel.text = buildString {
            append(SensingStatus.shared.state.name)
            val action = PermissionModel.next(currentStates())
            if (action !is PermissionAction.Proceed) append("  (permissions: $action)")
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
