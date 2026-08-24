package com.dsrc.phone

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Ties [PermissionModel]'s constants to the permissions actually declared.
 *
 * Without this, every permission test asserts against the same constant it is
 * testing, so a typo in the string defines its own truth and passes. The runtime
 * consequence is not a crash but something worse: `checkSelfPermission` on an
 * undeclared permission is permanently denied and `requestPermissions` returns denied
 * without showing a dialog, which walks the flow straight to the Settings dead end
 * the never-asked/permanently-denied split exists to avoid.
 */
class ManifestPermissionsTest {

    private val manifest: String by lazy {
        val path = System.getProperty("dsrc.manifest")
            ?: error("dsrc.manifest is not set; the build must pass the manifest path")
        val file = File(path)
        require(file.isFile) { "manifest not found at $path" }
        // Comments stripped, or a commented-out declaration still satisfies every
        // `contains` below: the permission strings are fully qualified, so
        // `<!-- <uses-permission android:name="android.permission.CAMERA" /> -->` passed
        // three tests and the Gradle manifest gate alike, for an app with no CAMERA
        // permission at all.
        file.readText().replace(Regex("""<!--.*?-->""", RegexOption.DOT_MATCHES_ALL), "")
    }

    private fun declares(permission: String): Boolean =
        manifest.contains("""android:name="$permission"""")

    @Test
    fun `every runtime permission the model requires is declared`() {
        // Swept across the levels the app supports so the API-33 branch is covered
        // rather than only whatever level the test host happens to report.
        for (sdk in 29..35) {
            for (permission in PermissionModel.required(sdk)) {
                assertTrue(
                    "sdk $sdk requires $permission but the manifest does not declare it",
                    declares(permission),
                )
            }
        }
    }

    @Test
    fun `the permission constants are the real platform strings`() {
        // Pinned as literals, not via the constants, so a typo in either place fails.
        assertTrue(declares("android.permission.CAMERA"))
        assertTrue(declares("android.permission.ACCESS_FINE_LOCATION"))
        assertTrue(declares("android.permission.POST_NOTIFICATIONS"))
    }

    @Test
    fun `foreground service permissions are declared for the types the service uses`() {
        // startForeground() throws SecurityException without the base permission, and
        // the typed ones are mandatory from API 34 for a camera|location service.
        assertTrue(declares("android.permission.FOREGROUND_SERVICE"))
        assertTrue(declares("android.permission.FOREGROUND_SERVICE_CAMERA"))
        assertTrue(declares("android.permission.FOREGROUND_SERVICE_LOCATION"))
    }

    @Test
    fun `the service declares both foreground types it starts with`() {
        // enterForeground() passes CAMERA or LOCATION; the platform kills the service
        // if a type it passes is not declared.
        assertTrue(manifest.contains("""android:foregroundServiceType="camera|location""""))
    }
}
