package com.dsrc.phone

import android.app.ActivityManager
import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The real service on a real platform.
 *
 * The lifecycle logic is unit-tested through [SensingStateMachine], but the parts that
 * only exist against Android -- going to the foreground, releasing on an ignored
 * intent, reconciling a stale status on a new instance -- have no JVM equivalent. A
 * plain JVM test can construct the service, but every platform call returns a default
 * (SDK_INT reads 0), so it would exercise branches the phone never takes.
 */
@RunWith(AndroidJUnit4::class)
class SensingServiceTest {

    @get:Rule
    val permissions: GrantPermissionRule = GrantPermissionRule.grant(
        PermissionModel.CAMERA,
        PermissionModel.FINE_LOCATION,
    )

    private val context: Context get() = ApplicationProvider.getApplicationContext()

    @Before
    fun resetStatus() {
        SensingService.stop(context)
        await(SensingState.IDLE)
    }

    @After
    fun stopSensing() {
        SensingService.stop(context)
        await(SensingState.IDLE)
    }

    private fun await(state: SensingState, timeoutMs: Long = 5_000) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (SensingStatus.shared.state == state) return
            Thread.sleep(50)
        }
        assertEquals("timed out waiting for $state", state, SensingStatus.shared.state)
    }

    private fun serviceIsResident(): Boolean {
        val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        @Suppress("DEPRECATION")
        return manager.getRunningServices(64).any {
            it.service.className == SensingService::class.java.name
        }
    }

    @Test
    fun startReachesRunningInTheForeground() {
        SensingService.start(context)
        await(SensingState.RUNNING)
        assertTrue("service should be resident while running", serviceIsResident())
    }

    @Test
    fun stopReturnsToIdle() {
        SensingService.start(context)
        await(SensingState.RUNNING)
        SensingService.stop(context)
        await(SensingState.IDLE)
    }

    @Test
    fun startingTwiceDoesNotStartTwice() {
        SensingService.start(context)
        await(SensingState.RUNNING)
        SensingService.start(context)
        Thread.sleep(500)
        assertEquals(SensingState.RUNNING, SensingStatus.shared.state)
    }

    @Test
    fun aDuplicateStopLeavesNoResidentService() {
        // The zombie case: startService creates the service to deliver an intent even
        // when the state machine ignores it, so an ignored stop must still release.
        SensingService.start(context)
        await(SensingState.RUNNING)
        SensingService.stop(context)
        await(SensingState.IDLE)

        SensingService.stop(context)
        Thread.sleep(1_000)
        assertEquals(SensingState.IDLE, SensingStatus.shared.state)
        assertTrue("a duplicate stop left the service resident", !serviceIsResident())
    }

    @Test
    fun anUnknownActionLeavesNoResidentService() {
        context.startService(
            android.content.Intent(context, SensingService::class.java).setAction("com.dsrc.phone.NONSENSE")
        )
        Thread.sleep(1_000)
        assertEquals(SensingState.IDLE, SensingStatus.shared.state)
        assertTrue("an unknown action left the service resident", !serviceIsResident())
    }

    @Test
    fun sensingCanBeRestartedAfterStopping() {
        repeat(2) {
            SensingService.start(context)
            await(SensingState.RUNNING)
            SensingService.stop(context)
            await(SensingState.IDLE)
        }
    }
}
