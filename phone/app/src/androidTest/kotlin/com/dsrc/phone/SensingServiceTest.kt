package com.dsrc.phone

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
 * intent, reconciling a stale published state -- have no JVM equivalent. A plain JVM
 * test can construct the service, but every platform call returns a default (SDK_INT
 * reads 0), so it would exercise branches the phone never takes.
 */
@RunWith(AndroidJUnit4::class)
class SensingServiceTest {

    // Tied to the model rather than a hardcoded pair, so the suite does not silently
    // pin itself to API <= 32: on API 33+ the model also requires POST_NOTIFICATIONS,
    // and without granting it the permission gate refuses every start.
    @get:Rule
    val permissions: GrantPermissionRule = GrantPermissionRule.grant(
        *PermissionModel.required(Build.VERSION.SDK_INT).toTypedArray()
    )

    private val context: Context get() = ApplicationProvider.getApplicationContext()

    @Before
    fun reset() {
        SensingService.permissionOverride = null
        SensingService.enterForegroundOverride = null
        quiesce()
    }

    @After
    fun tearDown() {
        SensingService.permissionOverride = null
        SensingService.enterForegroundOverride = null
        quiesce()
    }

    /**
     * Stop sensing and wait until the service is actually gone.
     *
     * Waiting on the published state alone is not enough: `stop()` sends an intent
     * asynchronously, and if the state already reads IDLE the wait returns immediately
     * while the intent is still queued. It then lands in the middle of the next test --
     * which is how a stale stop arrived after a test had set up its own state and
     * cleared it.
     */
    private fun quiesce() {
        SensingService.stop(context)
        pollUntil(5_000) { !serviceIsResident() && SensingStatus.shared.state == SensingState.IDLE }
        assertFalse("service still resident after quiesce", serviceIsResident())
        assertEquals(SensingState.IDLE, SensingStatus.shared.state)
    }

    private fun await(state: SensingState, timeoutMs: Long = 5_000) {
        pollUntil(timeoutMs) { SensingStatus.shared.state == state }
        assertEquals("timed out waiting for $state", state, SensingStatus.shared.state)
    }

    /** Polls rather than sleeping a fixed margin, so the wait is a deadline not a guess. */
    private fun pollUntil(timeoutMs: Long, condition: () -> Boolean): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return true
            Thread.sleep(50)
        }
        return condition()
    }

    private fun serviceIsResident(): Boolean {
        val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        @Suppress("DEPRECATION")
        return manager.getRunningServices(64).any {
            it.service.className == SensingService::class.java.name
        }
    }

    /**
     * Whether the service is actually in the foreground.
     *
     * Residency and foreground-ness are different facts: a merely-created service is
     * resident, so checking only residency passed for a service that never called
     * startForeground at all.
     *
     * Read from the service record rather than by looking for our notification. The
     * notification check is unreliable on the *first* post after install: the channel
     * is created and the record enqueued asynchronously, so `getActiveNotifications`
     * can return before it is visible. (It is first-post latency, not the rate limiting
     * it first looked like -- measured over 14 start/stop cycles, only cycle 0 failed
     * and all 13 later ones passed, which is the opposite shape to a rate limiter.)
     * The service record is the stronger fact and never failed across those cycles.
     */
    private fun serviceIsForeground(): Boolean {
        val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        @Suppress("DEPRECATION")
        return manager.getRunningServices(64).any {
            it.service.className == SensingService::class.java.name && it.foreground
        }
    }

    @Test
    fun startReachesRunningInTheForeground() {
        SensingService.start(context)
        await(SensingState.RUNNING)
        assertTrue("service should be resident while running", serviceIsResident())
        assertTrue(
            "the service should be in the foreground while running",
            pollUntil(3_000) { serviceIsForeground() },
        )
    }

    @Test
    fun stoppingLeavesTheForeground() {
        SensingService.start(context)
        await(SensingState.RUNNING)
        assertTrue(pollUntil(3_000) { serviceIsForeground() })

        SensingService.stop(context)
        await(SensingState.IDLE)
        assertTrue(
            "the service should leave the foreground on stop",
            pollUntil(3_000) { !serviceIsForeground() },
        )
    }

    @Test
    fun stopReturnsToIdle() {
        // Kept distinct from stoppingLeavesTheForeground: this is the test that
        // distinguishes stopSelf(lastStartId) from the bare stopSelf(), which is the
        // line that makes a stop racing a newer start safe.
        SensingService.start(context)
        await(SensingState.RUNNING)
        SensingService.stop(context)
        await(SensingState.IDLE)
        assertTrue(pollUntil(5_000) { !serviceIsResident() })
    }

    @Test
    fun aStartRacingAStopIsNotDiscarded() {
        // What stopSelf(lastStartId) is for. The bare overload destroys the service
        // even when a newer start has already been queued, so the start the user just
        // asked for is thrown away with the stop.
        SensingService.start(context)
        await(SensingState.RUNNING)
        repeat(10) {
            SensingService.stop(context)
            SensingService.start(context)
        }
        assertTrue(
            "a start queued behind a stop must survive it",
            pollUntil(10_000) { SensingStatus.shared.state == SensingState.RUNNING && serviceIsForeground() },
        )
    }

    @Test
    fun aStaleActiveStateIsNotAdoptedByANewInstance() {
        // The cell the other reconcile tests miss: published-active plus a *non-Stop*
        // intent. Adopting an active state here would leave requiresService true, so
        // the trailing release() never fires -- a service resident forever having never
        // entered the foreground, under a UI still claiming RUNNING.
        SensingStatus.shared.set(SensingState.RUNNING)
        context.startService(Intent(context, SensingService::class.java).setAction("com.dsrc.phone.NONSENSE"))
        assertTrue(
            "an active published state must be corrected, not adopted",
            pollUntil(5_000) { SensingStatus.shared.state == SensingState.IDLE },
        )
        assertTrue("the service must not linger", pollUntil(5_000) { !serviceIsResident() })
    }

    @Test
    fun aSecondStartDoesNotDisturbARunningSession() {
        // Named for what it checks. Idempotence of the transition itself is pinned by
        // SensingStateMachineTest; here the point is that the running session survives
        // a duplicate intent rather than being torn down and rebuilt.
        SensingService.start(context)
        await(SensingState.RUNNING)
        SensingService.start(context)
        assertFalse(
            "state should not leave RUNNING on a duplicate start",
            pollUntil(1_000) { SensingStatus.shared.state != SensingState.RUNNING },
        )
        assertTrue("the running session should still be in the foreground", serviceIsForeground())
    }

    @Test
    fun aDuplicateStopLeavesNoResidentService() {
        // startService creates the service to deliver an intent even when the state
        // machine ignores it, so an ignored stop must still release.
        SensingService.start(context)
        await(SensingState.RUNNING)
        SensingService.stop(context)
        await(SensingState.IDLE)

        SensingService.stop(context)
        assertTrue(
            "a duplicate stop left the service resident",
            pollUntil(5_000) { !serviceIsResident() },
        )
        assertEquals(SensingState.IDLE, SensingStatus.shared.state)
    }

    @Test
    fun anUnknownActionLeavesNoResidentService() {
        context.startService(Intent(context, SensingService::class.java).setAction("com.dsrc.phone.NONSENSE"))
        assertTrue(
            "an unknown action left the service resident",
            pollUntil(5_000) { !serviceIsResident() },
        )
        assertEquals(SensingState.IDLE, SensingStatus.shared.state)
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

    // -- reconciling the published state with a new instance ---------------------

    @Test
    fun aStaleActiveStateIsCorrectedByANewInstance() {
        // The published state is process-global while the machine is per-instance. An
        // active state cannot be true for a brand-new instance, and leaving it would
        // show a live session with an inert Stop button.
        SensingStatus.shared.set(SensingState.RUNNING)
        SensingService.stop(context)
        await(SensingState.IDLE)
    }

    @Test
    fun aRecordedFailureSurvivesUntilItIsCleared() {
        // The other direction: a terminal state is a fact the previous instance
        // recorded, and it is the only place the failure is visible. An unrelated
        // intent must not erase it.
        SensingStatus.shared.set(SensingState.STOPPED_ERROR)
        context.startService(Intent(context, SensingService::class.java).setAction("com.dsrc.phone.NONSENSE"))
        Thread.sleep(750)
        assertEquals(SensingState.STOPPED_ERROR, SensingStatus.shared.state)
        assertTrue(pollUntil(5_000) { !serviceIsResident() })
    }

    @Test
    fun stopClearsARecordedFailure() {
        // And Stop does clear it. This is the user-visible path for the machine's
        // terminal-Stop rows: the intent always lands on a new instance, so the machine
        // has to adopt the published terminal state for Stop to mean anything.
        SensingStatus.shared.set(SensingState.STOPPED_ERROR)
        SensingService.stop(context)
        await(SensingState.IDLE)
    }

    // -- teardown by the platform ------------------------------------------------

    @Test
    fun aPlatformTeardownReleasesTheWorkerThreads() {
        // stopService goes straight to onDestroy without passing through the state
        // machine, so onSensingDown was never called and both worker threads outlived
        // the service. Every task-removal and low-memory kill leaked two. The camera
        // itself is released by the lifecycle unbinding, which is why threads were the
        // only symptom and nothing noticed.
        SensingService.start(context)
        await(SensingState.RUNNING)
        // Wait for the workers to exist: executor threads are created lazily on first
        // submit, so a control taken too early sees zero either way.
        assertTrue("no worker threads appeared", pollUntil(10_000) { workerThreads() > 0 })
        val whileRunning = workerThreads()

        context.stopService(android.content.Intent(context, SensingService::class.java))

        assertTrue(
            "worker threads outlived the service: $whileRunning still running",
            pollUntil(10_000) { workerThreads() == 0 },
        )
    }

    /**
     * Every thread the service owns, by name.
     *
     * This counted only `pool-` threads, which is the encoder and CameraX's analyser. The
     * link, the frame sender and the GPS looper all use *named* threads, so the one test
     * that claimed to pin teardown could not see three of the five resources -- and
     * deleting `holder.start()`, `sender.start()` or `locations.start()` left the whole
     * instrumented suite green while sensing transmitted nothing at all.
     *
     * Named prefixes are asserted individually below rather than summed, because a total
     * hides which one went missing.
     */
    private fun workerThreads(): Int = threadsNamed(WORKER_PREFIXES)

    private fun threadsNamed(prefixes: List<String>): Int {
        val all = arrayOfNulls<Thread>(Thread.activeCount() * 2 + 32)
        val n = Thread.enumerate(all)
        return (0 until n).count { index ->
            val name = all[index]?.name ?: return@count false
            prefixes.any { name.startsWith(it) }
        }
    }

    private companion object {
        /** The encoder and CameraX's analyser. */
        const val POOL = "pool-"
        /** SessionHolder's reconnect thread. */
        const val LINK = "dsrc-link"
        /** CameraFrameSender's drain thread. */
        const val SENDER = "dsrc-camera-send"
        /** GpsLocationSource's HandlerThread. */
        const val GPS = "dsrc-gps"

        val WORKER_PREFIXES = listOf(POOL, LINK, SENDER, GPS)
    }

    @Test
    fun everyResourceStartsWhenSensingStartsAndStopsWhenItStops() {
        // Named per resource, because a summed census hides which one never started. Each
        // of these was individually deletable with the suite green: no link, no frames
        // reaching the wire, no GPS registration.
        SensingService.start(context)
        await(SensingState.RUNNING)

        for (prefix in WORKER_PREFIXES) {
            assertTrue(
                "nothing named '$prefix' started, so that resource is doing nothing",
                pollUntil(10_000) { threadsNamed(listOf(prefix)) > 0 },
            )
        }

        context.stopService(android.content.Intent(context, SensingService::class.java))

        for (prefix in WORKER_PREFIXES) {
            assertTrue(
                "a thread named '$prefix' outlived the service",
                pollUntil(10_000) { threadsNamed(listOf(prefix)) == 0 },
            )
        }
    }

    // -- a failed foreground transition -----------------------------------------

    @Test
    fun aRefusedForegroundTransitionIsRecordedAndDoesNotKillTheProcess() {
        // This test passing at all is the assertion that matters: if the process died
        // the whole instrumentation run would abort. It died before, because
        // startForegroundService() is a promise to call startForeground(), and the
        // ActivityManager throws the instant the service is brought down with that
        // promise outstanding -- so the catch below ran, recorded the failure, and then
        // the teardown killed the app 1 ms later. Not making the promise is the fix.
        SensingService.enterForegroundOverride = { throw SecurityException("probe: refused") }
        SensingService.start(context)
        await(SensingState.STOPPED_ERROR)
        assertTrue("a failed start must not leave the service resident", pollUntil(5_000) { !serviceIsResident() })
        assertFalse("a failed start must not be in the foreground", serviceIsForeground())
    }

    @Test
    fun sensingStartsOnceTheForegroundTransitionWorksAgain() {
        SensingService.enterForegroundOverride = { throw SecurityException("probe: refused") }
        SensingService.start(context)
        await(SensingState.STOPPED_ERROR)

        SensingService.enterForegroundOverride = null
        SensingService.start(context)
        await(SensingState.RUNNING)
        assertTrue(pollUntil(3_000) { serviceIsForeground() })
    }

    // -- the permission gate ----------------------------------------------------

    @Test
    fun aMissingPermissionRefusesTheStart() {
        // The grant rule satisfies every real requirement, so the refusal branch has no
        // active premise without this seam -- the whole gate never ran under test.
        SensingService.permissionOverride = { listOf(PermissionModel.CAMERA) }
        SensingService.start(context)
        await(SensingState.STOPPED_PERMISSION_REVOKED)
        assertTrue(
            "a refused start must not leave the service in the foreground",
            pollUntil(5_000) { !serviceIsForeground() },
        )
        assertTrue(pollUntil(5_000) { !serviceIsResident() })
    }

    @Test
    fun sensingStartsOnceThePermissionIsBack() {
        SensingService.permissionOverride = { listOf(PermissionModel.CAMERA) }
        SensingService.start(context)
        await(SensingState.STOPPED_PERMISSION_REVOKED)

        SensingService.permissionOverride = null
        SensingService.start(context)
        await(SensingState.RUNNING)
    }
}
