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
        clearSeams()
        quiesce()
    }

    @After
    fun tearDown() {
        clearSeams()
        quiesce()
    }

    /**
     * Every seam and every companion counter, in one place -- five seams, not the six
     * this used to claim.
     *
     * Four of the six seams were left to each test's own `finally`, and the counters are
     * process-global and were never reset -- so a test reading one was reading whatever the
     * test before it happened to leave. Nothing leaked today, which is exactly the
     * condition under which this stops being true silently.
     */
    private fun clearSeams() {
        SensingService.permissionOverride = null
        SensingService.enterForegroundOverride = null
        SensingService.teardownFailureOverride = null
        SensingService.comeUpFailureOverride = null
        SensingService.startedFailureOverride = null
        SensingService.teardownFailures.set(0)
        SensingService.shutdownFailures.set(0)
        SensingService.resourcesHeldAfterTeardown = -1
        SensingService.lastShutdownFailure = null
        SensingService.statsReadBeforeStop.set(0)
        SensingStatus.shared.listenerFailures.set(0)
        SensingStatus.shared.lastListenerFailure = null
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
        /** ImuSource's HandlerThread. */
        const val IMU = "dsrc-imu"

        // Adding a prefix here is how a new resource gets teardown coverage: every test
        // that counts threads iterates this list, so the come-up, the failed start, the
        // failed teardown and the restart all cover it without a new test.
        val WORKER_PREFIXES = listOf(POOL, LINK, SENDER, GPS, IMU)
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

    @Test
    fun aFailedStartLeavesNothingRunning() {
        // react() reaches the terminal-stopped states through release() alone, so before
        // the guard, a throw part-way through onSensingUp left every allocation above it
        // live until onDestroy -- and cleanup depended on stopSelf(lastStartId) winning,
        // which the startId overload exists precisely to *lose* when a start is queued
        // behind it. A retried Start then overwrote every field and orphaned the first
        // set: a link thread reconnecting forever, a sender polling forever, a GNSS callback
        // still registered.
        //
        // **This does not pin the guard, and the name no longer claims it does.** Removing
        // either the re-entrancy release or the failure catch leaves this green, because
        // release() -> stopSelf() -> onDestroy() -> onSensingDown() cleans up anyway. The
        // window the guard closes needs the service to *survive* its own stopSelf, which
        // takes a start queued behind it -- a ~2 ms race neither the validator nor I could
        // force from a test. What is verified here is the weaker, still-worth-having claim:
        // a failed start leaves nothing running, by whatever route.
        SensingService.comeUpFailureOverride = { throw IllegalStateException("a failed come-up") }
        try {
            SensingService.start(context)
            await(SensingState.STOPPED_ERROR)

            // Nothing may be left running after a failed start.
            for (prefix in WORKER_PREFIXES) {
                assertTrue(
                    "a thread named '$prefix' survived a failed start",
                    pollUntil(10_000) { threadsNamed(listOf(prefix)) == 0 },
                )
            }

            // And a retry must not stack a second set on top of the first.
            repeat(3) {
                SensingService.start(context)
                await(SensingState.STOPPED_ERROR)
            }
            for (prefix in WORKER_PREFIXES) {
                val count = threadsNamed(listOf(prefix))
                assertTrue(
                    "three failed starts left $count threads named '$prefix'",
                    count == 0,
                )
            }
        } finally {
            SensingService.comeUpFailureOverride = null
            context.stopService(android.content.Intent(context, SensingService::class.java))
        }
    }

    @Test
    fun aRestartLeavesNoMoreThreadsThanOneRunDoes() {
        // This was not a restart. The second Start was offered from RUNNING, which the
        // machine ignores -- no react(STARTING), no come-up, no second allocation ever
        // happened -- so deleting the second start and its await left the test passing. It
        // has to stop first.
        SensingService.start(context)
        await(SensingState.RUNNING)
        for (prefix in WORKER_PREFIXES) {
            assertTrue("nothing named '$prefix' started", pollUntil(10_000) { threadsNamed(listOf(prefix)) > 0 })
        }

        val path = java.util.Collections.synchronizedList(mutableListOf<SensingState>())
        val recorder = SensingStatus.Listener { path.add(it) }
        SensingStatus.shared.addListener(recorder)
        try {
            SensingService.stop(context)
            await(SensingState.IDLE)
            SensingService.start(context)
            await(SensingState.RUNNING)
        } finally {
            SensingStatus.shared.removeListener(recorder)
        }
        Thread.sleep(500)
        // That a second run happened at all. Counting threads cannot see it: if the second
        // come-up never runs, the first run's workers are still there and satisfy every
        // bound below. That is exactly how this test used to pass with the restart deleted.
        assertTrue(
            "sensing never went down and back up; states were ${synchronized(path) { path.toList() }}",
            synchronized(path) { path.toList() }
                .dropWhile { it != SensingState.IDLE }
                .contains(SensingState.RUNNING),
        )

        for (prefix in WORKER_PREFIXES) {
            val count = threadsNamed(listOf(prefix))
            // pool- covers the encoder and CameraX's analyser, so two is its healthy count.
            val limit = if (prefix == POOL) 2 else 1
            // Both bounds: an upper bound alone passes for a restart that started nothing.
            assertTrue(
                "restarting left $count threads named '$prefix', wanted 1..$limit",
                count in 1..limit,
            )
        }
        context.stopService(android.content.Intent(context, SensingService::class.java))
        for (prefix in WORKER_PREFIXES) {
            assertTrue("'$prefix' outlived the service", pollUntil(10_000) { threadsNamed(listOf(prefix)) == 0 })
        }
    }


    @Test
    fun aFailedTeardownStillClearsEverythingAndARestartDoesNotDoubleIt() {
        // The route I argued was unreachable, and it is not. onSensingDown used to run its
        // releases in sequence and null the fields last, so a throw part-way through
        // skipped every release behind it and left them all set -- and react(STOPPING)
        // catches exactly that and offers Failed, which the machine turns into
        // STOPPED_ERROR, from which Start goes to STARTING and straight back into
        // onSensingUp. A live CameraX binding, a live encoder executor, and a SessionHolder
        // whose threads and socket are still up, all orphaned, with the Jetson seeing a
        // second session displace the first.
        //
        // My argument covered only the paths where teardown succeeds. The failure path is
        // the one the state machine models sixty lines above the guard I removed on the
        // strength of it.
        //
        // The seam fails the *first* release, so what this pins is that the ten behind it
        // still run. Nothing here asserts on the exception: the observable is that no
        // worker thread survives a teardown whose first step threw.
        SensingService.teardownFailures.set(0)
        // One-shot, and it has to be. onSensingDown runs twice per stop -- once from
        // react(STOPPING) and again from onDestroy, which is the safety net for a service
        // destroyed without a Stop -- and onDestroy lands asynchronously after the state
        // reaches IDLE. A seam that fires every time makes the count 1 or 2 depending on
        // which side of the assertion the platform gets to, so either number would have
        // been a race dressed as a constant.
        SensingService.teardownFailureOverride = {
            SensingService.teardownFailureOverride = null
            throw IllegalStateException("release refused")
        }
        try {
            SensingService.start(context)
            await(SensingState.RUNNING)
            for (prefix in WORKER_PREFIXES) {
                assertTrue("nothing named '$prefix' started", pollUntil(10_000) { threadsNamed(listOf(prefix)) > 0 })
            }

            SensingService.stop(context)
            // IDLE, not STOPPED_ERROR: a failed release is counted rather than propagated,
            // so teardown completes. Letting it escape instead killed the whole process
            // from onDestroy, which calls onSensingDown with no catch of its own -- found
            // by writing this test, and the reason the guard is per-step.
            await(SensingState.IDLE)
            assertEquals(
                "the seam has to have actually fired, or this test proves nothing",
                1L,
                SensingService.teardownFailures.get().toLong(),
            )

            for (prefix in WORKER_PREFIXES) {
                assertTrue(
                    "a thread named '$prefix' survived a teardown whose first step threw",
                    pollUntil(10_000) { threadsNamed(listOf(prefix)) == 0 },
                )
            }

            // And the restart the machine allows must not stack a second set of workers on
            // top of an orphaned first.
            SensingService.teardownFailureOverride = null
            SensingService.start(context)
            await(SensingState.RUNNING)
            Thread.sleep(500)
            for (prefix in WORKER_PREFIXES) {
                val count = threadsNamed(listOf(prefix))
                assertTrue(
                    "restarting after a failed teardown left $count threads named '$prefix'",
                    count <= if (prefix == POOL) 2 else 1,
                )
            }
        } finally {
            SensingService.teardownFailureOverride = null
            context.stopService(android.content.Intent(context, SensingService::class.java))
        }
    }


    @Test
    fun aFailureOfferedFromRunningDoesNotLeaveTheWholeSetLive() {
        // The route the removed re-entrancy guard covered, and my enumeration missed it a
        // second time. react(STARTING)'s try encloses handle(Started) as well as
        // onSensingUp(), so a throw after come-up has already succeeded is caught as a
        // start failure and offered as Failed while the machine is RUNNING. The machine
        // accepts that arm and nothing on the route called onSensingDown, so all seven
        // fields and every worker stayed live -- and STOPPED_ERROR then accepts Start, so
        // the second come-up ran on top of the first.
        //
        // Driven through the seam rather than through a throwing status listener. The
        // listener was the trigger the validator used, and it worked -- but containing
        // listener failures (which is a fix in its own right) closed that door, and a test
        // built on it then passed for the wrong reason: I mutated the teardown away and all
        // 51 tests stayed green. The arm is still reachable by construction, so it gets a
        // seam of its own.
        //
        // start() before throwing, because that is what makes AMS advance the
        // last-delivered startId past the one release() hands stopSelf(lastStartId): the
        // stop is declined and the same instance takes the queued Start. Without it the
        // service is destroyed and a fresh process hides the leak.
        val path = java.util.Collections.synchronizedList(mutableListOf<SensingState>())
        val recorder = SensingStatus.Listener { path.add(it) }
        SensingStatus.shared.addListener(recorder)
        SensingService.startedFailureOverride = {
            SensingService.startedFailureOverride = null
            SensingService.start(context)
            throw RuntimeException("something offered Failed while we were RUNNING")
        }
        try {
            SensingService.start(context)
            await(SensingState.RUNNING)
            Thread.sleep(1_000)

            // The route, before its aftermath. Thread counts alone cannot tell the intended
            // path from the arm being dead: make RUNNING + Failed unreachable in the machine
            // and the service simply stays RUNNING with its first set of workers, which
            // satisfies every count assertion below just as well. Round 5 proved that by
            // doing it -- all 53 tests passed. So the states are asserted as a subsequence:
            // RUNNING, then STOPPED_ERROR, then back up through STARTING to RUNNING.
            val wanted = listOf(
                SensingState.RUNNING,
                SensingState.STOPPED_ERROR,
                SensingState.STARTING,
                SensingState.RUNNING,
            )
            val seen = synchronized(path) { path.toList() }
            var index = 0
            for (state in seen) if (index < wanted.size && state == wanted[index]) index++
            assertTrue(
                "the failure never took the RUNNING -> STOPPED_ERROR -> STARTING -> RUNNING " +
                    "route; states were $seen",
                index == wanted.size,
            )
            for (prefix in WORKER_PREFIXES) {
                val count = threadsNamed(listOf(prefix))
                // Both bounds, for the same reason as the restart test: an upper bound
                // alone passes for a service running nothing.
                assertTrue(
                    "a failure offered from RUNNING left $count threads named '$prefix', " +
                        "wanted 1..${if (prefix == POOL) 2 else 1}",
                    count in 1..(if (prefix == POOL) 2 else 1),
                )
            }
        } finally {
            SensingStatus.shared.removeListener(recorder)
            SensingService.startedFailureOverride = null
            context.stopService(android.content.Intent(context, SensingService::class.java))
        }
    }


    @Test
    fun teardownLeavesTheServiceHoldingNoResourceReferences() {
        // Round 4: deleting all seven field assignments left every teardown test green,
        // because with each release independently guarded a stale field is behaviourally
        // inert -- the object is stopped and the next come-up overwrites it. What it costs
        // is memory, and a retained CameraPipeline is not small: its whole ring buffer of
        // encoded frames, an encoder executor and a socket, held for the life of the
        // process. Asserted as the property it is rather than through a consequence it
        // does not have.
        SensingService.resourcesHeldAfterTeardown = -1
        SensingService.start(context)
        await(SensingState.RUNNING)
        SensingService.stop(context)
        await(SensingState.IDLE)

        assertTrue(
            "teardown never ran: the recorder is still at its initial value",
            pollUntil(5_000) { SensingService.resourcesHeldAfterTeardown >= 0 },
        )
        assertEquals(
            "teardown finished still holding resource references",
            0L,
            SensingService.resourcesHeldAfterTeardown.toLong(),
        )
    }

    @Test
    fun anOrdinaryStopReleasesEveryResourceWithoutOneRefusing() {
        // Round 4 found that a release *failing* is observable only for the four
        // thread-owning steps: making the camera pipeline throw on stop left the suite
        // green, because nothing looks at those objects afterwards and the
        // thread census cannot see a pipeline. The count closes that in aggregate -- a
        // release that starts throwing in production now fails the suite, whichever of the
        // fourteen it is, instead of being a log line nobody reads.
        SensingService.teardownFailures.set(0)
        SensingService.start(context)
        await(SensingState.RUNNING)
        SensingService.stop(context)
        await(SensingState.IDLE)
        Thread.sleep(300)

        assertEquals(
            "a release step threw during an ordinary stop",
            0L,
            SensingService.teardownFailures.get().toLong(),
        )
        // And the stats were read after the stops. Reading them first made abandoned,
        // refusedStopped and the buffer's discarded structurally zero on every call, at the
        // only place production reads them -- so the teardown log said "encoder backlog"
        // for frames that had in fact been abandoned and counted.
        assertEquals(
            "a pipeline's stats were logged before it was stopped",
            0L,
            SensingService.statsReadBeforeStop.get().toLong(),
        )
    }

}
