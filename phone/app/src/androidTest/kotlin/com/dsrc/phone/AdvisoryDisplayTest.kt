package com.dsrc.phone

import android.view.ViewGroup
import android.widget.TextView
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import com.dsrc.phone.ui.AdvisoryHolder
import com.dsrc.transport.AdvisoryMessage
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * What the driver actually sees, asserted on the views.
 *
 * Task 17 deferred an Activity harness and task 23 needed it: every other assertion about
 * the advisory goes through `SensingService.advisories`, which proves the holder is right
 * and says nothing about whether the text reaches a pixel. The defect that made the case
 * lived exactly in that gap — coming back to the app repainted the last advisory, however
 * old, because the labels kept their text across `onStop` and the resume traversal drew
 * before the posted tick ran.
 */
@RunWith(AndroidJUnit4::class)
class AdvisoryDisplayTest {

    @get:Rule
    val permissions: GrantPermissionRule = GrantPermissionRule.grant(
        android.Manifest.permission.CAMERA,
        android.Manifest.permission.ACCESS_FINE_LOCATION,
    )

    private val advisory = AdvisoryMessage(
        captureMonoNs = 1,
        recSpeedMps = 13.4,
        recSpeedDisplay = 30.4,
        currentSpeedDisplay = 28.0,
        units = "kmh",
        headwayTargetS = 2.0,
        laneText = "Keep lane",
        mergeText = "",
        trafficText = "Moderate",
        confidence = 0.87,
        confidenceLabel = "high",
        action = mapOf(
            "lane_preference" to "keep",
            "merge_mode" to "normal",
            "desired_speed_bin" to "nominal",
            "desired_headway_bin" to "normal",
        ),
    )

    @Before
    fun clearHolder() {
        SensingService.advisories.clear()
        SensingService.advisories.start()
    }

    @After
    fun stopHolding() {
        SensingService.advisories.clear()
    }

    @Test
    fun theJetsonsOwnStringsReachTheScreen() {
        SensingService.advisories.accept(advisory, android.os.SystemClock.elapsedRealtimeNanos())

        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val text = labels(activity)
                // The number and its units, unformatted. A phone that rounded 30.4 to 30
                // while the Jetson meant 30.4 would be showing a recommendation nobody made.
                assertTrue("wanted '30.4 kmh' among $text", text.any { it == "30.4 kmh" })
                assertTrue("the advice line is missing from $text", text.any { it.contains("Keep lane") })
                assertTrue(text.any { it == "high" })
            }
        }
    }

    @Test
    fun comingBackToTheAppDoesNotRepaintAStaleAdvisory() {
        // The defect. Backgrounded with an advisory on screen, returning minutes later
        // painted the old one for a frame or two before blanking -- measured at two
        // consecutive frames spanning ~29 ms showing an advisory that had expired three and
        // a half seconds earlier. The labels are blanked on the way out and refreshed
        // synchronously on the way in, so no path leaves stale text to be drawn.
        SensingService.advisories.accept(advisory, android.os.SystemClock.elapsedRealtimeNanos())

        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { assertTrue(labels(it).any { text -> text == "30.4 kmh" }) }

            // Away, and past the expiry.
            scenario.moveToState(Lifecycle.State.CREATED)
            Thread.sleep(AdvisoryHolder.MAX_AGE_NS / 1_000_000 + 500)

            // While backgrounded, nothing stale may be left sitting in the views.
            scenario.onActivity { activity ->
                assertEquals(
                    "the labels kept their text while backgrounded, so the resume traversal " +
                        "had something stale to paint",
                    emptyList<String>(),
                    labels(activity).filter { it.isNotBlank() },
                )
            }

            scenario.moveToState(Lifecycle.State.RESUMED)
            scenario.onActivity { activity ->
                assertEquals(
                    "a stale advisory was repainted on the way back in",
                    emptyList<String>(),
                    labels(activity).filter { it.isNotBlank() },
                )
            }
        }
    }

    @Test
    fun afreshAdvisoryAfterReturningIsShown() {
        // The other direction: blanking on the way out must not stop a current advisory
        // being drawn on the way back.
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.moveToState(Lifecycle.State.CREATED)
            SensingService.advisories.accept(advisory, android.os.SystemClock.elapsedRealtimeNanos())
            scenario.moveToState(Lifecycle.State.RESUMED)

            scenario.onActivity { activity ->
                assertTrue(
                    "a current advisory was not drawn after returning: ${labels(activity)}",
                    labels(activity).any { it == "30.4 kmh" },
                )
            }
        }
    }

    /**
     * The text on the three advisory labels, and nothing else.
     *
     * Addressed by tag. Walking every `TextView` also collects the state line and both
     * buttons -- buttons are TextViews -- so "nothing is displayed" would have asserted the
     * screen was empty rather than that the panel was.
     */
    private fun labels(activity: MainActivity): List<String> {
        val wanted = setOf(
            MainActivity.TAG_SPEED, MainActivity.TAG_ADVICE, MainActivity.TAG_CONFIDENCE,
        )
        val found = mutableListOf<String>()
        fun walk(view: android.view.View) {
            if (view is TextView && view.tag in wanted) found.add(view.text.toString())
            if (view is ViewGroup) for (i in 0 until view.childCount) walk(view.getChildAt(i))
        }
        walk(activity.findViewById(android.R.id.content))
        return found
    }
}
