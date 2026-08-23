package com.dsrc.phone

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PermissionModelTest {

    private val camera = PermissionModel.CAMERA
    private val location = PermissionModel.FINE_LOCATION
    private val notifications = PermissionModel.POST_NOTIFICATIONS

    // -- required ------------------------------------------------------------

    @Test
    fun `notifications are required from api 33`() {
        assertTrue(PermissionModel.required(33).contains(notifications))
        assertTrue(PermissionModel.required(35).contains(notifications))
    }

    @Test
    fun `notifications are not requestable below api 33`() {
        assertFalse(PermissionModel.required(32).contains(notifications))
        assertFalse(PermissionModel.required(29).contains(notifications))
    }

    @Test
    fun `camera and location are required at every supported level`() {
        for (sdk in 29..35) {
            val required = PermissionModel.required(sdk)
            assertTrue("sdk $sdk", required.contains(camera))
            assertTrue("sdk $sdk", required.contains(location))
        }
    }

    // -- classify ------------------------------------------------------------

    @Test
    fun `granted is granted`() {
        assertEquals(
            PermissionState.GRANTED,
            PermissionModel.classify(granted = true, shouldShowRationale = false, hasAsked = true),
        )
    }

    @Test
    fun `a granted permission ignores the rationale flag`() {
        // The platform's rationale value is unspecified for a granted permission, so
        // it must not be able to change the answer.
        for (rationale in listOf(true, false)) {
            for (asked in listOf(true, false)) {
                assertEquals(
                    PermissionState.GRANTED,
                    PermissionModel.classify(true, rationale, asked),
                )
            }
        }
    }

    @Test
    fun `rationale means denied once and askable again`() {
        assertEquals(
            PermissionState.DENIED_CAN_ASK,
            PermissionModel.classify(granted = false, shouldShowRationale = true, hasAsked = true),
        )
    }

    @Test
    fun `never asked and permanently denied differ only by our own record`() {
        // This is the whole reason AskedPermissions exists: the two platform signals
        // are identical in both cases.
        assertEquals(
            PermissionState.NEVER_ASKED,
            PermissionModel.classify(granted = false, shouldShowRationale = false, hasAsked = false),
        )
        assertEquals(
            PermissionState.DENIED_PERMANENTLY,
            PermissionModel.classify(granted = false, shouldShowRationale = false, hasAsked = true),
        )
    }

    // -- next ----------------------------------------------------------------

    @Test
    fun `all granted proceeds`() {
        val states = mapOf(
            camera to PermissionState.GRANTED,
            location to PermissionState.GRANTED,
        )
        assertEquals(PermissionAction.Proceed, PermissionModel.next(states))
    }

    @Test
    fun `only never asked means request`() {
        val states = mapOf(
            camera to PermissionState.NEVER_ASKED,
            location to PermissionState.GRANTED,
        )
        assertEquals(PermissionAction.Request(listOf(camera)), PermissionModel.next(states))
    }

    @Test
    fun `a soft denial means explain first`() {
        val states = mapOf(
            camera to PermissionState.DENIED_CAN_ASK,
            location to PermissionState.GRANTED,
        )
        assertEquals(PermissionAction.Rationale(listOf(camera)), PermissionModel.next(states))
    }

    @Test
    fun `a permanent denial outranks a soft one`() {
        // Asking again does nothing, so an "allow" button would be a dead end.
        val states = mapOf(
            camera to PermissionState.DENIED_CAN_ASK,
            location to PermissionState.DENIED_PERMANENTLY,
        )
        assertEquals(PermissionAction.OpenSettings(listOf(location)), PermissionModel.next(states))
    }

    @Test
    fun `a permanent denial outranks an unasked permission`() {
        val states = mapOf(
            camera to PermissionState.NEVER_ASKED,
            location to PermissionState.DENIED_PERMANENTLY,
        )
        assertEquals(PermissionAction.OpenSettings(listOf(location)), PermissionModel.next(states))
    }

    @Test
    fun `a soft denial outranks an unasked permission`() {
        val states = mapOf(
            camera to PermissionState.NEVER_ASKED,
            location to PermissionState.DENIED_CAN_ASK,
        )
        assertEquals(PermissionAction.Rationale(listOf(location)), PermissionModel.next(states))
    }

    @Test
    fun `every permission in a returned action is named, not just the first`() {
        val states = mapOf(
            camera to PermissionState.NEVER_ASKED,
            location to PermissionState.NEVER_ASKED,
            notifications to PermissionState.NEVER_ASKED,
        )
        val action = PermissionModel.next(states) as PermissionAction.Request
        assertEquals(3, action.permissions.size)
    }

    @Test
    fun `returned permissions are sorted so the action is comparable`() {
        val states = mapOf(
            notifications to PermissionState.NEVER_ASKED,
            camera to PermissionState.NEVER_ASKED,
        )
        val action = PermissionModel.next(states) as PermissionAction.Request
        assertEquals(action.permissions.sorted(), action.permissions)
    }

    @Test
    fun `no requirements proceeds rather than failing`() {
        assertEquals(PermissionAction.Proceed, PermissionModel.next(emptyMap()))
    }

    // -- sanity --------------------------------------------------------------

    @Test
    fun `the flow terminates in Proceed from any starting point`() {
        // Walking the flow the way a user does: whatever the initial states, granting
        // whatever the action names must eventually reach Proceed and not loop.
        val initial = listOf(
            PermissionState.NEVER_ASKED,
            PermissionState.DENIED_CAN_ASK,
            PermissionState.DENIED_PERMANENTLY,
        )
        for (a in initial) for (b in initial) {
            var states = mapOf(camera to a, location to b)
            var steps = 0
            while (PermissionModel.next(states) != PermissionAction.Proceed) {
                val named = when (val action = PermissionModel.next(states)) {
                    is PermissionAction.Request -> action.permissions
                    is PermissionAction.Rationale -> action.permissions
                    is PermissionAction.OpenSettings -> action.permissions
                    PermissionAction.Proceed -> emptyList()
                }
                assertTrue("an action must name at least one permission", named.isNotEmpty())
                states = states + named.associateWith { PermissionState.GRANTED }
                steps++
                assertTrue("flow did not converge from $a/$b", steps <= 4)
            }
        }
    }
}
