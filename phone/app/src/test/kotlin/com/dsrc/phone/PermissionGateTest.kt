package com.dsrc.phone

import org.junit.Assert.assertEquals
import org.junit.Test

class PermissionGateTest {

    private val camera = PermissionModel.CAMERA
    private val location = PermissionModel.FINE_LOCATION

    @Test
    fun `nothing is missing when everything is granted`() {
        assertEquals(emptyList<String>(), PermissionGate.missing(listOf(camera, location)) { true })
    }

    @Test
    fun `everything required is missing when nothing is granted`() {
        assertEquals(
            listOf(camera, location).sorted(),
            PermissionGate.missing(listOf(camera, location)) { false },
        )
    }

    @Test
    fun `only the ungranted ones are reported`() {
        assertEquals(
            listOf(location),
            PermissionGate.missing(listOf(camera, location)) { it == camera },
        )
    }

    @Test
    fun `the result is sorted so the log line is stable`() {
        val required = listOf(PermissionModel.POST_NOTIFICATIONS, camera, location)
        val missing = PermissionGate.missing(required) { false }
        assertEquals(missing.sorted(), missing)
    }

    @Test
    fun `an empty requirement list is satisfied`() {
        assertEquals(emptyList<String>(), PermissionGate.missing(emptyList()) { false })
    }

    @Test
    fun `the gate is asked about exactly the required permissions`() {
        // It must not consult anything it was not given, or a revoked permission the
        // app does not require would block startup.
        val consulted = mutableListOf<String>()
        PermissionGate.missing(listOf(camera)) { consulted.add(it); true }
        assertEquals(listOf(camera), consulted)
    }

    @Test
    fun `the gate refuses a start for every level's requirement set`() {
        // Whatever the platform level, a missing requirement must be caught -- the
        // API-33 notification permission included.
        for (sdk in 29..35) {
            val required = PermissionModel.required(sdk)
            assertEquals(required.sorted(), PermissionGate.missing(required) { false })
        }
    }
}
