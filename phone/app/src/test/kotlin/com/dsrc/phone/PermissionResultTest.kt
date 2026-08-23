package com.dsrc.phone

import org.junit.Assert.assertEquals
import org.junit.Test

class PermissionResultTest {

    private val camera = PermissionModel.CAMERA
    private val location = PermissionModel.FINE_LOCATION

    @Test
    fun `a granted permission is granted, not refused`() {
        // Inverting this split classifies a permission the user just allowed as
        // permanently denied, and offers a Settings trip for it.
        val split = PermissionResult.split(mapOf(camera to true))
        assertEquals(setOf(camera), split.granted)
        assertEquals(emptySet<String>(), split.refused)
    }

    @Test
    fun `a denied permission is refused, not granted`() {
        val split = PermissionResult.split(mapOf(camera to false))
        assertEquals(emptySet<String>(), split.granted)
        assertEquals(setOf(camera), split.refused)
    }

    @Test
    fun `a partial grant sends each permission to exactly one side`() {
        val split = PermissionResult.split(mapOf(camera to true, location to false))
        assertEquals(setOf(camera), split.granted)
        assertEquals(setOf(location), split.refused)
    }

    @Test
    fun `a dismissed dialog splits into nothing`() {
        // An empty result is what arrives when the user swipes the dialog away; it must
        // record nothing rather than recording everything as refused.
        val split = PermissionResult.split(emptyMap())
        assertEquals(emptySet<String>(), split.granted)
        assertEquals(emptySet<String>(), split.refused)
    }

    @Test
    fun `the two sides are disjoint and cover the input`() {
        val result = mapOf(camera to true, location to false, PermissionModel.POST_NOTIFICATIONS to true)
        val split = PermissionResult.split(result)
        assertEquals(emptySet<String>(), split.granted intersect split.refused)
        assertEquals(result.keys, split.granted + split.refused)
    }
}
