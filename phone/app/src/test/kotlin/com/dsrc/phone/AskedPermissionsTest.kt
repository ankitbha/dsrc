package com.dsrc.phone

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AskedPermissionsTest {

    private class FakeStore : AskedPermissions.Store {
        val keys = mutableSetOf<String>()
        var writes = 0
        override fun contains(key: String) = key in keys
        override fun put(newKeys: Set<String>) {
            writes++
            keys.addAll(newKeys)
        }
    }

    @Test
    fun `nothing has been asked on a fresh install`() {
        val asked = AskedPermissions(FakeStore())
        assertFalse(asked.hasAsked(PermissionModel.CAMERA))
    }

    @Test
    fun `marking records every permission in the batch`() {
        val store = FakeStore()
        val asked = AskedPermissions(store)
        asked.markAsked(listOf(PermissionModel.CAMERA, PermissionModel.FINE_LOCATION))
        assertTrue(asked.hasAsked(PermissionModel.CAMERA))
        assertTrue(asked.hasAsked(PermissionModel.FINE_LOCATION))
    }

    @Test
    fun `marking does not record permissions that were not asked about`() {
        val asked = AskedPermissions(FakeStore())
        asked.markAsked(listOf(PermissionModel.CAMERA))
        assertFalse(asked.hasAsked(PermissionModel.FINE_LOCATION))
    }

    @Test
    fun `an empty batch writes nothing`() {
        val store = FakeStore()
        AskedPermissions(store).markAsked(emptyList())
        assertEquals(0, store.writes)
    }

    @Test
    fun `marking twice is idempotent`() {
        val store = FakeStore()
        val asked = AskedPermissions(store)
        asked.markAsked(listOf(PermissionModel.CAMERA))
        asked.markAsked(listOf(PermissionModel.CAMERA))
        assertTrue(asked.hasAsked(PermissionModel.CAMERA))
        assertEquals(1, store.keys.size)
    }

    @Test
    fun `the record is what separates never-asked from permanently denied`() {
        // The pair of platform signals is identical in both cases; only this record
        // distinguishes them, so the classification is checked through it.
        val store = FakeStore()
        val asked = AskedPermissions(store)
        val permission = PermissionModel.CAMERA

        assertEquals(
            PermissionState.NEVER_ASKED,
            PermissionModel.classify(false, false, asked.hasAsked(permission)),
        )
        asked.markAsked(listOf(permission))
        assertEquals(
            PermissionState.DENIED_PERMANENTLY,
            PermissionModel.classify(false, false, asked.hasAsked(permission)),
        )
    }
}
