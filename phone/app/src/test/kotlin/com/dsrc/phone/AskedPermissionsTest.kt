package com.dsrc.phone

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AskedPermissionsTest {

    private class FakeStore : AskedPermissions.Store {
        val keys = mutableSetOf<String>()
        var writes = 0
        var removals = 0
        override fun contains(key: String) = key in keys
        override fun put(newKeys: Set<String>) {
            writes++
            keys.addAll(newKeys)
        }
        override fun remove(oldKeys: Set<String>) {
            removals++
            keys.removeAll(oldKeys)
        }
    }

    private val camera = PermissionModel.CAMERA
    private val location = PermissionModel.FINE_LOCATION

    @Test
    fun `nothing has been refused on a fresh install`() {
        assertFalse(AskedPermissions(FakeStore()).hasAsked(camera))
    }

    @Test
    fun `a refusal records every permission in the batch`() {
        val asked = AskedPermissions(FakeStore())
        asked.markRefused(listOf(camera, location))
        assertTrue(asked.hasAsked(camera))
        assertTrue(asked.hasAsked(location))
    }

    @Test
    fun `a refusal does not record permissions outside the batch`() {
        val asked = AskedPermissions(FakeStore())
        asked.markRefused(listOf(camera))
        assertFalse(asked.hasAsked(location))
    }

    @Test
    fun `an empty batch touches the store in neither direction`() {
        val store = FakeStore()
        val asked = AskedPermissions(store)
        asked.markRefused(emptyList())
        asked.clearRefused(emptyList())
        assertEquals(0, store.writes)
        assertEquals(0, store.removals)
    }

    @Test
    fun `recording the same refusal twice is idempotent`() {
        val store = FakeStore()
        val asked = AskedPermissions(store)
        asked.markRefused(listOf(camera))
        asked.markRefused(listOf(camera))
        assertTrue(asked.hasAsked(camera))
        assertEquals(1, store.keys.size)
    }

    @Test
    fun `a grant clears an earlier refusal`() {
        val asked = AskedPermissions(FakeStore())
        asked.markRefused(listOf(camera))
        asked.clearRefused(listOf(camera))
        assertFalse(asked.hasAsked(camera))
    }

    @Test
    fun `clearing one permission leaves the others recorded`() {
        val asked = AskedPermissions(FakeStore())
        asked.markRefused(listOf(camera, location))
        asked.clearRefused(listOf(camera))
        assertFalse(asked.hasAsked(camera))
        assertTrue(asked.hasAsked(location))
    }

    @Test
    fun `the record is what separates never-asked from permanently denied`() {
        // The pair of platform signals is identical in both cases; only this record
        // distinguishes them.
        val asked = AskedPermissions(FakeStore())
        assertEquals(
            PermissionState.NEVER_ASKED,
            PermissionModel.classify(false, false, asked.hasAsked(camera)),
        )
        asked.markRefused(listOf(camera))
        assertEquals(
            PermissionState.DENIED_PERMANENTLY,
            PermissionModel.classify(false, false, asked.hasAsked(camera)),
        )
    }

    @Test
    fun `a permission granted then revoked can be prompted for again`() {
        // The case a grow-only record got wrong. Granting then revoking -- in Settings,
        // or by Android's automatic reset for unused apps -- used to read as permanently
        // denied for the rest of the install, so the app could only ever offer a
        // Settings trip where the platform would have shown a prompt.
        val asked = AskedPermissions(FakeStore())

        asked.markRefused(listOf(camera))
        assertEquals(
            PermissionState.DENIED_PERMANENTLY,
            PermissionModel.classify(false, false, asked.hasAsked(camera)),
        )

        asked.clearRefused(listOf(camera))          // user grants
        assertEquals(
            PermissionState.GRANTED,
            PermissionModel.classify(true, false, asked.hasAsked(camera)),
        )

        // ... later revoked outside the app: askable again, not a dead end.
        assertEquals(
            PermissionState.NEVER_ASKED,
            PermissionModel.classify(false, false, asked.hasAsked(camera)),
        )
        assertEquals(
            PermissionAction.Request(listOf(camera)),
            PermissionModel.next(mapOf(camera to PermissionState.NEVER_ASKED)),
        )
    }
}
