package com.dsrc.phone

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The real SharedPreferences adapter.
 *
 * The unit tests drive [AskedPermissions] through a fake store, which leaves the
 * adapter itself unexercised -- and a `remove` that silently did nothing would turn
 * every permanent denial into an endless re-prompt without failing anything.
 */
@RunWith(AndroidJUnit4::class)
class AskedPermissionsStoreTest {

    private val permission = PermissionModel.CAMERA

    private fun freshStore(): AskedPermissions {
        val prefs = ApplicationProvider.getApplicationContext<Context>()
            .getSharedPreferences("test-permissions", Context.MODE_PRIVATE)
        prefs.edit().clear().commit()
        return AskedPermissions(prefs)
    }

    @Before
    fun sanity() {
        assertFalse(freshStore().hasAsked(permission))
    }

    @Test
    fun aRefusalPersists() {
        val asked = freshStore()
        asked.markRefused(listOf(permission))
        assertTrue(asked.hasAsked(permission))
    }

    @Test
    fun aRefusalSurvivesANewInstanceOverTheSameStore() {
        val prefs = ApplicationProvider.getApplicationContext<Context>()
            .getSharedPreferences("test-permissions", Context.MODE_PRIVATE)
        prefs.edit().clear().commit()
        AskedPermissions(prefs).markRefused(listOf(permission))
        assertTrue("the record must outlive the object that wrote it", AskedPermissions(prefs).hasAsked(permission))
    }

    @Test
    fun clearingActuallyRemoves() {
        val asked = freshStore()
        asked.markRefused(listOf(permission))
        asked.clearRefused(listOf(permission))
        assertFalse("clearRefused must reach the store, not just the object", asked.hasAsked(permission))
    }

    @Test
    fun clearingOnePermissionLeavesAnother() {
        val asked = freshStore()
        asked.markRefused(listOf(permission, PermissionModel.FINE_LOCATION))
        asked.clearRefused(listOf(permission))
        assertFalse(asked.hasAsked(permission))
        assertTrue(asked.hasAsked(PermissionModel.FINE_LOCATION))
    }
}
