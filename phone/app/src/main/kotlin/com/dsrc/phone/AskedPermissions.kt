package com.dsrc.phone

import android.content.SharedPreferences

/**
 * Which permissions this install has actually prompted for.
 *
 * Needed because `shouldShowRequestPermissionRationale` returns false both for a
 * permission never requested and one denied for good, and the right response to
 * those is opposite: ask, or send the user to Settings. The platform keeps no record,
 * so the app keeps its own.
 *
 * Wrapped behind an interface-shaped constructor so the logic is testable with a
 * plain map instead of an Android SharedPreferences.
 */
class AskedPermissions(private val store: Store) {

    /** The slice of SharedPreferences this needs, so tests do not need Android. */
    interface Store {
        fun contains(key: String): Boolean
        fun put(keys: Set<String>)
    }

    constructor(prefs: SharedPreferences) : this(SharedPrefsStore(prefs))

    fun hasAsked(permission: String): Boolean = store.contains(permission)

    fun markAsked(permissions: Collection<String>) {
        if (permissions.isEmpty()) return
        store.put(permissions.toSet())
    }

    private class SharedPrefsStore(private val prefs: SharedPreferences) : Store {
        override fun contains(key: String): Boolean = prefs.getBoolean(key, false)
        override fun put(keys: Set<String>) {
            prefs.edit().apply { keys.forEach { putBoolean(it, true) } }.apply()
        }
    }
}
