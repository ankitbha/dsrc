package com.dsrc.phone

/**
 * Splits a permission-request result into what to record and what to forget.
 *
 * Pure, because inverting it is silent and expensive: recording a *granted* permission
 * as refused makes the app classify it permanently denied and offer a Settings trip
 * for something the user just allowed.
 */
object PermissionResult {

    data class Split(val granted: Set<String>, val refused: Set<String>)

    fun split(result: Map<String, Boolean>): Split =
        Split(
            granted = result.filterValues { it }.keys,
            refused = result.filterValues { !it }.keys,
        )
}
