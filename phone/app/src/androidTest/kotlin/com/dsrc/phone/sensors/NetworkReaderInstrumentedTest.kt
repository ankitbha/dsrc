package com.dsrc.phone.sensors

import android.content.Context
import android.net.ConnectivityManager
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The adapter half, against a real `ConnectivityManager`.
 *
 * The rule that decides what a reading means is unit-tested through [NetworkReader.readingFor]
 * and [NetworkReader.readingFrom], which take the platform as two functions. What only a
 * handset can answer is whether those functions return anything: whether the system service
 * is obtainable, whether `ACCESS_NETWORK_STATE` is actually granted to the installed package
 * rather than merely written in the manifest, and whether `hasTransport` answers for the
 * transports this build probes.
 *
 * Without this the field could be inert on the device and every unit test would still pass,
 * with the drive recording `permission_denied` on every sample and nothing failing.
 */
@RunWith(AndroidJUnit4::class)
class NetworkReaderInstrumentedTest {

    private fun connectivity(): ConnectivityManager? =
        ApplicationProvider.getApplicationContext<Context>()
            .getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager

    @Test
    fun theSystemServiceIsObtainable() {
        // `no_manager` for a whole drive would be this, and it is the one reason that says
        // nothing at all about the network.
        assertNotNull(connectivity())
    }

    @Test
    fun theReadingIsNeitherPermissionDeniedNorAnUnhandledThrow() {
        // The two reasons that would mean the field never works on this handset, as opposed
        // to the ones that describe a moment. `permission_denied` in particular is a build
        // defect: it would be true of every sample of every drive, and the manifest entry
        // alone does not prove the installed package holds it.
        val reading = NetworkReader.from(connectivity())
        assertTrue(
            "reading was ${reading.value} / ${reading.absentReason}",
            reading.absentReason != NetworkReader.REASON_PERMISSION_DENIED &&
                reading.absentReason != NetworkReader.REASON_ACCESSOR_RAISED &&
                reading.absentReason != NetworkReader.REASON_NO_MANAGER,
        )
    }

    @Test
    fun aReadingCarriesAValueOrAReasonAndNeverBoth() {
        val reading = NetworkReader.from(connectivity())
        assertTrue(
            "value=${reading.value} reason=${reading.absentReason}",
            (reading.value == null) != (reading.absentReason == null),
        )
    }

    @Test
    fun aValueNamesOnlyTransportsThisBuildDeclares() {
        // The value is a `+`-joined list, so a receiver keying on it needs every part to be
        // in the declared set. A part that is not would be a name invented here.
        val reading = NetworkReader.from(connectivity())
        val value = reading.value
        if (value == null) {
            assertTrue(
                "absent for an undeclared reason: ${reading.absentReason}",
                reading.absentReason in NetworkReader.ABSENCE_REASONS,
            )
            return
        }
        assertNull(reading.absentReason)
        val declared = NetworkReader.TRANSPORTS.map { it.second }.toSet()
        value.split(NetworkReader.SEPARATOR).forEach { part ->
            assertTrue("undeclared transport name '$part' in '$value'", part in declared)
        }
    }
}
