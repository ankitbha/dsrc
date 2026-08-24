package com.dsrc.transport

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class RateCommandTest {

    private val rates = mapOf("camera_hz" to 5.0, "gps_hz" to 1.0, "imu_hz" to 50.0, "here_hz" to 0.2)

    private fun command(here: HereQuery? = null) = RateCommand(
        captureMonoNs = 1,
        rates = rates,
        trigger = "thermal",
        shadow = false,
        here = here,
    )

    private val query = HereQuery(
        `in` = "corridor:40.7,-74.0;40.8,-74.1;r=200",
        locationRef = "shape",
        lat = 40.7128,
        lon = -74.0060,
        radiusM = 9_000.0,
    )

    @Test
    fun `a here query survives a round trip field for field`() {
        // The reconciliation harness compares refusal *reasons*, so every accepted case is
        // value-blind -- swapping lat and lon on decode passed the whole suite. That would
        // put a wrong position in query_lat and query_lon on every here frame, silently,
        // and the phone echoes those straight onto the wire.
        val decoded = RateCommand.fromWire(command(query).toExtensions(), ByteArray(0)).here!!

        assertEquals(query.`in`, decoded.`in`)
        assertEquals(query.locationRef, decoded.locationRef)
        assertEquals(query.lat, decoded.lat)
        assertEquals(query.lon, decoded.lon)
        assertEquals(query.radiusM, decoded.radiusM)
    }

    @Test
    fun `absent means no change, and is not an error`() {
        // The whole reason the field could be added without a coordinated flag day.
        val decoded = RateCommand.fromWire(command().toExtensions(), ByteArray(0))
        assertNull(decoded.here)
    }

    @Test
    fun `an explicit null is absent, not a refusal`() {
        // Python reads a JSON null as absent, so refusing it here would make the two
        // decoders disagree on a command a sender could plausibly write.
        val withNull = command().toExtensions() + ("here" to JsonValue.Null)
        assertNull(RateCommand.fromWire(withNull, ByteArray(0)).here)
    }

    @Test
    fun `a longitude off the globe is refused, as a latitude is`() {
        for (bad in listOf("lat" to 91.0, "lat" to -90.5, "lon" to 181.0, "lon" to -180.5)) {
            val entries = (command(query).toExtensions()["here"] as JsonValue.Obj).entries +
                (bad.first to JsonValue.Real(bad.second))
            val broken = command().toExtensions() + ("here" to JsonValue.Obj(entries))
            val error = kotlin.test.assertFailsWith<MessageError>("${bad.first}=${bad.second}") {
                RateCommand.fromWire(broken, ByteArray(0))
            }
            assertEquals(RefusalReason.OUT_OF_RANGE, error.reason)
        }
    }
}
