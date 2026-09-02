package android.hardware

import android.content.Context
import android.content.ContextWrapper
import android.os.Handler
import com.dsrc.phone.config.SensingConfig
import com.dsrc.phone.log.FailureKinds
import com.dsrc.phone.sensors.ImuSource
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `imu.no_hardware` reached, before this file, only a logcat line at teardown -- one
 * of the five Kotlin failure kinds the plan named as untestable at JVM level without
 * Robolectric. `SensorManager`'s constructor is package-private, so a class declared
 * in this package (`android.hardware`, not `com.dsrc.phone.sensors`) can construct
 * and subclass it directly: no mocking library, no shadow layer, needed.
 *
 * `ContextWrapper`, not `Context` itself, is the base class `FakeContext` extends --
 * `Context` is abstract with dozens of members `ImuSource` never calls, and
 * `ContextWrapper` already implements all of them by delegating to a base context
 * this test never needs to supply.
 */
private class FakeSensorManager(
    private val accelerometer: Sensor?,
    private val gyroscope: Sensor?,
) : SensorManager() {

    val registered = mutableMapOf<Sensor, SensorEventListener>()

    override fun getDefaultSensor(type: Int): Sensor? = when (type) {
        Sensor.TYPE_ACCELEROMETER -> accelerometer
        Sensor.TYPE_GYROSCOPE -> gyroscope
        else -> null
    }

    override fun registerListener(
        listener: SensorEventListener,
        sensor: Sensor,
        samplingPeriodUs: Int,
        maxReportLatencyUs: Int,
        handler: Handler,
    ): Boolean {
        registered[sensor] = listener
        return true
    }
}

private class FakeContext(private val sensorManager: SensorManager) : ContextWrapper(null) {
    override fun getSystemService(name: String): Any? =
        if (name == Context.SENSOR_SERVICE) sensorManager else null
}

class ImuJvmHardwareTest {

    @Test
    fun `no accelerometer or gyroscope reports imu_no_hardware and registers nothing`() {
        val manager = FakeSensorManager(accelerometer = null, gyroscope = null)
        val context = FakeContext(manager)
        val source = ImuSource(
            context, SensingConfig(),
            appClock = { 0L }, monoClock = { 0L },
        )

        val kinds = mutableListOf<String>()
        source.start(
            onReading = {}, onUnpaired = {},
            onFailure = { kind, _ -> kinds.add(kind) },
        )

        assertEquals(listOf(FailureKinds.IMU_NO_HARDWARE), kinds)
        assertTrue("no sensor should ever have been registered", manager.registered.isEmpty())
    }

    @Test
    fun `a device with only a gyroscope is still imu_no_hardware`() {
        // `ImuSource.start` refuses on `accelerometer == null || gyroscope == null` --
        // one sensor present is exactly as unusable as none, and the detail string
        // should say which came back null. `Sensor()`'s own package-private
        // constructor is enough here: this sensor is never dispatched through, only
        // checked for nullness and used as a map key.
        val manager = FakeSensorManager(accelerometer = null, gyroscope = Sensor())
        val context = FakeContext(manager)
        val source = ImuSource(
            context, SensingConfig(),
            appClock = { 0L }, monoClock = { 0L },
        )

        var detail: String? = null
        source.start(
            onReading = {}, onUnpaired = {},
            onFailure = { kind, d -> assertEquals(FailureKinds.IMU_NO_HARDWARE, kind); detail = d },
        )
        assertTrue("detail must name what was missing: $detail", detail!!.contains("accelerometer=null"))
        assertTrue("no sensor should ever have been registered", manager.registered.isEmpty())
    }
}
