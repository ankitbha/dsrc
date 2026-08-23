package com.dsrc.phone.sensors

import android.os.Handler
import android.os.Looper
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * A lifecycle a test can drive, so CameraX can be bound without a service or Activity.
 *
 * `bindToLifecycle` only opens the camera once the owner reaches STARTED, so a test
 * that forgets to start it sees no frames and no error at all.
 */
class TestLifecycleOwner : LifecycleOwner {

    private val registry = LifecycleRegistry(this)

    override val lifecycle: Lifecycle get() = registry

    fun start() = onMain { registry.currentState = Lifecycle.State.RESUMED }

    fun stop() = onMain { registry.currentState = Lifecycle.State.DESTROYED }

    /** LifecycleRegistry enforces main-thread access; instrumented tests run elsewhere. */
    private fun onMain(block: () -> Unit) {
        val latch = CountDownLatch(1)
        Handler(Looper.getMainLooper()).post {
            block()
            latch.countDown()
        }
        latch.await(5, TimeUnit.SECONDS)
    }
}
