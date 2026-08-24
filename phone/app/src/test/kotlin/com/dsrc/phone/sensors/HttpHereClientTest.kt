package com.dsrc.phone.sensors

import com.dsrc.transport.HereQuery
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.net.URLConnection
import java.net.URLStreamHandler

/**
 * [HttpHereClient.fetch] itself, driven through a stub that opens no socket.
 *
 * The key guarantee was asserted on the static `urlFor` helper, which is not what `fetch`
 * records — rewriting the recorded URL to include the key put the production key into
 * `request_url` on every frame with the whole suite green. That is the most consequential
 * claim this class makes, and it was untested where it counts.
 *
 * No network. A `URLStreamHandler` answers the connection, so nothing resolves a name or
 * opens a socket. The HERE key is shared with Nash production and a test suite has no
 * business spending its quota.
 */
class HttpHereClientTest {

    private val query = HereQuery(
        `in` = "corridor:40.7,-74.0;40.8,-74.1;r=200",
        locationRef = "shape",
        lat = 40.7128,
        lon = -74.0060,
        radiusM = 9_000.0,
    )

    /** Records the URL actually requested and answers however the test says. */
    private class Stub(
        val status: Int = 200,
        val contentType: String? = "application/json",
        val body: ByteArray = """{"results":[]}""".toByteArray(),
        /**
         * What `errorStream` serves, when it differs from `inputStream`.
         *
         * They used to be one stream, which made the test for "the failure body comes from
         * errorStream" unable to fail: always reading inputStream instead gave the same
         * bytes, so the mutation was invisible.
         */
        val errorBody: ByteArray = """{"error":"unset"}""".toByteArray(),
        val bodyFails: Boolean = false,
        val connectFails: Boolean = false,
    ) : URLStreamHandler() {
        var requested: String? = null
        var disconnected = false

        override fun openConnection(url: URL): URLConnection {
            requested = url.toString()
            return object : HttpURLConnection(url) {
                override fun connect() {
                    if (this@Stub.connectFails) throw SocketTimeoutException("connect timed out")
                }

                override fun getResponseCode(): Int {
                    connect()
                    return this@Stub.status
                }

                // Qualified. Unqualified, `contentType` resolves to Kotlin's synthetic
                // property for this very getter rather than to the Stub's field, and the
                // method calls itself until the stack goes.
                override fun getContentType(): String? = this@Stub.contentType

                override fun getInputStream(): InputStream = stream(this@Stub.body)

                override fun getErrorStream(): InputStream = stream(this@Stub.errorBody)

                private fun stream(bytes: ByteArray): InputStream =
                    if (this@Stub.bodyFails) {
                        object : InputStream() {
                            override fun read(): Int = throw SocketTimeoutException("read timed out")
                        }
                    } else {
                        ByteArrayInputStream(bytes)
                    }

                override fun disconnect() {
                    this@Stub.disconnected = true
                }

                override fun usingProxy() = false
            }
        }
    }

    private fun fetch(stub: Stub, key: String = "s3cr3t"): HereCall {
        var now = 0L
        val client = HttpHereClient(
            apiKey = key,
            monoClock = { now += 1_000_000; now },
            urlOpener = { spec -> URL(null, spec, stub) },
        )
        return client.fetch(query)
    }

    @Test
    fun `the recorded url has no key and the requested one does`() {
        val stub = Stub()
        val call = fetch(stub)

        assertFalse(
            "the key reached request_url, which goes on the wire and into every artifact: " +
                call.requestUrl,
            call.requestUrl.contains("s3cr3t"),
        )
        assertFalse(call.requestUrl.contains("apiKey"))
        assertTrue(
            "the request was made without a key, which would make the check above true " +
                "for the wrong reason: ${stub.requested}",
            stub.requested!!.contains("apiKey=s3cr3t"),
        )
    }

    @Test
    fun `a response that stalls mid-body keeps the status it already had`() {
        // The defect this class was written for. A corridor reply arrives in pieces on a
        // cellular link, so a read can time out after the status line. Reported as
        // NO_RESPONSE, such a call was byte-identical to one where HERE was never reached
        // -- same status, same empty body, same null content type, and both clocks showing
        // about the read timeout either way.
        val call = fetch(Stub(status = 200, bodyFails = true))

        assertEquals("the phone saw a 200 and must say so", 200L, call.status.toLong())
        assertEquals("and what it learned about the response", "application/json", call.contentType)
    }

    @Test
    fun `a call that never connects is the only thing reported as no response`() {
        val call = fetch(Stub(connectFails = true))

        assertEquals(HttpHereClient.NO_RESPONSE, call.status)
        assertEquals(null, call.contentType)
        assertEquals(0, call.body.size)
    }

    @Test
    fun `a failure body is read from the error stream, not discarded`() {
        // HERE puts JSON on a 429 and it is the more informative of the two bodies. Always
        // reading inputStream instead would turn every 429 into a status-0 timeout.
        val explanation = """{"error":"Too Many Requests"}""".toByteArray()
        val call = fetch(
            Stub(
                status = 429,
                // Deliberately different, so reading the wrong stream is visible. With both
                // serving the same bytes this test could not fail.
                body = """{"results":[]}""".toByteArray(),
                errorBody = explanation,
            )
        )

        assertEquals(429, call.status)
        assertArrayEquals(explanation, call.body)
    }

    @Test
    fun `a successful body rides through untouched`() {
        val body = """{"results":[{"currentFlow":{"speed":13.4}}]}""".toByteArray()
        assertArrayEquals(body, fetch(Stub(body = body)).body)
    }

    @Test
    fun `the connection is released whatever happens`() {
        for (stub in listOf(Stub(), Stub(status = 500), Stub(bodyFails = true))) {
            fetch(stub)
            assertTrue("a connection was left open", stub.disconnected)
        }
    }

    @Test
    fun `the two stamps bracket the call even when it fails`() {
        val call = fetch(Stub(connectFails = true))
        assertTrue(
            "${call.requestMonoNs} .. ${call.responseMonoNs}",
            call.responseMonoNs > call.requestMonoNs,
        )
    }

    @Test
    fun `a blank key is refused at construction`() {
        // Not a stream of 401s against a key shared with Nash production.
        for (key in listOf("", "   ")) {
            val failure = runCatching {
                HttpHereClient(apiKey = key, monoClock = { 0 })
            }.exceptionOrNull()
            assertTrue("a blank key was accepted", failure is IllegalArgumentException)
        }
    }
}
