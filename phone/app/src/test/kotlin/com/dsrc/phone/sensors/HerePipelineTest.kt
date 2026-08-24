package com.dsrc.phone.sensors

import com.dsrc.phone.config.SensingConfig
import com.dsrc.transport.HereQuery
import com.dsrc.transport.HereResponse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HerePipelineTest {

    private val ms = 1_000_000L

    private val query = HereQuery(
        `in` = "corridor:40.7,-74.0;40.8,-74.1;r=200",
        locationRef = "shape",
        lat = 40.7128,
        lon = -74.0060,
        radiusM = 9_000.0,
    )

    /** A client that records what it was asked and answers however the test says. */
    private class FakeClient(
        private val status: Int = 200,
        private val body: ByteArray = """{"results":[]}""".toByteArray(),
        private val contentType: String? = "application/json",
    ) : HereClient {
        val asked = mutableListOf<HereQuery>()
        override fun fetch(query: HereQuery): HereCall {
            asked.add(query)
            return HereCall(
                requestUrl = HttpHereClient.urlFor(query, key = null),
                status = status,
                contentType = contentType,
                body = body,
                requestMonoNs = 1_000,
                responseMonoNs = 1_000 + 40 * 1_000_000,
            )
        }
    }

    private class RefusingClient : HereClient {
        var calls = 0
        override fun fetch(query: HereQuery): HereCall {
            calls++
            error("this client must not be called")
        }
    }

    private fun pipeline(
        hz: Double = 1_000.0,
        client: HereClient? = FakeClient(),
        accept: Boolean = true,
    ): Triple<HerePipeline, MutableList<HereResponse>, MutableList<ByteArray>> {
        val responses = mutableListOf<HereResponse>()
        val bodies = mutableListOf<ByteArray>()
        var now = 0L
        val p = HerePipeline(
            config = SensingConfig(hereHz = hz),
            client = client,
            monoClock = { now += 10 * ms; now },
        ) { response, body -> responses.add(response); bodies.add(body); accept }
        return Triple(p, responses, bodies)
    }

    // -- the key ---------------------------------------------------------------

    @Test
    fun `the api key is not in the url that goes on the wire`() {
        // `request_url` is a wire field and lands in every artifact the drive produces --
        // the frame, the Jetson's log, whatever anyone later greps. A key that leaks there
        // leaks everywhere, and this one is shared with Nash production.
        //
        // Asserted by content rather than by checking a redaction ran, because a redaction
        // that stopped matching would leak silently.
        val recorded = HttpHereClient.urlFor(query, key = null)
        assertFalse("the recorded url carries a key: $recorded", recorded.contains("apiKey"))
        assertFalse(recorded.contains("s3cr3t"))

        // And the one used to make the call does carry it, or the request is unauthenticated
        // -- which would make the assertion above true for the wrong reason.
        val live = HttpHereClient.urlFor(query, key = "s3cr3t")
        assertTrue("the live url has no key: $live", live.contains("apiKey=s3cr3t"))
    }

    @Test
    fun `the commanded query shape reaches the url verbatim`() {
        val url = HttpHereClient.urlFor(query, key = null)
        assertTrue(url, url.contains("locationReferencing=shape"))
        // Percent-encoded, because a corridor is full of characters a query string cannot
        // carry raw. Decoding it back is what proves it is the same expression.
        val encoded = url.substringAfter("in=").substringBefore("&")
        assertEquals(query.`in`, java.net.URLDecoder.decode(encoded, "UTF-8"))
    }

    // -- no call until told -----------------------------------------------------

    @Test
    fun `nothing is called before a query is commanded`() {
        // A default query shape would be the phone originating a sensing decision, which is
        // the rule the whole configuration surface exists to enforce. Asserted on a client
        // that fails the test if it is called at all, not on a counter that could be wrong.
        val refusing = RefusingClient()
        val (p, responses, _) = pipeline(client = refusing)

        repeat(5) { assertFalse(p.tick()) }

        assertEquals(0, refusing.calls)
        assertEquals("nothing may be sent", 0, responses.size)
        assertEquals(5, p.stats.unconfigured)
        assertFalse(p.stats.configured)
        assertTrue("${p.stats}", p.stats.balances)
    }

    @Test
    fun `a commanded query starts the calls`() {
        val fake = FakeClient()
        val (p, responses, _) = pipeline(client = fake)
        p.tick()
        assertEquals(0, fake.asked.size)

        p.setQuery(query)
        assertTrue(p.tick())

        assertEquals(listOf(query), fake.asked)
        assertEquals(1, responses.size)
        assertTrue(p.stats.configured)
    }

    @Test
    fun `a null query leaves the current one alone`() {
        // "Absent means no change" is what lets the field be optional. Applying null as a
        // clear would let a command about rates silently stop the HERE stream.
        val fake = FakeClient()
        val (p, _, _) = pipeline(client = fake)
        p.setQuery(query)
        p.setQuery(null)
        p.tick()
        assertEquals(listOf(query), fake.asked)
    }

    @Test
    fun `no api key disables the modality without failing the drive`() {
        val (p, responses, _) = pipeline(client = null)
        p.setQuery(query)

        repeat(3) { assertFalse(p.tick()) }

        assertEquals(0, responses.size)
        assertEquals("counted where a missing query is counted", 3, p.stats.unconfigured)
        assertFalse("a query alone is not enough without a client", p.stats.configured)
    }

    // -- a failed call is data --------------------------------------------------

    @Test
    fun `a failure is forwarded with its status, not swallowed`() {
        // A 429, a 500 and a timeout are three different facts about the drive, and the
        // receiver can only tell them apart if the phone forwards them. Counting and
        // dropping would leave a counter nobody can correlate with a moment.
        for (status in listOf(429, 500, HttpHereClient.NO_RESPONSE)) {
            val (p, responses, bodies) = pipeline(client = FakeClient(status = status, body = ByteArray(0)))
            p.setQuery(query)
            assertTrue(p.tick())

            assertEquals("status $status must reach the wire", status.toLong(), responses.single().status)
            assertEquals("an empty body is a fact too", 0, bodies.single().size)
            assertEquals(1, p.stats.calls)
            assertEquals(1, p.stats.errors)
        }
    }

    @Test
    fun `a success is not counted as an error`() {
        val (p, _, _) = pipeline(client = FakeClient(status = 200))
        p.setQuery(query)
        p.tick()
        assertEquals(1, p.stats.calls)
        assertEquals(0, p.stats.errors)
    }

    @Test
    fun `the body rides through untouched`() {
        val body = """{"results":[{"currentFlow":{"speed":13.4}}]}""".toByteArray()
        val (p, _, bodies) = pipeline(client = FakeClient(body = body))
        p.setQuery(query)
        p.tick()
        assertArrayEquals(body, bodies.single())
    }

    @Test
    fun `the frame records the position the jetson asked for, not zeros`() {
        // The phone cannot derive a position from `in`, which may be a corridor of a
        // hundred points. Zeros would be three wire fields claiming to be a query location
        // that is not one.
        val (p, responses, _) = pipeline()
        p.setQuery(query)
        p.tick()

        val response = responses.single()
        assertEquals(40.7128, response.queryLat, 1e-9)
        assertEquals(-74.0060, response.queryLon, 1e-9)
        assertEquals(9_000.0, response.queryRadiusM, 1e-9)
    }

    @Test
    fun `the two stamps bracket the call`() {
        val (p, responses, _) = pipeline()
        p.setQuery(query)
        p.tick()

        val response = responses.single()
        assertTrue(
            "${response.requestMonoNs} .. ${response.responseMonoNs}",
            response.responseMonoNs > response.requestMonoNs,
        )
        assertEquals("the capture stamp is the request", response.requestMonoNs, response.captureMonoNs)
    }

    // -- rate and accounting ----------------------------------------------------

    @Test
    fun `the commanded rate is honoured`() {
        // The clock advances 10 ms per tick, so 100 ticks is one simulated second. At 10 Hz
        // that is about ten calls, not a hundred.
        val fake = FakeClient()
        val (p, _, _) = pipeline(hz = 10.0, client = fake)
        p.setQuery(query)
        repeat(100) { p.tick() }

        assertTrue("made ${fake.asked.size} calls, wanted about 10", fake.asked.size in 9..11)
        assertTrue("${p.stats}", p.stats.balances)
    }

    @Test
    fun `a tick after stop is refused and counted where it happened`() {
        val refusing = RefusingClient()
        val (p, _, _) = pipeline(client = refusing)
        p.setQuery(query)
        p.stop()

        assertFalse(p.tick())
        assertEquals(0, refusing.calls)
        assertEquals(1, p.stats.refusedStopped)
        assertEquals("a stopped tick is not an unconfigured one", 0, p.stats.unconfigured)
    }

    @Test
    fun `every heading is asserted, not only the sum`() {
        // Distinct counts, so the four assertions are not interchangeable -- an equal-count
        // version of this test on another modality passed with two headings swapped.
        val (p, _, _) = pipeline(hz = 10.0)
        repeat(2) { p.tick() }                       // unconfigured: 2
        p.setQuery(query)
        p.tick()                                     // accepted: 1
        repeat(3) { p.tick() }                       // gated: 3, inside the 100 ms period
        p.stop()
        repeat(4) { p.tick() }                       // refusedStopped: 4

        val stats = p.stats
        assertEquals(10, stats.seen)
        assertEquals(1, stats.accepted)
        assertEquals(3, stats.gated)
        assertEquals(2, stats.unconfigured)
        assertEquals(4, stats.refusedStopped)
        assertTrue("$stats", stats.balances)
    }

    @Test
    fun `a reply the transport refuses is counted apart from a delivered one`() {
        val (p, _, _) = pipeline(accept = false)
        p.setQuery(query)
        p.tick()

        assertEquals(1, p.stats.refusedBySink)
        assertEquals(0, p.stats.delivered)
        assertTrue("${p.stats}", p.stats.acceptedBalances)
    }

    @Test
    fun `a reply the transport accepts is counted as delivered`() {
        val (p, _, _) = pipeline(accept = true)
        p.setQuery(query)
        p.tick()

        assertEquals(1, p.stats.delivered)
        assertEquals(0, p.stats.refusedBySink)
        assertTrue("${p.stats}", p.stats.acceptedBalances)
    }

    private fun assertArrayEquals(expected: ByteArray, actual: ByteArray) {
        assertEquals(expected.toList(), actual.toList())
    }
}
