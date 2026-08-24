package com.dsrc.phone.sensors

import com.dsrc.transport.HereQuery
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * One HERE call, as it came back.
 *
 * A failed call is data. A 429, a timeout and a 200 with an empty body are three different
 * facts about the drive, and the receiver can only tell them apart if the phone forwards
 * rather than swallows. [status] is zero for a call that never got a response, which no
 * HTTP status can collide with.
 */
data class HereCall(
    val requestUrl: String,
    val status: Int,
    val contentType: String?,
    val body: ByteArray,
    val requestMonoNs: Long,
    val responseMonoNs: Long,
) {
    // ByteArray in a data class: equals/hashCode compare by identity, which would make two
    // structurally identical calls unequal. Nothing compares these, and generating the
    // members would be worse than saying so.
    override fun equals(other: Any?) = this === other
    override fun hashCode() = System.identityHashCode(this)
}

/** Makes the call. An interface so no test has to touch the network. */
interface HereClient {
    fun fetch(query: HereQuery): HereCall
}

/**
 * The real client.
 *
 * One attempt, no retry. A retry loop against a key shared with Nash production is a
 * production incident with our name on it, and the rate gate offers another call in five
 * seconds anyway — which *is* the retry, rate-limited by construction.
 *
 * The key never appears in [HereCall.requestUrl]. That field goes on the wire and into
 * every artifact the drive produces, so a key that leaks there leaks everywhere: the URL is
 * built twice, once with the key for the connection and once without it for the record.
 */
class HttpHereClient(
    private val apiKey: String,
    private val monoClock: () -> Long,
    private val connectTimeoutMs: Int = 5_000,
    private val readTimeoutMs: Int = 10_000,
    /**
     * Opens the connection. Injectable so `fetch` itself can be tested without a socket.
     *
     * The key guarantee was asserted on the `urlFor` helper, which is not what `fetch`
     * records -- rewriting the recorded URL to carry the key put it into `request_url` on
     * every frame with the whole suite green. The seam exists so that claim is checked
     * where it is made.
     */
    private val urlOpener: (String) -> URL = { URL(it) },
) : HereClient {

    init {
        // A hard refusal at construction, not a stream of 401s. The key is shared with Nash
        // production, so a build that quietly ships without one and then hammers the API
        // with unauthenticated requests is the failure worth preventing loudest.
        require(apiKey.isNotBlank()) {
            "no HERE api key: set here.apiKey in local.properties"
        }
    }

    override fun fetch(query: HereQuery): HereCall {
        val recorded = urlFor(query, key = null)
        val requestAt = monoClock()
        var connection: HttpURLConnection? = null
        // Held outside the try, and that is the point. A corridor reply arrives in pieces
        // on a cellular link, so a read can time out *after* the status line has been read
        // -- and with these inside the try, such a call was reported as NO_RESPONSE with an
        // empty body and a null content type, byte-identical to one where HERE was never
        // reached. Both clocks show about the read timeout in either case, so no field was
        // left to separate them, and the guarantee NO_RESPONSE exists for -- that it means
        // no response at all -- was false in the one case where the phone had seen one.
        var status = NO_RESPONSE
        var contentType: String? = null
        var body = ByteArray(0)
        return try {
            connection = (urlOpener(urlFor(query, apiKey)).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = connectTimeoutMs
                readTimeout = readTimeoutMs
            }
            status = connection.responseCode
            contentType = connection.contentType
            // errorStream on a failure, inputStream on success. HERE puts a JSON body on
            // both, and the failure body is the more informative of the two.
            body = (if (status in 200..299) connection.inputStream else connection.errorStream)
                ?.use { it.readBytes() } ?: ByteArray(0)
            HereCall(recorded, status, contentType, body, requestAt, monoClock())
        } catch (e: IOException) {
            // A timeout, a DNS failure, no route. Forwarded as a frame rather than counted
            // and dropped, because a receiver cannot correlate a counter with a moment --
            // and carrying whatever was learned before the failure, so a stalled body is
            // distinguishable from a network that was never there.
            HereCall(recorded, status, contentType, body, requestAt, monoClock())
        } finally {
            connection?.disconnect()
        }
    }

    companion object {
        /** No response at all. Distinct from every HTTP status. */
        const val NO_RESPONSE = 0

        const val BASE = "https://data.traffic.hereapi.com/v7/flow"

        /**
         * The request URL, with the key or without it.
         *
         * `key = null` builds the form that goes on the wire. Two calls rather than one
         * redaction pass: a pass that stopped matching would leak silently, and this cannot
         * — the recorded URL is never built from the key at all.
         */
        fun urlFor(query: HereQuery, key: String?): String {
            val parameters = buildList {
                add("in=${encode(query.`in`)}")
                add("locationReferencing=${encode(query.locationRef)}")
                if (key != null) add("apiKey=${encode(key)}")
            }
            return "$BASE?${parameters.joinToString("&")}"
        }

        private fun encode(value: String) = java.net.URLEncoder.encode(value, "UTF-8")
    }
}
