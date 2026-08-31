package com.dsrc.phone.config

import com.dsrc.transport.Json
import com.dsrc.transport.JsonValue
import java.io.File
import java.io.IOException

/**
 * Where the phone connects, and how hard it tries.
 *
 * Separate from [SensingConfig] because the two have different owners. The Jetson
 * commands every sensing setting; the link address is a property of how the two
 * boxes are cabled together on a given day, and the Jetson cannot tell the phone
 * where to find the Jetson.
 *
 * The default host is loopback because the in-car path is `adb reverse`, which puts
 * the Jetson's listener on a device-local port. Over Tailscale the host is the
 * Jetson's tailnet address instead. Either way the phone dials: the Jetson's Tegra
 * kernel has no `CONFIG_NF_CONNTRACK_MARK`, so Tailscale cannot install its connmark
 * rules and the Jetson cannot originate traffic to tailnet peers.
 */
data class LinkConfig(
    val host: String = "127.0.0.1",
    val port: Int = DEFAULT_PORT,
    val connectTimeoutMs: Int = 3_000,
    /**
     * Read timeout for the handshake only, cleared once the session is up.
     *
     * The handshake needs one because a peer that accepts the connection and then
     * says nothing would otherwise block the link thread forever, with `stop()`
     * unable to reach it -- there is no session to close yet. Afterwards it must be
     * cleared: the reader is *supposed* to block indefinitely, and silence is the
     * watchdog's business, judged against the spec's 5 s stall timeout rather than
     * whatever this happens to be set to.
     */
    val handshakeTimeoutMs: Int = 5_000,
    val firstBackoffMs: Long = 500,
    val maxBackoffMs: Long = 8_000,
) {
    init {
        require(host.isNotBlank()) { "host is blank" }
        require(port in 1..65535) { "port is $port, outside 1..65535" }
        require(connectTimeoutMs > 0) { "connectTimeoutMs is $connectTimeoutMs" }
        require(handshakeTimeoutMs > 0) { "handshakeTimeoutMs is $handshakeTimeoutMs" }
        require(firstBackoffMs > 0) { "firstBackoffMs is $firstBackoffMs" }
        require(maxBackoffMs >= firstBackoffMs) {
            "maxBackoffMs $maxBackoffMs is below firstBackoffMs $firstBackoffMs"
        }
    }

    companion object {
        /** Matches `DEFAULT_PORT` in `deployment/jetson/transport/tcp.py`. */
        const val DEFAULT_PORT = 47811

        /**
         * The file a link address is pushed to, in the app's external files directory:
         *
         *     adb push link.json /sdcard/Android/data/com.dsrc.phone/files/link.json
         *
         * A file rather than a build-config value, because the address is a property of
         * how the two machines are connected on a given day and a value that changes per
         * drive should not require a build. A file rather than an intent extra, because
         * an extra means exporting a service that is deliberately `exported="false"`.
         */
        const val FILE_NAME = "link.json"

        /**
         * Read the pushed address, or the defaults when no file is present.
         *
         * Read once, at service start. An address that changes during a session is a
         * reconnect, not a configuration edit.
         *
         * A malformed file is refused rather than defaulted. A mistyped address that
         * quietly became `127.0.0.1` would connect to nothing and present as a link
         * failure, which is the one reading that sends the search in the wrong
         * direction.
         *
         * [Loaded.source] is for the operator, in the log and on the status line. It is
         * deliberately not carried to the Jetson: the run record answers the same
         * question from the other end and from a harder fact, because the session's
         * peer address is `127.0.0.1` when the phone dialled loopback and a 100.x
         * address when it crossed the tailnet. A second channel saying the same thing
         * could disagree with the first.
         */
        fun load(directory: File?): Loaded {
            val file = directory?.let { File(it, FILE_NAME) }
            if (file == null || !file.exists()) {
                return Loaded(LinkConfig(), Source.DEFAULT)
            }
            val text = try {
                file.readText()
            } catch (e: IOException) {
                throw IllegalArgumentException("${file.path} could not be read: ${e.message}")
            }
            val root = try {
                Json.decode(text)
            } catch (e: Exception) {
                throw IllegalArgumentException("${file.path} is not JSON: ${e.message}")
            }
            if (root !is JsonValue.Obj) {
                throw IllegalArgumentException(
                    "${file.path} is ${root::class.simpleName}, expected a JSON object"
                )
            }
            val host = when (val h = root.entries["host"]) {
                null -> throw IllegalArgumentException("${file.path} has no `host`")
                is JsonValue.Text -> h.value
                else -> throw IllegalArgumentException(
                    "${file.path} `host` is ${h::class.simpleName}, expected a string"
                )
            }
            val port = when (val p = root.entries["port"]) {
                null -> DEFAULT_PORT
                // Range-checked as a Long before narrowing. `Long.toInt()` keeps the
                // low 32 bits rather than saturating, so 4295015107 became 47811 and
                // 4294967297 became 1, and both were then accepted by `init` -- a
                // mistyped value silently becoming a different valid one, which is the
                // failure this whole load path is written to avoid.
                is JsonValue.Num -> {
                    if (p.value < 1 || p.value > 65535) {
                        throw IllegalArgumentException(
                            "${file.path} `port` is ${p.value}, outside 1..65535"
                        )
                    }
                    p.value.toInt()
                }
                else -> throw IllegalArgumentException(
                    "${file.path} `port` is ${p::class.simpleName}, expected a number"
                )
            }
            // Through the same `init` requirements as any other instance, so a pushed
            // file cannot reach a state a constructed one could not.
            return Loaded(LinkConfig(host = host, port = port), Source.FILE)
        }
    }

    /** Where an address came from. */
    enum class Source { FILE, DEFAULT }

    /**
     * A config and its provenance.
     *
     * The source travels with the value because a run that silently used loopback and
     * one that was pointed at a Jetson must not read alike in the record.
     */
    data class Loaded(val config: LinkConfig, val source: Source)
}
