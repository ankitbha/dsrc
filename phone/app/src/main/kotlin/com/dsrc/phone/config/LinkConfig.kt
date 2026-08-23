package com.dsrc.phone.config

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
    }
}
