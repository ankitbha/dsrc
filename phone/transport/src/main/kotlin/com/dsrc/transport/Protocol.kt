package com.dsrc.transport

/**
 * Constants shared with the Python implementation in `deployment/jetson/transport/`.
 *
 * These are the numbers `specs/transport_protocol.md` fixes. A test reads them back
 * out of the spec, so the two cannot drift silently: if the spec changes and this
 * file does not, the build fails rather than the phone quietly framing something the
 * Jetson refuses.
 */
object Protocol {
    const val VERSION = 1

    const val MAX_PAYLOAD_BYTES = 4194304
    const val MAX_HEADER_BYTES = 8192

    /**
     * The largest read a session's reader may ask for.
     *
     * The stall timeout is measured on completed reads rather than completed frames,
     * so this bound is what stops a large frame on a slow link from looking like a
     * dead peer.
     */
    const val MAX_READ_BYTES = 8192

    const val KEEPALIVE_INTERVAL_S = 1.0
    const val STALL_TIMEOUT_S = 5.0
}
