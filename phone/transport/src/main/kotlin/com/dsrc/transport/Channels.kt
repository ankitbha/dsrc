package com.dsrc.transport

/** Drain order. `high` before `normal` before `bulk`, round-robin within a tier. */
enum class Priority { HIGH, NORMAL, BULK }

/** What to do when a channel's queue is full. */
enum class Overflow {
    /**
     * Drop the oldest queued message and enqueue the new one.
     *
     * Recency wins because for every reliable channel here a newer message is worth
     * more than an older one. The drop is counted.
     */
    RELIABLE,

    /** The queue holds one message; a new one replaces any unsent one, counted as dropped. */
    LATEST_WINS,
}

/** Documentation only: the transport does not refuse a frame for arriving the wrong way. */
enum class Direction { UP, DOWN, BOTH }

data class ChannelPolicy(
    val id: String,
    val direction: Direction,
    val priority: Priority,
    val overflow: Overflow,
    val depth: Int,
)

/**
 * The channel table from `specs/transport_protocol.md`.
 *
 * Every channel MUST have a policy: there is no default for an unknown one, and a frame
 * naming a channel that is not here is a protocol error rather than something to guess
 * at. A test reads the table back out of the spec so the two cannot drift.
 */
object Channels {

    const val CONTROL = "control"
    const val RATE_CMD = "rate_cmd"
    const val ADVISORY = "advisory"
    const val GPS = "gps"
    const val IMU = "imu"
    const val HERE = "here"
    const val TELEMETRY = "telemetry"
    const val CAMERA = "camera"

    val ALL: List<ChannelPolicy> = listOf(
        ChannelPolicy(CONTROL, Direction.BOTH, Priority.HIGH, Overflow.RELIABLE, 8),
        ChannelPolicy(RATE_CMD, Direction.DOWN, Priority.HIGH, Overflow.RELIABLE, 16),
        ChannelPolicy(ADVISORY, Direction.DOWN, Priority.HIGH, Overflow.LATEST_WINS, 1),
        ChannelPolicy(GPS, Direction.UP, Priority.NORMAL, Overflow.RELIABLE, 64),
        ChannelPolicy(IMU, Direction.UP, Priority.NORMAL, Overflow.RELIABLE, 256),
        ChannelPolicy(HERE, Direction.UP, Priority.NORMAL, Overflow.RELIABLE, 16),
        ChannelPolicy(TELEMETRY, Direction.UP, Priority.NORMAL, Overflow.RELIABLE, 32),
        ChannelPolicy(CAMERA, Direction.UP, Priority.BULK, Overflow.LATEST_WINS, 1),
    )

    private val byId: Map<String, ChannelPolicy> = ALL.associateBy { it.id }

    fun isKnown(id: String): Boolean = id in byId

    fun policy(id: String): ChannelPolicy =
        byId[id] ?: throw FramingError("unknown channel '$id'")

    /** Channel ids in drain order: by priority tier, then as the table lists them. */
    val drainOrder: List<ChannelPolicy> = ALL.sortedBy { it.priority.ordinal }
}
