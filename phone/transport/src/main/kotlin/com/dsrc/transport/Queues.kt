package com.dsrc.transport

/** A message waiting for the writer. */
class Outbound(
    val channel: String,
    val sequence: Long,
    val extensions: Map<String, JsonValue>,
    val payload: ByteArray,
    /**
     * The sender's monotonic clock **at enqueue**, per the spec's header table.
     *
     * Carried on the message rather than read by the writer. Reading it at write time
     * instead folded the queueing delay into the field the spec defines as excluding it,
     * so `t_mono_ns - t_capture_mono_ns` -- which the spec names as a valid subtraction
     * for queueing latency -- measured capture-to-write and came out as the queue's own
     * depth. It also collapsed the distinction from `t_wire_mono_ns`, which exists
     * precisely to be the later of the two.
     */
    val monoNs: Long,
    val wallNs: Long,
    /** Whether the writer should stamp `t_wire_mono_ns` just before the bytes leave. */
    val wantsWireStamp: Boolean = false,
    /** Reserved keys this message is allowed to carry, by exact name. */
    val allowReserved: Set<String> = emptySet(),
)

/** Per-channel counters. Nothing is dropped silently. */
data class ChannelCounters(
    val enqueued: Long = 0,
    val dropped: Long = 0,
    val sent: Long = 0,
    /**
     * Queued when the session ended, so never sent and never dropped.
     *
     * Its own field because it was previously nothing at all: a message orphaned by
     * `close()` appeared only in the *derived* `pending`, on a session that had ended, and
     * Python's own comment for the equivalent counter says why -- "deriving a loss by
     * subtraction is how a counting bug hides".
     */
    val abandoned: Long = 0,
) {
    val pending: Long get() = enqueued - dropped - sent - abandoned
}

/**
 * The outbound side: one queue per channel, with the table's priority and overflow.
 *
 * Two rules from `specs/transport_protocol.md` decide almost everything here.
 *
 * `seq` is assigned **at enqueue, before any overflow decision**, which is what makes a
 * gap in received sequence numbers the peer's evidence that the sender dropped
 * something. Assigning it at send time instead would renumber the survivors and hide
 * every drop.
 *
 * And the hello spends `control` sequence 0, so a session's own control traffic
 * continues from 1. A peer that restarted control at 0 would duplicate the hello's
 * number, and the gap rule detects nothing — it only fires on a sequence *greater* than
 * expected — so the divergence would be silent and would offset every control-channel
 * gap statistic for the life of the session.
 */
class OutboundQueues {

    private val lock = Any()

    private val queues: Map<String, ArrayDeque<Outbound>> =
        Channels.ALL.associate { it.id to ArrayDeque<Outbound>() }

    private val nextSequence: MutableMap<String, Long> =
        Channels.ALL.associate { it.id to 0L }.toMutableMap()

    private val counters: MutableMap<String, ChannelCounters> =
        Channels.ALL.associate { it.id to ChannelCounters() }.toMutableMap()

    /** Round-robin cursor per priority tier, so no channel starves an equal peer. */
    private val cursor: MutableMap<Priority, Int> =
        Priority.entries.associateWith { 0 }.toMutableMap()

    /**
     * Enqueue a message, assigning its sequence number.
     *
     * @return the sequence assigned, and the message displaced by overflow if any.
     */
    fun enqueue(
        channel: String,
        extensions: Map<String, JsonValue>,
        payload: ByteArray,
        monoNs: Long,
        wallNs: Long,
        wantsWireStamp: Boolean = false,
        allowReserved: Set<String> = emptySet(),
    ): Enqueued = synchronized(lock) {
        val policy = Channels.policy(channel)
        val queue = queues.getValue(channel)

        // Before the overflow decision, deliberately.
        val sequence = nextSequence.getValue(channel)
        nextSequence[channel] = sequence + 1
        val message = Outbound(
            channel, sequence, extensions, payload, monoNs, wallNs, wantsWireStamp, allowReserved,
        )

        var displaced: Outbound? = null
        var displacedCount = 0
        when (policy.overflow) {
            Overflow.RELIABLE ->
                if (queue.size >= policy.depth) {
                    // Oldest out: for every reliable channel here a newer message is
                    // worth more than an older one.
                    displaced = queue.removeFirst()
                    displacedCount = 1
                }
            Overflow.LATEST_WINS ->
                while (queue.size >= policy.depth) {
                    displaced = queue.removeFirst()
                    // Counted per message, not per enqueue. The loop can shed more than
                    // one, and a single increment would undercount. Unreachable at depth 1,
                    // which is every latest_wins channel today, and wrong the moment a
                    // depth changes -- the kind of latent miscount that surfaces as a
                    // drop rate that does not add up.
                    displacedCount++
                }
        }
        queue.addLast(message)

        val previous = counters.getValue(channel)
        counters[channel] = previous.copy(
            enqueued = previous.enqueued + 1,
            dropped = previous.dropped + displacedCount,
        )
        return Enqueued(sequence, displaced)
    }

    data class Enqueued(val sequence: Long, val displaced: Outbound?)

    /**
     * The next message to write, or null when everything is empty.
     *
     * Strict priority across tiers, round-robin within one. A saturated high or normal
     * tier can starve bulk indefinitely and that is accepted: in this system the high
     * and normal tiers carry heartbeats, commands and small sensor records, and cannot
     * saturate a link that is carrying camera frames at all.
     */
    fun poll(): Outbound? = synchronized(lock) {
        for (tier in Priority.entries) {
            val tierChannels = Channels.inTier(tier)
            if (tierChannels.isEmpty()) continue
            val start = cursor.getValue(tier)
            for (offset in tierChannels.indices) {
                val index = (start + offset) % tierChannels.size
                val queue = queues.getValue(tierChannels[index].id)
                if (queue.isNotEmpty()) {
                    // Advance past the channel just served, so the next poll at this
                    // tier starts with its neighbour.
                    cursor[tier] = (index + 1) % tierChannels.size
                    val message = queue.removeFirst()
                    val previous = counters.getValue(message.channel)
                    counters[message.channel] = previous.copy(sent = previous.sent + 1)
                    return message
                }
            }
        }
        return null
    }

    /**
     * Reserve `control` sequence 0 for the hello.
     *
     * Called by the session as it sends the hello, so ordinary control traffic starts
     * at 1 without the caller having to know why.
     */
    fun reserveHelloSequence(): Long = synchronized(lock) {
        val sequence = nextSequence.getValue(Channels.CONTROL)
        nextSequence[Channels.CONTROL] = sequence + 1
        val previous = counters.getValue(Channels.CONTROL)
        counters[Channels.CONTROL] = previous.copy(
            enqueued = previous.enqueued + 1,
            sent = previous.sent + 1,
        )
        return sequence
    }

    /**
     * Draw a sequence number without queueing anything.
     *
     * For a frame the transport writes directly -- a keepalive -- which still consumes a
     * sequence number like any other. Enqueueing and immediately polling looks equivalent
     * and is not: `enqueue` appends and `poll` takes the head, so it returns whatever was
     * already waiting.
     */
    fun nextSequenceFor(channel: String): Long = synchronized(lock) {
        Channels.policy(channel)   // refuse an unknown channel here, not later
        val sequence = nextSequence.getValue(channel)
        nextSequence[channel] = sequence + 1
        val previous = counters.getValue(channel)
        counters[channel] = previous.copy(enqueued = previous.enqueued + 1, sent = previous.sent + 1)
        return sequence
    }

    fun counters(): Map<String, ChannelCounters> = synchronized(lock) { counters.toMap() }

    /** Messages enqueued and not yet handed to the writer. */
    fun pending(): Long = synchronized(lock) { queues.values.sumOf { it.size.toLong() } }

    /**
     * The real depth of one channel's queue.
     *
     * Exists so [ChannelCounters.pending] can be checked against something. That value is
     * *derived* -- `enqueued - dropped - sent` -- which made the identity
     * `enqueued == dropped + sent + pending` expand to `enqueued == enqueued`: true for
     * every input, including inputs where the counters disagree with the queue. Two
     * mutations survived the whole suite behind it, both of which inflate `control`'s
     * apparent backlog by one per keepalive.
     */
    fun depth(channel: String): Long = synchronized(lock) {
        queues.getValue(channel).size.toLong()
    }

    fun isEmpty(): Boolean = pending() == 0L

    /**
     * Discard everything queued and count it as abandoned.
     *
     * Called once as the session ends. Without it a queued message left no trace but a
     * derived `pending` on a dead session, which is indistinguishable from a counting bug.
     */
    fun abandonAll(): Long = synchronized(lock) {
        var total = 0L
        for ((id, queue) in queues) {
            if (queue.isEmpty()) continue
            val count = queue.size.toLong()
            queue.clear()
            total += count
            val previous = counters.getValue(id)
            counters[id] = previous.copy(abandoned = previous.abandoned + count)
        }
        return total
    }
}

/** One arriving frame, with the receipt stamps the reader took. */
class Received(
    val frame: Frame,
    /**
     * The reader's own clocks, both read at one instant on arrival.
     *
     * Carried rather than re-read at handling time because the timebase requires it: the
     * initiator computes the responder's service interval as `t3 - t2`, and a `t2` taken
     * when a handler got round to the message instead of when it arrived makes that
     * difference arbitrary. Python builds the same pair in `_record_inbound` for the same
     * reason. It was equivalent here only while delivery was synchronous.
     */
    val recvMonoNs: Long,
    val recvWallNs: Long,
)

/** Per-channel inbound counters. */
data class InboundCounters(
    val received: Long = 0,
    val delivered: Long = 0,
    val dropped: Long = 0,
    val refused: Long = 0,
    val abandoned: Long = 0,
)

/**
 * The inbound side: one queue per channel, same policies and depths as outbound.
 *
 * The spec asks for this ("Inbound queues use the same policies and depths") and there was
 * none: `onFrame` ran on the reader thread with nothing between. A handler that blocked
 * stopped the reader, froze the stall timer's evidence of progress, and the watchdog ended
 * a healthy session as `STALLED` with a null cause -- the phone tearing down a working link
 * over its own slowness and then reconnecting to displace itself.
 */
class InboundQueues {

    private val lock = Any()

    private val queues: Map<String, ArrayDeque<Received>> =
        Channels.ALL.associate { it.id to ArrayDeque<Received>() }

    private val counters: MutableMap<String, InboundCounters> =
        Channels.ALL.associate { it.id to InboundCounters() }.toMutableMap()

    private val cursor: MutableMap<Priority, Int> =
        Priority.entries.associateWith { 0 }.toMutableMap()

    /** @return the message displaced by overflow, if any. */
    fun offer(message: Received): Received? = synchronized(lock) {
        val channel = message.frame.channel
        val policy = Channels.policy(channel)
        val queue = queues.getValue(channel)

        var displaced: Received? = null
        var displacedCount = 0
        when (policy.overflow) {
            Overflow.RELIABLE ->
                if (queue.size >= policy.depth) {
                    displaced = queue.removeFirst()
                    displacedCount = 1
                }
            Overflow.LATEST_WINS ->
                while (queue.size >= policy.depth) {
                    displaced = queue.removeFirst()
                    displacedCount++
                }
        }
        queue.addLast(message)
        val previous = counters.getValue(channel)
        counters[channel] = previous.copy(
            received = previous.received + 1,
            dropped = previous.dropped + displacedCount,
        )
        return displaced
    }

    /** Strict priority across tiers, round-robin within one -- the outbound rule. */
    fun poll(): Received? = synchronized(lock) {
        for (tier in Priority.entries) {
            val tierChannels = Channels.inTier(tier)
            if (tierChannels.isEmpty()) continue
            val start = cursor.getValue(tier)
            for (offset in tierChannels.indices) {
                val index = (start + offset) % tierChannels.size
                val queue = queues.getValue(tierChannels[index].id)
                if (queue.isNotEmpty()) {
                    cursor[tier] = (index + 1) % tierChannels.size
                    return queue.removeFirst()
                }
            }
        }
        return null
    }

    fun countDelivered(channel: String) = synchronized(lock) {
        val previous = counters.getValue(channel)
        counters[channel] = previous.copy(delivered = previous.delivered + 1)
    }

    fun countRefused(channel: String) = synchronized(lock) {
        val previous = counters.getValue(channel)
        counters[channel] = previous.copy(refused = previous.refused + 1)
    }

    fun abandonAll(): Long = synchronized(lock) {
        var total = 0L
        for ((id, queue) in queues) {
            if (queue.isEmpty()) continue
            val count = queue.size.toLong()
            queue.clear()
            total += count
            val previous = counters.getValue(id)
            counters[id] = previous.copy(abandoned = previous.abandoned + count)
        }
        return total
    }

    fun counters(): Map<String, InboundCounters> = synchronized(lock) { counters.toMap() }

    fun pending(): Long = synchronized(lock) { queues.values.sumOf { it.size.toLong() } }

    fun depth(channel: String): Long = synchronized(lock) { queues.getValue(channel).size.toLong() }
}
