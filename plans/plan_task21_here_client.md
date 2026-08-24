# Task 21 — HERE client

## Short version

The phone calls the HERE Traffic API on a timer, forwards the raw response body on the
`here` channel with both clocks around the call, and interprets none of it. The wire
message (`HereResponse`) and the channel policy already exist from task 19; this task is
the client, its configuration, and its failure accounting.

Three things decide whether this is worth having, and only one of them is about HTTP.

1. **The query shape is the Jetson's, and today the phone has no way to be told it.**
   `specs/transport_protocol.md` is explicit that "the HERE query shape and
   location-referencing mode" is set upstream, because the state that would justify
   choosing is all on the Jetson. It is equally explicit that `rate_cmd` carries only
   `rates`, that a non-frequency setting "cannot be smuggled into `rates`", and that
   carrying one "means a sibling object and a new header field". So this task either
   widens the downlink or the phone starts inventing sensing decisions. It widens the
   downlink.

2. **The key is shared with Nash production.** A test that hits the live API spends a
   production quota on a research prototype, and a retry loop that hits it in anger is a
   production incident with my name on it. No test touches the network. The HTTP call sits
   behind an interface with a fake, and the one live check is a separate opt-in target with
   a hard call budget.

3. **A failed call is data.** A 429, a timeout and a 200 with an empty body are three
   different facts about the drive, and the receiver can only tell them apart if the phone
   forwards rather than swallows. `status` is on the wire and the body rides in the payload
   untouched, empty body included.

**Scope boundary.** No parsing of the response, no flow-model, no map matching, no caching
or deduplication of responses, no retry-with-backoff beyond a single bounded attempt. The
phone is a courier. `telemetry`'s `here_calls` / `here_errors` are populated here; the rest
of `telemetry` is task 24.

## Open decisions — taken, not asked

| Decision | Taken | Why |
|---|---|---|
| Where the query shape comes from | **Widen `rate_cmd` with an optional `here` sibling object** | The spec names this as the mechanism and names the failure mode of doing it late: "an old receiver tolerates a new sender; but a new receiver `require`s the field and refuses an old sender's command". Optional from the start avoids the flag day entirely, and it is the spec's own recommendation ("receiver last, or the field optional from the start"). |
| What that object carries | `in` (the location-referencing string HERE takes verbatim), `radius_m`, `locationReferencing` | Enough to express a corridor or a circle without the phone composing either. The phone concatenates them into a URL and does not otherwise read them. |
| What the phone does before any command arrives | **Nothing.** No calls until told. | A default query shape is the phone originating a sensing decision, which is the rule this whole surface exists to enforce. A drive that produces no `here` frames because nobody configured one is a legible outcome; a drive that produces frames for a corridor nobody chose is not. |
| Rate | `here_hz` through the same `RateGate` as the other three | "At the commanded rate" has to mean one thing on every modality. Default 0.2 Hz. |
| Timeouts | Connect 5 s, read 10 s, one attempt, no retry | A retry against a shared production key is the wrong default. The gate will offer another call in 5 s anyway, which *is* the retry, and it is rate-limited by construction. |
| Failure representation | Forwarded as a frame with `status` and an empty payload | The alternative is a counter the receiver cannot correlate with a moment. A timeout gets `status = 0`, which no HTTP response can collide with. |
| The key itself | From `local.properties` into `BuildConfig`, never committed | Same handling as any credential. A missing key is a hard startup refusal for this modality, not a stream of 401s. |
| HTTP client | `HttpURLConnection` behind an interface | The app has no HTTP dependency today and one call site does not earn OkHttp. The interface is what tests need, not the library. |

## Shape

- **`HereQuery`** — the commanded shape, parsed from `rate_cmd`'s new `here` object. A
  value class, not a URL: the URL is built at call time so the key never lands in a field
  that gets logged.
- **`HereClient`** — an interface with one method, `fetch(query): HereCall`, and a
  `HttpHereClient` implementation. `HereCall` carries status, content type, body bytes and
  the two monotonic stamps. The interface exists so every test can drive the failure cases
  the live API will not produce on demand.
- **`HerePipeline`** — the rate gate and the counters, the same shape as the other three:
  `seen`, `accepted`, `gated`, `refusedStopped`, `delivered`, `refusedBySink`, plus
  `calls`, `errors`, and `unconfigured` (the gate opened and there was no query to run).
- **Wiring** — `SensingService` starts it with the others, releases it as another
  independent `release(...)` step, and logs its stats after the stop.

## The downlink widening

Three places change together, and the order matters: **senders last**.

1. Kotlin `RateCommand.fromWire` learns an *optional* `here` object. Absent means "no
   change", which is distinct from present-and-empty.
2. Python `RateCommand.from_wire` learns the same, with the same optionality, and
   `scripts/refusal_reasons.py` gains cases for a malformed one so the two decoders are
   reconciled on it.
3. Golden vectors gain a command carrying one, and a command carrying none.

Nothing on either side *sends* the new field until both sides read it. That is the whole
of the coordination, and writing it down is most of the work.

## Tests

JVM, none of it touching the network:

- Rate: `here_hz` honoured over a simulated run; a re-commanded rate re-anchors. Both
  bounds asserted — a bound that only rules out "too slow" passes for a client hammering a
  production key.
- The unconfigured path: the gate opens, no query is set, nothing is called, and it is
  counted. Asserted by the call count on a fake that fails the test if called at all.
- Failure cases the live API will not produce on demand: 429, 500, a timeout, a 200 with an
  empty body, a 200 with a body larger than one frame's payload budget. Each forwarded, each
  distinguishable at the receiver.
- The URL: built from the commanded shape, and **the key is not in `request_url`**. This is
  the one that matters — `request_url` goes on the wire and into logs, and a key that leaks
  there leaks to every artifact the drive produces. Asserted by content, not by a redaction
  pass that could silently stop matching.
- The two stamps bracket the call, driven apart deliberately so the assertion can fail.
- Accounting: `seen == accepted + gated + refusedStopped + unconfigured`, per heading and
  not only as a sum, and both identities failing in both directions.

Instrumented, on the device: that a call is actually made and a frame reaches the pipeline
— the assertion the IMU work needed three rounds to arrive at. A fake client injected
through the service, so the device test is about the wiring and not about HERE.

Live, opt-in only (`-Ddsrc.here.live=true`): one call, one assertion that the status is
200, a hard budget of one call per invocation. Never in the default suite. This exists so
the URL shape is checked against the real API once, not so the API is tested.

## What this task does not settle

- **Whether the commanded query shape is any good.** The phone cannot tell, by design.
- **Response size against the frame budget.** A corridor reply can be large; the plan
  measures it rather than assuming, and the measurement is the deliverable, not a limit.
- **What the Jetson does with the body.** Out of scope on this side of the link.
