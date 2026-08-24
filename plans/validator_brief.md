# Validator brief — behavioural correctness only

Report a finding **only** if you can name a concrete way the software behaves wrongly:
a wrong value on the wire, a message lost or duplicated, a resource leaked, a crash, a
hang, a stream that silently stops, or a decision taken on data that does not mean what
the code thinks it means.

State each finding as: **what goes wrong, in terms an operator or the peer would
notice** — then the file and line, then how you established it.

## Not findings. Do not report these.

- A test that could be stronger, unless you can show the code is *actually wrong*
  underneath it. "This mutation survived" is evidence only when the mutation describes a
  defect someone could plausibly write; a contrived rewrite that no one would make is not.
- Comment, KDoc, or plan wording that overstates or has drifted. Note it in one line at
  the end under "doc drift" if you must, and move on.
- Mutation-harness anchors, test naming, assertion argument order, unused test seams.
- Races in counters that exist only for a log line, unless the wrong number would cause a
  wrong decision.
- Style, structure, duplication, or "this would be cleaner as".
- Boundary conditions that are unreachable from any caller. Say why they are unreachable
  instead, in one line.

## Ranking

Rank strictly by consequence: what breaks, how often, and whether anyone would notice.
A defect that silently corrupts data outranks one that crashes loudly.

## Verdict

**SIGN OFF** unless there is at least one behavioural defect. A short report is the
expected outcome for code that has already been through a round. Do not pad it.
