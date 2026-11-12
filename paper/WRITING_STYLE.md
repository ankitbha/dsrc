# Writing style for this paper

Three sources, listed in order of authority. Each rule says where it came from, so
a rule that turns out to be wrong can be checked against its own evidence.

1. **Ankit, directly.** Part A. Given as instructions on 18 and 19 September 2026,
   not inferred from any text.
2. **Ankit's own paragraphs in this paper.** Part C. The abstract, introduction
   paragraphs one to four, the division-of-labour paragraph, and the two figure
   captions.
3. **Research statements read at the level being applied for.** Part B, carried
   down from `applications/research_statement/WRITING_RULES.md`, keeping only the
   rules that transfer from a statement to a paper.

**How the two files relate.** Part C originated here and was carried up into the
statement file. Part A originated as instructions for the statement and is carried
down here, because the instructions are about writing rather than about statements.
When either file changes, check whether the other needs the same change, and keep
the direction of flow recorded so the two do not drift into disagreement.

**Provenance matters here.** An earlier version of this file was derived partly
from AI-written paragraphs already in the draft, which produced rules describing
the assistant's habits rather than the author's. Before adding a rule, confirm the
sentence it comes from was written by the author.

---

## Part A. These come first.

A1 is the most important rule in this file. A2 and A3 are about what the sentences
contain rather than how they are phrased.

### A1. Simple and easy to follow. The paper tells a story; it is not a technical report.

Write the way you would say it if you were giving the talk. The reader should
follow every sentence on one pass, and should not feel reported at.

**Why this rule exists.** Ankit: "When I say simple and direct, I don't want the
tone to be reporting or recording, it should be storytelling. Each document is
telling a story, it is not a tech report." A clipped sentence is not more precise
than a full one. It is the same content with the connective tissue removed, and
removing it makes the reader do work the writer should have done.

**The worked example.** "Neither term implies the other" belongs in a technical
report and, in Ankit's words, "feel[s] rude to read". Write "These two terms do not
imply each other." Same meaning, same length, and the reader is being spoken to
rather than filed at.

**The test.** Read the sentence aloud as though presenting it. If you would not say
it that way to a room, rewrite it.

**What this rule is not.** It is not licence for poetry, ornament or long
constructions. The words stay short and literal. "Direct" is the wrong target,
because directness taken far enough becomes the telegraphic register this rule
rejects. The target is **simple and easy to understand**.

**The rule applies at content scale, not only at sentence scale.** When the text
makes a claim about how the material is organised, that claim is spelled out for
every item it covers. **A structural promise must be kept explicitly.** Find every
sentence that claims an organising principle, meaning the ones containing *each*,
*different*, *respectively*, *in turn*, *one at a time*, *in that order*, and
confirm the text then assigns every item its place. If the reader has to work out
the assignment, the promise was made and not kept.

**Signs that a sentence has drifted into the report register:**

- Articles and pronouns dropped: "Reconstruction is not attribution."
- A nominalisation standing in for a verb.
- A bare negative opening a sentence: "Neither term implies the other."
- A fragment that asserts without connecting to what came before.
- A sentence that announces what an example is about to do instead of telling it.
  In a talk nobody says "this example makes it concrete"; they say what happened.
  **Delete the announcement and start the example.** Grep for *makes this
  concrete*, *makes it easy to see*, *shows up clearly in*, *illustrates*, *is a
  good example of*, *is where this becomes visible*.
- A numbered signpost taken from the paper's own outline. Say what connects the
  material, not where it sits in the filing system.
- Compression applied to meet a page limit. **When the paper runs long, cut
  content, not the connective tissue of a sentence.** Drop a measurement, a
  paragraph or a table. Do not turn full sentences into telegrams to save lines.

### A2. No context jumping

Every sentence must be followable by a reader who has not read the cited papers,
using only the sentences that precede it in this paper. A sentence that requires
knowing a citation is a defect even when it is true.

**Why this rule exists.** Language models chain sentences that are each locally
plausible while assuming a referent the reader was never given. The text reads as
connected to whoever already knows the work, and as a list of assertions to
everyone else. A reviewer is in the second group.

**The test.** A reader with no access to the cited papers reads the section and
reports any sentence they cannot follow. A reader who knows the work cannot detect
the defect, because they supply the missing context themselves.

**The failure this came from.** An earlier draft named five results in sequence.
Ankit: "you have made a laundry list of papers and are rattling off stuff one by
one. As a reader who has not read all of my papers, they will likely throw out this
statement at this point."

### A3. Every what carries its why

Every sentence reporting an action carries why that action rather than another.
Every sentence reporting a result carries why the result happened, or what follows
from it.

**Why this rule exists.** The prose is simple and literal, with no euphemism and no
poetic writing, so the interest cannot come from the language. It has to come from
the content: the reason, the mechanism, the consequence. A paper that reports only
what was done and what was found is dry, and the remedy is more content, never more
adjectives.

**The test.** Take any sentence reporting an action or a result and ask "why that,
and what follows?" If the paper does not answer inside the same paragraph, the
sentence is dry.

**A sentence in this paper that already satisfies the rule.** "A capacity drop is
not a free parameter, because real queue discharge runs 5 to 20\% below free-flow
capacity, which is a property of traffic rather than of a model." The fleet choice
arrives with the reason it was not free to make.

---

## Part B. Structure, carried down from the statement rules

Only the rules that transfer from a research statement to a paper are kept here.
The statement file's B1, B3, B5, B6, B9, B10 and B11 are about a statement's own
front matter and do not apply.

**B1. Name the organising concept, and repeat the phrase verbatim.** A description
is not a name. This paper names "the minimum viable deployment" and should use that
exact phrase wherever the concept recurs, so a reader has a phrase to use about the
work when the authors are not in the room.

**B2. One sentence per prior result, in a fixed order: mechanism, number, venue.**
The result and the record arrive together rather than the record being deferred to
a trailing list.

> Self-regulating cars learns a policy that assigns a desired speed to each
> super-segment of a highway network, and reports a 5\% throughput gain and a 13\%
> delay reduction against no control in PTV Vissim.

**B3. State the organising axis early, then make it the literal section
structure.** A frame the sections do not follow is worse than no frame.

**B4. Name one recurring mechanism that runs through the work.** This is what makes
a heterogeneous set of components read as one system rather than a parts list.

---

## Part C. Sentence and paragraph

Derived from paragraphs Ankit wrote himself. Each rule quotes its evidence.

### C1. The introduction is a funnel, one step per paragraph

The order is: the phenomenon, then the name for it, then the field broadly, then
one specific prior work, then what that work measured, then its limitation, then
this paper's question and answer. Each paragraph does exactly one of those steps.
Do not open with the contribution. Paragraph one is the phenomenon:

> A free-flow road network can jam without any apparant cause, without a crash or
> a fixed bottleneck.

### C2. Every paragraph opens by stating its own job

The first sentence is a claim, not a warm-up.

> Earlier work introducing self-regulating cars built such a controller and
> measured it.

### C3. Write the connective out

The logical relation between two sentences is stated, never left for the reader to
infer. The connectives in use are "For example", "Therefore", "While", "because".

> Several prior studies have focussed on mitigating such traffic flow
> instabilities. For example, learned controllers that smooth mixed-autonomy
> traffic report gains at low penetration in microscopic simulators [...]

### C4. A citation can be the subject of the sentence

Prior work is an actor, not a parenthetical parked at the end of a claim.

> \cite{bhardwaj2023understanding} notes that the fundamental diagram of traffic
> flow provides an explanation for the underlying process of sudden traffic jams.

### C5. Name a term explicitly when it first appears

Definition is a marked event with its own sentence.

> This phenomenon in the literature is coined as a sudden traffic jam.

> We call this the minimum viable deployment and discuss it in
> Section~\ref{sec:vision}.

### C6. Concede, then pivot, in one sentence

Prior work is credited before its limitation is named.

> While the results shown by \cite{self_regulating_cars} and others are promising,
> they remain limited to simulations.

### C7. Announce a list before giving it

Use "as follows:" or a colon, then the parallel items.

> The division of labour inside the rig is as follows: the phone captures and
> forwards, the Jetson interprets and decides.

### C8. Every number carries what it is measured against

A bare number does not appear.

**The comparator is itself a number, and A3 applies to it too.** Writing "12 parts
per million against the 50 the design assumes" satisfies this rule and still fails
A3, because the reader is now told to measure against 50 and never told why 50.
A comparator that is a property of the world is stated as one: 50 parts per million
is the top of the ordinary range for the initial accuracy of a quartz crystal at
25 C. A comparator that is a number the authors chose has to say so, and say what
it was chosen from. If neither can be written, the comparator is not evidence and
should not be asserted.

> In PTV Vissim the policy raised throughput by 5\% and cut average delay by 13\%
> against no control.

> the phone-to-Jetson path a 95th percentile of 116.19\,ms against a 200\,ms target

### C9. Hardware is named with model, and price where price is the point

> a phone the driver already owns (OnePlus Nord N10), an onboard computer (NVIDIA
> Jetson Orin Nano - US\$250), and a public traffic API (HERE free tier)

### C10. "We" for what the authors do; the system described impersonally

> In this paper, we ask what is the minimal agumentation needed to an ordinary
> vehicle [...]

The artifact never acts in first person:

> The phone supplies the camera, GPS and inertial unit and carries the display;
> the Jetson runs perception, the policy and the safety gate [...]

### C11. Point at every float in the prose, and say what it holds

> Our proposed solution is summarized in Table~\ref{tab:bom}.

### C12. An appositive can carry the evaluation

A short clause between commas does the arguing, so the sentence stays one sentence.

> Deployment, an extremely important and difficult step for impact, is generally
> treated as an engineering detail rather than as a research question.

### C13. Captions run title, then panels, then legend

1. A title sentence that is a noun phrase: "The deployment prototype."
2. The panels in order, first person allowed: "In (b), we show the camera
   perception module."
3. The reading conventions last: "The leader in the ego lane is amber and the other
   tracked vehicles are cyan."

### C14. Source layout

One paragraph is one physical line of LaTeX, and a `\caption{}` is one line.
Wrapping is done by the editor, not by hard line breaks, because hard breaks make
commenting and editing harder. `tabular` rows, `tikzpicture` statements and
`\section`/`\label` keep their own lines.

---

## Naming

The language rule is **simple and easy to understand**. Not "plain", which suggests
the content should be dull. Not "direct" either, because directness pushed far
enough produces the telegraphic register A1 rejects. No euphemism and no poetic
writing, but the paper is meant to be read the way a person would say it. A1 governs
how it sounds, A2 governs whether the reader can follow it, and A3 governs whether
it is worth following.

## What this file does not claim

No rule here sets a sentence-length limit.

Rules about defining the contribution through lists of what is absent, and about
placing scope limits early, were removed in an earlier revision. They were derived
from AI-written paragraphs, not from the author's own.
