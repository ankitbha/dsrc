# Writing style for this paper

Rules derived from the paragraphs Ankit wrote himself: the abstract, introduction
paragraphs one to four, the division-of-labour paragraph, and the two figure
captions. Every rule below quotes the sentence it was derived from, so a rule
that turns out to be wrong can be checked against its own evidence.

**Provenance matters here.** An earlier version of this file was derived partly
from AI-written paragraphs already in the draft, which produced rules describing
the assistant's habits rather than the author's. Before adding a rule, confirm
the sentence it comes from was written by the author.

## 1. The introduction is a funnel, one step per paragraph

The order is: the phenomenon, then the name for it, then the field broadly, then
one specific prior work, then what that work measured, then its limitation, then
this paper's question and answer. Each paragraph does exactly one of those steps.

Do not open with the contribution. Paragraph one is the phenomenon:

> A free-flow road network can jam without any apparant cause, without a crash or
> a fixed bottleneck.

## 2. Every paragraph opens by stating its own job

The first sentence is a claim, not a warm-up.

> Earlier work introducing self-regulating cars built such a controller and
> measured it.

> While the results shown by \cite{self_regulating_cars} and others are
> promising, they remain limited to simulations.

## 3. Write the connective out

The logical relation between two sentences is stated, never left for the reader
to infer. The connectives in use are "For example", "Therefore", "While",
"Another work in this space".

> Several prior studies have focussed on mitigating such traffic flow
> instabilities. For example, learned controllers that smooth mixed-autonomy
> traffic report gains at low penetration in microscopic simulators [...]

> Therefore, a controller that holds density below that point protects
> throughput [...]

## 4. A citation can be the subject of the sentence

Prior work is referred to as an actor, not parked at the end of a claim.

> \cite{bhardwaj2023understanding} notes that the fundamental diagram of traffic
> flow provides an explanation for the underlying process of sudden traffic jams.

> Another work in this space introduced the concept of self-regulating
> cars~\cite{self_regulating_cars}.

## 5. Name a term explicitly when it first appears

Definition is a marked event with its own sentence.

> This phenomenon in the literature is coined as a sudden traffic jam.

> We call this the minimum viable deployment and discuss it in
> Section~\ref{sec:vision}.

## 6. Concede, then pivot

Prior work is credited before its limitation is named, in one sentence.

> While the results shown by \cite{self_regulating_cars} and others are
> promising, they remain limited to simulations.

## 7. Announce a list before giving it

Use "as follows:" or a colon, then the parallel items.

> Our answer has three parts: a phone the driver already owns (OnePlus Nord N10),
> an onboard computer (NVIDIA Jetson Orin Nano - US\$250), and a public traffic
> API (HERE free tier).

> The division of labour inside the rig is as follows: the phone captures and
> forwards, the Jetson interprets and decides.

## 8. Every number carries what it is measured against

A bare number does not appear.

> In PTV Vissim the policy raised throughput by 5\% and cut average delay by 13\%
> against no control.

> the phone-to-Jetson path a 95th percentile of 116.19\,ms against a 200\,ms
> target

> the deployed configuration gains $230 \pm 44$\,veh/h over no control while the
> published self-regulating cars configuration gains $224 \pm 61$\,veh/h

## 9. Hardware is named with model, and price where it is the point

> a phone the driver already owns (OnePlus Nord N10), an onboard computer (NVIDIA
> Jetson Orin Nano - US\$250), and a public traffic API (HERE free tier)

## 10. "We" for what the authors do; the system is described impersonally

> In this paper, we ask what is the minimal agumentation needed to an ordinary
> vehicle [...]

> Figure~\ref{fig:teaser} is the prototype we built, installed and running.

The artifact never acts in first person:

> The phone supplies the camera, GPS and inertial unit and carries the display;
> the Jetson runs perception, the policy and the safety gate [...]

## 11. Point at every float in the prose, and say what it holds

> Our proposed solution is summarized in Table~\ref{tab:bom}.

> Figure~\ref{fig:teaser} is the prototype we built, installed and running.

## 12. An appositive can carry the evaluation

A short clause between commas does the arguing, so the sentence stays one
sentence.

> Deployment, an extremely important and difficult step for impact, is generally
> treated as an engineering detail rather than as a research question.

## 13. Captions run title, then panels, then legend

1. A title sentence that is a noun phrase: "The deployment prototype." "The
   deployment, at two scales."
2. The panels in order, first person allowed: "In (b), we show the camera
   perception module." "In (b), we detail one of those vehicles."
3. The reading conventions last: "The leader in the ego lane is amber and the
   other tracked vehicles are cyan." "The dashed box is the set of vehicle
   systems that are not touched in this design."

## 14. Source layout

One paragraph is one physical line of LaTeX, and a `\caption{}` is one line.
Wrapping is done by the editor, not by hard line breaks, because hard breaks
make commenting and editing harder. `tabular` rows, `tikzpicture` statements and
`\section`/`\label` keep their own lines.

## What this file does not claim

Rules about defining the contribution through lists of what is absent, and about
placing scope limits early, were removed. They were derived from AI-written
paragraphs, not from the author's own.
