# Response to the RA-L review: three requests the revision does not take up

The review asks for changes that this revision makes (the mechanism for unnecessary
interventions and partial compliance, the full ablation including a camera-only arm, a model
of driver response, the title and the abstract). This note covers the three requests it does
not take up. For each, it states the request, the reason it is not done, and what the
revision offers in its place.

## 1. Sensitivity to the length and placement of the approach roads (Major Concern 6)

**Request.** Vary the 3 km approach roads, since they give the advisory space and time to act
on arriving traffic.

**Why the approach roads are not a parameter of the system.** They are part of the demand
model. In SUMO a burst is a schedule of insertions, so without roads upstream its vehicles
appear on the entry links at rates that real roads cannot deliver: the 9,000 veh/h burst sends
744 veh/h onto a one-lane entry. On a real highway those vehicles arrive along the roads before
the entrance, and the approach roads stand in for them. Their only requirement is to be long
enough that no queue reaches their far end, because a queue that does would make SUMO hold
vehicles off the map. At 3 km, no queue reached the far end in any evaluation run, and the
longest queue on an approach in the qualification runs was 832 m. Making the roads longer
therefore does not change what the network receives. Making them shorter breaks that
requirement, so the result would measure SUMO's insertion backlog rather than the advisory.

**What the measurements already charge.** Equipped vehicles on the approach roads follow the
advisory, as they would on the real roads upstream, and the time they spend there is not
counted in jam time or in travel time in the network. Holding vehicles back upstream would
lower the flow served at the exit over the two-hour run, and throughput at the exit is
reported for every controller.

**Placement.** Moving where the advisory can first reach vehicles amounts to a different
geography, for example an entrance fed directly by an urban street. That is a property of a
second network, which point 3 covers.

## 2. Background on Self-Regulating Cars, and a jam-prevention guarantee from the AAAI paper or the SICON analysis (main change 1)

**Request.** Add background on Self-Regulating Cars and its vision for sudden jams, and either
use the AAAI paper to guarantee that the system prevents jams or import the SICON analysis.

**The AAAI paper has no guarantee to import.** Its results are simulation measurements
(throughput, delay and stops on a Vissim network); it states no theorem about jam prevention.
The letter already cites it for the network model and for the form of the global layer's
reward.

**The SICON analysis is about a different controller.** Its theorems hold for an admission
law that sets delayed inflow caps on a merge tree in a cell transmission model. The system in
this letter has a Q-learned speed plan and a one-step speed correction from the camera, not an
inflow cap. A theorem about the admission law would not describe this system, and presenting it
as if it did would make the overclaim that Major Concern 4 warns against. Obtaining the
guarantee would require replacing the global layer with the admission law, which is a
different paper. The SICON manuscript is also unpublished.

**Background on the authors' earlier work.** Review is double-anonymous, so a section on the
authors' own earlier vision would have to be written as third-party background, and the six
pages are used by the system, its timing analysis and its evaluation. The letter positions
itself against prior controllers in the related-work section instead.

**What the revision offers.** Section IV presents its three propositions as a timing analysis:
when a feed-only controller cannot act before onset, how often the local layer must sample,
and why a jam near capacity does not drain. They motivate the architecture and claim no
closed-loop guarantee, which is the option the review itself calls entirely reasonable for
RA-L.

## 3. A second road network, a controlled driver experiment, and ranging ground truth for the camera (Major Concerns 1, 3 and 6)

**Request.** Add an independent network or a structured sweep of demand and capacity; run a
controlled experiment with real driver responses; validate the occupancy estimate against
independently measured vehicle positions.

**Why not in this letter.**
- *Second network.* A second network needs its own model, feed process, approach boundary and
  selection runs before any evaluation, which is a study of its own and does not fit in six
  pages. The evaluation instead covers two kinds of disturbance (bursts of three sizes and
  stopped vehicles at two locations) on a network operated near capacity, where jams do not
  drain by themselves.
- *Driver experiment.* An experiment in which an advisory influences a driver needs
  institutional review board approval, which the project does not hold, and one or a few
  instrumented vehicles cannot change how traffic flows on a highway. A closed-course study
  would measure compliance but not the network effect the letter evaluates.
- *Ranging ground truth.* The phone's focal length was not recorded on the drives, so the
  absolute distance scale cannot be recovered from the existing data; it needs new drives with
  a ranging sensor.

**What the revision offers in their place.**
- A driver-response model in simulation, with a reaction time, a compliance offset, a bounded
  deceleration and variation between drivers, at two settings, alongside the existing sweep of
  the share of drivers who follow (50% to 100%). Its parameters are stated as assumptions,
  because no driver acted on an advisory during the drives. With the lax setting (median
  reaction 4 s, 15% above the advice, 1.0 m/s^2) the largest burst left 46.7 jammed
  segment-minutes against 3.7 with exact compliance and 208.4 under no control; on every seed
  and both triggers the system still left less jam than no control.
- The cost of camera error on control, measured by running the same mechanism on true
  occupancy in the camera's window: two more jam-free runs of 75 and 0.5 fewer jammed
  segment-minutes per run. With the feed gate the system's jam changed by at most 1.8
  segment-minutes at camera distance scales of 0.8 and 1.25, where the earlier ungated layer
  left 25.0 more than the lookahead at scale 0.8.
- Section V's statement that no driver acted on an advisory, so the drives establish the
  feasibility of the sensing, computation and display pipeline and the traffic results are
  simulated.
