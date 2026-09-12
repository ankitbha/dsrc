# Related Materials — Citation Map for the HotNets Vision Paper

This folder collects the prior work we will lean on for the vision paper
**"Towards Self-Regulating Cars: Edge AI for Infrastructure-Free Traffic Control"**
(HotNets submission). Papers are grouped **by the argument they support**, not
alphabetically, so each entry says *where it goes and what job it does*.

## The paper's spine (so the roles below make sense)

- **Spine (Pillar 1):** highway traffic control should move off roadside infrastructure
  **and off cloud coordinators**, into the vehicles — infrastructure-free, **local-aggregate
  V2V**, decentralized, edge-executed. The sharp framing: this **decentralizes our own
  centralized predecessor** (the AAAI SRC system), removing the central coordinator and its
  per-vehicle cloud dependence, while replacing command-style coordination with compact
  vehicle-native aggregate-state exchange.
- **Communication model:** vehicles exchange or reconstruct only bounded aggregate traffic
  state: nearby AV count/density/mean speed/lane distribution, queue/congestion estimates,
  segment-level target speed, and merge pressure. They do **not** exchange raw video, stable
  identities, direct commands, AV-to-lane assignments, or coordinated lane-occupation plans.
  The control decision remains independently computed and safety-filtered on each vehicle.
- **Enabler 1:** a **differentiable digital twin of the road network** (the world model) —
  how you *train* decentralized controllers (train centrally on the twin, execute on local
  sensing + aggregate V2V = CTDE) and how each vehicle maintains an **ego-centric rolling
  digital twin** at runtime. Payoffs, in order: improved local observability, sample-efficient
  learning/adaptation, and safety shielding. Kept at *idea* altitude here; the methods depth
  is a future ML paper.
- **Enabler 2:** a **runtime safety/etiquette layer** — how you *deploy* credibly and
  non-obstructively in mixed human traffic, using the ego-centric twin to reject unsafe,
  stale, obstructive, or low-confidence advisories.
- **Why now:** rising AV/robotaxi penetration (the substrate) × commodity edge AI (the
  enabler), while today's growth defaults to *selfish* autonomy and leaves the network-control
  externality untapped.

Role tags used below: **[reuse-own]** our prior result to build on · **[support]** cite to
back a claim · **[differentiate]** the prior-art wall we must distinguish ourselves from.

---

## 0. Our own foundation — the predecessor we decentralize

### AAAI 2026 — "Self-Regulating Cars: Automating Traffic Control in Free Flow Road Networks" · **[in folder: `00832-BhardwajA.pdf`]**
Bhardwaj, Asim, Chauhan, Zaki, Subramanian (NYU + IIT Delhi). Code: <https://github.com/ankitbha/Self-Regulating-Cars>
- **The predecessor.** A **centralized** DQN sets desired speeds on 12 super-segments (2–3 km)
  of the Mainz, Germany highway in PTV Vissim, broadcast to vehicles via a **cloud service that
  assumes per-vehicle internet connectivity**. Results vs no-control: **+5% throughput, −13% avg
  delay, roughly −3% total stops**; partial-adoption ablation (25/50/75%) shows monotonic gains.
- **How it fits:** the thing the HotNets paper *pushes further*. Our contribution = remove the
  central coordinator + cloud dependence, execute per-vehicle on local edge sensing plus
  aggregate V2V state. Reuse its
  **throughput-optimality argument** (density-below-critical + maximize-speed) as our physical
  backbone, and its **partial-adoption table** as sparse-fleet evidence. **Honesty:** its
  "infrastructure-free" means *no new roadside hardware only*; ours is the stronger
  infrastructure-**and-cloud-coordinator**-free claim. **[reuse-own]**

### Bhardwaj et al. 2023 — "Understanding sudden traffic jams: From emergence to impact" · **[in folder: `1-s2.0-S2352728522000148-main.pdf`]**
*Development Engineering* 8:100105. `refs.bib`: `bhardwaj2023understanding`.
- **The problem-and-physics paper.** Defines *sudden traffic jams*: a small **burst** (not
  oversubscription) pushes density past a **critical threshold**, triggering a self-reinforcing
  **"spiraling effect"** into a stable low-capacity equilibrium (recovery needs draining). Gives
  the **traffic curve** (exit-rate vs density) with optimal operating point `B*/C*` — the rigorous
  origin of "critical density" and the two principles. n-1 **merges** are the congestion locus;
  empirical across NYC/Nairobi/São Paulo; aggressive driving → tighter/steeper curves → worse jams.
- **How it fits:** Intro (the failure mode we target) + physical backbone (with the AAAI
  throughput-optimality theorem, replaces the commented-out damping math) + why-highways-jam-at-merges
  + an **equity motivation** (infra-free control is worth most where roadside infra is unaffordable —
  Nairobi ~3× jam-time, ~0.5× capacity utilization) + human-driver-robustness framing. **Caveat:**
  measurement/characterization only — *no controller*; it's the problem paper, not evidence the
  solution works. **[reuse-own] [support]**

---

## 1. Sparse-AVs-as-actuators — premise validation *and* the prior-art wall

These are the closest neighbors. Read them to write the differentiation paragraph; our
local-aggregate, no-cloud-coordinator, commodity-edge stance must hold against them (note several
rely on a **centralized planner**, command-style coordination, and/or **purpose-built AVs** —
which is exactly our differentiation axis).

### Stern et al. 2018 — "Dissipation of stop-and-go waves via control of autonomous vehicles: Field experiments"
*Transportation Research Part C* 89:205–221. <https://arxiv.org/abs/1705.01693> · <https://doi.org/10.1016/j.trc.2018.02.005>
- Single AV (<5% penetration) damps stop-and-go waves on a real ring track.
- **How it fits:** the flagship *physical* proof of the sparse-fleet premise (Intro + "why it
  works"). **[support]**, and **[differentiate]**: theirs is a hand-designed controller on an
  instrumented test track; ours is learned, decentralized, commodity-edge, on open networks.

### Vinitsky et al. 2018 (CoRL) — "Benchmarks for Reinforcement Learning in Mixed-Autonomy Traffic"
PMLR v87. <https://proceedings.mlr.press/v87/vinitsky18a/vinitsky18a.pdf>
- Ring / figure-8 / merge / bottleneck mixed-autonomy RL benchmarks.
- **How it fits:** the benchmark lineage a HotNets reviewer will invoke. Related work — cite and
  distinguish (they benchmark *centralized-training* control in a research simulator; we target
  decentralized *edge deployment* with local aggregate state, no roadside infrastructure, and no
  cloud coordinator). **[differentiate]**

### Wu et al. — "Flow: A Modular Learning Framework for Mixed Autonomy Traffic"
IEEE Transactions on Robotics, 2021. <https://arxiv.org/abs/1710.05465> · <https://ieeexplore.ieee.org/document/9489303/>
- *The* framework paper; shows learned control with 4–7% AVs can eliminate stop-and-go.
- **How it fits:** if we are read as "a framework," we compete with this — so we must **not** frame
  as a framework, but as an infrastructure-free/edge *deployment vision*. Related work. **[differentiate]**

### Kreidieh, Wu, Bayen 2018 (ITSC) — "Dissipating stop-and-go waves in closed and open networks via deep RL"
IEEE ITSC, pp. 1475–1480. <https://www.researchgate.net/publication/329619310>
- Extends wave-dissipation to merge/open networks.
- **How it fits:** shows the approach works beyond the ring (supports our multi-topology ambition);
  related work. **[support] [differentiate]**

### CIRCLES / MegaVanderTest — 100-AV I-24 open-road field test
- Field experiment: <https://arxiv.org/abs/2402.17043>
- **Explicit *local* controllers (decentralized, non-RL — closest to our thesis; read this):** <https://arxiv.org/abs/2310.18151>
- Methodology/design: <https://arxiv.org/abs/2404.15533> · Consortium: <https://circles-consortium.github.io/>
- The largest real-world AV traffic-smoothing deployment to date.
- **How it fits:** simultaneously our strongest feasibility evidence and our biggest novelty threat.
  Crucially, CIRCLES uses a **centralized Speed Planner + decentralized controllers** and
  **purpose-built instrumented AVs** — our differentiation is *no cloud/roadside coordinator,
  aggregate-state V2V rather than command exchange, commodity retrofit hardware, advisory-to-human*.
  The "explicit local controllers" paper is the decentralized subset to engage most carefully.
  **[differentiate]** (this is *the* paragraph to nail)

### Yan, Kreidieh, Vinitsky, Bayen, Wu 2023 — human-compatible advisory autonomy · **already in `refs.bib`: `Yan_2023`**
- **How it fits:** precedent for the human-in-the-loop *advisory* interface (deployment section).
  Distinguish: we target on-vehicle edge inference with aggregate-state V2V and without a central
  controller. **[support] [differentiate]**

---

## 2. Traffic-flow theory — the physical "why it works" backbone

Replaces the currently commented-out viscosity/damping math in `architecture.tex` with a
textbook-solid foundation (fundamental diagram + critical density).

### Sugiyama et al. 2008 — "Traffic jams without bottlenecks: experimental evidence…"
*New J. Phys.* 10:033001. <https://iopscience.iop.org/article/10.1088/1367-2630/10/3/033001>
- The phantom-jam ring experiment.
- **How it fits:** empirical evidence that jams form from fluctuations alone — motivates smoothing
  as the intervention. Intro. **[support]**

### Traffic-flow classics (already cited in the AAAI paper; add PDFs only for a complete folder)
- **Greenshields 1935** — fundamental diagram (HRB Proc. vol 14).
- **Lighthill & Whitham 1955** (LWR kinematic waves) <https://doi.org/10.1098/rspa.1955.0089> · **Richards 1956** (Operations Research).
- **How they fit:** the flow = density × speed relationship and shockwave theory underpinning the
  fundamental diagram and the world model's *edge-field* (spatial, intra-segment) representation. **[support]**

---

## 3. Networking heritage — the HotNets-native framing

### Tassiulas & Ephremides 1992 — "Stability properties of constrained queueing systems… maximum throughput in multihop radio networks"
IEEE Trans. Automatic Control 37(12). <https://doi.org/10.1109/9.182479>
- The backpressure / max-throughput scheduling result.
- **How it fits:** anchors "traffic control = network backpressure/routing" for a *networking*
  audience — the intellectual bridge that makes this a HotNets paper, not a transportation paper.
  Our AAAI merge proof-of-concept is backpressure borrowed from here. Method framing. **[support]**

### Local aggregate V2V — the communication substrate this paper should formalize
- **Repo contract:** `specs/observation_schema.md` and `plans/plan_simulations.md` already define
  `cooperation_mode = local_aggregate`. The paper should formalize this as **semantic aggregate
  exchange**: vehicles communicate compact traffic-state summaries, not commands.
- **Allowed state:** local density, mean speed, queue estimate, downstream congestion estimate,
  segment-level target speed, merge pressure, nearby AV count/density/mean speed/lane distribution.
- **Disallowed state:** raw video, identity-stable neighbor tracking, direct V2V action messages,
  AV-to-lane assignments, joint lane-occupation plans, coordinated roadblock formations.
- **How it fits:** this is the HotNets heart of the revised paper. The network problem is not
  high-bandwidth perception sharing; it is designing a low-rate, privacy-preserving, loss-tolerant
  aggregate-state substrate from which each vehicle builds its own ego-centric digital twin.

---

## 4. Advisory / human-in-the-loop / incentives — deployment + community agenda

### Cho, Li, Kim, Wu 2023 — "Temporal Transfer Learning for Traffic Optimization with Coarse-Grained Advisory Autonomy"
<https://arxiv.org/abs/2312.09436>
- **How it fits:** advisory-autonomy precedent; supports the coarse (discrete speed-bin) advisory
  interface and transfer-across-conditions story. Deployment section. **[support]**

### Fridman 2018 — "Human-Centered Autonomous Vehicle Systems: Principles of Effective Shared Autonomy"
<https://arxiv.org/abs/1810.01835>
- **How it fits:** framing for driver-in-the-loop advisory (not full actuation). Deployment section. **[support]**

### Hasan, Chakraborty, Wu, Driggs-Campbell — cooperative/advisory congestion mitigation (flagged in our own `related.tex` review notes)
- "Towards Co-operative Congestion Mitigation": <https://arxiv.org/abs/2302.09140>
- IEEE Xplore item <https://ieeexplore.ieee.org/abstract/document/10422444> *(confirm exact title on IEEE Xplore before citing)*
- **How it fits:** the dashboard-style advisory interface + cooperative-mitigation angle the
  reviewer said to cite. Related work / deployment. **[support] [differentiate]**

### Prabhakar et al. — incentive mechanisms for decongestion (Stanford "societal networks"; flagged in `related.tex`)
- "An Incentive Mechanism for Decongesting the Roads: A Pilot Program in Bangalore" (INSTANT): <https://web.stanford.edu/~balaji/papers/09anincentive.pdf>
- "INSINC: A Platform for Managing Peak Demand in Public Transit": <https://web.stanford.edu/~balaji/papers/13INSINC.pdf>
- Reviewer-flagged DOIs (confirm titles): <https://dl.acm.org/doi/10.1145/2637364.2592014> · <https://ieeexplore.ieee.org/document/6736646> · <https://dl.acm.org/doi/abs/10.1145/2465529.2465766>
- **How it fits:** the **adoption/incentive** thread — how a voluntary advisory system actually gets
  people to comply (insurance discounts, rideshare partnerships, nudges). Feeds the community agenda
  (open questions) and the deployment section. **[support]**

---

## Differentiation cheat-sheet (the wall, and our answer)

| Prior work | What it is | Our distinguishing axis |
|---|---|---|
| AAAI SRC (ours) | centralized RL + cloud broadcast + per-vehicle internet connectivity | remove cloud/roadside coordinator → **vehicle-native edge + local aggregate V2V** |
| Stern 2018 | hand-tuned single-AV, instrumented test track | **learned, commodity-edge, open networks, advisory-to-human** |
| Vinitsky / Flow / Kreidieh | RL mixed-autonomy in research simulators | **edge *deployment* vision with aggregate-state execution, not a simulator framework** |
| CIRCLES / MegaVanderTest | centralized Speed Planner + purpose-built AVs | **no cloud/roadside planner; aggregate-state V2V; retrofit commodity hardware** |
| Cooperative-vehicular MARL | dense P2P V2V policies / learned coordination | **bounded aggregate-state exchange, no command exchange or joint lane plans** |

## Status of files in this folder

- **Present:** `00832-BhardwajA.pdf` (AAAI SRC), `1-s2.0-S2352728522000148-main.pdf` (sudden jams 2023),
  `claude_conversation.pdf` (internal review notes — not a citation).
- **To fetch (links above):** Stern 2018, Vinitsky 2018, Flow, Kreidieh 2018, CIRCLES (3), Sugiyama 2008,
  Tassiulas–Ephremides, Cho 2023, Fridman 2018, Hasan/Wu advisory, Prabhakar incentives.
- **Already in `vision_paper/refs.bib`:** `bhardwaj2023understanding`, `Yan_2023` (plus the perception/
  edge-AI stack from the SenSys draft).
- **Deliberately excluded** (per project steer — non-ML-centric vision): pure RL-method papers
  (MAPPO / IPPO / PPO / DQN).
