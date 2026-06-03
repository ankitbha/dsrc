# DSRC Project Task List

Approach these tasks sequentially. Each item is intentionally broad enough to be planned and implemented as its own focused work unit.

1. **Confirm simulator integration**
   Make sure the installed `highway_env` package runs correctly from this repo, including reset/step/render, multi-agent actions, and basic straight/merge environments.

2. **Define the project interface**
   Standardize the environment wrapper API, v2 action format, observation format, config loading, and controller interface before building many experiments. The v2 action contract is `desired_speed_bin`, `desired_headway_bin`, `lane_preference`, and `merge_mode`; lane preferences are conservative and must pass a safety/etiquette layer. Include active vehicle lifecycle semantics, aggregate-only cooperation, the CTDE integrated safety contract, and forward-weighted safety metrics.

3. **Build the topology ladder**
   Implement or wrap the four target road setups in order: ring road, straight highway, merge/bottleneck, then inverted tree.

4. **Add traffic demand generation**
   Create reproducible spawning, inflow/outflow handling, AV penetration rates, branch splits, low/medium/high/burst demand profiles, and explicit active/inactive vehicle lifecycle handling after vehicles leave the topology.

5. **Implement vehicle roles and behavior profiles**
   Separate AVs from regular vehicles, then add cautious, normal, aggressive, and heterogeneous human-driver settings.

6. **Create metrics and logging**
   Log throughput, travel time, speed variance, queues, hard braking, lane use, collisions, segment metrics, fairness, follower disruption, lane-change dwell, all-lane low-speed occupancy, and rolling-roadblock score before serious control work.

7. **Implement local sensing for AVs**
   Give each AV local observations first, including local aggregate AV cooperation fields with neutral fallback when no AVs are nearby. Then add noisy sensing, limited range, latency, and density/speed estimates.

8. **Add the safety, etiquette, and physical-control layer**
   Convert speed/headway bins and conservative lane preferences into bounded, safe vehicle behavior with acceleration limits, headway control, lane-change dwell, safe front/rear gap checks, follower-disruption blocks, low-speed-uncongested blocks, and emergency overrides. Integrate safety-aware masking/penalties with CTDE controllers, and keep a separate external safety path for baselines/RVs/human drivers where needed.

9. **Build the baseline ladder**
   Add human-only, random AVs, selfish AVs, density lookup, local dynamic speed advisory, local speed harmonization, local backpressure-inspired speed metering, and cooperative smoothing.

10. **Run topology-by-topology validation**
    For each topology, verify spawning, routing, exits, detector counts, metrics, baseline behavior, and logical directional sanity checks such as selfish AV early speed, density/smoothing behavior under high demand, merge gap creation, branch fairness, and low rolling-roadblock scores before moving to the next topology.

11. **Train the CTDE policy**
    Add shared PPO, IPPO, and MAPPO RL for local AV policies once environments and baselines are stable. The actor should learn smooth speed/headway targets with conservative lane preferences; MAPPO may use a centralized critic during training, but learned-policy execution must remain local-only and must not learn obstruction, lane hogging, or coordinated roadblock behavior.

12. **Evaluate and compare experiments**
    Sweep topology, demand level, AV penetration, human-driver model, sensing noise, and baselines.

13. **Analyze and visualize results**
    Generate plots for time series, speed heatmaps, queues, throughput, branch fairness, merge delay, and spillback.

14. **Package reproducible experiments**
    Clean up configs, scripts, output structure, seeds, and README instructions so experiments can be rerun consistently.
