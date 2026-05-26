# DSRC Project Task List

Approach these tasks sequentially. Each item is intentionally broad enough to be planned and implemented as its own focused work unit.

1. **Confirm simulator integration**
   Make sure the installed `highway_env` package runs correctly from this repo, including reset/step/render, multi-agent actions, and basic straight/merge environments.

2. **Define the project interface**
   Standardize the environment wrapper API, action format, observation format, config loading, and controller interface before building many experiments.

3. **Build the topology ladder**
   Implement or wrap the four target road setups in order: ring road, straight highway, merge/bottleneck, then inverted tree.

4. **Add traffic demand generation**
   Create reproducible spawning, inflow/outflow handling, AV penetration rates, branch splits, and low/medium/high/burst demand profiles.

5. **Implement vehicle roles and behavior profiles**
   Separate AVs from regular vehicles, then add cautious, normal, aggressive, and heterogeneous human-driver settings.

6. **Create metrics and logging**
   Log throughput, travel time, speed variance, queues, hard braking, lane use, collisions, segment metrics, and fairness before serious control work.

7. **Implement local sensing for AVs**
   Give each AV local observations first, then add noisy sensing, limited range, latency, and density/speed estimates.

8. **Add the safety/control layer**
   Convert desired-speed and desired-lane commands into bounded, safe vehicle behavior with lane-change checks and emergency overrides.

9. **Build the baseline ladder**
   Add human-only, random AVs, selfish AVs, density lookup, dynamic speed limits, AV-mediated speed limits, backpressure, and cooperative smoothing.

10. **Run topology-by-topology validation**
    For each topology, verify spawning, routing, exits, detector counts, metrics, and baseline behavior before moving to the next topology.

11. **Train the CTDE policy**
    Add centralized-training/decentralized-execution RL for local AV policies once environments and baselines are stable.

12. **Evaluate and compare experiments**
    Sweep topology, demand level, AV penetration, human-driver model, sensing noise, and baselines.

13. **Analyze and visualize results**
    Generate plots for time series, speed heatmaps, queues, throughput, branch fairness, merge delay, and spillback.

14. **Package reproducible experiments**
    Clean up configs, scripts, output structure, seeds, and README instructions so experiments can be rerun consistently.
