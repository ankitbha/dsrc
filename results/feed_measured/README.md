# Block A on the measured feed process

Block A of the feed-agreement program (`plans/2026-10-15-feed-agreement-plan.md`), re-run on the
measured afternoon polling session of 2026-10-16 instead of the drive-log placeholder that
`results/feed/A.jsonl` used. Run from an archive of commit `d699ac2` (so the headers record no
commit), checkpoint `results/checkpoints/mainz_here_best.pt`, actuation `max_speed`, seeds 16-30,
2,500 s, 150 episodes each, zero collisions and zero collision teleports in both.

- `afternoon_nj/`: the artefact's default session, `afternoon_20261016_nj` (the session with the
  most versions; `g_min` 28 s, `L_bar` 84.0 s, `J` 55.1 s).
- `afternoon_mainz/`: `--feed-session afternoon_20261016_mainz` (`g_min` 60 s, `L_bar` 89.5 s,
  `J` 55.1 s), the area of the simulated network.

The artefact both replayed is `results/jam/feed_process_afternoon.json` (sha256 prefix `8f5430bf`).

| rule | share of vehicles on a different version, NJ | same, Mainz |
|---|---|---|
| no rule, query every 60 s | 0.394 | 0.251 |
| no rule, query every 10 s | 0.065 | 0.042 |
| version-keyed, `D` from the lemma | 0.000 | 0.000 |
| version-keyed, `D` 20 s shorter | 0.016 | 0.016 |
