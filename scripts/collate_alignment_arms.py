"""One table from the arm logs, so the write-up is not hand-assembled.

Each arm answers a different question and they are only interpretable together:

  vehicle / shared   the configuration under test -- a team reward that does not
                     respond to an individual action IS the credit-assignment problem
  segment / shared   does making the agent the ROAD help, holding the reward global
  segment / own      does matching the REWARD's level to the agent's level help
  vehicle / local    the same question one level down

Only the two `own`/`local` rows have a reward that differs between agents, and they
are the only ones that separate "the reward does not respond to individual actions"
from "nothing responds to individual actions".
"""
import re, statistics
from pathlib import Path

S = Path("/private/tmp/claude-502/-Users-ankit-nash-Desktop-ankit-summer-2026/"
         "74a3b6a3-b76c-4322-9224-4a3f73d81187/scratchpad")

ARMS = [
    ("vehicle / shared, 1 s", "align_sumo.log", None),
    ("vehicle / shared, 60 s", "align_src.log", None),
    ("segment / shared", "segagent3.log", None),
    ("segment / OWN", "segown2.log", None),
    # This log labels each row with its arm, so a label is needed to pick rows out.
    # Without one the arm silently reported "no rows" and would have been left out of
    # the final table -- an omission that looks exactly like a measurement not run.
    ("vehicle / shared (paired)", "localalign.log", "team only"),
    ("vehicle / LOCAL blend", "localalign.log", "local 0.5"),
]
#: `seed n measured null sd z` -- the layout the alignment and segment scripts print.
row = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-+][\d.]+)")
#: `<label> <seed> <n> <z> <corr>` -- the layout the paired local-reward script prints.
labelled = re.compile(r"^\s*(.+?)\s+(\d+)\s+(\d+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s*$")

print(f"{'arm':>24} {'seeds':>6} {'mean z':>9} {'se':>6} {'per seed':>26}  reward")
for label, filename, arm_label in ARMS:
    path = S / filename
    if not path.exists() or not path.read_text().strip():
        print(f"{label:>24} {'--':>6} {'(not yet)':>9}")
        continue
    lines = path.read_text().splitlines()
    if arm_label is None:
        zs = [float(m.group(6)) for line in lines if (m := row.match(line))]
        corrs = []
    else:
        hits = [m for line in lines
                if (m := labelled.match(line)) and m.group(1).strip() == arm_label]
        zs = [float(m.group(4)) for m in hits]
        corrs = [float(m.group(5)) for m in hits]
    if not zs:
        print(f"{label:>24} {'--':>6} {'(no rows)':>9}")
        continue
    se = statistics.stdev(zs) / len(zs) ** 0.5 if len(zs) > 1 else float("nan")
    shared = "shared" if "shared" in label else "UNSHARED"
    extra = (f"  corr {statistics.fmean(corrs):+.4f}" if corrs else "")
    print(f"{label:>24} {len(zs):>6} {statistics.fmean(zs):>+9.2f} {se:>6.2f} "
          f"{str([round(z, 2) for z in zs]):>26}  {shared}{extra}")

print("\nz is the measured policy-gradient norm's distance from a null that resamples")
print("the action at the same state and keeps the advantage. Zero means the advantage")
print("says nothing about which action was taken.")
