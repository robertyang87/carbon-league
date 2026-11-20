"""Secondary comparator: map-held-out official residual mean, matched by seat and exact step."""
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
source = HERE / "out_of_map.json.gz"
rows = json.loads(gzip.decompress(source.read_bytes()))
group = defaultdict(dict)
for seed, seat, t, cash, final, _ in rows:
    by_map = group[seat, t]
    if seed in by_map:
        raise ValueError("Duplicated map/seat/step")
    by_map[seed] = final - cash
if len(group) != 100 or any(len(pool) != 10 for pool in group.values()):
    raise ValueError("Expected ten distinct maps for each matched seat and time")
stats = defaultdict(list)
for seed, seat, t, cash, final, fitted in rows:
    matched = group[seat, t]
    correction = mean(value for key, value in matched.items() if key != seed)
    baseline = cash + correction
    stats[seed].append((abs(fitted-final), abs(baseline-final), abs(cash-final)))
names = ("fitted_mae", "peer_mae", "cash_mae")
folds = [{"seed": seed, **dict(zip(names, map(mean, zip(*values))))} for seed,values in sorted(stats.items())]
report = {
    "note": "Secondary untuned historical map-held-out diagnostic. Nine other physical maps, the identical seat and the identical step supply the mean remaining official margin added to visible cash. Both seats and the whole trajectory of the held terrain are excluded.",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "by_map": folds,
    "map_means": {k: mean(row[k] for row in folds) for k in names},
    "fitted_better_than_peer_maps": sum(row["fitted_mae"] < row["peer_mae"] for row in folds),
}
out = HERE / "seat_stage_baseline.json"
serialized = json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n"
if out.exists() and out.read_text() != serialized:
    raise ValueError("Conflicting secondary baseline")
out.write_text(serialized)
print(report["map_means"], report["fitted_better_than_peer_maps"])
