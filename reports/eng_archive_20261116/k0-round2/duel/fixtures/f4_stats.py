"""Acceptance 4: clustered statistics self-check on constructed rows
with known per-game values. Reference t interval computed independently
via scipy on the game means; reference cluster bootstrap implemented
independently from the spec (resample games with replacement, average
per-game means, percentile). Run from the stage dir."""
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402

GAMES = [("g1", [0.10, 0.20]), ("g2", [-0.05]),
         ("g3", [0.30, 0.10, 0.20]), ("g4", [0.0, 0.05])]
rows = [{"game": g, "du": v} for g, vs in GAMES for v in vs]
gm = np.array([np.mean(vs) for _, vs in GAMES])

# independent t-interval reference (over game means, df = G-1)
mean = float(gm.mean())
sd = float(gm.std(ddof=1))
crit = float(sps.t.ppf(0.975, len(gm) - 1))
half = crit * sd / np.sqrt(len(gm))

cs = D.cluster_stats(rows, "du")
assert cs["n_rows"] == 8 and cs["n_games"] == 4, cs
assert cs["game_mean"] == mean, (cs["game_mean"], mean)
assert cs["game_ci95_t"] == [mean - half, mean + half], cs["game_ci95_t"]
assert cs["t_exact"] is True, cs
assert cs["sigma_game"] == sd, (cs["sigma_game"], sd)

# independent cluster-bootstrap reference (same registered seed)
rng = np.random.default_rng(995400)
idx = rng.integers(0, len(gm), size=(10000, len(gm)))
means = gm[idx].mean(axis=1)
ref = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
got = D.cluster_bootstrap_ci(rows, 995400, "du")
assert got == ref, (got, ref)
print(f"F4 PASS: game_mean={mean} t_ci95={cs['game_ci95_t']} "
      f"boot_ci95={got} (bitwise vs independent references)")
