"""Score physical location conditional on actual continuing heavy, undelivered status.

Reuses frozen held-out forecasts. This is a post-gate diagnostic with no tuning and no
counterfactual opponent response. Factual command replay only reconstructs public geometry
for the original reference kernel; no agent or search is called.
"""

from collections import Counter, defaultdict, deque
import math
from statistics import mean
import gate
from evaluate import reference
from train import HERE, PROTOCOL, check, extract, forecast

ARMS = ("registered_geometry", "learned")
METRICS = ("brier", "true_cell_logloss")


def from_full(brier, truth_mass, other_mass):
    if not (0 <= other_mass < 1 and 0 < truth_mass <= 1 - other_mass + 1e-9):
        raise ValueError("invalid conditional probability")
    survive = 1 - other_mass
    squared_without_other = brier + 2 * truth_mass - 1 - other_mass**2
    return 1 - 2 * truth_mass / survive + squared_without_other / survive**2, -math.log(truth_mass / survive)


def direct(probs, y):
    den = sum(p for c, p in probs.items() if c != -1)
    sub = {c: p / den for c, p in probs.items() if c != -1}
    return sum((p - int(c == y)) ** 2 for c, p in sub.items()), -math.log(sub[y])


def verify_close(a, b):
    if len(a) != len(b) or any(not math.isclose(x, y, abs_tol=3e-12) for x, y in zip(a, b)):
        raise ValueError("conditional Brier disagrees with physical-only probability vector")


def game(g, weights):
    raw_path = extract.ROOT / g["source"]
    check(extract.sha(raw_path) == g["source_sha256"], "different original game")
    raw = extract.load(raw_path)
    assert g["forecast_schema"] == [
        "t",
        "uid",
        "h",
        "actual",
        "top_class",
        "p_actual",
        "p_other",
        "brier_fit",
        "brier_base",
    ]
    by_t = defaultdict(list)
    for row in g["forecasts"]:
        by_t[row[0]].append(row)
    cfg = raw["game_configuration"]
    s = g["seat"]
    sim = extract.FastCarbon(cfg).reset(seed=g["seed"])
    counts = {h: Counter() for h in (3, 5)}
    verified = set()
    hist = deque(maxlen=10)
    for t, joint in enumerate(raw["actions"]):
        obs = sim.obs(s)
        targets = obs["players"][1 - s]
        bases = targets[1].values()
        for _, uid, h, y, _, p_y, p_o, fit, base in by_t[t]:
            v = counts[h]
            v["total_forecasts"] += 1
            if y == -1:
                v["other"] += 1
                continue
            v["observed_spatial"] += 1
            origin = targets[2][uid][0]
            locations = forecast._BALLS[h][origin]
            if y not in locations:
                raise ValueError("held-out target outside reachable set")
            geometric = reference(locations, origin, bases)
            obs_brier = sum(p * p for p in geometric.values()) - 2 * geometric[y] + 1
            verify_close([base], [obs_brier])
            old = from_full(base, geometric[y], geometric[-1])
            new = from_full(fit, p_y, p_o)
            verify_close(old, direct(geometric, y))
            if h not in verified:
                ph = forecast.predict(weights, obs, list(hist), uid, h, cfg)
                verify_close([p_y, p_o], [ph[y], ph[-1]])
                verify_close(new, direct(ph, y))
                verified.add(h)
            for arm, scores in zip(ARMS, (old, new)):
                for metric, z in zip(METRICS, scores):
                    v[arm + "_" + metric] += z
        hist.append(obs)
        sim.step(*joint)
    if verified != {3, 5} or list(sim.cash) != g["score"]:
        raise ValueError("invalid score or insufficient spot checks")
    return counts


def means(c):
    n = c["observed_spatial"]
    return {
        "all_forecasts": c["total_forecasts"],
        "actual_OTHER": c["other"],
        "actual_spatial": n,
        **{a: {m: c[a + "_" + m] / n for m in METRICS} for a in ARMS},
        "learned_minus_geometry": {
            m: (c["learned_" + m] - c["registered_geometry_" + m]) / n for m in METRICS
        },
    }


def equal_maps(rows):
    return {
        "maps": len(rows),
        "all_forecasts": sum(r["all_forecasts"] for r in rows),
        "actual_spatial": sum(r["actual_spatial"] for r in rows),
        **{
            arm: {m: mean(r[arm][m] for r in rows) for m in METRICS}
            for arm in (*ARMS, "learned_minus_geometry")
        },
        "maps_with_lower_learned_error": {
            m: sum(r["learned_minus_geometry"][m] < 0 for r in rows) for m in METRICS
        },
    }


def main():
    basegate = extract.load(HERE / "gate.json")
    weights = extract.load(HERE / "weights.json")
    assert weights["protocol_sha256"] == PROTOCOL and basegate["weights_sha256"] == extract.sha(
        HERE / "weights.json"
    )
    maps = {}
    arms = {}
    batches = {}
    checks = 0
    sources = {}
    for b in (4, 5):
        gids = []
        for g in gate.load_games(b):
            key = f"{g['seed']}:{g['seat']}"
            seed = str(g["seed"])
            assert basegate["covered_artifacts"][key] == g["artifact_sha256"]
            c = game(g, weights)
            checks += 2
            sources[key] = g["artifact_sha256"]
            arms[key] = {str(h): means(c[h]) for h in (3, 5)}
            dst = maps.setdefault(seed, {3: Counter(), 5: Counter()})
            for h in (3, 5):
                dst[h].update(c[h])
            if seed not in gids:
                gids.append(seed)
        batches[str(b)] = {str(h): equal_maps([means(maps[seed][h]) for seed in gids]) for h in (3, 5)}
    whole = {str(h): equal_maps([means(m[h]) for m in maps.values()]) for h in (3, 5)}
    out = dict(
        estimand="Held-out original trajectories where the target still had >=100 cargo and never unloaded by the exact registered horizon. Condition on actual spatial outcome, erase OTHER, and independently renormalize each frozen distribution. Physical cell Brier is squared error summed across all cells. Log loss uses natural logs. Lower is better for both metrics.",
        interpretation="Retrospective conditional forecast quality only, excluding forecasts whose actual class was OTHER. Not an action gain, win result or change to the frozen gate.",
        averaging="Within map pool physical seats and qualifying worker-turns. Give independent maps equal weight across each batch and overall; temporally adjacent roots may overlap.",
        protocol_sha256=PROTOCOL,
        gate_sha256=extract.sha(HERE / "gate.json"),
        weights_sha256=basegate["weights_sha256"],
        forecast_artifacts_sha256=sources,
        conditional_reconstruction="recover sum of squared spatial mass as full Brier + 2*p(true) - 1 - p(OTHER)^2; divide by (1-p(OTHER))^2; subtract 2*p(true)/(1-p(OTHER)) and add 1",
        independent_probability_vector_checks=checks,
        by_map_physical_seat=arms,
        by_map={seed: {str(h): means(m[h]) for h in (3, 5)} for seed, m in maps.items()},
        by_batch_map_equal=batches,
        overall_map_equal=whole,
    )
    extract.write(HERE / "spatial_check.json", out)
    print(whole)


if __name__ == "__main__":
    main()
