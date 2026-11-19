"""Recompute hybrid phase and Sapling parent/child comparisons from scored development games."""
from pathlib import Path
from statistics import mean
import json

HERE = Path(__file__).resolve().parent
TIMES = (60, 100, 200, 299)
METRICS = ("cash", "own_positive_delivery_carbon", "alive_trees", "new_trees_planted", "cargo")


def read(path):
    return json.loads((HERE / path).read_text())


def number(x):
    return round(x, 4)


def read_hybrid():
    first = read("it50_v3_exam.json")["games"]
    assert len(first) == 1
    game = first[0]
    rows = {(game["seed"], game["candidate_seat"]): {
        int(t): (values["candidate"], values["opponent"]) for t, values in game["snapshots"].items()
    }}
    originals = ["it50_v3_seat1_repeat.json"]
    originals += [f"it50_v3_fresh_games/{seed}_{seat}.json" for seed in (620098121, 620098122) for seat in (0, 1)]
    for name in originals:
        game = read(name)
        seat = game["seat"]
        key = (game["map"], seat)
        assert key not in rows and game["status"] == "complete" and game["actual_transitions"] == 299
        rows[key] = {int(t): (v[seat], v[1-seat]) for t, v in game["snapshots_by_step_and_seat"].items()}
    assert set(rows) == {(seed, seat) for seed in (620098101, 620098121, 620098122) for seat in (0, 1)}
    return rows


def stage(rows, t):
    metrics = {}
    for k in METRICS:
        own = mean(xs[t][0][k] for xs in rows.values())
        other = mean(xs[t][1][k] for xs in rows.values())
        metrics[k] = dict(candidate=number(own), examiner=number(other), difference=number(own-other))
    return metrics


def matched_ppo():
    models = {}
    for model in ("v3", "win_cash"):
        games = read(f"tree_exam_{model}/results.json")["games"]
        models[model] = {(g["seed"], g["candidate_seat"]): g for g in games}
        assert len(games) == len(models[model]) == 4
    assert set(models["v3"]) == set(models["win_cash"])
    out = []
    for seed, seat in sorted(models["v3"]):
        key = (seed, seat)
        a, b = (m[key] for m in models.values())
        diff = {}
        for t in (100, 299):
            left, right = (x["snapshots"][str(t)]["candidate"] for x in (a, b))
            diff[str(t)] = {k: number(right[k] - left[k]) for k in METRICS}
        out.append(dict(map=seed, seat=seat, parent_margin=number(a["margin"]), child_margin=number(b["margin"]),
                        margin_change=number(b["margin"]-a["margin"]), child_minus_parent_snapshots=diff))
    return out


def main():
    rows = read_hybrid()
    stages = {t: stage(rows, t) for t in TIMES}
    midgame = []
    for (seed, seat), data in sorted(rows.items()):
        def gap(t, k):
            a, b = data[t]
            return a[k] - b[k]
        midgame.append(dict(map=seed, seat=seat, change_100_to_200_in_candidate_lead={
            k: number(gap(200,k) - gap(100,k)) for k in METRICS}))
    scores = read("it50_v3_summary.json")["scored_games"]
    assert {(g["map"], g["seat"]) for g in scores} == set(rows) and len(scores) == len(rows)
    for g in scores:
        row = rows[g["map"], g["seat"]][299]
        assert abs(g["margin"] - (row[0]["cash"] - row[1]["cash"])) < 1e-7
    tree_income = [mean(g["tree_income"][i] for g in scores) for i in (0,1)]
    ppo = matched_ppo()
    print(json.dumps(dict(
        scope="Development games only. Summary snapshots and scorer ledgers; hybrid sources store hashes, not a full action trace. No action-level causal identification.",
        hybrid_by_stage=stages,
        hybrid_midgame_by_map_seat=midgame,
        hybrid_midgame_lead_declines=sum(x["change_100_to_200_in_candidate_lead"]["cash"] < 0 for x in midgame),
        hybrid_final_mean_tree_income=[number(x) for x in tree_income],
        hybrid_final_mean_tree_income_difference=number(tree_income[0]-tree_income[1]),
        ppo_same_map_seat=ppo,
        ppo_mean_margin_change=number(mean(g["margin_change"] for g in ppo)),
        ppo_own_snapshot_changes={t: {k: number(mean(g["child_minus_parent_snapshots"][t][k] for g in ppo))
                                      for k in METRICS} for t in ("100", "299")},
    ), indent=2))


if __name__ == "__main__":
    main()
