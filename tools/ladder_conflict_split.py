#!/usr/bin/env python3
"""ladder_conflict_split -- split a parsed ladder corpus by opponent "wall" and by conflict intensity.

Input: a parsed-games JSON list as produced by the scout parse2 toolchain
(analysis/echo4/echo4_parsed243_v2.json, analysis/echo5/echo5_parsed29_annotated.json):
each game has opp, won, margin, deaths, recruits, tree_steps, cash50/100/200/270, cash_final
and the same metrics for the opponent under opp_metrics.

Prints (JSON lines):
  wall     opponents never beaten (>= --min-games games, 0 wins): games, share of all losses
  groups   our/opponent means of the key metrics for games vs the wall and vs everyone else
  deaths   win rate and margin by terciles of our own worker deaths
  corr     Pearson correlations of a few per-game metrics with the margin
Descriptive only: opponent identity and conflict intensity are confounded in ladder data.
"""
import argparse, collections, json, statistics as st

import numpy as np

KEYS = ["cash50", "cash100", "cash200", "cash270", "cash_final", "tree_steps", "plants", "recruits", "deaths",
        "peak_cargo", "end_cargo", "end_trees", "first_plant"]


def means(rows):
    out = {}
    for k in KEYS:
        try:
            out[k] = [round(st.mean(r[k] for r in rows), 1), round(st.mean(r["opp_metrics"][k] for r in rows), 1)]
        except (KeyError, TypeError, st.StatisticsError):
            out[k] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed")
    ap.add_argument("--min-games", type=int, default=3)
    a = ap.parse_args()
    g = json.load(open(a.parsed))
    by = collections.defaultdict(list)
    for r in g:
        by[r["opp"]].append(r)
    wall = sorted(o for o, v in by.items() if len(v) >= a.min_games and not any(r["won"] for r in v))
    wg = [r for o in wall for r in by[o]]
    rest = [r for r in g if r["opp"] not in wall]
    n_loss = sum(not r["won"] for r in g)
    print(json.dumps({"games": len(g), "wins": len(g) - n_loss, "losses": n_loss, "wall_opponents": wall,
                      "wall_games": len(wg), "wall_share_of_losses": round(len(wg) / max(1, n_loss), 3),
                      "wall_mean_margin": round(st.mean(r["margin"] for r in wg), 1) if wg else None,
                      "rest_games": len(rest),
                      "rest_winrate": round(sum(r["won"] for r in rest) / max(1, len(rest)), 3)}, ensure_ascii=False))
    for label, rows in (("vs_wall", wg), ("vs_rest", rest)):
        if rows:
            print(json.dumps({"group": label, "n": len(rows), "ours_vs_opp": means(rows)}))
    d = np.array([r["deaths"] for r in g], float)
    q1, q2 = np.quantile(d, [1 / 3, 2 / 3])
    for label, sel in (("low", d < q1), ("mid", (d >= q1) & (d < q2)), ("high", d >= q2)):
        rows = [r for r, s in zip(g, sel) if s]
        if rows:
            print(json.dumps({"deaths_tercile": label, "n": len(rows),
                              "our_deaths": round(st.mean(r["deaths"] for r in rows), 1),
                              "opp_deaths": round(st.mean(r["opp_metrics"]["deaths"] for r in rows), 1),
                              "winrate": round(sum(r["won"] for r in rows) / len(rows), 3),
                              "mean_margin": round(st.mean(r["margin"] for r in rows), 1),
                              "our_cash50": round(st.mean(r["cash50"] for r in rows), 1),
                              "our_tree_steps": round(st.mean(r["tree_steps"] for r in rows), 1)}))
    m = np.array([r["margin"] for r in g], float)
    cols = {"our_deaths": d, "opp_deaths": np.array([r["opp_metrics"]["deaths"] for r in g], float),
            "our_cash50": np.array([r["cash50"] for r in g], float),
            "our_tree_steps": np.array([r["tree_steps"] for r in g], float),
            "opp_tree_steps": np.array([r["opp_metrics"]["tree_steps"] for r in g], float)}
    print(json.dumps({"corr_with_margin": {k: round(float(np.corrcoef(v, m)[0, 1]), 3) for k, v in cols.items()}}))


if __name__ == "__main__":
    main()
