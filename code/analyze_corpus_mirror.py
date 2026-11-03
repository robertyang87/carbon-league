#!/usr/bin/env python3
"""analyze_corpus_mirror -- mirror self-play observations from
corpus-batch-1 (base_v53 vs base_v53, 40 games), for the data card
and as Plan-A archetype material.

Per game, from the trace frames only:
  winner / margin (player0 - player1), per player:
    stranded   final cargo still on surviving workers (frame 300)
    live_trees trees owned at frame 300 (unliquidated assets)
    deaths     worker ids that disappear between consecutive frames
    last30     cash gained over steps 270..300 (frame 270 -> 300)
    peak_cargo max over frames of total carried cargo
Loser profile: means of loser vs winner on each metric, plus the
share of decisive games in which the loser is worse on that metric
(for stranded/live_trees/deaths higher = worse; for last30 lower =
worse).  The metric with the highest separation share is the most
common collapse ledger item.

Run: python3 analyze_corpus_mirror.py
"""

import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def game_metrics(trace):
    frames = trace["frames"]
    out = {}
    for p in (0, 1):
        pre = f"player-{p}-"
        deaths = 0
        peak_cargo = 0.0
        for t in range(300):
            now = {w for w in frames[t]["workers"]
                   if w.startswith(pre)}
            nxt = {w for w in frames[t + 1]["workers"]
                   if w.startswith(pre)}
            deaths += len(now - nxt)
            cargo = sum(v[1] for w, v in frames[t]["workers"].items()
                        if w.startswith(pre))
            peak_cargo = max(peak_cargo, cargo)
        last = frames[300]
        stranded = sum(v[1] for w, v in last["workers"].items()
                       if w.startswith(pre))
        live_trees = sum(1 for v in last["trees"].values()
                         if v[1] == p)
        out[p] = {
            "cash": last["cash"][p],
            "stranded": stranded,
            "live_trees": live_trees,
            "deaths": deaths,
            "last30": last["cash"][p] - frames[270]["cash"][p],
            "peak_cargo": peak_cargo,
        }
    return out


def main():
    manifest = json.loads(
        (HERE / "corpus_batch1_manifest.json").read_text())
    games = [g for g in manifest if g["tag"] == "cb1_mirror"]
    rows = []
    for g in games:
        with gzip.open(HERE / "traces" / g["file"], "rt") as fh:
            trace = json.load(fh)
        m = game_metrics(trace)
        rows.append({"seed": g["seed"], "side": g["side"], "m": m})

    p0w = sum(1 for r in rows if r["m"][0]["cash"] > r["m"][1]["cash"])
    p1w = sum(1 for r in rows if r["m"][1]["cash"] > r["m"][0]["cash"])
    draws = len(rows) - p0w - p1w
    margins = np.array([r["m"][0]["cash"] - r["m"][1]["cash"]
                        for r in rows])
    same_seed = {}
    for r in rows:
        same_seed.setdefault(r["seed"], []).append(
            r["m"][0]["cash"] - r["m"][1]["cash"])
    identical_pairs = sum(1 for v in same_seed.values()
                          if len(v) == 2 and v[0] == v[1])

    metrics = ["stranded", "live_trees", "deaths", "last30",
               "peak_cargo"]
    worse_high = {"stranded", "live_trees", "deaths"}
    profile = {}
    decisive = [r for r in rows
                if r["m"][0]["cash"] != r["m"][1]["cash"]]
    for k in metrics:
        lw, ww, sep = [], [], 0
        for r in decisive:
            loser = 0 if r["m"][0]["cash"] < r["m"][1]["cash"] else 1
            lv, wv = r["m"][loser][k], r["m"][1 - loser][k]
            lw.append(lv)
            ww.append(wv)
            worse = lv > wv if k in worse_high else lv < wv
            sep += 1 if worse else 0
        profile[k] = {
            "loser_mean": float(np.mean(lw)),
            "winner_mean": float(np.mean(ww)),
            "loser_worse_share": sep / len(decisive) if decisive else 0.0,
        }

    result = {
        "games": len(rows),
        "player0_wins": p0w, "player1_wins": p1w, "draws": draws,
        "player0_winrate_decisive": (
            p0w / (p0w + p1w) if (p0w + p1w) else None),
        "margin_p0_minus_p1_mean": float(margins.mean()),
        "margin_abs_mean": float(np.abs(margins).mean()),
        "margin_std": float(margins.std()),
        "same_seed_side_pairs_identical_margin": identical_pairs,
        "loser_profile": profile,
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    (HERE / "corpus_batch1_mirror_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
