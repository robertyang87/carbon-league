#!/usr/bin/env python3
"""extract_bc_v2 -- expanded Claude-family BC extraction with the
corrected (obs_t -> act_{t+1}) pairing, over the UNION of:

  * replays/index.json          (public snapshot, list format, has order)
  * replays/accumulated/index.json (harvested family games, dict format,
                                    NO order field)

Deduped by match id (snapshot entry wins -- it carries the true order).
For accumulated games the missing `order` is derived from the replay
file itself: replay["rewards"][p] is replay-player p's final score and
the index scores are per slot, so order is the assignment minimizing
|rewards[p] - scores[order[p]]| (verified against snapshot games with
known order -- see bc_shadow_v2_report.md).

Everything else (canonical arrays, slot sorting, corrected pairing)
reuses extract_bc_fixed.extract_player_samples_fixed verbatim. The
misaligned extract_bc.py pairing is NOT used anywhere.

Output: bc_data/claude_family_v2.npz + bc_data/claude_family_v2_manifest.json

Run: python3 extract_bc_v2.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_bc as X  # noqa: E402
from extract_bc_fixed import extract_player_samples_fixed  # noqa: E402

ROOT = X.ROOT
REPLAYS = ROOT / "replays"


def derive_order(replay, scores):
    """Return (order, gap): order[p] = slot of replay player p, chosen
    by matching the replay's per-player final score to the per-slot
    index scores. Primary source: replay["rewards"]; when that is
    None (unfinished/aborted records), fall back to the final step's
    shared players-block cash (cash IS the score in this game)."""
    rew = replay.get("rewards") or [None, None]
    if rew[0] is None or rew[1] is None:
        players = replay["steps"][-1][0]["observation"]["players"]
        rew = [float(players[0][0]), float(players[1][0])]
    if scores.get("a") is None or scores.get("b") is None:
        return None, None
    d_ab = abs(rew[0] - scores["a"]) + abs(rew[1] - scores["b"])
    d_ba = abs(rew[0] - scores["b"]) + abs(rew[1] - scores["a"])
    if d_ab <= d_ba:
        return ["a", "b"], abs(d_ab - d_ba)
    return ["b", "a"], abs(d_ab - d_ba)


def build_registry():
    """match_id -> meta dict in snapshot-index shape (plus 'src')."""
    reg = {}
    snap = json.loads((REPLAYS / "index.json").read_text(encoding="utf-8"))
    for m in snap:
        reg[m["id"]] = {
            "id": m["id"], "file": m["file"],
            "strategies": m["strategies"], "scores": m["scores"],
            "winner": m.get("winner"), "order": list(m["order"]),
            "src": "snapshot", "order_gap": None,
        }
    acc = json.loads(
        (REPLAYS / "accumulated" / "index.json").read_text(encoding="utf-8"))
    n_acc_new = 0
    for mid, m in acc.items():
        if mid in reg:
            continue
        path = REPLAYS / "accumulated" / m["file"]
        replay = X.load_game(path)
        order, gap = derive_order(replay, m["scores"])
        if order is None:
            print(f"WARN: cannot derive order for accumulated {mid}, "
                  f"skipping")
            continue
        reg[mid] = {
            "id": mid, "file": "accumulated/" + m["file"],
            "strategies": {"a": {"name": m["a"]}, "b": {"name": m["b"]}},
            "scores": m["scores"], "winner": m.get("winner"),
            "order": order, "src": "accumulated", "order_gap": gap,
        }
        n_acc_new += 1
    print(f"registry: {len(reg)} games "
          f"({len(reg) - n_acc_new} snapshot, {n_acc_new} accumulated-only)")
    return reg


def main():
    reg = build_registry()
    # sanity: verify derive_order against snapshot games with known order
    cache = {}
    ok = bad = amb = 0
    for meta in reg.values():
        if meta["src"] != "snapshot":
            continue
        rp = X.load_game(REPLAYS / meta["file"])
        derived, gap = derive_order(rp, meta["scores"])
        if derived is None or gap is None:
            amb += 1
        elif gap < 0.02:
            amb += 1
        elif derived == meta["order"]:
            ok += 1
        else:
            bad += 1
    print(f"order-derivation check on snapshot: ok={ok} bad={bad} "
          f"ambiguous(gap<0.02)={amb}")
    amb_acc = [m["id"] for m in reg.values()
               if m["src"] == "accumulated" and (m["order_gap"] or 0) < 0.02]
    print(f"accumulated games with ambiguous derived order: "
          f"{len(amb_acc)} {amb_acc}")

    claude_games = []  # (meta, player)
    for meta in reg.values():
        for slot in ("a", "b"):
            if X.CLAUDE_RE.search(meta["strategies"][slot]["name"]):
                claude_games.append((meta, X.player_of_slot(meta, slot)))
    games_per_strategy = Counter(
        meta["strategies"][X.slot_of_player(meta, p)]["name"]
        for meta, p in claude_games)
    print("family sides per strategy (games):")
    for name, n in games_per_strategy.most_common():
        print(f"  {n:3d}  {name}")
    uniq_games = {meta["id"] for meta, _ in claude_games}
    print(f"family sides: {len(claude_games)} over "
          f"{len(uniq_games)} unique games")

    counters = Counter()
    samples = []
    for meta, player in claude_games:
        if meta["file"] not in cache:
            cache[meta["file"]] = X.load_game(REPLAYS / meta["file"])
        samples.extend(extract_player_samples_fixed(
            meta, cache[meta["file"]], player, counters))
    out = X.OUT_DIR / "claude_family_v2.npz"
    np.savez_compressed(out, **X.pack(samples))
    print(f"claude_family_v2: {len(samples)} samples; "
          f"counters={dict(counters)}")

    pts = Counter(s["strategy"] for s in samples)
    games_seen = defaultdict(set)
    for s in samples:
        games_seen[s["strategy"]].add(s["file"])
    print("decision points per strategy:")
    manifest = {"total_samples": len(samples), "strategies": {}}
    for name, n in pts.most_common():
        print(f"  {n:6d} pts  {len(games_seen[name]):3d} games  {name}")
        manifest["strategies"][name] = {
            "decision_points": int(n),
            "games": len(games_seen[name]),
        }
    manifest["games"] = [
        {"id": m["id"], "file": m["file"], "src": m["src"],
         "a": m["strategies"]["a"]["name"],
         "b": m["strategies"]["b"]["name"],
         "order": m["order"], "order_gap": m["order_gap"]}
        for m in reg.values()
        if any(X.CLAUDE_RE.search(m["strategies"][s]["name"])
               for s in ("a", "b"))]
    with open(X.OUT_DIR / "claude_family_v2_manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("wrote", out, "+ manifest")


if __name__ == "__main__":
    main()
