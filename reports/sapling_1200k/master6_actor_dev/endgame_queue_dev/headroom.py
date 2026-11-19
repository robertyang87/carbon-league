"""Score every allowed final joint move against the already-submitted original opponent action."""
from collections import Counter
from itertools import product
from pathlib import Path
import statistics

from study import HERE, REPLAY, FastCarbon, protocol, read, sha, twin_step, write
from policy import _operator_grid, _traffic, dist, home_dist


def _edits(parent, workers, combo):
    return {w: c for w, c in zip(workers, combo) if c != (parent.get(w) or "NONE")}


def _apply(parent, overrides):
    joint = dict(parent)
    for uid, cmd in overrides.items():
        if cmd == "NONE":
            joint.pop(uid, None)
        else:
            joint[uid] = cmd
    return joint


def _inventory(obs, horizon, cfg):
    totals = []
    for _, bases, workers, _ in obs["players"]:
        returned = []
        for uid, (cell, cargo, kind) in workers.items():
            d = home_dist(cell, bases.values(), cfg["size"])
            if kind == "COLLECTOR" and cargo >= 1 and 0 < d <= horizon:
                returned.append({"uid":uid, "cell":cell, "cargo":cargo, "distance":d})
        totals.append(returned)
    return totals


def inspect(game):
    t, seat, cfg = 298, game["seat"], game["config"]
    row = {k:game[k] for k in ("seed","seat","split","batch","steps","terrain")}
    row["inventory"] = {t: _inventory(both[seat], 299-int(t), cfg)
                        for t, both in game["history"].items() if int(t) in (297,298)}
    if "298" not in game["states"]:
        row["missing"] = True
        return row
    state = game["states"]["298"]
    obs = game["history"]["298"][seat]
    own = game["orders"]["298"][seat]
    enemy = game["orders"]["298"][1-seat]
    opts, meta = _operator_grid(obs, own, cfg, 1)
    parent_conflicts, parent_banks = _traffic(obs, own, cfg)
    sim = FastCarbon(cfg)
    trials = []
    for combo in product(*opts.values()):
        edits = _edits(own, opts, combo)
        candidate = _apply(own, edits)
        conflicts, banks = _traffic(obs, candidate, cfg)
        if not conflicts.issubset(parent_conflicts):
            continue
        joint = [None, None]
        joint[seat], joint[1-seat] = candidate, enemy
        sim.set_state(state)
        result = twin_step(sim, joint)
        trials.append({
            "edits": edits, "shared_bank": banks, "cash": result["cash"], "official": result["score"],
            "margin": round(result["score"][seat] - result["score"][1-seat], 2),
            "flow": result["flow"], "deaths": result["deaths"]})
    parent = next(c for c in trials if not c["edits"])
    if parent["official"] != game["original"]["score"]:
        raise ValueError("original last move no longer reproduces")
    best = max(trials, key=lambda c:(c["margin"],-len(c["edits"])))
    row.update(missing=False,operators=meta,base_shared=parent_banks,trials=trials,best=best["edits"],
               base_margin=parent["margin"],best_delta=round(best["margin"]-parent["margin"],2))
    return row


def summarize(rows):
    data = [r for r in rows if not r.get("missing", False)]
    by_map = {}
    for row in data:
        by_map.setdefault(row["terrain"],[]).append(row["best_delta"])
    return {
        "games": len(rows), "available_last_decisions":len(data), "maps":len(by_map),
        "legal_offers_including_parent":sum(len(r["trials"]) for r in data),
        "games_with_changed_offer":sum(len(r["trials"])>1 for r in data),
        "games_with_hindsight_gain":sum(r["best_delta"]>0 for r in data),
        "largest_hindsight_gain":max((r["best_delta"] for r in data),default=None),
        "average_best_gain_map_equal":statistics.mean(statistics.mean(v) for v in by_map.values()) if by_map else None,
        "games_with_hindsight_win_but_parent_loss":sum(r["best_delta"]+r["base_margin"]>0 and r["base_margin"]<0 for r in data),
        "games_with_any_synchronized_offer":sum(any(sq["inbound_loaded"]>=2 for q in r["trials"] for sq in q["shared_bank"]) for r in data),
        "root_principal": {
            t: {who: {
                "positive": len([g for g in rows if g["inventory"].get(t,[[],[]])[g["seat"]^other]]),
                "two_loaded": sum(len(g["inventory"].get(t,[[],[]])[g["seat"]^other]) >= 2 for g in rows),
                "sum":sum(w["cargo"] for g in rows for w in g["inventory"].get(t,[[],[]])[g["seat"]^other])}
                for who,other in (("sapling",0),("examiner",1))} for t in ("297","298")}
    }


if __name__ == "__main__":
    protocol()
    game_list=read(REPLAY)["games"]
    rows=[inspect(g) for g in game_list]
    directory=Path(__file__).resolve().parent
    write(directory/"last_turn_ceiling.json.gz", {"roots": rows, "sealed_replay_sha256":sha(REPLAY),
          "policy_sha256":sha(HERE/"policy.py"),"protocol_sha256":sha(HERE/"protocol.json")})
    brief={split:summarize([r for r in rows if r["split"] == split]) for split in ("diagnostic","evaluation")}
    write(directory/"last_turn_ceiling_summary.json",brief)
    print(brief)
