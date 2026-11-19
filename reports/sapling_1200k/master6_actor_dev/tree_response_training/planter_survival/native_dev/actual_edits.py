"""Replay changed orders once after development, measuring one-step worker survival and foregone purchases."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import extract


def game(p):
    row = p["candidate"]
    s, seat = p["seed"], p["seat"]
    tape = HERE / "candidate" / f"raw_{s}_{seat}.json.gz"
    if extract.sha(tape) != row["transcript_sha256"]:
        raise ValueError("candidate tape changed")
    raw = extract.load(tape)
    edits = {r["step"]: r for r in row["attribution"]["decisions"]}
    if len(edits) != row["attribution"]["actual_edits"]:
        raise ValueError("multiple changes in a turn")
    cfg = raw["game_configuration"]
    sim = extract.FastCarbon(cfg).reset(seed=s)
    alt = extract.FastCarbon(cfg)
    out = []
    for t, joint in enumerate(raw["actions"]):
        x = edits.get(t)
        if not x:
            sim.step(*joint)
            continue
        st, snapshot = sim.full_observation(), sim.get_state()
        uid = x["uid"]
        origin = st["players"][seat][2][uid][0]
        if x["old"] != "NONE" or joint[seat].get(uid) != x["move"]:
            raise ValueError("wrong factual change")
        alt.set_state(snapshot)
        held = [a.copy() for a in joint]
        del held[seat][uid]
        v1 = extract.step(alt, st, cfg, held)
        factual = extract.step(sim, st, cfg, joint)
        a, kind, cost = extract.label(st, v1, seat, uid, origin, "NONE", cfg["treeLifespan"])
        m, mkind, mcost = extract.label(
            st, factual, seat, uid, extract.destination(origin, x["move"]), x["move"], cfg["treeLifespan"]
        )
        if mcost or mkind:
            raise ValueError("moving worker booked tree cost")
        out.append(
            dict(
                t=t,
                uid=uid,
                cell=origin,
                move=x["move"],
                p_stay=x["p_stay"],
                p_move=x["p_move"],
                v1_hold_dies=a,
                actual_move_dies=m,
                foregone_kind=kind,
                foregone_price=cost,
            )
        )
    if list(sim.cash) != raw["last_post_cash"] or len(out) != len(edits):
        raise ValueError("factual terminal cash or edit count differs")
    return dict(
        seed=s,
        seat=seat,
        raw_sha256=extract.sha(tape),
        changed=len(out),
        saved=sum(x["v1_hold_dies"] and not x["actual_move_dies"] for x in out),
        harmed=sum(not x["v1_hold_dies"] and x["actual_move_dies"] for x in out),
        foregone_purchases=sum(x["foregone_price"] > 0 for x in out),
        foregone_debit=sum(x["foregone_price"] for x in out),
        orders=out,
    )


def main():
    path = HERE / "comparison.json"
    results = json.loads(path.read_bytes())
    rows = [game(p) for p in results["pairs"]]
    report = dict(
        schema="posthoc-actual-veto-one-step/1",
        comparison_sha256=extract.sha(path),
        methodology="At each actual changed candidate state, restore v1 NONE for the focal planter only. Preserve the actually submitted simultaneous commands. Official Board and FastCarbon must agree independently. This does not estimate terminal utility under folded-back actions.",
        games=rows,
        totals={
            k: sum(x[k] for x in rows)
            for k in ("changed", "saved", "harmed", "foregone_purchases", "foregone_debit")
        },
    )
    dest = HERE / "actual_edits.json"
    if dest.exists() and extract.load(dest) != report:
        raise ValueError("existing posthoc report differs")
    extract.write(dest, report)
    print(report["totals"])


if __name__ == "__main__":
    main()
