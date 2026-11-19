"""Describe actual tree ownership, income and pressure in seven frozen Top Five games."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

EXPECTED = "d132c41d4b93de18f6db8ee7393426fd83a564bc34cf33c789c2a63df2bbaa7f"
MANIFEST = Path(__file__).resolve().parents[2] / "data/sapling_top5_h2h_corpus/later7.json"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def distance(a, b, size):
    dx, dy = abs(a % size - b % size), abs(a // size - b // size)
    return min(dx, size - dx) + min(dy, size - dy)


def score(steps, seat, size):
    births = {}
    seized = lost = 0
    income = 0.
    last = len(steps) - 1
    for t in range(1, last + 1):
        before, after = [steps[k][0]["observation"] for k in (t-1, t)]
        prev = before["players"]
        own, opp = after["players"][seat], after["players"][1-seat]
        old, new = prev[seat][3], own[3]
        for tid, (cell, age) in new.items():
            amount = float(after["trees"][tid][1])
            income += amount
            if tid in prev[1-seat][3]:
                seized += 1
            elif tid not in old:
                if tid in births or age != 1:
                    raise ValueError("invalid tree birth")
                enemy_planters = [w[0] for w in prev[1-seat][2].values() if w[2] == "PLANTER"]
                births[tid] = dict(plant_step=t, cell=cell, cash_before_turn=prev[seat][0],
                                    cash_at_end_of_turn=own[0], income_first_tenure=0.,
                                    nearest_enemy_planter_pre=min((distance(cell,c,size) for c in enemy_planters), default=None))
            if tid in births and "end" not in births[tid]:
                births[tid]["income_first_tenure"] += amount
        for tid in old.keys() - new.keys():
            stolen = tid in opp[3]
            lost += stolen
            if tid in births and "end" not in births[tid]:
                births[tid].update(end=t, reason="seized" if stolen else "expired")
    for tree in births.values():
        if "end" not in tree:
            tree.update(end=last+1, reason="censored")
        tree["observed_first_tenure"] = tree["end"] - tree["plant_step"]
    return dict(trees=births, first_plant=min((s["plant_step"] for s in births.values()), default=None),
                total_tree_income=income, total_times_lost=lost, total_times_seized=seized)


def summarize(records):
    starts = [r["first_plant"] for r in records if r["first_plant"] is not None]
    trees = [v for r in records for v in r["trees"].values()]
    def tier(subset):
        return dict(plants=len(subset),
                    lost_by_ten=sum(t["reason"]=="seized" and t["observed_first_tenure"]<=10 for t in subset),
                    early_censored=sum(t["reason"]=="censored" and t["observed_first_tenure"]<10 for t in subset),
                    first_tenure_income=sum(t["income_first_tenure"] for t in subset))
    return dict(seats=len(records), planted_seats=len(starts), median_first_plant=median(starts) if starts else None,
                planted_by_t100=sum(t<=100 for t in starts), plants=len(trees),
                end_of_plant_turn_below_30=sum(t["cash_at_end_of_turn"]<30 for t in trees),
                censored=sum(t["reason"]=="censored" for t in trees),
                lost_first_tenure=sum(t["reason"]=="seized" for t in trees),
                nearest_enemy_planter_at_most_two=tier([t for t in trees if t["nearest_enemy_planter_pre"] is not None and t["nearest_enemy_planter_pre"]<=2]),
                nearest_enemy_planter_far_or_absent=tier([t for t in trees if t["nearest_enemy_planter_pre"] is None or t["nearest_enemy_planter_pre"]>2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    manifest=MANIFEST.read_bytes()
    if digest(manifest)!=EXPECTED:
        raise ValueError("unknown later7 manifest")
    games=[]
    for card in json.loads(manifest)["games"]:
        blob=(args.raw_dir/card["source"]).read_bytes()
        if digest(blob)!=card["sha256"]:
            raise ValueError("unexpected replay")
        env=json.loads(gzip.decompress(blob))
        if env["statuses"]!=["DONE","DONE"] or card["end_step"]!=len(env["steps"])-1:
            raise ValueError("incomplete replay")
        seats=[score(env["steps"],s,env["configuration"]["size"]) for s in (0,1)]
        games.append(dict(id=card["id"],seed=card["seed"],replay_sha256=card["sha256"],
                          names=card["seat_names"],seats=seats))
    rows=[r for g in games for r in g["seats"]]
    report=dict(manifest_sha256=EXPECTED,evaluator_sha256=digest(Path(__file__).read_bytes()),
                scope="All consecutive observed transitions in seven archived mutual matches. "
                "Game/seat rows share maps. Income is credited absorption during the first uninterrupted ownership period. "
                "Endpoint-owned trees are right censored. Closing cash on a planting turn includes all other credits and debits.",
                games=games,summary=summarize(rows))
    with args.out.open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps(report["summary"]))


if __name__=="__main__":
    main()
