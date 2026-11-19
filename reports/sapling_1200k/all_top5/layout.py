"""Cross-map layout, supply and comeback evidence, one record per ladder game."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

TIMES = (20, 40, 60, 100, 150, 200, 250)
SIDE_FIELDS = ("cash", "collectors", "planters", "trees", "remaining_tree_turns_30",
               "cargo_total", "cargo_near_bank", "cargo_approach_threatened", "blocked_bases")
CASH_CLOSE = 300


def distance(a, b, size):
    dx, dy = abs(a % size - b % size), abs(a // size - b // size)
    return min(dx, size - dx) + min(dy, size - dy)


def moves(p, size):
    x, y = p % size, p // size
    return (p, (x-1)%size+y*size, (x+1)%size+y*size, x+(y-1)%size*size, x+(y+1)%size*size)


def snapshot(obs, cfg):
    players = obs["players"]
    size = cfg["size"]
    values = []
    for seat in range(2):
        cash, homes, workers, trees = players[seat]
        collectors = [(p,c) for p,c,role in workers.values() if role == "COLLECTOR"]
        enemy = [(p,c) for p,c,role in players[1-seat][2].values() if role == "COLLECTOR"]
        locations = {p for side in players for p,_,_ in side[2].values()}
        bank_distance = lambda p: min(distance(p,b,size) for b in homes.values())
        nearby, exposed = 0., 0.
        for p, cargo in collectors:
            if cargo <= 0:
                continue
            d = bank_distance(p)
            if d <= 3:
                nearby += cargo
            dangerous = {cell for q, ec in enemy if ec <= cargo and distance(p,q,size) <= 2 for cell in moves(q, size)}
            toward = [cell for cell in moves(p, size)[1:] if bank_distance(cell) < d]
            if toward and all(cell in dangerous for cell in toward):
                exposed += cargo
        values.append([cash, len(collectors), len(workers)-len(collectors), len(trees),
                       sum(max(0,min(30,cfg["treeLifespan"]-age)) for p,age in trees.values()),
                       round(sum(c for _,c in collectors),2), round(nearby,2), round(exposed,2),
                       sum(b in locations for b in homes.values())])
    return values


def layout(case, source):
    compressed = source.read_bytes()
    digest = hashlib.sha256(compressed).hexdigest()
    if case.get("sha256") and case["sha256"] != digest:
        raise ValueError("unexpected replay bytes")
    try:
        from pydantic_core import from_json
    except ImportError:
        from_json = json.loads
    replay = from_json(gzip.decompress(compressed))
    cfg, frames = replay["configuration"], replay["steps"]
    rewards = replay["rewards"]
    if len(rewards) != 2 or any(x is None for x in rewards):
        raise ValueError("replay has no ordinary numeric terminal rewards")
    if list(rewards) != case["score_by_seat"]:
        raise ValueError("score card and replay disagree")
    last = len(frames)-1
    snapshots = {str(t):snapshot(frames[t][0]["observation"], cfg) for t in TIMES if t <= last}
    return {"id":case["id"],"sha256":digest,"seed":cfg["randomSeed"],"steps":last,
            "scores":rewards,"statuses":replay["statuses"],"snapshots":snapshots}


def group_counts(games):
    stats = defaultdict(Counter)
    for case in games:
        for seat, player_id in enumerate(case["seat_strategy_ids"]):
            if player_id is None:
                continue
            name = case["seat_names"][seat]
            delta = case["score_by_seat"][seat] - case["score_by_seat"][1-seat]
            outcome = "win" if delta > 0 else "loss" if delta < 0 else "draw"
            for label in (name, name+"/"+case["cohort"], name+"/seat"+str(seat)):
                stats[label][outcome] += 1
                stats[label]["games"] += 1
                stats[label]["margin_sum"] += delta
    return {key:{"games":v["games"],"win":v["win"],"draw":v["draw"],"loss":v["loss"],
                 "mean_margin":round(v["margin_sum"]/v["games"],2)} for key,v in sorted(stats.items())}


def relation(cases, rows):
    """Each map enters a stratum once, including games in which both players rank."""
    entries = defaultdict(list)
    fields = {k:i for i,k in enumerate(SIDE_FIELDS)}
    for case in cases:
        record = rows.get(case["id"])
        if not record:
            continue
        for t, pair in record["snapshots"].items():
            diff = {k:pair[0][i]-pair[1][i] for k,i in fields.items()}
            candidate_lead = {"more_living_trees":diff["trees"],
                              "more_workers":diff["planters"]+diff["collectors"],
                              "more_cargo_within_3":diff["cargo_near_bank"],
                              "more_tree_life_next_30":diff["remaining_tree_turns_30"],
                              "less_cargo_with_all_shortest_approaches_threatened":-diff["cargo_approach_threatened"]}
            for k,d in candidate_lead.items():
                if d == 0:
                    continue
                leader = 0 if d>0 else 1
                margin = record["scores"][leader]-record["scores"][1-leader]
                for subgroup in ("all",case["cohort"]):
                    entries[(t,subgroup,"all_cash",k)].append(margin)
                    if abs(diff["cash"])<=CASH_CLOSE:
                        entries[(t,subgroup,"cash_gap_within_300",k)].append(margin)
    results=[]
    for (t,cohort,condition,metric),values in sorted(entries.items()):
        results.append(dict(step=int(t),cohort=cohort,condition=condition,metric=metric,games=len(values),
                            favored_seat_wins=sum(x>0 for x in values),draws=values.count(0),
                            favored_seat_losses=sum(x<0 for x in values),
                            median_official_margin=round(statistics.median(values),2)))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True,help="root containing replays/<game>.json.gz")
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_bytes())
    cases=manifest["games"]
    if len(cases)!=len({g["id"] for g in cases}):
        raise ValueError("duplicate games")
    args.work.mkdir(parents=True,exist_ok=True)
    rows={}; errors={}
    for g in cases:
        path=args.cache/g["path"]
        saved=args.work/(g["id"]+".json")
        if not path.exists():
            continue
        try:
            if saved.is_file():
                r=json.loads(saved.read_bytes())
                if not g.get("sha256") or r["sha256"] == g["sha256"]:
                    rows[g["id"]]=r
                    continue
            r=layout(g,path)
            tmp=saved.with_suffix(".tmp")
            tmp.write_text(json.dumps(r,allow_nan=False,separators=(",",":")))
            tmp.replace(saved)
            rows[g["id"]]=r
        except (ValueError,KeyError,OSError,TypeError) as exc:
            errors[g["id"]]=str(exc)
    evidence={"git_revision":manifest["revision"],"covered":len(rows),"total":len(cases),
              "field_order":SIDE_FIELDS,"snapshot_steps":TIMES,"missing":sorted({r["id"] for r in cases}-set(rows)-set(errors)),
              "errors":errors,"scorecards":group_counts(cases),"relationships":relation(cases,rows),
              "games":[rows[k] for k in sorted(rows)]}
    tmp=args.out.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(evidence,allow_nan=False,ensure_ascii=False,separators=(",",":")).encode(),mtime=0))
    tmp.replace(args.out)
    print("covered",len(rows),"of",len(cases),"errors",len(errors),flush=True)


if __name__=="__main__":
    main()
