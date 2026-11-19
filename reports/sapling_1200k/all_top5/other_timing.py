"""Recruitment and delivery timing in actual Top Five decisions, using no new games."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from economy_extract import distances, reconstruct, visible_ledger

HERE = Path(__file__).resolve().parent
ROLES = {"RECCOLLECTOR": "COLLECTOR", "RECPLANTER": "PLANTER"}
try:
    from pydantic_core import from_json as loads
except ImportError:
    loads = json.loads


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sample(games, rank):
    """Stratify mutual matchups by better-ranked player's seat, external by elite player and seat."""
    grouped = defaultdict(list)
    for g in games:
        if not g["clean_terminal"]:
            continue
        names = g["seat_names"]
        focal = min((n for n in names if n in rank), key=rank.get)
        s = names.index(focal)
        key = (focal, s, names[1-s]) if g["cohort"] == "mutual" else (focal, s)
        grouped[key].append(g)
    result = []
    for key, gs in grouped.items():
        ordered = sorted(gs, key=lambda g: digest(("other-timing-v1|" + g["id"]).encode()))
        # Six distinct external opponents per player/seat or one map per mutual stratum.
        found = set()
        for g in ordered:
            n = g["seat_names"][1-key[1]]
            if n not in found:
                result.append(g)
                found.add(n)
                if len(found) == (1 if len(key) == 3 else 6):
                    break
    return sorted(result, key=lambda g: g["id"])


def delivery(games):
    by_id = {g["id"]: g for g in games}
    seen = set()
    counts = defaultdict(Counter)
    labels = defaultdict(lambda: defaultdict(set))
    for path in sorted((HERE / "economy_data").glob("*.jsonl.gz")):
        with gzip.open(path, "rb") as stream:
            for line in stream:
                record = loads(line)
                g = by_id[record["id"]]
                if g["id"] in seen or record["replay_sha256"] != g["sha256"]:
                    raise ValueError(g["id"])
                seen.add(g["id"])
                if not g["clean_terminal"]:
                    continue
                for s, sid in enumerate(g["seat_strategy_ids"]):
                    if sid is None:
                        continue
                    for tag, (hits, total) in record["record"]["alerts"][s].items():
                        keys = [(g["cohort"], "all", tag), (g["cohort"], str(s), tag)]
                        for key in keys:
                            c = counts[key]
                            c["hits"] += hits
                            c["opportunities"] += total
                            if total:
                                c["appearances_with_alert"] += 1
                                labels[key]["maps"].add(g["id"])
                                labels[key]["opponents"].add(g["seat_names"][1-s])
    if seen != set(by_id):
        raise ValueError("economic archive incomplete")
    return {"/".join(k): dict(c, maps_with_alert=len(labels[k]["maps"]), opponents_with_alert=len(labels[k]["opponents"]))
            for k, c in sorted(counts.items())}


def recruit_rows(env, g):
    frames, cfg = env["steps"], env["configuration"]
    end = len(frames)-1
    dist = distances(cfg["size"])
    rows = []
    for t in range(end):
        ps = frames[t][0]["observation"]["players"]
        pn = frames[t+1][0]["observation"]["players"]
        roles = Counter(w[2] for p in ps for w in p[2].values())
        prices = {role: cfg["rec"+role.title()+"Cost"] + roles[role]*cfg["rec"+role.title()+"IncreaseCost"]
                  for role in ROLES.values()}
        for s, sid in enumerate(g["seat_strategy_ids"]):
            if sid is None:
                continue
            cash, bases, workers, _ = ps[s]
            enemies = ps[1-s][2]
            count = Counter(w[2] for w in workers.values())
            roster = len(workers)
            newborns = {uid: w for uid, w in pn[s][2].items() if uid not in workers}
            cmds = frames[t+1][s]["action"] or {}
            for bid, base in bases.items():
                can = {r: roster < cfg["workerLimit"] and cash >= p for r, p in prices.items()}
                cmd = cmds.get(bid, "NONE")
                role = ROLES.get(cmd)
                hired = bool(role and can[role])
                new = [uid for uid, w in newborns.items() if w[0] == base and w[2] == role]
                if len(new)>1 or new and not hired:
                    raise ValueError((g["id"], t, bid, "contradictory newborn"))
                occ = ("own" if any(w[0] == base for w in workers.values()) else
                       "enemy" if any(w[0] == base for w in enemies.values()) else "empty")
                enemy_distance = min((dist[base][w[0]] for w in enemies.values()), default=None)
                alive5 = None if not hired or t+5 > end else bool(new and new[0] in frames[t+5][0]["observation"]["players"][s][2])
                rows.append([t,s,bid,base,cfg["episodeSteps"]-1-t,round(ps[s][0],2),round(cash,2),
                             count["COLLECTOR"],count["PLANTER"],roster,len(enemies),occ,enemy_distance,
                             prices["COLLECTOR"],prices["PLANTER"],can["COLLECTOR"],can["PLANTER"],
                             cmd,hired,new[0] if new else None,alive5])
                if hired:
                    roster += 1
                    cash -= prices[role]
    return rows


def illustrate(env,g):
    fs, cfg = env["steps"], env["configuration"]
    n = len(fs)-1
    try:
        tx = reconstruct(fs,cfg)[2]
    except (ValueError,KeyError,TypeError,IndexError):
        tx = visible_ledger(fs,cfg)[2]
    dist = distances(cfg["size"])
    ps = [f[0]["observation"]["players"] for f in fs]
    found = {}
    for s, sid in enumerate(g["seat_strategy_ids"]):
        if sid is None:
            continue
        last = {}
        for t in range(3,n-9):
            home = list(ps[t][s][1].values())
            for uid, (cell, load, role) in sorted(ps[t][s][2].items()):
                if role != "COLLECTOR" or load < 200 or t-last.get(uid,-20)<10:
                    continue
                d = min(dist[cell][b] for b in home)
                old = ps[t-3][s][2].get(uid)
                if not 4<=d<=8 or old is None:
                    continue
                old_d = min(dist[old[0]][b] for b in home)
                last[uid] = t
                nxt = ps[t+1][s][2].get(uid)
                if nxt is None:
                    continue
                nd = min(dist[nxt[0]][b] for b in home)
                action = (fs[t+1][s]["action"] or {}).get(uid,"NONE")
                receipts = [[j+1, round(val,2)] for j in range(t,t+10) for seat, who, val in tx[j]
                            if (seat,who)==(s,uid) and val>0]
                if nd < d and receipts:
                    tag = "toward_then_deposit"
                elif action == "NONE" and nxt[0]==cell and nxt[1]>load and not receipts:
                    tag = "stay_and_more_cargo_no_ten_step_deposit"
                else:
                    continue
                if tag in found:
                    continue
                path = []
                for i in sorted({t-3,t,t+1,t+3,t+5,t+10} | {j for j, _ in receipts}):
                    worker=ps[i][s][2].get(uid)
                    path.append([i,worker[0],round(worker[1],2),min(dist[worker[0]][b] for b in home)] if worker else [i,None,None,None])
                found[tag]=dict(id=g["id"],seed=g["seed"],seat=s,player=g["seat_names"][s],opponent=g["seat_names"][1-s],t=t,
                                planned_turns_left=cfg["episodeSteps"]-1-t,uid=uid,cash=ps[t][s][0],home=home,
                                cell=cell,cargo=load,home_distance=d,past3_distance=old_d,first_action=action,
                                nearest_enemy_distance=min((dist[cell][w[0]] for w in ps[t][1-s][2].values()),default=None),
                                positive_bank_next10=receipts,path=path)
                if len(found)==2:
                    return found
    return found


COL = "t seat base_id cell turns_left initial_cash spendable_cash collectors planters roster_before_base enemy_workers occupancy nearest_enemy collector_price planter_price can_collector can_planter action accepted new_uid alive5".split()


def main(cache):
    m = json.loads((HERE/"manifest.json").read_bytes())
    rank = {r["name"]:r["rank"] for r in m["top5"]}
    maps = sample(m["games"],rank)
    stats = defaultdict(Counter)
    grouping = defaultdict(lambda: defaultdict(set))
    cases = {}
    raw_rows = {}
    for g in maps:
        b = (cache/(g["id"]+".json.gz")).read_bytes()
        if digest(b)!=g["sha256"]:
            raise ValueError(g["id"])
        env = loads(gzip.decompress(b))
        raw_rows[g["id"]]=rows=recruit_rows(env,g)
        if g["cohort"]=="mutual" and not cases:
            found=illustrate(env,g)
            if len(found)==2:
                cases=found
        for row in rows:
            t,s,bid,base,rem,initial,cash,col,plan,roster,foes,occ,ed,pc,pp,cc,cp,act,ac,uid,alive=row
            age = "opening_0_20" if t<20 else "mid_20_249" if t<250 else "late_250_plus"
            for key in ((g["cohort"],"all"),(g["cohort"],age),(g["cohort"],"occupancy_"+occ),
                        (g["cohort"],"enemy_d_le1" if ed is not None and ed<=1 else "enemy_d_gt1_or_none"),
                        (g["cohort"],"seat"+str(s))):
                c=stats[key]
                c["decisions"]+=1
                c["legal"]+=bool(cc or cp)
                c["accepted"]+=ac
                c["accepted_collector"]+=ac and act=="RECCOLLECTOR"
                c["accepted_planter"]+=ac and act=="RECPLANTER"
                c["rejected"]+=act in ROLES and not ac
                c["alive1"]+=bool(uid)
                c["eligible_alive5"]+=alive is not None
                c["alive5"]+=bool(alive)
                grouping[key]["maps"].add(g["id"])
                grouping[key]["opponents"].add(g["seat_names"][1-s])
    if len(cases)!=2:
        raise ValueError("no two trace types in same mutual map")
    compact=lambda x:json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    sealed=gzip.compress(compact(dict(fields=COL,rows_by_game=raw_rows)),compresslevel=6,mtime=0)
    (HERE/"other_timing_rows.json.gz").write_bytes(sealed)
    result=dict(input_revision=m["revision"],manifest_sha256=digest((HERE/"manifest.json").read_bytes()),row_file_sha256=digest(sealed),
                source_games=len(m["games"]),normal_terminal_games=sum(g["clean_terminal"] for g in m["games"]),
                sample_ids=[g["id"] for g in maps],sample_maps=dict(Counter(g["cohort"] for g in maps)),delivery_alerts=delivery(m["games"]),
                recruits={"/".join(k):dict(c,maps=len(grouping[k]["maps"]),opponents=len(grouping[k]["opponents"])) for k,c in sorted(stats.items())},
                delivery_examples=cases)
    (HERE/"other_timing.json").write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cache",type=Path,required=True,help="directory of archived <id>.json.gz members (fetch.py --restore)")
    main(p.parse_args().cache)
