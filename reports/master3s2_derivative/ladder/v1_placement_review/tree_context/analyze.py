"""Plant contexts for one frozen, complete ds-pro-v1 placement snapshot."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent.parent
sys.path.insert(0, str(LADDER / "tree_payback"))
from labels import tree_rows
from runtime import distance, neighbors, quoted_price

SNAPSHOT = "805f6ab04fd81959bf43fd1db00d1d7724ffa5e5082c7a1251e6c6405bf2b338"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plant_context(frame, seat, c, size):
    roster = frame[0]["observation"]["players"]
    own, enemy = roster[seat], roster[1-seat]
    cover = sorted((distance(p,c,size), str(uid), cargo) for uid,(p,cargo,k) in own[2].items() if k=="COLLECTOR")
    planters = sorted((distance(p,c,size), str(uid), p) for uid,(p,_,k) in enemy[2].items() if k=="PLANTER")
    ring = neighbors(c,size)
    seen = [frame[seat]["observation"]["carbon"][z] for z in ring]
    if not all(v is not None and float(v)>=0 for v in seen):
        raise ValueError("Planted cell should have four visible neighbors from its active planter")
    nearest_enemy = planters[0][0] if planters else None
    empty = [u for u in cover if u[2]==0]
    nearest_empty = empty[0][0] if empty else None
    return dict(enemy_planters=planters, own_collectors=cover, nearest_enemy=nearest_enemy,
                nearest_empty=nearest_empty, own_empty_within_one=any(a<=1 for a,_,_ in empty),
                own_any_collector_within_one=any(a<=1 for a,_,_ in cover),
                visible_neighbor_cells=ring, visible_neighbor_carbon=seen, resource=sum(seen),
                risk=nearest_enemy is not None and nearest_enemy<=2 and not any(d<=1 for d,_,_ in empty))


def all_rows(cards):
    rows=[]
    for card in cards:
        path=LADDER/"replays"/f"{card['match_id']}-{card['game']}.json.gz"
        raw=path.read_bytes()
        if digest(raw)!=card["replay_sha256"]:
            raise ValueError(f"Replay identity mismatch: {path}")
        game=json.loads(gzip.decompress(raw))
        seat=card["seat"]; cfg=game["configuration"]; frames=game["steps"]
        if game["rewards"][seat]!=card["score"][0] or card["statuses"]!=game["statuses"]:
            raise ValueError("Scores or replay status mismatch")
        for e in tree_rows(game):
            if e["seat"]!=seat:
                continue
            t=e["t"]; c=e["cell"]; frame=frames[t]
            public=frame[0]["observation"]["players"]
            if public[seat][2][e["planter"]][::2] != [c,"PLANTER"]:
                raise ValueError("Incorrect decision pose")
            if abs(quoted_price(public,seat,cfg)-e["cost"])>1e-6:
                raise ValueError("Actual price and quote disagree")
            context=plant_context(frame,seat,c,cfg["size"])
            row={k:e[k] for k in ("tree","t","cell","planter","break_even_t","end_t","exit","gross","cost","net","repaid","seized_unrepaid")}
            rows.append(dict(match=card["match_id"],game=card["game"],seat=seat,opponent=card["opponent"],outcome=card["outcome"],**row,**context))
    if len(set((r["match"],r["tree"]) for r in rows))!=len(rows):
        raise ValueError("Duplicate tree identity")
    return rows


def summarize(records):
    games=defaultdict(list)
    for r in records:
        games[r["match"]].append(r)
    names=("repaid","seized_unrepaid","seized_after_repaid")
    counts=[]
    for rows in games.values():
        counts.append([len(rows),sum(r["repaid"] for r in rows),sum(r["seized_unrepaid"] for r in rows),
                       sum(r["exit"]=="seized" and r["repaid"] for r in rows)])
    if not counts:
        return {"games":0,"trees":0}
    matrix=np.array(counts); games_n=len(games)
    rates=matrix[:,1:]/matrix[:,0:1]
    out={"games":games_n,"trees":len(records),"exit":dict(Counter(r["exit"] for r in records))}
    out["receipts"]={
        "total_original_tenure_net":sum(r["net"] for r in records),
        "unrecovered_spend_when_seized":sum(-r["net"] for r in records if r["seized_unrepaid"]),
        "net_on_seized_repaid":sum(r["net"] for r in records if r["exit"]=="seized" and r["repaid"]),
    }
    # Resample whole games. All trees of a selected game move together, with equal game weight.
    if games_n>1:
        rng=np.random.default_rng(734291)
        indices=rng.integers(0,games_n,size=(4000,games_n))
        ci=np.quantile(rates[indices].mean(axis=1),[.025,.975],axis=0)
    for i,k in enumerate(names):
        out[k]=dict(n=int(matrix[:,i+1].sum()),per_tree=float(matrix[:,i+1].sum()/len(records)),
                    equal_game=float(rates[:,i].mean()),equal_game_cluster_95=ci[:,i].tolist() if games_n>1 else None)
    return out


def strata(r):
    d=r["nearest_enemy"]
    yield "all"
    yield "high_risk" if r["risk"] else "other"
    for radius in (1,2,3):
        yield f"enemy_le{radius}="+str(d is not None and d<=radius)
    ring="none_or_gt3" if d is None or d>3 else str(d)
    yield "enemy_ring="+ring
    cover="empty_near" if r["own_empty_within_one"] else "no_empty_near"
    yield cover
    yield "any_collector_near="+str(r["own_any_collector_within_one"])
    yield "enemy_le2="+str(d is not None and d<=2)+"/"+cover
    phase = r["t"]//100
    price = "protected20" if r["cost"]<=20.000001 else "20_to_lt40" if r["cost"]<40 else "ge40"
    resource = "lt80" if r["resource"]<80 else "80_to_lt160" if r["resource"]<160 else "ge160"
    for a,b in (("phase",str(phase)),("price",price),("visible_carbon",resource)):
        yield f"{a}={b}"
        yield f"{a}={b}/high_risk="+str(r["risk"])


def pack(path, obj):
    raw=(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    path.write_bytes(gzip.compress(raw,mtime=0) if path.suffix==".gz" else raw)


def main():
    snap=(HERE.parent/"snapshot.json").read_bytes()
    assert digest(snap)==SNAPSHOT
    ledger=json.loads(gzip.decompress((HERE.parent/"ledger.json.gz").read_bytes()))
    cards=ledger["registered_games"]
    assert ledger["snapshot_sha256"]==SNAPSHOT and Counter(c["outcome"] for c in cards)=={"W":26,"L":6}
    rows=all_rows(cards)
    eligible=defaultdict(list)
    for r in rows:
        for key in strata(r):
            for subset in ("all",r["outcome"]):
                eligible[subset,key].append(r)
    pools={subset:{key:summarize(rs) for (group,key),rs in eligible.items() if group==subset} for subset in ("all","W","L")}
    per_game={c["match_id"]:dict(opponent=c["opponent"],outcome=c["outcome"],all=summarize([r for r in rows if r["match"]==c["match_id"]]),
              high_risk=summarize([r for r in rows if r["match"]==c["match_id"] and r["risk"]])) for c in cards}
    paired={}
    for group in ("all","W","L"):
        both=[]
        for match,c in per_game.items():
            if group!="all" and c["outcome"]!=group:
                continue
            in_game=[r for r in rows if r["match"]==match]
            hi=[r for r in in_game if r["risk"]]; other=[r for r in in_game if not r["risk"]]
            if hi and other:
                both.append(sum(r["repaid"] for r in hi)/len(hi)-sum(r["repaid"] for r in other)/len(other))
        diffs=np.array(both)
        rng=np.random.default_rng(734291)
        ci=np.quantile(diffs[rng.integers(0,len(both),(4000,len(both)))].mean(axis=1),[.025,.975])
        paired[group]=dict(games=len(both),mean_high_minus_other=float(diffs.mean()),cluster_95=ci.tolist())
    pack(HERE/"trees.json.gz",dict(snapshot_sha256=SNAPSHOT,
         receipts="v1 first ownership through first seizure, withering, or terminal",rows=rows))
    pack(HERE/"counts.json",dict(snapshot_sha256=SNAPSHOT,groups=pools,games=per_game,paired_within_games=paired))
    for subset in pools:
        for key in ("all","high_risk","other","enemy_le2=True/empty_near","enemy_le2=True/no_empty_near"):
            v=pools[subset].get(key,{})
            print(subset,key,v.get("trees"),[v[o]["n"] for o in ("repaid","seized_unrepaid","seized_after_repaid") if o in v])


if __name__=="__main__":
    main()
