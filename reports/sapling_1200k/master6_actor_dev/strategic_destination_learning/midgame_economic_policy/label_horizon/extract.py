"""Replay trained decision roots only, tracking the two treasuries and the first tenure of trees acquired at the root."""
from collections import defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from pydantic_core import from_json

HERE=Path(__file__).resolve().parent
LANE=HERE.parent
ACTOR=LANE.parents[1]
ALL=ACTOR.parent/"all_top5"
CACHE=Path("/workspace/sapling-all-top5-20261119/replays")
sys.path.insert(0,str(ALL))
from economy_extract import reconstruct, KEYS
SLOTS=(30,50,70,90,110)


def signature(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    a=path.read_bytes()
    return from_json(gzip.decompress(a) if path.suffix==".gz" else a)


def write(path,data):
    raw=(json.dumps(data,sort_keys=True,allow_nan=False,ensure_ascii=False,separators=(",",":"))+"\n").encode()
    raw=gzip.compress(raw,mtime=0) if path.suffix==".gz" else raw
    tmp=path.with_suffix(".pending")
    tmp.write_bytes(raw)
    tmp.replace(path)


def need(ok,msg):
    if not ok:
        raise ValueError(msg)


def obs(frames,t):
    return frames[t][0]["observation"]


def tenure(frames,t,seat):
    current=obs(frames,t+1)
    prev=obs(frames,t)
    new=current["players"][seat][3];old=prev["players"][seat][3]
    total=len(frames)-1;out=[]
    for tree in sorted(set(new)-set(old)):
        kind="seize" if tree in prev["players"][1-seat][3] else "plant"
        worker=current["trees"][tree][0]
        revenues=[]
        for j in range(t+1,total+1):
            q=obs(frames,j)
            if tree not in q["players"][seat][3]:
                break
            revenues.append(float(q["trees"][tree][1]))
        out.append(dict(tree=tree,kind=kind,worker=worker,cell=new[tree][0],age_at_purchase=new[tree][1],
                        first_tenure_steps=len(revenues),rent20=sum(revenues[:20]),rent60=sum(revenues[:60]),
                        rent_observed_terminal=sum(revenues)))
    return out


def game_rows(card,rows):
    name=card["id"]
    path=CACHE/(name+".json.gz")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==card["sha256"],"replay identity")
    game=from_json(gzip.decompress(raw))
    frames=game["steps"];T=len(frames)-1;seat=card["seat"]
    need(T==card["turns"] and sha256(bytes(obs(frames,0)["full_carbon"])).hexdigest()==card["terrain_sha256"],"terrain or duration")
    rewards=game["rewards"]
    need(all(isinstance(v,(int,float)) for v in rewards) and game["statuses"]==["DONE","DONE"],"official terminal")
    clocks=[row["t"] for row in rows]
    need(clocks==[t for t in SLOTS if t+20<=T], "unexpected root allocation")
    stop=min(T,max(clocks)+60)
    reason=None
    try:
        events,_,_,_=reconstruct(frames[:stop+1],game["configuration"])
        cash=np.cumsum(np.concatenate([np.zeros((1,2,8)),np.array(events)]),axis=0)
    except ValueError as e:
        reason=str(e);cash=None
    out=[]
    for r in rows:
        t=r["t"];final=obs(frames,T)["players"]
        horizons={"root":[t,t+1],"early":[t,t+20],"late":[t+20,t+60] if t+60<=T else None}
        sides=[]
        for s in (seat,1-seat):
            assets=tenure(frames,t,s)
            before=obs(frames,t)["players"][s]
            rr=dict(initial_cash=before[0],workers=len(before[2]),trees=len(before[3]),
                    carried=sum(w[1] for w in before[2].values()),root_receipts=assets,root_order=(frames[t+1][s]["action"] or {}),
                    official_score=rewards[s],terminal_bank=final[s][0],
                    after_20_to_terminal_official=rewards[s]-obs(frames,t+20)["players"][s][0],
                    after_20_to_terminal_bank=final[s][0]-obs(frames,t+20)["players"][s][0])
            for h,span in horizons.items():
                if span is None:
                    rr[h]=None
                    continue
                a,b=span
                money=obs(frames,b)["players"][s][0]-obs(frames,a)["players"][s][0]
                fields=dict(net_cash=money)
                if cash is not None:
                    comp=cash[b,s]-cash[a,s]
                    need(abs(comp.sum()-money)<.015, f"unbalanced replay {name} {s} {span}")
                    fields.update({k:float(v) for k,v in zip(KEYS,comp)})
                    fields["tree_spend"]=-(fields["plant"]+fields["seize"])
                    fields["tree_net"]=fields["tree"]-fields["tree_spend"]
                rr[h]=fields
            if cash is not None:
                need((rr["root"]["plant"]<0)==any(a["kind"]=="plant" for a in assets),"tree planting provenance")
                need((rr["root"]["seize"]<0)==any(a["kind"]=="seize" for a in assets),"tree takeover provenance")
            sides.append(rr)
        need(abs(sides[0]["early"]["net_cash"]-sides[1]["early"]["net_cash"]-r["target"])<1e-5,"training label differs")
        out.append(dict(gid=name,terrain=card["terrain_sha256"],seed=card["seed"],teacher=card["teacher"],opponent=card["opponent"],
                        seat=seat,source_sha256=card["sha256"],t=t,turns=T,early_terminal=T<299,complete_60=t+60<=T,
                        verified_ledger=cash is not None,missing_ledger_reason=reason,fold=r["fold"],opportunity=r["opportunity"],
                        residual=r["residual"],target=r["target"],weight=r["weight"],sides=sides))
    return out


def main():
    cfg=read(LANE/"protocol.json")
    pp=ACTOR/"broader_elite_backbone_dev/panel.json"
    need(signature(pp)==cfg["panel"],"changed source panel")
    adv=read(LANE/"advantage.json.gz")
    inputs=defaultdict(list)
    for r in adv:
        need(r["split"]=="train" and r["supported"],"unexpected nontraining label")
        inputs[r["gid"]].append(r)
    train={q["id"]:q for q in read(pp)["games"] if q["split"]=="train" and q["id"] in inputs}
    need(len(train)==len(inputs)==len({q["terrain_sha256"] for q in train.values()})==234,"training map coverage")
    all_rows=[]
    for i,g in enumerate(sorted(inputs),1):
        rows=game_rows(train[g],sorted(inputs[g],key=lambda r:r["t"]))
        all_rows.extend(rows)
        if i%25==0:
            print("replayed",i,flush=True)
    need(len(all_rows)==len(adv)==1167,"training root coverage")
    write(HERE/"training_cash.json.gz",all_rows)


if __name__=="__main__":
    main()
