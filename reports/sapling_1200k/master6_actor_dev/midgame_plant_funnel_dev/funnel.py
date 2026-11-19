"""Rebuild 18 archived Master3s2 games. Separate public opportunity, action, and actual cash debit."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from turning_point_ledger import FastCarbon, CashEvents

def load(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix==".gz" else path.read_bytes())

def save(path,value):
    data=json.dumps(value,indent=None if path.suffix==".gz" else 2,allow_nan=False,sort_keys=True).encode()+b"\n"
    path.write_bytes(gzip.compress(data,mtime=0) if path.suffix==".gz" else data)

def price(me,enemy,cfg):
    if len(me[3])<=cfg["playerTreeProtectiveNumber"]:
        return cfg["plantCost"]
    alive=sum(age<cfg["treeLifespan"] for pl in (me,enemy) for _,age in pl[3].values())
    return cfg["plantCost"]+cfg["plantCostInflationRatio"]*cfg["plantCostInflationBase"]**alive

def targets(obs,seat,uid,nb,cfg,actual_ground):
    """Each physical candidate gets independent pre-action public and retrospective exact richness."""
    me,them=obs["players"][seat],obs["players"][1-seat]
    pos=me[2][uid][0]
    used={cell for pl in (me,them) for cell in pl[1].values()}
    used.update(cell for pl in (me,them) for cell,_ in pl[3].values())
    used.update(w[0] for pl in (me,them) for key,w in pl[2].items() if key!=uid)
    enemy_reach={q for p,_,_ in them[2].values() for q in (p,*nb[p])}
    seen={q for p,_,role in me[2].values() for q in ((p,) if role=="COLLECTOR" else nb[p])}
    options={}
    for q in (pos,*nb[pos]):
        if q in used:continue
        around=nb[q]; measured=[obs["carbon"][c] for c in around if c in seen]
        # A sum >= 120 guarantees mean >=30 even when unobserved cells later prove empty.
        lower_mean=sum(measured)/4.
        options[q]=(lower_mean,sum(actual_ground[c] for c in around)/4,len(measured),q not in enemy_reach)
    return options

def phase_counts(raw,c,detail):
    cfg=raw["game_configuration"];seat=c["seat"];sim=FastCarbon(cfg).reset(seed=raw["seed"])
    expected=raw['official_score_by_seat'];T=len(raw["actions"])
    out=[Counter(),Counter()]
    frame=[]
    for t,act in enumerate(raw["actions"]):
        if t<=200:
            public=[sim.obs(s) for s in (0,1)]
            full=list(sim.carbon)
        sim.cash=CashEvents(sim.cash);sim.step(*act)
        if t<=200:
            e=sim.cash.events
            frame.append((public,full,[dict(a) for a in act],{x[3]:x[0] for x in e if x[0] in ("plant","seize")},
                          [dict(sim.obs(s)["players"][s][2]) for s in (0,1)]))
    if max(abs(a-b) for a,b in zip(expected,sim.cash))>.011:raise ValueError("terminal cash mismatch")
    if T<201:raise ValueError("window truncated")
    for t in range(100,200):
        ob,full,pair,settled,arrivals=frame[t]
        for s,d in enumerate(out):
            me,en=ob[s]["players"][s],ob[s]["players"][1-s]
            seen_seize=set();seen_plant=set()
            pt=price(me,en,cfg)
            affordable=me[0]>=pt
            planters=[(u,w[0]) for u,w in me[2].items() if w[2]=="PLANTER"]
            d["side_turns"]+=1;d["turns_with_planter"]+=bool(planters)
            next_events=frame[t+1][3]
            for uid,pos in planters:
                actions=pair[s]
                cmd=actions.get(uid)
                to=sim._move_nbr[cmd][pos] if cmd in sim._move_nbr else pos
                stayed=pos==to;planted=settled.get(uid)=="plant";seized=settled.get(uid)=="seize"
                if planted:seen_plant.add(uid)
                if seized:seen_seize.add(uid)
                opt=targets(ob[s],s,uid,sim._nbr4,cfg,full)
                current=opt.get(pos)
                arrived=arrivals[s].get(uid)
                on_target=bool(arrived and arrived[0]==to)
                d["planter_turns"]+=1;d["affordable"]+=affordable
                d["reachable_vacant"]+=bool(opt)
                d["reachable_vacant_affordable"]+=bool(opt) and affordable
                d["standing_vacant"]+=current is not None
                d["standing_vacant_affordable"]+=current is not None and affordable
                d["stayed_vacant_affordable"]+=current is not None and affordable and stayed
                d["plant_after_stayed_vacant_affordable"]+=bool(current and affordable and stayed and planted)
                d["plant_after_stay_unaffordable"]+=bool(current and not affordable and stayed and planted)
                d["plant"]+=planted;d["seize"]+=seized
                if current and not affordable:d["standing_vacant_unaffordable_stayed"]+=stayed
                d["root_affordable_plant"]+=affordable and planted
                d["standing_enemy_tree"]+=pos in {q for q,_ in en[3].values()}
                d["moved"]+=not stayed
                d["died"]+=not bool(arrived)
                flags={}
                for threshold in (10,30):
                    g=str(threshold)+"_"
                    public={p for p,(lower,actual,n,safe) in opt.items() if lower>=threshold and safe}
                    oracle={p for p,(lower,actual,n,safe) in opt.items() if actual>=threshold and safe}
                    contested={p for p,(lower,actual,n,safe) in opt.items() if lower>=threshold and not safe}
                    if not public<=oracle:raise ValueError("visibility lower bound")
                    if current and current[2]!=4:raise ValueError("planter's four adjacent cells should be visible")
                    d[g+"public_reachable"]+=bool(public);d[g+"public_neighbor"]+=bool(public-{pos})
                    d[g+"oracle_reachable"]+=bool(oracle);d[g+"only_oracle"]+=bool(oracle) and not public
                    d[g+"contested_reachable"]+=bool(contested)
                    d[g+"public_reachable_affordable"]+=bool(public) and affordable
                    d[g+"standing_rich_safe"]+=pos in public
                    d[g+"standing_rich_safe_affordable"]+=pos in public and affordable
                    d[g+"standing_rich_contested_affordable"]+=pos in contested and affordable
                    d[g+"rich_safe_stayed"]+=pos in public and affordable and stayed
                    d[g+"rich_safe_plant"]+=pos in public and affordable and planted
                    d[g+"rich_safe_left"]+=pos in public and affordable and not stayed
                    d[g+"rich_safe_left_to_enemy_tree"]+=pos in public and affordable and not stayed and to in {q for q,_ in en[3].values()}
                    d[g+"rich_contested_stayed"]+=pos in contested and affordable and stayed
                    d[g+"rich_contested_plant"]+=pos in contested and affordable and planted
                    away=affordable and pos not in public and bool(public)
                    d[g+"near_only"]+=away
                    d[g+"near_only_moved_to_public"]+=away and to in public
                    d[g+"near_only_arrived"]+=away and to in public and on_target
                    d[g+"near_only_planted_next"]+=away and to in public and on_target and next_events.get(uid)=="plant"
                    d[g+"produced_on_unaffordable_rich"]+=pos in public and not affordable and planted
                    if threshold==30:flags=dict(public=sorted(public),contested=sorted(contested),oracle=sorted(oracle))
                detail.append(dict(t=t,s=s,uid=uid,pos=pos,cmd=cmd,affordable=affordable,opening_cash=me[0],plant_price=pt,
                                   on_target=on_target,settled=settled.get(uid),next_settled=next_events.get(uid),**flags))
            debits=Counter(settled.get(uid) for uid,_ in planters)
            if debits["plant"]!=len(seen_plant) or debits["seize"]!=len(seen_seize):raise ValueError("ambiguous receipts")
        if set(settled)!={r['uid'] for r in detail if r['t']==t and r['settled']}:raise ValueError("missed planter debit")
    return {"sapling":dict(out[seat]),"master3s2":dict(out[1-seat])}

def main():
    cases=load(HERE.parent/"tactical_exchange_dev/paired.json.gz");assert len(cases)==18
    parts=[];all_roots=[]
    for c in cases:
        path=ROOT/c["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=c["source_sha256"]:raise ValueError("source identity")
        raw=load(path)
        if not raw["examiner_md5"].startswith("57f9a594") or raw["candidate_seat"]!=c["seat"]:raise ValueError("examiner or side")
        det=[];co=phase_counts(raw,c,det)
        parts.append(dict(path=c['path'],sha256=c['source_sha256'],seed=c["seed"],candidate_seat=c["seat"],pack=raw["pack_sha256"],counts=co))
        all_roots.append(dict(path=c["path"],seed=c["seed"],candidate_seat=c["seat"],roots=det))
    def aggregate(games):
        return {s:dict(sum((Counter(g["counts"][s]) for g in games),Counter())) for s in ("sapling","master3s2")}
    m=sorted(set(g["seed"] for g in parts))
    rep=dict(games=parts,seeds=m,window=[100,199],aggregate=aggregate(parts),
             by_map={str(k):aggregate([g for g in parts if g["seed"]==k]) for k in m})
    save(HERE/"counts.json",rep);save(HERE/"opportunities.json.gz",all_roots)
    print({s:{k:v for k,v in d.items() if k in ("planter_turns","plant","seize","30_rich_safe_stayed","30_rich_safe_plant","30_standing_rich_safe_affordable","30_near_only","30_near_only_moved_to_public","30_near_only_planted_next","30_oracle_reachable","30_public_reachable")} for s,d in rep["aggregate"].items()})

if __name__=="__main__":main()
