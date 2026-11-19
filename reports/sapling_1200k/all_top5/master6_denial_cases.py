"""Reproduce matched early cashflow case studies and exact one-command official-Board forks."""
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

from context import Corpus, HERE, ROOT
from defense import DEST, DIST, MOVE

GAME_SOURCE = ROOT / "code/env_src/carbon_challenge-master"
sys.path.insert(0, str(GAME_SOURCE))
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.helpers import Board

ACTIONS = ("STAY", "UP", "RIGHT", "DOWN", "LEFT")
# Each control is the same external registered name and physical seat. Three contrastive mechanisms.
CASES = [
    ("First111", 0, "1841c9fb5bf77087b4129693495faac3-1", 68, 13, 16,
     "95739c59503d79c2a4aa65af8fe41fcf-1", 75, 9),
    ("stephen-K4TacSearch-v1", 0, "be51be0daa809b964da213cb1bc99842-1", 48, 6, 2,
     "82d1b6c28582f94beec041550168faf0-1", 46, 7),
    ("claude-opus55-cloud-Echo6", 1, "4abeffae2b78b601f00be3980084ea0d-1", 23, 0, 3,
     "44e3a7995cfa7899bf8b57a40952eb57-1", 25, 0),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def site(uid, seat):
    return f"player-{seat}-worker-{uid}"


def totals(players):
    return [dict(cash=p[0], bases=list(p[1].values()), collectors=sum(x[2]=="COLLECTOR" for x in p[2].values()),
                 planters=sum(x[2]=="PLANTER" for x in p[2].values()), trees=len(p[3]),
                 cargo=sum(x[1] for x in p[2].values())) for p in players]


def describe(players, seat, uid, cmd):
    workers=players[seat][2]
    if uid not in workers:
        return None
    p,c,role=workers[uid]
    base=list(players[seat][1].values())
    enemy=players[1-seat]
    threats={q for pos,load,kind in enemy[2].values() if kind=="COLLECTOR" and load<=c for q in DEST[pos]}
    safe=[q for q in DEST[p] if q not in threats and q not in enemy[1].values()]
    near_trees=[[s,key,pos,age] for s,k in enumerate(players) for key,(pos,age) in k[3].items() if DIST[p][pos]<=2]
    near_units=[[s,key,*vec] for s,k in enumerate(players) for key,vec in k[2].items() if DIST[p][vec[0]]<=2]
    return dict(id=uid,pos=p,coord_row_col=divmod(p,15),cargo=c,role=role,command=cmd or "STAY",
                destination=DEST[p][MOVE[cmd]],nearest_own_base_steps=min(DIST[p][b] for b in base),
                geometric_safe_destinations=safe,nearby_trees_radius2=near_trees,nearby_workers_radius2=near_units)


def commands(game,t):
    return game.labels(t,0,["actions"])["joint_commands_from_t"]


def visible_trace(game, t, seat, victim, enemy):
    rows=[]
    for k in (t-10,t-5,t-3,t-1,t,t+1):
        pub=game.decision(k,seat,lookbacks=())["at"]["observation"]
        ps=pub["players"];act=commands(game,k)
        rows.append(dict(t=k,physical_seat_stocks=totals(ps),
                    carrier=describe(ps,seat,victim,act[seat].get(victim)),
                    opponent=describe(ps,1-seat,enemy,act[1-seat].get(enemy)) if enemy else None))
    return rows


def prefix(game,t):
    cfg=game.metadata()["configuration"]
    idgen.reset()
    state=copy.deepcopy(game._frames[0][0]["observation"])
    state["carbon"]=state["full_carbon"]
    for k in range(t):
        state=Board(state,cfg,commands(game,k)).next().observation
        check(state,game,k+1)
    return state,copy.deepcopy(idgen._id_dict),cfg


def check(state,game,k):
    obs=game._frames[k][0]["observation"]
    if (state["players"],state["trees"],state["carbon"])!=(obs["players"],obs["trees"],obs["full_carbon"]):
        raise ValueError(f"official board disagrees at t={k}")


def forks(game,t,seat,victim,hunter,edits):
    """Run each alternative from the same verified state, restoring engine entity ID counters as well."""
    state,ids,cfg=prefix(game,t)
    joint=commands(game,t)
    results=[]
    for who,uid,choice in [(None,None,None),*edits]:
        idgen._id_dict.clear()
        idgen._id_dict.update(ids)
        a=copy.deepcopy(joint)
        if uid is not None:
            if choice=="STAY":
                a[who].pop(uid,None)
            else:
                a[who][uid]=choice
            if a==joint:
                continue
        after=Board(state,cfg,a).next().observation
        if uid is None:
            check(after,game,t+1)
        ps=after["players"]
        results.append(dict(changed_seat=who,changed_entity=uid,command=choice,
                          stocks=totals(ps),carrier=ps[seat][2].get(victim),
                          opponent=ps[1-seat][2].get(hunter),prefix_and_factual_frames_verified=t+2 if uid is None else None))
    return results


def branches(game,t,seat,carrier,enemy=None):
    units=[(seat,carrier)]+([(1-seat,enemy)] if enemy else [])
    legal=[(s,uid,a) for s,uid in units for a in ACTIONS]
    return forks(game,t,seat,carrier,enemy,legal)


def choose_control(game,t,s,uid):
    before=game._obs(t,s)["players"]
    factual=commands(game,t)
    p,q,kind=before[s][2][uid]
    dest=DEST[p][MOVE[factual[s].get(uid)]]
    friends=before[s][1]
    others=[DEST[pos][MOVE[factual[k].get(w)]]
            for k in (0,1) for w,(pos,_,_) in before[k][2].items() if w!=uid]
    if (kind!="COLLECTOR" or q<=0 or dest==p or dest not in friends.values() or dest in others
            or any(factual[s].get(b) for b,pos in friends.items() if pos==dest)):
        raise ValueError("control must be an uncontested normal deposit")
    observed=game._obs(t+1,s)["players"][s][2].get(uid)
    if not observed or observed[0]!=dest or observed[1]!=0:
        raise ValueError("control did not actually bank")
    return q


def get_game(corpus,gid,seat,name):
    meta=corpus.provenance(gid)
    if meta["seat_names"][seat]!=name:
        raise ValueError("physical seat mismatch")
    return corpus.open(gid),dict(id=gid,names=meta["seat_names"],replay_sha256=corpus.rows[gid]["replay_sha256"])


def main():
    corpus=Corpus()
    controls=HERE/"master6_opponent_controls.json"
    strata=json.loads(controls.read_bytes())["groups"]["21_50"]["strata"]
    defense=HERE/"defense_traces.jsonl.gz"
    selected={r[2] for r in CASES}
    lots={r["id"]:r["lots"] for r in map(json.loads,gzip.open(defense,"rt")) if r["id"] in selected}
    result=[]
    for name,s,gid,t,n,m,cid,ct,cn in CASES:
        st=next(v for v in strata if v["opponent"]==name and v["seat"]==s)
        if gid not in st["arm_ids"][0] or cid not in st["arm_ids"][1]:
            raise ValueError("cases not in the common opponent/seat stratum")
        v=site(n,s);h=site(m,1-s);cv=site(cn,s)
        g,a=get_game(corpus,gid,s,name)
        cg,b=get_game(corpus,cid,s,name)
        a.update(root=t,victim=v,opposing_worker=h,lookbacks=visible_trace(g,t,s,v,h),
                 official_one_command_forks=branches(g,t,s,v,h))
        lot=next((e for e in lots[gid] if e["t"]==t+1 and e["victim"]==v and e["hunter"]==h),None)
        a["traced_enemy_cargo_lot"]=lot
        control_assets=cg._obs(ct,s)["players"]
        focal_position=control_assets[s][2][cv][0]
        nearest_enemy=min((w for w,row in control_assets[1-s][2].items() if row[2]=="COLLECTOR"),
                          key=lambda w:(DIST[control_assets[1-s][2][w][0]][focal_position],w))
        b.update(root=ct,carrier=cv,normal_deposit=choose_control(cg,ct,s,cv),
                 nearest_opponent_at_root=nearest_enemy,lookbacks=visible_trace(cg,ct,s,cv,nearest_enemy),
                 official_one_command_forks=branches(cg,ct,s,cv))
        if name.endswith("Echo6"):
            if lot is not None:raise ValueError("empty collector recorded as a positive seizure")
            initial=g._obs(20,s)["players"][s];base=next(k for k,pos in initial[1].items() if pos==154)
            ft=forks(g,20,s,v,h,[(s,base,a) for a in ("STAY","RECCOLLECTOR")])
            other_id=site(3,s)
            other_forks=branches(g,21,s,other_id,site(4,1-s))
            last_failure=[totals(g._obs(k,s,terminal=True)["players"])[s] for k in range(24,g.turns+1)]
            assets=("cash","collectors","planters","cargo","trees")
            constant={key:sorted({r[key] for r in last_failure}) for key in assets}
            a["funding_and_extinction"]={"t20_recruiting_base":base,"t20_before":totals(g._obs(20,s)["players"]),
                "t20_actual_base_command":commands(g,20)[s][base],"t20_one_base_change":ft,
                "t21_other_empty_worker":visible_trace(g,21,s,other_id,site(4,1-s)),
                "t21_one_worker_change":other_forks,"t24_through_terminal_stocks":constant,
                "official_end":g.turns,"plantCost":g.metadata()["configuration"]["plantCost"],
                "seizeCost":g.metadata()["configuration"]["seizeCost"],"recCollectorCost":g.metadata()["configuration"]["recCollectorCost"]}
        elif lot is None:raise ValueError("missing physically traced seized cargo")
        result.append(dict(opponent=name,seat=s,master6=a,control=b))
    out={"scope":"Purposive common-name/same-seat case studies, not a prevalence estimate. Only factual continuations have final results. Edited branches stop after one joint transition.",
         "source_sha256":{p.name:sha(p) for p in (Path(__file__),HERE/"context.py",controls,defense,GAME_SOURCE/"zerosum_env/envs/carbon/helpers.py")},
         "cases":result}
    (HERE/"master6_denial_cases.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
