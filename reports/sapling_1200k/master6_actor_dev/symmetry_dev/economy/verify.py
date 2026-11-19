"""Independently account for every official Board balance write and ownership interval."""
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
BASE = DEV.parents[1]
ROOT = BASE.parents[1]
TEN = DEV.parent / "tree_response_training/ten_map_economy"
ENGINE = ROOT / "code/env_src/carbon_challenge-master"
sys.path.insert(0, str(ENGINE))
from zerosum_env import make
from zerosum_env.envs.carbon import helpers, idgen

ACCOUNTS = {"player._cash = max(player._cash - rec_collector_cost, 0)":"hire_collector",
            "player._cash = max(player._cash - rec_planter_cost, 0)":"hire_planter",
            "worker.cell.tree.player._cash += worker.carbon":"tree_collision_cash",
            "worker.cell.recrtCenter.player._cash += worker.carbon":"base_collision_cash",
            "recrtCenter.player._cash += worker._carbon - configuration.smelt_cost":"base_deposit",
            "worker.player._cash = worker.player._cash - player_plant_cost[worker.player_id]":"plant",
            "worker.player._cash -= configuration.seize_cost":"seize",
            "tree.player._cash += tree_carbon":"tree_absorption",
            "player._cash = round(player._cash, 2)":"rounding"}


def read(path):
    blob=path.read_bytes()
    return json.loads(gzip.decompress(blob) if path.suffix==".gz" else blob)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def write(path, value):
    raw=json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    if path.suffix==".gz":
        raw=gzip.compress(raw,mtime=0)
    with tempfile.NamedTemporaryFile(dir=HERE,delete=False) as out:
        out.write(raw)
    Path(out.name).replace(path)


@contextmanager
def ledger():
    code=helpers.Board.next.__code__
    lines,start=inspect.getsourcelines(helpers.Board.next)
    sites={start+i:ACCOUNTS[line.strip()] for i,line in enumerate(lines) if line.strip() in ACCOUNTS}
    check(len(sites)==len(ACCOUNTS),"changed official balance-write sites")
    events=[]
    old=helpers.Player.__setattr__
    def observe(player,name,value):
        if name=="_cash" and hasattr(player,"_cash"):
            f=sys._getframe(1)
            check(f.f_code==code and f.f_lineno in sites,"unmapped writer")
            k=sites[f.f_lineno]
            local=f.f_locals
            wid=None
            pos=None
            tid=None
            if k=="tree_absorption":
                tid=local["tree"].id
            elif k.startswith("hire_"):
                wid=local["recrtCenter"].id
            else:
                worker=local.get("worker")
                if worker is not None and k!="rounding":
                    wid=worker.id
                    pos=worker.position.to_index(local["configuration"].size)
                if k=="tree_collision_cash":
                    tid=worker.cell.tree.id
                if k=="base_deposit":
                    k="own_delivery" if player.id==worker.player_id else "opponent_worker_delivery_to_this_base"
            events.append((k,player.id,float(value-player._cash),wid,pos,tid))
        return old(player,name,value)
    helpers.Player.__setattr__=observe
    try:
        yield events
    finally:
        helpers.Player.__setattr__=old


def cash(obs):
    return [x[0] for x in obs["players"]]


def trees(obs):
    return {tid:(p,tuple(info)) for p,assets in enumerate(obs["players"]) for tid,info in assets[3].items()}


def initial(tape):
    np.random.seed(tape["seed"])
    env=make("carbon", configuration={**tape["game_configuration"],"randomSeed":tape["seed"]})
    obs=dict(env.states[0].observation)
    check(obs["full_carbon"]==tape["initial_full_carbon"],"initial map mismatch")
    obs["carbon"]=obs["full_carbon"]
    idgen.reset()
    return obs


def replay(path,cert,source_ref=None):
    tape=read(path)
    check(sha(path)==cert["transcript_sha256"] and
          all(v==int(v) and 0<=v<256 for v in tape["initial_full_carbon"]) and
          hashlib.sha256(bytes(map(int,tape["initial_full_carbon"]))).hexdigest()==cert["map_sha256"],"tape identity")
    s=cert["seat"]
    state=initial(tape)
    prev_trees=trees(state)
    check(not prev_trees,"unexpected initial tree")
    times=[]; money=[Counter(),Counter()]; phases=[]; holdings=[]; active={}; heads=[]; sparse=[]
    with ledger() as events:
        for t,joint in enumerate(tape["actions"]):
            if t%50==0:
                phase=[Counter(),Counter()]; exposure=[Counter(),Counter()]
            if source_ref is None:
                heads.append({"cash":cash(state),"workers":[a[2] for a in state["players"]],
                              "bases":[a[1] for a in state["players"]],"trees":[a[3] for a in state["players"]],
                              "joint_actions":joint})
            for owner,_ in prev_trees.values():
                exposure[owner]["tree_turns"]+=1
            for owner,assets in enumerate(state["players"]):
                for worker in assets[2].values():
                    exposure[owner]["collector_turns" if worker[2]=="COLLECTOR" else "planter_turns"]+=1
            events.clear()
            before=cash(state)
            state=helpers.Board(state,tape["game_configuration"],joint).next().observation
            post=trees(state)
            payments={}; after=cash(state);step_cash=[0.,0.]
            for k,p,v,w,pos,tid in events:
                money[p][k]+=v;phase[p][k]+=v;step_cash[p]+=v
                if k in ("plant","seize"):
                    payments[p,pos]=(k,w,-v)
                elif k=="tree_collision_cash" and v:
                    active[tid]["collision"]+=v
            check(all(abs(after[p]-before[p]-step_cash[p])<1e-5 for p in (0,1)),"official cash mismatch")
            for tid,(owner,_) in prev_trees.items():
                if tid not in post or post[tid][0]!=owner:
                    h=active.pop(tid)
                    h.update(end=t+1,exit="seized_by_opponent" if tid in post else "expired")
                    holdings.append(h)
            for tid,(owner,(pos,age)) in post.items():
                if tid not in active:
                    k,w,price=payments.pop((owner,pos))
                    active[tid]=dict(tid=tid,pos=pos,owner=owner,start=t+1,age_at_entry=age,origin=k,
                                     worker=w,price=price,gross=0.,collision=0.)
            check(not payments,"unbound payment")
            for tid,h in active.items():
                h["gross"]+=state["trees"][tid][1]
            if source_ref is None:
                sparse.append([e for e in events if e[0] not in ("rounding","tree_absorption") and abs(e[2])>1e-10])
            prev_trees=post
            times.append(after[s]-after[1-s])
            if (t+1)%50==0 or t+1==len(tape["actions"]):
                phases.append({"post":t+1,"accounts":[dict(a) for a in phase],
                               "turn_exposure":[dict(a) for a in exposure],"cash_lead":times[-1]})
    for h in active.values():
        h.update(end=len(times),exit="terminal")
        holdings.append(h)
    check(after==cert["official_reward_by_seat"]==tape["last_post_cash"] and times[-1]==tape["margin"],"terminal mismatch")
    result={"seed":tape["seed"],"seat":s,"pack_sha256":tape["pack_sha256"],"transcript_sha256":sha(path),
            "map_sha256":cert["map_sha256"],"steps":len(times),"physical_final_cash":after,
            "margin":times[-1],"physical_cash_channels":[dict(a) for a in money],"phases":phases,
            "holdings":holdings}
    if source_ref:
        original,tenures=source_ref
        for p,q in zip(original["phases"],phases,strict=True):
            check(p["post"]==q["post"] and abs(p["cash_lead"]-q["cash_lead"])<1e-5,"phase cash")
            for seat in (0,1):
                for k in set(p["accounts"][seat])|set(q["accounts"][seat]):
                    check(abs(p["accounts"][seat].get(k,0)-q["accounts"][seat].get(k,0))<1e-5,"account "+k)
                check(p["turn_exposure"][seat]==q["turn_exposure"][seat],"exposure")
        by={(h["tid"],h["start"]):h for h in holdings}
        check(len(by)==len(tenures),"different number of tree tenures")
        for original_hold in tenures:
            h=by[original_hold["tid"],original_hold["start"]]
            for k in ("pos","owner","end","exit","worker","origin","age_at_entry"):
                check(original_hold[k]==h[k],"different tree history: "+k)
            for k in ("price","gross","collision"):
                check(abs(original_hold[k]-h[k])<1e-5,"different tree earnings: "+k)
            check(original_hold["preturns"]==h["end"]-h["start"],"different duration")
    else:
        before=[0.]+times[:-1]
        t=max(range(len(times)),key=lambda i:abs(times[i]-before[i]))
        result["largest_cash_jump"]={"pre":t, "post":t+1,"own_minus_enemy_gain":times[t]-before[t],
                     "states":[dict(pre=k,**heads[k]) for k in sorted({max(0,t-d) for d in (10,5,3,1,0)})],
                     "cash_events":sparse[t]}
        result["step_leads"]=times
    return result


def main():
    old=read(TEN/"games.json.gz");old_rows={(x["seed"],x["seat"]):x for x in old["games"]}
    tenure={(x["seed"],x["seat"]):x["holdings"] for x in read(TEN/"tenures.json.gz")}
    training=[]
    for f,h in old["source_files"].items():
        source=ROOT/f
        check(sha(source)==h,"source receipt changed")
        r=read(source)
        certs=read(source.parent/"source/official_replay.json")["runs"]["source"]["games"]
        for c in certs:
            key=c["seed"],c["seat"]
            game=next(g for g in r["games"] if (g["seed"],g["seat"])==key)
            result=replay(ROOT/game["raw_path"],c,(old_rows[key],tenure[key]))
            training.append({"seed":key[0],"seat":key[1],"steps":result["steps"],"holdings":len(result["holdings"]),
                             "transcript_sha256":result["transcript_sha256"]})
    dev={}
    for arm in ("parent","candidate"):
        certs=read(DEV/arm/"official_replay.json")["runs"][arm]["games"]
        dev[arm]=[replay(DEV/arm/f"raw_{c['seed']}_{c['seat']}.json.gz",c) for c in certs]
    check(len(training)==20 and all(len(games)==4 for games in dev.values()),"incomplete")
    data={"source":{"official_helpers_sha256":sha(Path(helpers.__file__)),"training_games_sha256":sha(TEN/"games.json.gz"),
                     "training_tenures_sha256":sha(TEN/"tenures.json.gz"),"comparison_sha256":sha(DEV/"comparison.json"),
                     "implementation_sha256":sha(Path(__file__))},"training_independent_checks":training,"development":dev}
    write(HERE/"official_accounts.json.gz",data)
    print("verified",sum(g["steps"] for g in training),sum(g["holdings"] for g in training),sum(g["steps"] for a in dev.values() for g in a))


if __name__=="__main__":
    main()
