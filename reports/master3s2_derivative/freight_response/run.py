"""Run both exploratory suffixes from each preregistered public root."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = D.parents[1]
P = json.loads((HERE / "protocol.json").read_bytes())
STUDY = ROOT / "reports/sapling_1200k/master6_actor_dev"
LEAGUE = STUDY / "cargo_tree_action_learning_dev/native_actor_corpus/league_policy"
sys.path[:0] = [str(LEAGUE/"native_return_intake/live_curriculum/inprocess_snapshot"), str(D/"opponent_policy")]
import original_state
import official_state


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


econ=load("freight_account", LEAGUE / "natural_original/economy.py")
from zerosum_env import make
import numpy as np


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(name,value):
    dest = HERE / name
    raw=(json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    if dest.suffix==".gz":
        raw=gzip.compress(raw,mtime=0)
    temporary=dest.with_suffix(".partial")
    temporary.write_bytes(raw)
    temporary.replace(dest)


def public(env,s,bank):
    shared=env.states[0].observation
    return dict(step=shared.step,player=s,carbon=env.states[s].observation.carbon,
                players=shared.players,trees=shared.trees,remainingOverageTime=bank[s])


def digest(obs):
    return sha((json.dumps(obs,separators=(",",":"),allow_nan=False)+"\n").encode())


def measured(mod,obs,cfg):
    stats=mod._S_STATS
    before={k:stats[k] for k in ("searched","single","aborted","changed")}
    prev=stats.pop("last",None)
    wall,cpu=time.perf_counter(),time.process_time()
    try:
        act=mod.agent(obs,cfg)
    finally:
        wall,cpu=time.perf_counter()-wall,time.process_time()-cpu
    last=stats.get("last")
    if last is None and prev is not None:
        stats["last"]=prev
    return act,dict(wall=wall,cpu=cpu,K=1 if last is None else int(last[0]),
                    H=0 if last is None else int(last[1]),rolls=0 if last is None else int(last[2]),
                    cost=list(mod._S_COST),**{k:stats[k]-v for k,v in before.items()})


def root_call(mod, obs, cfg):
    menu=[]
    def capture(code,offset,ret):
        values=sys._getframe(1).f_locals
        menu.extend(copy.deepcopy(values.get("cands",[ret])))
    mon=sys.monitoring
    tid=4
    mon.use_tool_id(tid,"native search menu")
    try:
        mon.register_callback(tid,mon.events.PY_RETURN,capture)
        mon.set_local_events(tid,mod._search.__code__,mon.events.PY_RETURN)
        res=measured(mod,obs,cfg)
    finally:
        mon.set_local_events(tid,mod._search.__code__,0)
        mon.register_callback(tid,mon.events.PY_RETURN,None)
        mon.free_tool_id(tid)
    check(res[0] in menu,"missing authorized native menu")
    return res,menu


def coverage(mod,obs,act):
    s=obs["player"]
    players=obs["players"]
    own=players[s][2]
    foe=players[1-s]
    bases=set(foe[1].values())
    scored=[]
    for uid,(pos,freight,role) in foe[2].items():
        if role!="COLLECTOR" or freight<=0:continue
        d=min(mod.dist(pos,b) for b in bases)
        possible={pos} | {target for k in range(1,5) if min(mod.dist(target:=mod.move(pos,k),b) for b in bases)<d}
        covered=set()
        for wid,(origin,weight,kind) in own.items():
            if kind!="COLLECTOR" or weight>=freight:continue
            target=mod.move(origin,mod.WIDX[act.get(wid) or "NONE"])
            if target in possible-bases:covered.add(target)
        if covered:
            scored.append(dict(enemy=uid,cargo=freight,possible=sorted(possible),covered=sorted(covered),
                               score=freight*len(covered)/len(possible)))
    return sum(x["score"] for x in scored),scored


def generate_root(mod,obs,cfg):
    original,menu=root_call(mod,obs,cfg)
    default=original[0]; rows=[]
    for i,cmd in enumerate(menu):
        value,reasons=coverage(mod,obs,cmd)
        changes=sum(default.get(k)!=cmd.get(k) for k in default.keys()|cmd.keys())
        rows.append(dict(index=i,score=value,changed=changes,coverage=reasons,command=cmd))
    base=coverage(mod,obs,default)[0];best=max(rows,key=lambda x:(x["score"],-x["changed"],-x["index"]))
    return original,rows,best if best["score"]>base else None


def run(ticket):
    seed,seat,t=ticket["seed"],ticket["seat"],ticket["t"]
    stem=f"{seed}_{seat}_{t}"
    bookbytes=(D/"dev1/hosted/settlement.json").read_bytes()
    check(sha(bookbytes)==P["historical_settlement_sha256"],"wrong host records")
    book=json.loads(bookbytes)
    game=next(g for g in book["games"] if (g["seed"],g["seat"])==(seed,seat))
    archive=D/"dev1/hosted"/game["archive_path"]
    arc=next(a for a in book["archives"] if a["id"]==game["archive_id"])
    check(sha(archive.read_bytes())==arc["sha256"],"archive mismatch")
    with zipfile.ZipFile(archive) as z:
        member="opponent_policy/"
        turns=z.read(member+"turns.jsonl")
        r=json.loads(z.read(member+"receipt.json"))
        check(sha(turns)==r["turns_sha256"] and game["status"]=="complete","invalid wire orders")
        req=z.read(member+"observations.jsonl.gz")
        check(sha(req)==game["public_observations_sha256"],"wrong wire observations")
    # Never parse the root or suffix orders while building the opportunity or history.
    past=[json.loads(x) for x in turns.splitlines()[:t]]
    frames=[json.loads(x) for x in gzip.decompress(req).splitlines()[:t+1]]
    cfg=econ.config()
    cfg.pop("randomSeed",None)
    np.random.seed(seed)
    env=make("carbon",configuration={**cfg,"randomSeed":seed})
    sim=econ.FastCarbon(cfg).reset(seed=seed)
    check(sha(bytes(env.states[0].observation.full_carbon))==game["terrain"]==sha(bytes(map(int,sim.carbon))),"terrain")
    paths=[None,None]
    paths[seat]=D/"cold_start/candidate.py"
    paths[1-seat]=ROOT/"data/master3s2_source.py"
    check(sha(paths[seat].read_bytes())==P["candidate_v2"] and sha(paths[1-seat].read_bytes())==P["original_sha256"],"pack bytes")
    mods=[load("freight_s"+str(s),p) for s,p in enumerate(paths)]
    for m in mods:
        check(m._TEMP==1. and m._SEED is None and m._S_ABORT==.58 and m._RX and m._SEARCH_OK,"native entry configuration")
    sim.cash=econ.CashEvents(sim.cash)
    costs=[]; verified=[]; bank=[5.,5.]
    for i,frame in enumerate(frames):
        obs=[public(env,s,bank) for s in (0,1)]
        hashed=[digest(o) for o in obs]
        check(hashed==frame["public_observation_sha256"] and obs==frame["observations"],f"public replay drift at {i}")
        check(sim.cash==[o[0] for o in obs[0]["players"]],"accounting drift")
        verified.append(hashed)
        if i==t:break
        acts=[q["response"]["action"] for q in past[i]["calls"]]
        check(past[i]["public_observation_sha256"]==hashed,"historical order binding")
        for s,m in enumerate(mods):
            players=obs[s]["players"]
            if i<3:
                _,c=measured(m,obs[s],cfg);costs.append(dict(step=i,seat=s,clock=c))
                if i==0:
                    m._ST["rng"]=np.random.RandomState((seed+37*s)%2**32)
                    m._ST["rng"].random_sample(len(players[s][1])+len(players[s][2]))
            else:
                m._ST["tr"].step(players,i,obs[s]["carbon"])
                m._ST["rng"].random_sample(len(players[s][1])+len(players[s][2]))
            m._ST["tr"].set_actions(players,acts[s]);m._ST["prev"]=players
        env.step(acts,[]);sim.step(*acts)
        bank=past[i]["overage_remaining_seconds"]
    stamp=original_state.h([m._ST for m in mods])
    our,menu,alt=generate_root(mods[seat],obs[seat],cfg)
    selected=dict(seed=seed,t=t,seat=seat,source_archive=arc["sha256"],terrain=game["terrain"],
                  protocol_sha256=sha((HERE/"protocol.json").read_bytes()),historical_request_digests=verified,
                  native_prefix_state=stamp,calibration=costs,root_public=obs[seat],default=our,menu=menu,alternative=alt,
                  affinity=sorted(os.sched_getaffinity(0)),cgroup={k:Path("/sys/fs/cgroup",k).read_text().strip() for k in ("cpu.max","memory.max")})
    put(stem+"_root.json.gz",selected)
    print(seed, "candidates",len(menu),"deviation",bool(alt),flush=True)
    if not alt:return
    # Both branch policies see agreement at the same simultaneous-move boundary.
    other=measured(mods[1-seat],obs[1-seat],cfg)
    match=official_state.take(env,bank); shadow=copy.deepcopy(sim)
    snapshots=[copy.deepcopy(original_state.dynamic(m)) for m in mods]
    cfg_memos=[(m._RX_CFG[0],copy.deepcopy(m._RX_CFG[1])) for m in mods]
    for arm in ticket["arm_order"]:
        bank=official_state.restore(env,match)
        sim=copy.deepcopy(shadow)
        for s,m in enumerate(mods):
            for k,v in copy.deepcopy(snapshots[s]).items():setattr(m,k,v)
            m._RX_CFG[:]=[cfg_memos[s][0],copy.deepcopy(cfg_memos[s][1])]
            check(original_state.h(original_state.dynamic(m))==original_state.h(snapshots[s]),"unequal post-call native states")
        cmd=our[0] if arm=="default" else alt["command"]
        mods[seat]._ST["tr"].set_actions(env.states[0].observation.players,cmd)
        rows=[]
        while not env.done:
            views=[public(env,s,bank) for s in (0,1)]
            i=views[0]["step"]
            if i==t:
                pair=[None,None];pair[seat]=(cmd,our[1]);pair[1-seat]=other
            else:
                pair=[measured(mods[s],views[s],cfg) for s in (0,1)]
            orders=[v[0] for v in pair]
            clocks=[v[1] for v in pair]
            bank=[b-max(0,c["wall"]-1.) for b,c in zip(bank,clocks)]
            sim.cash.events.clear()
            sim.step(*orders);env.step(orders,[])
            next_cash=[v[0] for v in env.states[0].observation.players]
            check(next_cash==sim.cash,"official cash divergence")
            deposits=[sum(ev[2] for ev in sim.cash.events if ev[0]=="own_delivery" and ev[1]==s) for s in (0,1)]
            rows.append(dict(t=i,public_sha256=[digest(v) for v in views],orders=orders,clocks=clocks,
                             end_cash=next_cash,delivered=deposits))
        scores=[s.reward for s in env.states]
        result=dict(seed=seed,seat=seat,arm=arm,root=t,records=rows,scores=scores,official_margin=scores[seat]-scores[1-seat],
                    deliveries10=[sum(r["delivered"][s] for r in rows[:10]) for s in (0,1)],
                    cpu_affinity=sorted(os.sched_getaffinity(0)),status=[s.status for s in env.states],
                    bank_after=bank)
        put(stem+"_"+arm+".json.gz",result)
        print(seed,arm,scores,result["deliveries10"],flush=True)


if __name__=="__main__":
    os.environ["CARBON_DEBUG"]="1"
    check(os.sched_getaffinity(0)=={2},"run with taskset -c 2")
    for ticket in P["roots"]:
        run(ticket)
