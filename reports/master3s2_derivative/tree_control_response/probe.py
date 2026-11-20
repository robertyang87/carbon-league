"""Test public native competition and its mechanistic response to one enemy tree defender."""
import copy,gzip,hashlib,importlib.util,json,os,random,sys
from pathlib import Path

from build import build,sha,HERE
D=HERE.parent;repo=D.parents[1]
sys.path[:0]=[str(D/"opponent_policy"),str(repo/"reports/sapling_1200k/all_top5")]
from original_policy import module
from context import Game
spec=importlib.util.spec_from_file_location("tc_official",D/"freight_response/run.py")
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
COST=[0.004014796040266934,0.0007784288891318335,0.003623260498396121,0.000797088498075027]
def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix==".gz" else raw)

def unique_seat_target(env,side,target):
    return next((s for s,p in enumerate(env.states[0].observation.players) if any(v[0]==target for v in p[3].values())),None)

def local_branches(env,sim,tape,t,seat,plan,candidate):
    env.steps[:-1]=[None]*t;env.logs=[]
    archive=ref.official_state.take(env,[0,0])
    enemy=1-seat
    def run(key,first,opposition):
        ref.official_state.restore(env,archive);s=copy.deepcopy(sim);logs=[]
        src=plan["cell"];spot=plan["target"];uid=plan["worker"];defender=None
        for j in range(4):
            states=env.states[0].observation.players;mine=states[seat];foe=states[enemy]
            moves=[dict((k,v) for k,v in z["action"].items() if k in p[1] or k in p[2]) for p,z in zip(states,tape[t+j+1])]
            if j==0:moves[seat]=dict(first)
            else:
                worker=mine[2].get(uid)
                if worker and key in ("pursue","seize"):
                    pos=worker[0];opts=["NONE"] if pos==spot else list(candidate._ACTIONS)
                    a=min(opts,key=lambda a:candidate._TREE_LIB["distance"](candidate._tree_dest(pos,a,15),spot,15))
                    if a=="NONE":moves[seat].pop(uid,None)
                    else:moves[seat][uid]=a
            if opposition and j>=1:
                # Foe sees the first intervention's public successor. Re-task a surviving planter at that time.
                if not defender:
                    xs=[(candidate._TREE_LIB["distance"](w[0],spot,15),who) for who,w in foe[2].items() if w[2]=="PLANTER"]
                    defender=min(xs)[1] if xs else None
                worker=foe[2].get(defender)
                if worker:
                    opts=["NONE"] if worker[0]==spot else list(candidate._ACTIONS)
                    a=min(opts,key=lambda a:candidate._TREE_LIB["distance"](candidate._tree_dest(worker[0],a,15),spot,15))
                    if a=="NONE":moves[enemy].pop(defender,None)
                    else:moves[enemy][defender]=a
            s.cash.events.clear();s.step(*moves);env.step(moves,[])
            o=env.states[0].observation;back=s.obs(0)
            need(o.players==back["players"] and o.trees==back["trees"] and o.full_carbon==s.carbon,"Branch rules")
            logs.append(dict(state=t+j+1,wallet=[p[0] for p in o.players],target_owner=unique_seat_target(env,seat,spot),
                             plant_cell_owner=unique_seat_target(env,seat,src),actor_after=o.players[seat][2].get(uid),defender=defender,
                             defender_after=o.players[enemy][2].get(defender),events=[e for e in s.cash.events if e[0] in ("plant","seize")]))
            if env.done:break
        return logs
    # Physical commands that coincide at the first step can have a different declared continuation intention.
    results={k:{v:run(k,a,v=="enemy_defends") for v in ("recorded_others","enemy_defends")} for k,a in plan["roots"].items()}
    ref.official_state.restore(env,archive)
    return results


def capture_one(path,seat,label,requested,models):
    raw=path.read_bytes();game=json.loads(gzip.decompress(raw));tape=game["steps"];cfg=game["configuration"];public=Game(game)
    terrain=sha(bytes(tape[0][0]["observation"]["full_carbon"]));seed=cfg.get("randomSeed",0)
    parent,candidate=models;tracker=parent.BeliefTracker(parent._Bel,seat)
    random.seed(seed);ref.np.random.seed(seed);env=ref.make("carbon",configuration=cfg)
    start_carbon=tape[0][0]["observation"]["full_carbon"];env.states[0].observation.full_carbon=copy.deepcopy(start_carbon)
    sim=ref.econ.FastCarbon(cfg).reset(full_carbon=start_carbon);sim.cash=ref.econ.CashEvents(sim.cash)
    hist={};prev=None;roots=[];next_allowed=30;cfg_for_agent={k:v for k,v in cfg.items() if k!="randomSeed"}
    for t in range(min(len(tape)-5,251)):
        obs=public._obs(t,seat);ps=obs["players"]
        need(ps==env.states[0].observation.players and (sim.obs(0)["players"]==ps),"Historical physical mismatch")
        near=any(x[2]=="PLANTER" and any(candidate._TREE_LIB["distance"](x[0],z[0],15)<=2 for z in ps[1-seat][3].values()) for x in ps[seat][2].values())
        do=t in requested or (t>=next_allowed and sum(r["selected_by"]=="first_eligible_spacing25" for r in roots)<3 and near)
        if do:
            priv=int(sha((path.name+"/"+str(t)+"/tree-v1").encode())[:8],16)
            past={k:hist[t-k] for k in (3,5) if t-k in hist}
            def prep(model):
                model._ST=dict(tr=copy.deepcopy(tracker),prev=copy.deepcopy(prev),rng=ref.np.random.RandomState(priv));model._S_COST[:]=COST
                if hasattr(model,"_TREE_HIST"):model._TREE_HIST={seat:{t-k:v for k,v in past.items()}};model._TREE_LAST=None
            prep(parent);srch=parent._search
            parent._search=lambda obs,cfg,est,act,*args:act
            try:base=parent.agent(obs,cfg_for_agent)
            finally:parent._search=srch
            inspection=candidate._tree_assess(obs,cfg_for_agent,base,past)
            if inspection or t in requested:
                prep(parent);parent_command,pc=ref.measured(parent,obs,cfg_for_agent)
                prep(candidate);candidate_command,cc=ref.measured(candidate,obs,cfg_for_agent)
                rr=candidate._TREE_LAST
                row=dict(t=t,selected_by="retrospective_N34_anchor" if t in requested else "first_eligible_spacing25",p_rng=priv,
                         parent=parent_command,candidate=candidate_command,parent_clock=pc,candidate_clock=cc,command_changed=candidate_command!=parent_command)
                if rr:
                    p=rr["plan"];proposed=p["roots"];new=rr["new"]
                    assert inspection==p and rr["before"][0]==base
                    row.update(plan=p,new_to_native_menu=new,menu_before=rr["before"],menu_after=rr["after"],
                               selected_variants=[k for k,a in proposed.items() if a==candidate_command],
                               selected_new_variants=[k for k in new if proposed[k]==candidate_command],economy=local_branches(env,sim,tape,t,seat,p,candidate))
                roots.append(row)
                if t not in requested:next_allowed=t+25
        actual=[v["action"] for v in tape[t+1]]
        tracker.step(ps,t,obs["carbon"]);tracker.set_actions(ps,actual[seat]);prev=ps;hist[t]=ps
        if t-6 in hist:del hist[t-6]
        sim.cash.events.clear();sim.step(*actual);env.step(actual,[])
    return dict(archive=path.name,archive_sha256=sha(raw),terrain=terrain,seat=seat,group=label,roots=roots)


def main():
    import torch
    os.environ["CARBON_DEBUG"]="1";torch.set_num_threads(1);torch.set_num_interop_threads(1)
    base=(D/"cold_start/candidate.py").read_bytes();pkg=build()
    parent,candidate=module(base),module(pkg);parent._net();candidate._net()
    ladder=D/"ladder/replays";cases=[
        (ladder/"eec452f455f16f82bb432ce057daa25c-1.json.gz",1,"N34",[137,142]),
        (ladder/"bdbe59b04f71e43e53696874c8fd5958-1.json.gz",1,"Fable_V7",[]),
        (ladder/"3f0a1dfa0e1ac4a00a9f63f9cf644af9-1.json.gz",1,"Hajimi_v3pro",[])]
    maps=read(D/"ladder/tree_payback/maps.json")["tickets"]
    for i,ticket in enumerate([c for c in maps if c["split"]=="held"][:2]):
        p=Path("/workspace/sapling-all-top5-20261119/replays")/(ticket["id"]+".json.gz")
        need(sha(p.read_bytes())==ticket["compressed_sha256"],"Held source");cases.append((p,i,"held_top5",[]))
    result=dict(code=sha(Path(__file__).read_bytes()),parent_sha=sha(base),candidate_sha=sha(pkg),candidate_bytes=len(pkg),cases=[])
    for p,s,k,requested in cases:
        g=capture_one(p,s,k,requested,(parent,candidate));result["cases"].append(g)
        print(k,[(r["t"],r["command_changed"],r.get("selected_new_variants"),r.get("new_to_native_menu")) for r in g["roots"]],flush=True)
        (HERE/"probe.json").write_text(json.dumps(result,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
if __name__=="__main__":main()
