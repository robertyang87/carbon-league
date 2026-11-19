"""Lock two natural divergences before any independent examiner reply, then play both arms."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import inspect
import os
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parent
A=D.parent
ROOT=A.parents[2]
sys.path.insert(0,str(A/"elite_role_mix_dev"/"conditional"))
import responses as R
P={"parent":ROOT/"data/weights/sapling_1200k/master6_actor_v1.py","student":D/"student.py"}


def idle(mod,obs):
    state=inspect.getclosurevars(mod._AG).nonlocals["st"]
    assert set(state)=={"f"}
    state["f"].observe(obs)


def select():
    manifest=R.load(D/"manifest.json")
    fit=R.load(D/"fit.json")
    e=R.load(D/"entry.json.gz")
    assert fit["common_prefix_pass"] and e["manifest_sha256"]==R.sha(D/"manifest.json")
    assert all(e["packs"][name]==R.sha(file) for name,file in P.items())
    assert all(g["totals"]["student_vs_parent"]["joint_changed"]>0 for g in e["samples"] if g["primary"])
    cards=sorted((z for z in manifest["records"] if z["split"]=="held" and z["domain"]=="sapling_visited"),
                 key=lambda c:(c["terrain"],c["id"]))
    first=cards[0]
    more=[c for c in cards if c["terrain"]!=first["terrain"]]
    second=next((c for c in more if c["seat"]!=first["seat"]),more[0])
    cases=[]
    for c in [first,second]:
        source=ROOT/c["source"]
        assert R.sha(source)==c["source_sha256"]
        r=R.load(source); s=c["seat"]; cfg=r["game_configuration"]
        assert r["seed"]==c["seed"] and s==r["candidate_seat"] and r["initial_terrain_u8_sha256"]==c["terrain"] and not r["errors"]
        sim=R.FastCarbon(cfg).reset(seed=c["seed"])
        packs={n:R.live.module(path,"pre_"+n) for n,path in P.items()}
        found=None
        for t,joint in enumerate(r["actions"][:286]):
            obs=sim.obs(s)
            if t==0 or t>=270:
                proposals={n:R.live.clean(m.agent(obs,cfg)) for n,m in packs.items()}
                if t>=270 and proposals["student"]!=proposals["parent"]:
                    found=dict(t=t,proposals=proposals)
                    break
            else:
                for m in packs.values():idle(m,obs)
            sim.step(*joint)
        cases.append(dict(id=c["id"],terrain=c["terrain"],seed=c["seed"],seat=s,
                          source=c["source"],source_sha256=c["source_sha256"],root=found))
    R.write(HERE/"roots.json",dict(cases=cases,entry=R.sha(D/"entry.json.gz"),protocol=R.sha(HERE/"protocol.json"),
            runner=R.sha(Path(__file__)),manifest=R.sha(D/"manifest.json"),packs={n:R.sha(p) for n,p in P.items()}))


def run_arm(job):
    case,arm,cpu,warm=job
    os.sched_setaffinity(0,{cpu})
    s=case["seat"]; op=1-s; t0=case["root"]["t"]
    f=HERE/f"{case['id']}_{arm}.json.gz"
    bind=dict(roots=R.sha(HERE/"roots.json"),arm=arm,cpu=cpu,warm=warm)
    if f.exists():
        out=R.load(f)
        assert out["binding"]==bind and out["statuses"]==["DONE","DONE"]
        return out
    r=R.load(ROOT/case["source"]);cfg=r["game_configuration"]
    assert R.sha(ROOT/case["source"])==case["source_sha256"]
    env=R.live.begin(r)
    parent=R.live.module(P["parent"],"response_parent")
    master=R.live.module(R.live.SOURCE,"response_original")
    assert R.sha(R.live.SOURCE)==R.live.MASTER_SHA and master._SEED is None and master._TEMP==1 and master._RX and master._SEARCH_OK and master._S_ABORT==.58
    tr=master.BeliefTracker(master._Bel,op)
    prev=None
    for t,joint in enumerate(r["actions"][:t0]):
        view=R.live.public(env,s); other=R.live.public(env,op)
        tr.step(other["players"],t,other["carbon"])
        tr.set_actions(other["players"],joint[op]);prev=other["players"]
        if t==0:parent.agent(view,cfg)
        else:idle(parent,view)
        env.step(joint)
    rng=int(hashlib.sha256(f"source-prior-native|{case['terrain']}|{s}|{t0}".encode()).hexdigest()[:8],16)
    master._S_COST[:]=warm
    master._ST.clear()
    master._ST.update(tr=tr,prev=prev,rng=np.random.RandomState(rng))
    recorder=R.native_clock.Recorder(master); opp=recorder.wrap(master.agent)
    tape=[]
    while not env.done:
        own=R.live.clean(parent.agent(R.live.public(env,s),cfg))
        if not tape:
            assert own==case["root"]["proposals"]["parent"]
            own=case["root"]["proposals"][arm]
        joint=[None,None]; joint[s]=own; joint[op]=R.live.clean(opp(R.live.public(env,op),cfg))
        tape.append(joint);env.step(joint)
    status=[v.status for v in env.states]
    assert status==["DONE","DONE"] and len(tape)==len(recorder.calls)>0 and not recorder.errors
    scores=[v.reward for v in env.states]
    data=dict(binding=bind,statuses=status,score=scores,margin=scores[s]-scores[op],rng=rng,tape=tape,
         native_clock=R.native_clock.summarize(recorder.calls,master._S_ABORT,master._S_D2),
         calls=recorder.calls,cash=R.cash_summary(r,s,t0,tape,[p[0] for p in env.states[0].observation.players]))
    R.write(f,data)
    return data


def run():
    roots=R.load(HERE/"roots.json")
    assert roots["protocol"]==R.sha(HERE/"protocol.json") and roots["runner"]==R.sha(Path(__file__)) and all(roots["packs"][n]==R.sha(f) for n,f in P.items())
    result=[]
    if all(c["root"] for c in roots["cases"]):
        with R.LOCK.open("a") as slot:
            fcntl.flock(slot,fcntl.LOCK_EX)
            for i,c in enumerate(roots["cases"]):
                warm_file=HERE/f"warm_{c['id']}.json"
                if not warm_file.exists():
                    with ProcessPoolExecutor(1) as pool:
                        warm=pool.submit(R.native_warm).result()
                    R.write(warm_file,dict(root=R.sha(HERE/"roots.json"),cost=warm))
                warm=R.load(warm_file)
                assert warm["root"]==R.sha(HERE/"roots.json")
                with ProcessPoolExecutor(2) as pool:
                    a,b=pool.map(run_arm,[(c,"parent",i,warm["cost"]),(c,"student",1-i,warm["cost"])])
                assert a["rng"]==b["rng"] and a["binding"]["warm"]==b["binding"]["warm"]
                result.append(dict(id=c["id"],t=c["root"]["t"],seat=c["seat"],parent=a["margin"],student=b["margin"],
                    delta=b["margin"]-a["margin"],original_root_action_identical=a["tape"][0][1-c["seat"]]==b["tape"][0][1-c["seat"]],
                    opponent_order_differences=sum(j[1-c["seat"]]!=k[1-c["seat"]] for j,k in zip(a["tape"],b["tape"])),
                    original_calls=[len(a["calls"]),len(b["calls"])]))
    R.write(HERE/"result.json",dict(roots=R.sha(HERE/"roots.json"),rows=result,
                both_strictly_positive=len(result)==2 and all(z["delta"]>0 for z in result)))
    print(result,flush=True)


if __name__=="__main__":
    if sys.argv[1:] == ["select"]:select()
    elif sys.argv[1:] == ["run"]:run()
    else:raise SystemExit("choose select or run")
