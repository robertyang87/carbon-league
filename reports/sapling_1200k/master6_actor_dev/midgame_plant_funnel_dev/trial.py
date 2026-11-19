"""Freeze site roots and use registered live opponent/parent continuations without staging."""
from pathlib import Path
import fcntl
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
sys.path[:0]=[str(ACTOR/"tree_ground_advantage_dev"),str(ACTOR/"tactical_exchange_dev"),str(ACTOR/"capture_pressure_branches")]
import compare as runner
from decision import update_memory
from route import approach,pay
sr=runner.sr;live=runner.live;PACK=ACTOR/"planter_capture_full_dev";MODEL=ACTOR/"tree_ground_advantage_dev/weights.json"
W=sr.load(MODEL)["coef"]

def prepare():
    prog=sr.load(HERE/"site_protocol.json")
    if sr.sha(PACK/"candidate.py")!=prog["source_hash"]:raise ValueError("parent changed")
    considered=[];selection=[]
    for g in sr.load(ACTOR/"tactical_exchange_dev/paired.json.gz"):
        if g["seed"] not in prog["holdout_seeds"]:continue
        p=sr.ROOT/g["path"];raw=sr.load(p);s=g["seat"]
        if sr.sha(p)!=g["source_sha256"]:raise ValueError("source changed")
        if raw["pack_sha256"]!=prog["source_hash"]:continue
        cfg=raw["game_configuration"];sim=sr.FastCarbon(cfg).reset(seed=raw["seed"]);mem={};hits=[]
        for t,pair in enumerate(raw["actions"][:200]):
            obs=sim.obs(s);update_memory(obs,mem)
            root=approach(obs,mem,pair[s],cfg,W)
            if root:hits.append(root)
            sim.step(*pair)
        considered.append(dict(path=g["path"],sha=g["source_sha256"],seed=g["seed"],seat=s,hits=hits))
    counts=[(g["seed"],g["seat"],len(g["hits"])) for g in considered]
    assert len(considered)==4
    sr.retain(HERE/"approaches.json.gz",considered)
    used=set()
    for seat in (0,1):
        pick=next((g for g in sorted(considered,key=lambda x:(x["seed"],x["path"])) if g["seat"]==seat and g["seed"] not in used and g["hits"]),None)
        if not pick:break
        used.add(pick["seed"]);root=pick["hits"][0];p=sr.ROOT/pick["path"]
        r=sr.load(p.with_name(p.name.replace("raw_","receipt_").replace(".json.gz",".json")))
        if r["game"]["raw_sha256"]!=pick["sha"]:raise ValueError("official receipt disagrees")
        out=HERE/f"seat{seat}";out.mkdir(exist_ok=True)
        sr.retain(out/"official_replay.json",dict(runs={out.name:dict(games=[r["official"]])}))
        q=dict(seed=pick["seed"],seat=seat,root_t=root["t"],family="adjacent_site_purchase",source_dir=str(out.relative_to(sr.ROOT)),
               record=dict(seed=pick["seed"],seat=seat,source=pick["path"],source_sha256=pick["sha"]),root=root,
               pack_sha256=prog["source_hash"],route_sha=sr.sha(HERE/"route.py"),model_sha=sr.sha(MODEL),
               scorer_sha=sr.sha(MODEL.with_name("decision.py")),gate_sha=sr.sha(HERE/"site_protocol.json"))
        sr.retain(out/"live_protocol.json",q);live.prepare(out);selection.append(q)
    sr.retain(HERE/"selected_sites.json",selection);print(counts,[(q["seed"],q["seat"],q["root_t"]) for q in selection])

def play(s,arm):
    runner.HERE=HERE;runner.PRIOR=PACK
    q=sr.load(HERE/f"seat{s}/live_protocol.json");root=q["root"];r=q["record"]
    if any(sr.sha(f)!=q[k] for f,k in ((HERE/"route.py","route_sha"),(MODEL,"model_sha"),(MODEL.with_name("decision.py"),"scorer_sha"),(HERE/"site_protocol.json","gate_sha"))):raise ValueError("changed intervention")
    raw=sr.load(sr.ROOT/r["source"]);cfg=raw["game_configuration"];sim=sr.FastCarbon(cfg).reset(seed=q["seed"]);mem={}
    for t,pair in enumerate(raw["actions"][:root["t"]+1]):
        obs=sim.obs(s);update_memory(obs,mem)
        if t==root["t"] and approach(obs,mem,pair[s],cfg,W)!=root:raise ValueError("selection changed in replay")
        sim.step(*pair)
    original=live.module
    def patched(p,name):
        mod=original(p,name)
        if name=="recovery_live_student" and arm=="approach":
            original_agent=mod.agent
            def agent(obs,cfg):
                result=original_agent(obs,cfg)
                if obs["step"]==root["t"]+1:
                    update_memory(obs,mem);revised,reason=pay(obs,mem,result,cfg,root,W)
                    sr.retain(HERE/f"seat{s}/follow.json",dict(t=obs["step"],reason=reason,parent=result,executed=revised))
                    return revised
                return result
            mod.agent=agent
        return mod
    live.module=patched
    runner.play(s,arm)

if __name__=="__main__":
    if sys.argv[1:]==["prepare"]:prepare()
    elif len(sys.argv)>1:play(int(sys.argv[1]),sys.argv[2])
    else:
        if len(sr.load(HERE/"selected_sites.json"))!=2:raise ValueError("insufficient held sites")
        for seat in (0,1):
            with (runner.TREE/".continuation.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                jobs=[subprocess.Popen(["taskset","-c",str((seat+i)%2),sys.executable,__file__,str(seat),arm]) for i,arm in enumerate(("parent","approach"))]
                if any([j.wait() for j in jobs]):raise ValueError("native site continuations failed")
