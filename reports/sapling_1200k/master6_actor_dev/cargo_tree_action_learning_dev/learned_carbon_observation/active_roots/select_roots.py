"""Prospectively register public roots; the streaming reader decodes an action only after examining its state."""
from collections import Counter
import gzip
import inspect
import json
from pathlib import Path
import re
import sys
import types
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import F,ROOT,read,save,sha
sys.path.insert(0,str(HERE.parent/"action_preference"))
from support import digest, option, begin, public, clean

REG=read(HERE/"protocol.json")
PHYSICAL=("NONE","UP","RIGHT","DOWN","LEFT")
QUOTA=((0,"collector"),(1,"planter"),(1,"collector"),(0,"planter"))


def streaming_archive(path):
    # Decompressing/hashing file bytes is not a read of its scores. The interpreter
    # sees metadata before 'actions', then only the joint command consumed below.
    fd=gzip.open(path,"rt",encoding="utf-8")
    head=""
    marker=re.compile(r'"actions"\s*:\s*\[')
    while not marker.search(head):
        part=fd.read(1024)
        if not part:
            raise ValueError("Missing taped actions")
        head+=part
    m=marker.search(head)
    info=json.loads(head[:m.start()]+'"actions": null}')
    meta={k:info[k] for k in ("seed","game_configuration","initial_terrain_u8_sha256","initial_full_carbon")}
    def actions():
        buf=head[m.end():]
        decoder=json.JSONDecoder()
        try:
            while True:
                buf=buf.lstrip(" \t\r\n,")
                if buf.startswith("]"):
                    return
                try:
                    item,end=decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    chunk=fd.read(1024)
                    if not chunk:
                        raise EOFError("truncated tape")
                    buf+=chunk
                    continue
                buf=buf[end:]
                yield item
        finally:
            fd.close()
    return meta,actions()


def already_signed():
    # Only these top-level fields are parsed. Do not load any terminal scores.
    path=HERE.parent/"action_preference/support_rows.json.gz"
    raw=gzip.decompress(path.read_bytes()).decode()
    get=lambda k:[json.loads(v) for v in re.findall(r'^    "'+k+r'": ([^,\n]+),?$',raw,re.M)]
    keys=set(zip(get("terrain"),get("seat"),get("t"),strict=True))
    assert len(keys)==108
    return keys


class Pack:
    def __init__(self, path, signature):
        raw=path.read_bytes()
        assert sha(raw)==signature
        m=types.ModuleType("carbon_root_"+signature[:8]); self.m=m
        m.__file__=str(path)
        exec(compile(raw,str(path),"exec"),vars(m))
        # A read-only tap at the original dual-prefix decoder preserves literal agent output.
        old=m._return_decode
        def tap(x,aux,enc,apply_residual):
            self.inputs=(x,aux,enc)
            return old(x,aux,enc,apply_residual)
        m._return_decode=tap
        original=inspect.getsource(old)
        needle="        action = int(np.argmax(np.where(legal, combined, -np.inf)))"
        assert original.count(needle)==1
        code=original.replace(needle,needle+'\n        action = override(slot, action, legal, combined, prefix)')
        # Isolated decoder copy. All logits, prefix updates and network math come from the deployed code.
        self.ns={**vars(m)}
        exec(compile(code,str(path)+":forced","exec"),self.ns)
        self.decoder=self.ns["_return_decode"]

    def decode(self,fixed):
        x,aux,enc=self.inputs
        m=self.m
        # Hypothetical decodes keep diagnostic counters separate from the public agent.
        self.ns["_COUNTS"]={role: dict(counts) for role,counts in m._COUNTS.items()}
        def one(residual):
            trace={}
            def override(slot,argmax,legal,logits,prefix):
                action=int(fixed.get(slot,argmax))
                if not legal[action]:
                    raise ValueError("forced slot is illegal on its actual prefix")
                trace[slot]=dict(argmax=int(argmax),actual=action,legal=legal.astype(bool).tolist(),
                    logits=logits.astype(float).tolist(),cash=float(prefix.cash),remaining_slots=int(prefix.slots),
                    claimed=[int(p) for p in np.flatnonzero(prefix.reserved)],
                    plants=int(prefix.plants))
                return action
            self.ns["override"]=override
            return self.decoder(x,aux,enc,residual),trace
        y0,r0=one(False);y1,r1=one(True)
        protected=(np.arange(F.OWN)>=10)|(x["tok_kind"][:F.OWN]==1)
        reason=("protected_slot" if np.any(y0[protected]!=y1[protected]) else
                "new_conflict" if m._return_conflicts(x,y1)-m._return_conflicts(x,y0) else None)
        if not np.any(m._RETURN_THETA):
            reason="zero_residual"
        return (y0,r0,"baseline",reason) if reason else (y1,r1,"return",None)

    def current(self,obs,cfg):
        command=self.m.agent(obs,cfg)
        y,trace,path,reason=self.decode({})
        assert self.m._FEAT.to_actions(y)==command
        return command,y,trace,path,reason


def serialize_trace(trace,feat):
    return [{**v,"slot":int(k),"uid":feat.own_uid[k]} for k,v in sorted(trace.items())]


def inspect_tick(parent,student,obs,cfg,role,stats):
    pa,py,pt,pp,pr=parent.current(obs,cfg)
    free,sy,st,sp,sr=student.current(obs,cfg)
    pm,sm=parent.m,student.m
    px,paux,_=parent.inputs
    x,aux,_=student.inputs
    assert np.array_equal(px["maps"],x["maps"])
    old=x["maps"][0].astype(float)*(100/255)
    new=sm._carbon_view(x,sm._FEAT,F,sm._CARBON_WEIGHTS)["maps"][0].astype(float)*(100/255)
    seat=obs["player"]
    order=F.CANON if seat else F.GRID
    us=obs["players"][seat]
    passing=[]
    for slot in np.flatnonzero(x["tok_mask"][:10]):
        uid=sm._FEAT.own_uid[slot]
        physical,cargo,name=us[2][uid]
        if name.lower()!=role:
            continue
        stats["target_slots"]+=1
        old_cmd=pa.get(uid,"NONE")
        for k,cmd in enumerate(PHYSICAL):
            if cmd==old_cmd:
                continue
            stats["physical_alternatives_examined"]+=1
            vectors=[];footprint=set()
            for b in (old,new):
                arms=[]
                for act in (old_cmd,cmd):
                    v,cells=option(b,us[1],us[2],uid,act,order)
                    arms.append(v);footprint.update(cells)
                vectors.append(np.stack(arms))
            o,n=vectors
            delta=(n[1]-n[0])-(o[1]-o[0])
            if (max(abs(delta[0]),abs(delta[2])) if role=="collector" else abs(delta[1]))<1:
                continue
            stats["role_carbon_threshold"]+=1
            canon=k if seat==0 else int(F.CANON_MOVE[k])
            force={int(j):int(py[j]) for j in range(slot) if py[j]>=0}
            force[int(slot)]=canon
            try:
                y,track,branch,reject=student.decode(force)
            except ValueError:
                stats["illegal_forced_prefix"]+=1
                continue
            acts=sm._FEAT.to_actions(y)
            if sorted(u for u in pa.keys()|acts.keys() if pa.get(u)!=acts.get(u)) != [uid]:
                stats["teammate_would_change"]+=1
                continue
            if sm._return_conflicts(x,y)-pm._return_conflicts(px,py):
                stats["new_conflicting_pair"]+=1
                continue
            v=track[slot]
            logits=np.array(v["logits"])
            legal=np.array(v["legal"])
            soft=np.where(legal,np.exp(logits-max(logits[legal])),0)
            soft/=soft.sum()
            rank=1+int(np.sum(legal & (logits>logits[canon])))
            if rank>3 or soft[canon]<.08:
                stats["insufficient_prior"]+=1
                continue
            stats["qualified_options"]+=1
            candid=dict(focal_uid=uid,role=role,focal_slot=int(slot),cargo=float(cargo),
                physical_destinations=[int(F.DEST[physical,PHYSICAL.index(act)]) for act in (old_cmd,cmd)],
                carbon=dict(old=o.tolist(),student=n.tolist(),contrast_change=delta.tolist(),footprint_size=len(footprint),
                            changed_cells=int(sum(old[g]!=new[g] for g in footprint)),units="carbon; [destination, four-neighbor mean, shortest-to-nearest-own-base routes mean]"),
                incumbent=pa,alternative=acts,free_student=free,prior=dict(rank=rank,p_parent=float(soft[py[slot]]),p_alternative=float(soft[canon]),canonical_commands=[int(py[slot]),canon],legal=v["legal"],logits=v["logits"]),
                selected_paths=dict(incumbent=pp,incumbent_reject_reason=pr,free_student=sp,free_reject_reason=sr,forced_student=branch,forced_reject_reason=reject),
                budget_and_legality=dict(incumbent=serialize_trace(pt,pm._FEAT),forced_student=serialize_trace(track,sm._FEAT)),
                incumbent_conflicts=sorted(pm._return_conflicts(px,py)),alternative_conflicts=sorted(sm._return_conflicts(x,y)))
            passing.append(((-float(soft[canon]),uid,k),candid))
    return min(passing,key=lambda v:v[0])[1] if passing else None


def main():
    estimator=read(HERE.parent/"protocol.json")
    assert sha((HERE.parent/"protocol.json").read_bytes())==REG["estimator_protocol_sha256"]
    cards=[c for c in estimator["cards"] if c["split"]=="train"]
    assert len(cards)==10
    excluded=already_signed()
    parent=Pack(HERE.parents[1]/"candidate.py",REG["incumbent_sha256"])
    student=Pack(HERE.parent/"candidate.py",REG["carbon_student_sha256"])
    cases,coverage=[],[]
    for i,card in enumerate(cards):
        seat,role=QUOTA[i%4]
        source=ROOT/card["path"]
        assert sha(source.read_bytes())==card["sha256"]
        archive,tape=streaming_archive(source)
        assert archive["initial_terrain_u8_sha256"]==card["terrain"]
        env=begin(archive)
        cfg={**archive["game_configuration"],"randomSeed":None}
        prev=[];stats=Counter();case=None
        for t in range(97):
            obs=public(env,seat)
            assert set(obs)=={"step","player","players","carbon","trees","remainingOverageTime"} and obs["step"]==t
            if t==0:
                parent.current(obs,cfg);student.current(obs,cfg)
            elif t%4==0 and (card["terrain"],seat,t) not in excluded:
                stats["sampled_ticks"]+=1
                case=inspect_tick(parent,student,obs,cfg,role,stats)
            else:
                if t%4==0:stats["old_signed_tick_excluded"]+=1
                parent.m._FEAT.observe(obs);student.m._FEAT.observe(obs)
            if case is not None:
                prefix_sha=digest(prev); ob_sha=digest({k:v for k,v in obs.items() if k!="remainingOverageTime"})
                root_joint=next(tape)
                case.update(seed=card["seed"],seat=seat,t=t,terrain=card["terrain"],source=card["path"],
                            source_sha256=card["sha256"],prefix_sha256=prefix_sha,public_obs_sha256=ob_sha,
                            root_enemy=clean(root_joint[1-seat]),root_enemy_mode="the recorded archive command; read only after accepting this root")
                cases.append(case)
                break
            if t<96:
                both=next(tape)
                prev.append(both);env.step(both)
        tape.close()
        coverage.append(dict(seed=card["seed"],terrain=card["terrain"],seat=seat,role=role,selected=case is not None,through=t,counts=stats))
        print(card["seed"],seat,role,t,stats,flush=True)
        if len(cases)==4:break
    save(HERE/"roots.json",dict(protocol_sha256=sha((HERE/"protocol.json").read_bytes()),incumbent_sha256=REG["incumbent_sha256"],
        student_sha256=REG["carbon_student_sha256"],historical_signed_exclusions=len(excluded),cases=cases,coverage=coverage,
        unopened_train_maps=sorted({c["seed"] for c in cards}-{c["seed"] for c in coverage})))

if __name__=="__main__":
    main()
