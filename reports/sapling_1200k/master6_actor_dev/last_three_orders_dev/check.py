"""Public last-three-step options, with retrospective outcome scoring physically separate."""
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
ROOT=ACTOR.parents[2]
sys.path[:0]=[str(ACTOR.parent/"root_recovery_branches"),str(ROOT/"code")]
import live
from fastsim import FastCarbon

read=lambda p:json.loads(gzip.decompress(p.read_bytes()) if p.suffix==".gz" else p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=read(HERE/"protocol.json")
moves=["UP","RIGHT","DOWN","LEFT"]


def write(p,v):
    text=json.dumps(v,allow_nan=False,separators=(",",":")).encode()
    out=gzip.compress(text,mtime=0) if p.suffix==".gz" else text+b"\n"
    if p.exists():
        if p.read_bytes()!=out:raise ValueError("source or output changed")
    else:p.write_bytes(out)


def distance(a,b):
    u,v=abs(a%15-b%15),abs(a//15-b//15)
    return min(u,15-u)+min(v,15-v)


def reachable(s,side,left):
    return sum(w.carbon for w in s.workers.values() if w.player==side and w.is_collector and
               min(distance(w.pos,b) for _,b in s.bases[side])<=left)


def legal_commands(m,x,original):
    f=sys.modules["cnet2_feat"];a=m._FEAT.aux
    enc=[]
    for pol in m._NETS:
        k,q=pol.encode(x,m._HYBRID["style"]);enc.append((pol,k,q,pol.lin(q,"h0")))
    actions=[(None,original)]
    seen={tuple(sorted(original.items()))}
    signed=m._mixed_actions(x,a);assert m._FEAT.to_actions(signed)==original
    for slot in np.flatnonzero(x["tok_mask"][:f.OWN]):
        for choice in range(3 if slot>=10 else 5):
            if choice==signed[slot]:continue
            j=int(slot);base=m._forced(x,a,enc,False,j,choice)
            if base is None:continue
            adj=m._forced(x,a,enc,True,j,choice)
            protection=(np.arange(f.OWN)>=10)|(x["tok_kind"][:f.OWN]==1)
            acceptable=adj is not None and np.all(adj[protection]==base[protection]) and not(m._return_conflicts(x,adj)-m._return_conflicts(x,base))
            chosen=adj if acceptable else base
            if m._return_conflicts(x,chosen)-m._return_conflicts(x,signed):continue
            if not np.array_equal(chosen[:j],signed[:j]) or chosen[j]!=choice:continue
            cmd=m._FEAT.to_actions(chosen);key=tuple(sorted(cmd.items()))
            if key in seen:continue
            seen.add(key);actions.append(({'slot':j,'canonical_label':choice,'id':m._FEAT.own_uid[j]},cmd))
    return actions


def make_actor():
    m=live.module(ACTOR/"actor_partner_dev/candidate.py","terminal_experiment_parent")
    src=inspect.getsource(m._return_decode).replace("_return_decode(x, aux, encoded, apply_residual)", "_forced(x, aux, encoded, apply_residual, focal, override)")
    point='        action = int(np.argmax(np.where(legal, combined, -np.inf)))'
    assert src.count(point)==1
    src=src.replace(point,point+"\n        if slot == focal:\n            if not legal[override]:return None\n            action = override")
    exec(src,m.__dict__); return m


def proposals(view,previous,s):
    side=1-view["player"]; _,bases,units,_=view["players"][side]; prev=previous["players"][side][2]
    momentum={};home={}
    for u,(pos,cargo,role) in units.items():
        pre=prev.get(u,[pos])[0]
        for act in moves:
            if s._move_nbr[act][pre]==pos:
                momentum[u]=act;break
        if role=="COLLECTOR" and cargo>0:
            dist=min(distance(pos,b) for b in bases.values())
            options=[a for a in moves if min(distance(s._move_nbr[a][pos],b) for b in bases.values())<dist]
            if options:home[u]=options[0]
        elif u in momentum:home[u]=momentum[u]
    return [{},momentum,home]


def outcome(s,side):
    return s.cash[side]-s.cash[1-side]


def apply(s,snap,side,own,enemy):
    s.set_state(snap);joint=[None,None];joint[side]=own;joint[1-side]=enemy
    s.step(*joint)


def choose(view,previous,cfg,est,acts):
    side=view["player"];sim=FastCarbon(cfg); left=298-view["step"]
    values=np.empty((len(acts),6));j=0
    for carbon in [view["carbon"],est.tolist()]:
        public=dict(view,carbon=carbon);sim.load_observation(public);snapshot=sim.get_state()
        for enemy in proposals(view,previous,sim):
            for i,(_,act) in enumerate(acts):
                apply(sim,snapshot,side,act,enemy)
                score=outcome(sim,side)
                if left:score+=.85*(reachable(sim,side,left)-reachable(sim,1-side,left))
                values[i,j]=score
            j+=1
    d=values-values[:1];order=sorted(range(len(acts)),key=lambda i:(float(d[i].min()),float(d[i].mean()),-i),reverse=True)
    w=order[0] if d[order[0]].min()>3 else 0
    return w,d


def game(row):
    p=ROOT/row["source"];assert sha(p)==row["sha256"];g=read(p);s=FastCarbon(g["game_configuration"]).reset(seed=g["seed"]);side=g["candidate_seat"]
    assert list(s.carbon)==g["initial_full_carbon"];assert sha(ACTOR/"actor_partner_dev/candidate.py")==P["parent_sha256"]
    m=make_actor();output=[];previous=None
    for t,joint in enumerate(g["actions"]):
        obs=s.obs(side)
        if t==0:
            assert m.agent(obs,s.config)==joint[side]
        else:x=m._FEAT.observe(obs)
        if t>=296:
            left=299-t
            metrics=dict(t=t,remaining=left,pre_cash=list(s.cash),cargo_total=[sum(w.carbon for w in s.workers.values() if w.player==p and w.is_collector) for p in (side,1-side)],
                         cargo_reachable=[reachable(s,p,left) for p in (side,1-side)],recorded_actual=joint[side])
            if row["seed"] not in P["skip_alternative_seeds"]:
                parent=m._FEAT.to_actions(m._mixed_actions(x,m._FEAT.aux));assert parent==joint[side]
                alternatives=legal_commands(m,x,parent)
                sel,vals=choose(obs,previous,s.config,m._FEAT.est.copy(),alternatives)
                # No original opponent action or hidden carbon enters the preceding selection.
                shot=s.get_state();actual=[];copy=FastCarbon(s.config)
                for _,alt in alternatives:
                    apply(copy,shot,side,alt,joint[1-side])
                    if not copy.player_workers[0] or not copy.player_workers[1]:raise ValueError("non-normal branch needs official extinction handling")
                    actual.append(outcome(copy,side))
                if t==298 and abs(actual[0]-g["margin"])>.00001:raise ValueError("terminal mismatch")
                metrics.update(actions=[dict(focal=root,command=z,public_gain_range=[float(v.min()),float(v.max())],actual_one_step_cash_margin=v1) for (root,z),v,v1 in zip(alternatives,vals,actual)],
                    chosen=sel,actual_one_step_cash_gain=actual[sel]-actual[0],oracle_max_one_step_gain=max(actual)-actual[0])
            output.append(metrics)
        previous=obs;s.step(*joint)
    if list(s.cash)!=g["official_score_by_seat"]:raise ValueError("archived official cash mismatch")
    return dict(**row,reported_margin=g["margin"],terminal_cash=list(s.cash),terminal_unbanked=[sum(w.carbon for w in s.workers.values() if w.player==p and w.is_collector) for p in (side,1-side)],roots=output)


def main():
    out=HERE/"per_game";out.mkdir(exist_ok=True)
    for row in P["games"]:
        p=out/f"{row['seed']}_{row['seat']}.json.gz"
        if not p.exists():write(p,game(row))
        r=read(p);assert r["sha256"]==row["sha256"]
        print(row["seed"],row["seat"],[x.get("chosen") for x in r["roots"]],flush=True)


if __name__=="__main__":main()
