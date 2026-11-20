"""Quantify the parent's existing initial-map symmetry inference using only public actor history."""
import numpy as np
from visibility import D,HERE,load,read,data,obs,sha,report
from pathlib import Path

def no_propagation(self,cls,v0):
    if cls not in self.cls_v0:
        self.cls_v0[cls]=v0;self.cls_seen+=1
        if v0>0:self.nz_found+=1

def main():
    m=load(D/"cold_start/candidate.py");control=load(D/"cold_start/candidate.py")
    control.Belief._learn_class=no_propagation
    orig=m.Belief._learn_class;flags=set()
    def trace(self,cls,v0):
        before={i for i in m.CLS_MEMBERS[cls] if not self.known[i]}
        ret=orig(self,cls,v0)
        flags.update(i for i in before if self.known[i])
        return ret
    m.Belief._learn_class=trace
    cards=read(D/"ladder/ds-pro-v1-placement.json")["games"];maps=[];example=None
    phases=((0,0),(1,49),(50,149),(150,298),(0,298))
    for card in cards:
        r,h=data(card);s=card["candidate_seat"];frames=r["steps"];n=r["configuration"]["size"]
        basepos={p for side in frames[0][0]["observation"]["players"] for p in side[1].values()}
        bclass={m.CLS[p] for p in basepos}
        assert n==15 and basepos=={z for c in bclass for z in m.CLS_MEMBERS[c]}
        reflecting=[np.array([w for w in m.CLS_MEMBERS[c]],dtype=int) for c in m.CLS_MEMBERS]
        direct=set();siblings=set();when=[None,None]
        tr=[a.BeliefTracker(a._Bel,s) for a in (m,control)]
        bins={k:dict(steps=0,cell_times=0,known=np.zeros(2),sse=np.zeros(2),sibling_unseen=0,
                     sibling_abs_err=0.,sibling_error_gt_1=0,new_inferred=0,truth_reflection_everywhere=0) for k in phases}
        for t,f in enumerate(frames[:-1]):
            o=obs(f,s);ps=o["players"]
            visible=set()
            for pos,_,role in ps[s][2].values():
                visible.update([pos] if role=="COLLECTOR" else m.NB[pos])
            direct.update(visible);flags.clear()
            a,ka=tr[0].step(ps,t,o["carbon"]);b,kb=tr[1].step(ps,t,o["carbon"])
            newly=len(flags-direct);siblings.update(flags)
            unseen=sorted(siblings-direct)
            for i,mask in enumerate((ka,kb)):
                if when[i] is None and sum(mask)>=.95*n*n:when[i]=t
            # No true map or other side's private carbon is used by either tracker or ablation.
            truth=np.array(f[0]["observation"]["full_carbon"])
            asymmetric=any(np.ptp(truth[g])>1.e-8 for g in reflecting)
            if t==0:assert not asymmetric
            ae=np.abs(np.array(a)-truth);be=np.abs(np.array(b)-truth)
            if example is None and t>0:
                for p in sorted(visible):
                    q=next((q for q in m.CLS_MEMBERS[m.CLS[p]] if q not in direct and abs(truth[q]-truth[p])>1.),None)
                    if q is not None:
                        example=dict(game=card["match_id"],seat=s,t=t,visible_cell=p,visible_current=o["carbon"][p],ever_unseen_sibling=q,actual_sibling=float(truth[q]),native_estimate=a[q],sibling_known=ka[q],initial_class=m.CLS[p])
                        break
            for k in phases:
                if not k[0]<=t<=k[1]:continue
                v=bins[k];v["steps"]+=1;v["cell_times"]+=n*n;v["known"]+=np.array([sum(ka),sum(kb)])
                v["sse"]+=np.array([ae@ae,be@be]);v["sibling_unseen"]+=len(unseen)
                v["sibling_abs_err"]+=sum(float(ae[i]) for i in unseen);v["sibling_error_gt_1"]+=sum(bool(ae[i]>1) for i in unseen)
                v["new_inferred"]+=newly;v["truth_reflection_everywhere"]+=int(not asymmetric)
            for q in tr:q.set_actions(ps,frames[t+1][s]["action"] or {})
        def plain(v):
            return {k:x.tolist() if isinstance(x,np.ndarray) else x for k,x in v.items()}
        maps.append(dict(game=card["match_id"],map=card["map_sha256"],replay=h,bases=sorted(basepos),t_known_95=when,
                         totals={str(k):plain(v) for k,v in bins.items()}))
    out={}
    for k in phases:
        z=[g["totals"][str(k)] for g in maps]
        n=sum(v["cell_times"] for v in z);q=sum(v["sibling_unseen"] for v in z)
        out[str(k)]=dict(steps=sum(v["steps"] for v in z),total_cells=n,
                known_fraction=(np.sum([v["known"] for v in z],axis=0)/n).tolist(),
                mse=(np.sum([v["sse"] for v in z],axis=0)/n).tolist(),sibling_only_cell_times=q,
                sibling_only_mean_abs_error=sum(v["sibling_abs_err"] for v in z)/q if q else None,
                sibling_only_above_1=sum(v["sibling_error_gt_1"] for v in z),new_inferred_before_direct=sum(v["new_inferred"] for v in z),
                truth_reflection_everywhere=sum(v["truth_reflection_everywhere"] for v in z))
    generator=D.parents[1]/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.py"
    report(HERE/"symmetry.json",dict(generator=sha(generator.read_bytes()),parent=sha((D/"cold_start/candidate.py").read_bytes()),
          order=["parent","disable_sibling_propagation_only"],periods=out,maps=maps,current_copy_counterexample=example))
if __name__=="__main__":main()
