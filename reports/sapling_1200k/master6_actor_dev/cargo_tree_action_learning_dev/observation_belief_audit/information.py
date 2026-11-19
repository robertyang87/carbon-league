"""Exploratory decomposition of known-but-never-directly-observed squares by the first public fact."""
from collections import Counter,defaultdict
from types import SimpleNamespace
import numpy as np
from audit import HERE,FastCarbon,cards,modules,view,load,save

def instrument(m):
    class Attributed(m.Belief):
        def __init__(self,*args):
            super().__init__(*args)
            self.reason=np.full(225,'',dtype='U28')
            self.reason[self.known]='base'
            self.direct_cause=False
        def _op(self,i,op):
            before=self.known[i]
            super()._op(i,op)
            if not before and self.known[i]:self.reason[i]='public_reset'
        def _learn_class(self,cls,v0):
            before=[i for i in m.CLS_MEMBERS[cls] if not self.known[i]]
            super()._learn_class(cls,v0)
            for i in before:
                if self.known[i]:self.reason[i]='mirror_direct' if self.direct_cause else 'mirror_enemy_harvest'
        def observe(self,i,v):
            direct=not self.known[i];self.direct_cause=True
            try:super().observe(i,v)
            finally:self.direct_cause=False
            if direct:self.reason[i]='direct'
        def update(self,*args):
            super().update(*args)
            missing=np.asarray(self.known)&(self.reason=='')
            self.reason[missing]='enemy_harvest'
            assert np.array_equal(self.reason!='',np.asarray(self.known))
    return SimpleNamespace(Belief=Attributed,NB=m.NB)

def main():
    f,m=modules();bm=instrument(m)
    oi=load(HERE/'opportunities.json.gz');base=np.load(HERE/'ranks.npz')['values']
    additional=np.zeros((len(oi),5),np.float32);idx=0;tot=defaultdict(lambda:np.zeros(3));N=f.DEST[:,1:]
    for gi,(card,cfg,initial,actions,end) in enumerate(cards()):
        F=[f.Feat(s) for s in (0,1)]; tr=[m.BeliefTracker(bm,s) for s in (0,1)]
        sim=FastCarbon(cfg).reset(seed=card['seed'],full_carbon=initial)
        for t,pair in enumerate(actions):
            truth=np.asarray(sim.carbon)
            for s in (0,1):
                o=sim.obs(s);x,es,vis,known=view(f,m,F[s],tr[s],o,t);raw,vnet,sap,master=es
                gap=~vis&(F[s].last_step<0)&known
                sub=sap.copy()
                sub[gap]=f.q8((vnet[gap]/100.).astype(np.float32)).astype(np.float64)*(100/255)
                for fact in ('base','enemy_harvest','mirror_enemy_harvest','mirror_direct','public_reset','direct'):
                    select=gap&(tr[s].bel.reason==fact)
                    if select.any():
                        stat=tot[card['seed'],s,fact];stat[0]+=select.sum();stat[1]+=np.abs(sap[select]-truth[select]).sum();stat[2]+=np.abs(sub[select]-truth[select]).sum()
                while idx<len(oi) and oi[idx][:3]==[gi,s,t]:
                    _,_,_,_,uid,role,*_=oi[idx];d=f.DEST[o['players'][s][2][uid][0]]
                    values=np.stack((truth,sap,master,sub))[:,d] if role=='COLLECTOR' else np.stack((truth,sap,master,sub))[:,N[d]].mean(2)
                    assert np.max(np.abs(values[:3]-base[idx]))<1e-4
                    additional[idx]=values[3];idx+=1
                tr[s].set_actions(o['players'],pair[s])
            sim.step(*pair)
    assert idx==len(oi)
    stats=[];reg=defaultdict(list)
    for v in (tot,):
        for k,a in sorted(v.items()):stats.append([*k,*a.tolist()])
    for i,row in enumerate(oi):
        eligible=(row[7]&row[8]) & 31;mask=np.array([bool(eligible&(1<<k)) for k in range(5)])
        if mask.sum()<2:continue
        truth=base[i,0,mask]
        if np.ptp(truth)<1e-5:continue
        errors=[]
        for pred in (base[i,1,mask],base[i,2,mask],additional[i,mask]):
            tops=np.isclose(pred,pred.max(),atol=1e-5,rtol=0)
            errors.append(float(truth.max()-truth[tops].mean()))
        reg[row[5]].append(errors)
    save(HERE/'information.json',dict(by_seed_seat_source_and_first_public_fact=stats,columns=['seed','seat','first_known_fact','count','abs_error_Sapling_sum','abs_error_selective_sum'],
        selective_regret={role:dict(n=len(vals),sap_master_selective=np.mean(vals,axis=0).tolist()) for role,vals in reg.items()},
        qualification='Exploratory post-audit field substitution using master raw at student q8 resolution only on hidden, never-directly-observed, master-known squares. No network or terminal policy effect measured.'))
    np.savez_compressed(HERE/'selective_ranks.npz',values=additional)
if __name__=='__main__':main()
