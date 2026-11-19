"""Observe actual Sapling hidden vectors and differentiate issued joint commands after both real decoders and guard."""
import inspect
import numpy as np
from common import policy


def instrument(module):
    src=inspect.getsource(module._return_decode)
    old='action = int(np.argmax(np.where(legal, combined, -np.inf)))'
    assert src.count(old)==1
    src=src.replace('encoded, apply_residual):','encoded, apply_residual, select):').replace(old,'action = select(slot,combined,legal,states[1][3])')
    ns=dict(vars(module));exec(compile(src,'frozen-parent::conditioned-worker','exec'),ns)
    module._conditional_decode=ns['_return_decode']
    w=module._NETS[1].w
    module._orig_head={k:w[k].copy() for k in ('whead.weight','whead.bias')}
    return module


def apply(module,projection,delta):
    state=module._orig_head if projection is None else projection.inject(module._orig_head,delta)
    w=module._NETS[1].w
    for k in ('whead.bias','whead.weight'):w[k][:]=state[k]
    w['whead.weightT'][:]=state['whead.weight'].T


class Root(policy.Root):
    def inspect(self,residual,slot=None,forced=None):
        observations={}
        def select(s,logits,mask,h):
            if slot is None or s==slot:
                z=np.where(mask,logits,-np.inf);p=np.exp((z-z.max()).astype(np.float64));p/=p.sum()
                observations[s]=(p,h.copy())
            if s==slot and forced is not None:
                assert mask[forced];return forced
            return int(np.argmax(np.where(mask,logits,-np.inf)))
        labels=self.m._conditional_decode(self.x,self.aux,self.enc,residual,select)
        return labels,observations

    def physical(self,slot):
        pos=int(self.x['tok_pos'][slot]);load=float(self.x['tok_cargo'][slot])
        return [load,float(self.aux['dob'][pos]),int(self.aux['en_min'][pos]<load)]

    def choose(self,loaded):
        ref,_=self.inspect(False);corrected,detail=self.inspect(True)
        final,_=policy.guard(self.m,self.x,ref,corrected)
        assert np.array_equal(self.m._mixed_actions(self.x,self.aux),final)
        candidates=[]
        for s,(p,h) in detail.items():
            if s>=10 or self.x['tok_kind'][s]!=0 or sum(p>0)<=1:continue
            cargo,dist,danger=self.physical(s);eligible=(cargo>=40)==loaded
            nz=p[p>0];entropy=float(-nz@np.log(nz))
            candidates.append((eligible,entropy,-s))
        if not candidates:return None
        eligible,ent,negative=max(candidates);s=-negative
        return dict(slot=s,uid=self.f.own_uid[s],desired_loaded=loaded,matched=bool(eligible),entropy=ent,
                    context=self.physical(s),greedy=policy.key(self.f.to_actions(final)))

    def mass(self,slot,projection,delta):
        apply(self.m,projection,delta)
        versions=[];edges=[]
        for residual in (False,True):
            _,data=self.inspect(residual,slot)
            p,h=data[slot]
            jac=np.einsum('ij,k->ijk',.25*(np.diag(p)-np.outer(p,p)),projection.features(h)).reshape(5,15)
            cdf=np.cumsum(p);cdf[-1]=1.
            edges.extend((v,der) for v,der in zip(cdf[:-1],np.cumsum(jac,axis=0)[:-1]) if 0<v<1.)
            labels={i:self.inspect(residual,slot,i)[0] for i in np.flatnonzero(p>0)}
            versions.append((cdf,labels))
        starts=[(0.,np.zeros(15)),*sorted(edges,key=lambda v:v[0]),(1.,np.zeros(15))]
        combined={};segments=[]
        for (lo,a),(hi,b) in zip(starts,starts[1:]):
            if hi<=lo:continue
            commands=[m[int(np.searchsorted(c,lo,side='right'))] for c,m in versions]
            final,guarded=policy.guard(self.m,self.x,*commands)
            key=policy.key(self.f.to_actions(final));q,g=combined.get(key,(0.,np.zeros(15)))
            combined[key]=(q+(hi-lo),g+b-a)
            segments.append((float(lo),float(hi),key,bool(guarded)))
        assert abs(sum(q for q,g in combined.values())-1)<1e-12
        assert np.max(np.abs(sum(g for q,g in combined.values())))<1e-8
        return combined,segments
