"""Exact joint-command distribution when one slot of both real return decoders shares one uniform draw."""
import base64
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import zlib

import numpy as np

HERE=Path(__file__).resolve().parent
PACK=HERE.parent/'actor_partner_dev/candidate.py'
PARENT_SHA='c5ad3f43959a4812e84e89574ee0c6e69765ebb06a8b3ec93f839b0dde128dd3'
ZERO=np.zeros(5,dtype=np.float32)


def key(command):
    return json.dumps(command,sort_keys=True,separators=(',',':'))


def load(path=PACK):
    if path==PACK and hashlib.sha256(path.read_bytes()).hexdigest()!=PARENT_SHA:
        raise ValueError('frozen parent differs')
    spec=importlib.util.spec_from_file_location('research_carbon_blend',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy=sys.modules['cnet2_np'].Policy; module._NETS=[]
    for data,q,m in ((module._B1,module._Q1,module._M1),(module._W64,module._QMETA,module._WMETA)):
        w=module._dequantize(zlib.decompress(base64.b64decode(data)),q);w['meta']=np.array(m)
        module._NETS.append(policy(w))
    module._COUNTS={k:dict(valid_slots=0,disagree=0,selected_student_in_disagreement=0,selected_third_in_disagreement=0)
                    for k in ('base','planter','collector')}
    assert module._HYBRID['expert_weight']==.75 and module._HYBRID['student_weight']==.25
    node=inspect.getsource(module._return_decode)
    hook='action = int(np.argmax(np.where(legal, combined, -np.inf)))'
    assert node.count(hook)==1
    node=node.replace('encoded, apply_residual):','encoded, apply_residual, select):').replace(hook,'action = select(slot,combined,legal)')
    namespace=dict(vars(module))
    exec(compile(node,str(path)+'::single-uniform', 'exec'),namespace)
    module._one_slot_decode=namespace['_return_decode']
    module._initial_bias=module._NETS[1].w['whead.bias'].copy()
    return module


def set_delta(module,delta):
    module._NETS[1].w['whead.bias'][:]=module._initial_bias+np.asarray(delta,np.float32)


def guard(module,x,base,residual):
    n=len(base)
    protected=(np.arange(n)>=10)|(x['tok_kind'][:n]==1)
    reject=bool(np.any(base[protected]!=residual[protected]) or
                (np.any(base!=residual) and module._return_conflicts(x,residual)-module._return_conflicts(x,base)))
    return (base if reject else residual),reject


class Root:
    def __init__(self,module,feature,x):
        self.m=module;self.f=feature;self.x=x;self.aux=feature.aux
        self.enc=[]
        for p in module._NETS:
            tokens,ctx=p.encode(x,2);self.enc.append((p,tokens,ctx,p.lin(ctx,'h0')))

    def decode(self,residual,slot=None,forced=None):
        selected={}
        def choose(s,scores,legal):
            if s==slot or slot is None:
                masked=np.where(legal,scores,-np.inf)
                p=np.exp((masked-masked.max()).astype(np.float64));p/=p.sum()
                selected[s]=p
            if s==slot and forced is not None:
                assert legal[forced];return forced
            return int(np.argmax(np.where(legal,scores,-np.inf)))
        y=self.m._one_slot_decode(self.x,self.aux,self.enc,residual,choose)
        return y,selected

    def primary(self):
        a,_=self.decode(False);b,probs=self.decode(True)
        y,_=guard(self.m,self.x,a,b)
        actual=self.m._mixed_actions(self.x,self.aux)
        assert np.array_equal(y,actual),'instrumented and original greedy joint orders differ'
        slots=[s for s,p in probs.items() if s<10 and len(np.flatnonzero(p))>1]
        if not slots:return None
        def entropy(p):
            p=p[p>0];return -float(p@np.log(p))
        s=max(slots,key=lambda s:(entropy(probs[s]),-s))
        return s,entropy(probs[s]),self.f.to_actions(actual)

    def distribution(self,slot,delta):
        set_delta(self.m,delta)
        base={};endpoints=[];outputs={}
        for branch in (False,True):
            _,probs=self.decode(branch,slot)
            p=probs[slot]
            jac=.25*(np.diag(p)-np.outer(p,p))
            endpoints.extend((float(boundary),deriv) for boundary,deriv in zip(np.cumsum(p)[:-1],np.cumsum(jac,axis=0)[:-1])
                             if 0.<boundary<1.)
            full={i:self.decode(branch,slot,i)[0] for i in np.flatnonzero(p>0)}
            cdf=np.cumsum(p);cdf[-1]=1.
            base[branch]=(cdf,full)
        ends=[(0.,np.zeros(5)),*sorted(endpoints,key=lambda z:z[0]),(1.,np.zeros(5))]
        for (lo,glo),(hi,ghi) in zip(ends,ends[1:]):
            if hi<=lo:continue
            z=lo
            orders=[full[min(int(np.searchsorted(cdf,z,side='right')),4)] for cdf,full in base.values()]
            issued,_=guard(self.m,self.x,*orders)
            k=key(self.f.to_actions(issued))
            old,derivative=outputs.get(k,(0.,np.zeros(5)));outputs[k]=(old+hi-lo,derivative+ghi-glo)
        assert abs(sum(float(q) for q,d in outputs.values())-1)<1e-10
        assert np.max(np.abs(sum(d for q,d in outputs.values())))<1e-8
        return outputs
