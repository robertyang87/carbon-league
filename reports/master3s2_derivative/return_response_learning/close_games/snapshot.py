"""Portable, data-only encoding of this native controller's two belief classes and RandomState."""
import gzip
import json

import numpy as np

ALLOWED=('Belief','BeliefTracker','_Bel')
def encode(v):
    if type(v) in (int,float,str,bool) or v is None:return v
    if isinstance(v,np.random.RandomState):return ['rng',encode(v.get_state())]
    if isinstance(v,np.ndarray):return ['array',[v.dtype.str,v.tolist()]]
    if isinstance(v,type) and v.__name__ in ALLOWED:return ['class',v.__name__]
    if type(v).__name__ in ALLOWED:return [type(v).__name__,encode(vars(v))]
    if isinstance(v,dict):return ['map',[[encode(k),encode(x)] for k,x in v.items()]]
    if isinstance(v,(list,tuple,set)):return [type(v).__name__,[encode(x) for x in (sorted(v) if isinstance(v,set) else v)]]
    raise TypeError(type(v))
def decode(v,mod):
    if not isinstance(v,list):return v
    tag,x=v
    if tag=='rng':
        r=np.random.RandomState();r.set_state(decode(x,mod));return r
    if tag=='array':
        dtype,vals=x
        if dtype not in ('<u4','<i4','<f4','<f8'): raise ValueError(dtype)
        return np.array(vals,dtype=dtype)
    if tag=='class' and x in ALLOWED:return getattr(mod,x)
    if tag in ALLOWED:
        item=object.__new__(getattr(mod,tag));item.__dict__.update(decode(x,mod));return item
    if tag=='map':return {decode(k,mod):decode(a,mod) for k,a in x}
    if tag in ('list','tuple','set'):
        values=[decode(a,mod) for a in x]
        return {'list':list,'tuple':tuple,'set':set}[tag](values)
    raise ValueError(tag)
def read(ticket,subj,directory):
    text=gzip.decompress((directory/f"{ticket['seed']}_{ticket['seat']}_{ticket['step']}.json.gz").read_bytes())
    return decode(json.loads(text),subj)
