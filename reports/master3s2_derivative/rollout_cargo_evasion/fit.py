"""Fit the gated inverse temperature, then a bounded residual on the same correctly aligned labels."""
from collections import Counter
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from scipy.special import logsumexp,softmax
from source import init,collect
from register import H,read,dump,sha

PROTOCOL='5efc69e15ab3cf2e666ec8f2739b566da8b81b62086e63e7d02aa0c461fc548c'
def train():
    assert sha((H/'protocol.json').read_bytes())==PROTOCOL
    m=init();rs,seen=collect(m,read(H/'protocol.json')['train'],False)
    data=[(r['terrain'],r['lw'][pos],r['labels'][uid],x) for r in rs for uid,pos,x,_ in r['units']]
    ids,lw,y,x=map(np.array,zip(*data));x=x.astype(np.float64);lw=lw.astype(np.float64);N=len(y)
    counts=Counter(ids);weights=np.array([1/len(counts)/counts[k] for k in ids])
    temp=minimize_scalar(lambda b:weights@(logsumexp(lw*b,axis=1)-lw[np.arange(N),y]*b),bounds=(.05,2.),method='bounded')
    beta=float(temp.x);h=6;n=x.shape[2];g=np.random.default_rng(624739);init_w=g.normal(0,.16,(n,h))
    def unpack(z):return z[:n*h].reshape(n,h),z[n*h:n*h+h],z[n*h+h:]
    def obj(z):
        W,b,O=unpack(z);u=np.tanh(x@W+b);v=np.tanh(u@O);logits=lw*beta+2*v
        loss=weights@(logsumexp(logits,axis=1)-logits[np.arange(N),y])
        grad=softmax(logits,axis=1);grad[np.arange(N),y]-=1;dv=2*(1-v**2)*grad*weights[:,None]
        du=(dv[:,:,None]*O)*(1-u**2)
        return loss+.002*(np.mean(W**2)+np.mean(O**2)),np.r_[(np.einsum('nkd,nkh->dh',x,du)+.004*W/W.size).ravel(),du.sum(axis=(0,1)),np.einsum('nkh,nk->h',u,dv)+.004*O/O.size]
    z=np.r_[init_w.ravel(),np.zeros(h),g.normal(0,.08,h)]
    ret=minimize(obj,z,method='L-BFGS-B',jac=True,options={'maxiter':80,'ftol':1e-10})
    W,b,O=unpack(ret.x)
    spec=dict(beta=beta,w=W.astype(np.float32).tolist(),b=b.astype(np.float32).tolist(),out=O.astype(np.float32).tolist())
    wsha=dump(H/'weights.json',spec)
    print(dump(H/'fit.json',dict(protocol=PROTOCOL,weights=wsha,eligible_workers=N,eligible_roots=len(rs),maps=len(counts),
         terrain_counts=dict(counts),samples=seen,network_type=type(m._RNET).__name__,uses_distilled_runtime=m._RNET is not m._NET,
         native_NLL=float(weights@(logsumexp(lw,axis=1)-lw[np.arange(N),y])),temp_NLL=float(temp.fun),residual_penalized_objective=float(ret.fun),iterations=int(ret.nit))),wsha,flush=True)
if __name__=='__main__':train()
