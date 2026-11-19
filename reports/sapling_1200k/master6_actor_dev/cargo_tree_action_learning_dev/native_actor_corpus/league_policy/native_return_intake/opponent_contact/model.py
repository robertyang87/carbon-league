"""Matched five-way enemy worker action classifiers fit on registered training rows only.

fit(x, y, weights, history_cols) returns NumPy weight dictionaries under "full" and
"geometry"; predict(one_dictionary, x) returns the five unconditional movement
probabilities in the caller's fixed class order. geometry sees zero standardized
history inputs in both training and inference.
"""
import math
import numpy as np

HIDDEN=32
CLASSES=5
SEED=621758490
EPOCHS=12
BATCH=512
LR=0.001


def _indices(raw,width):
    items=list(raw)
    if any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<width for i in items):
        raise ValueError('Feature indices must be distinct valid column numbers')
    ids=np.asarray(items,dtype=np.int64)
    if len(np.unique(ids))!=len(ids):
        raise ValueError('Duplicate feature column')
    return ids


def _moments(x,w,continuous_cols):
    # Weighted parallel moment accumulation avoids constructing a full dense
    # float64 copy. No held rows are supplied for moment or binary detection.
    n,d=x.shape
    mu=np.zeros(d,dtype=np.float64);m2=np.zeros_like(mu)
    binary=np.ones(d,dtype=bool);total=0.
    for i in range(0,n,16384):
        a=x[i:i+16384].astype(np.float64,copy=False)
        ww=w[i:i+16384]
        positive=ww>0
        if not np.any(positive):continue
        keep=a[positive];ww=ww[positive]
        binary&=np.all((keep==0)|(keep==1),axis=0)
        block=float(ww.sum(dtype=np.float64))
        bm=np.einsum('i,ij->j',ww,keep)/block
        centered=keep-bm
        b2=np.einsum('i,ij,ij->j',ww,centered,centered)
        delta=bm-mu
        merged=block+total
        m2+=b2+(delta*delta)*(total*block/merged)
        mu+=delta*(block/merged)
        total=merged
    is_continuous=~binary
    if continuous_cols is not None:
        is_continuous[:]=False;is_continuous[_indices(continuous_cols,d)]=True
    var=np.maximum(m2/total,0.)
    center=np.where(is_continuous,mu,0)
    scale=np.where(is_continuous&(var>=1e-6),np.sqrt(var),1.)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError('Nonfinite training moments')
    return center,scale,is_continuous,var


def _working_features(x,center,scale,mask):
    return (((x-center)/scale)*mask).astype(np.float32,copy=False)


def fit(x,y,weights,history_cols,continuous_cols=None):
    """Fit both arms with independently recreated identical starts and minibatches.

    The caller supplies row weights whose total is equal across physical maps
    and across games within each map. The weighted likelihood uses one global
    training normalization; individual minibatches are never normalized by their
    own weight sum. Set continuous_cols explicitly if schema metadata are available.
    """
    import torch
    import torch.nn.functional as functional
    x=np.asarray(x)
    y=np.asarray(y)
    w=np.asarray(weights,dtype=np.float64)
    if x.ndim!=2 or not x.shape[0] or not x.shape[1] or x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise ValueError('Expected finite numeric training inputs [n,d]')
    n,d=x.shape
    if y.shape!=(n,) or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=CLASSES):
        raise ValueError('Training labels must be integer action IDs 0 through 4')
    if w.shape!=(n,) or not np.isfinite(w).all() or np.any(w<0) or not np.isfinite(w.sum()) or w.sum()<=0:
        raise ValueError('Training row weights are invalid')
    hist=_indices(history_cols,d)
    if not hist.size:raise ValueError('The matched history ablation needs at least one history column')
    center,scale,is_cont,var=_moments(x,w,continuous_cols)
    # The natural objective is sum(w_i * CE_i)/sum(w); normalizing globally
    # retains map/game weights with a uniformly shuffled epoch of rows.
    relative=w/(w.sum()/n)
    if not np.isfinite(relative.astype(np.float32)).all():raise ValueError('Row weights exceed float32')
    weights_t=torch.from_numpy(relative.astype(np.float32))
    labels=torch.from_numpy(y.astype(np.int64))
    cpu=torch.device('cpu')
    torch.set_num_threads(1)
    out={}
    for name in ('full','geometry'):
        active=np.ones(d,dtype=np.float32)
        if name=='geometry':active[hist]=0
        normalized=_working_features(x,center,scale,active)
        if not np.isfinite(normalized).all():raise ValueError('Nonfinite standardized training input')
        data=torch.from_numpy(normalized)
        # Isolate the weight-initialization generator from caller state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            model=torch.nn.Sequential(torch.nn.Linear(d,HIDDEN,device=cpu,dtype=torch.float32),torch.nn.Tanh(),
                                      torch.nn.Linear(HIDDEN,CLASSES,device=cpu,dtype=torch.float32))
        optimizer=torch.optim.Adam(model.parameters(),lr=LR,betas=(0.9,0.999),eps=1e-8,weight_decay=0.,foreach=False)
        rng=torch.Generator(device='cpu').manual_seed(SEED)
        trace=[]
        for _ in range(EPOCHS):
            order=torch.randperm(n,generator=rng)
            loss_total=0.
            for ix in order.split(BATCH):
                optimizer.zero_grad(set_to_none=True)
                per_row=functional.cross_entropy(model(data[ix]),labels[ix],reduction='none')
                summed=torch.sum(weights_t[ix]*per_row)
                if not bool(torch.isfinite(summed)):raise ValueError('Nonfinite training loss')
                (summed/len(ix)).backward()
                optimizer.step()
                loss_total+=float(summed.detach())
            trace.append(loss_total/n)
        def asarray(p):return p.detach().cpu().numpy().copy()
        out[name]=dict(w1=asarray(model[0].weight),b1=asarray(model[0].bias),
                       w2=asarray(model[2].weight),b2=asarray(model[2].bias),
                       center=center.copy(),scale=scale.copy(),active=active,continuous=is_cont.astype(np.uint8),
                       training_variance=var.copy(),epoch_weighted_online_nll=np.asarray(trace),
                       seed=np.int64(SEED),epochs=np.int64(EPOCHS),batch_size=np.int64(BATCH),
                       rows=np.int64(n),label_count=np.int64(CLASSES),learning_rate=np.float64(LR))
    return out


def predict(weights,x):
    """Return a finite [n,5] NumPy probability array without running the simulator."""
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=len(weights['center']) or not np.isfinite(x).all():
        raise ValueError('Invalid prediction input')
    probs=np.empty((len(x),CLASSES),dtype=np.float64)
    for i in range(0,len(x),8192):
        sl=slice(i,min(i+8192,len(x)))
        a=_working_features(x[sl],weights['center'],weights['scale'],weights['active'])
        hidden=np.tanh(a@weights['w1'].T+weights['b1'])
        logits=(hidden@weights['w2'].T+weights['b2']).astype(np.float64)
        exp=np.exp(logits-logits.max(axis=1,keepdims=True))
        probs[sl]=exp/exp.sum(axis=1,keepdims=True)
    if not np.isfinite(probs).all():raise ValueError('Nonfinite predictions')
    return probs
