"""Two-block policy readout for the original six-block Winograd network. Stored heads are [C, actions]."""
import numpy as np
import torch


def stages(net, h, blocks):
    n = h.shape[1]
    for U1, b1, g, gb, U2, b2 in blocks:
        t = torch.relu(net._conv(h[net.win], U1, b1, n))
        t = t + (t[net.board].mean(0) @ g + gb)
        h = torch.relu(h + net._conv(t[net.win], U2, b2, n))
    return h


def prefix(net, X, sid):
    n = X.shape[0]
    x = torch.from_numpy(X[:, :net.cin]).permute(2,3,0,1).reshape(225,n,net.cin)
    h = torch.relu(net._conv(x[net.winb], *net.stem, n) + net.emb[sid])
    return stages(net, h, net.blocks[:2])


def policy(hf, worker, base):
    def apply(head):
        return (hf @ head[0] + head[1]).permute(1,0,2).contiguous().numpy()
    return apply(worker), apply(base)


def value(net, hf, scalars):
    s = torch.from_numpy(scalars)
    z = torch.relu(hf @ net.v1[0] + net.v1[1])
    z = torch.cat([z.mean(0),z.amax(0),s],1)
    return (s[:,0] + (torch.relu(z @ net.v2[0].T + net.v2[1]) @ net.v3[0].T + net.v3[1])[:,0]).numpy()


@torch.inference_mode()
def frozen_targets(net, X, sid):
    """Return B×225×96 second-block features and B×225×{5,3} deep teacher logits, all NumPy arrays."""
    h = prefix(net, X, sid)
    shallow = h[net.board].permute(1,0,2).contiguous().numpy()
    deep = stages(net,h,net.blocks[2:])[net.board]
    return shallow, *policy(deep,net.hw,net.hb)


@torch.inference_mode()
def evaluate(net, X, sid, scalars, kind, mode, worker, base):
    """Return (own worker/base/value, simulated opponent worker/base). Batches have the original ordering."""
    if kind in ("root","own") or mode == "original":
        lw,lb,v = net(X,sid,scalars)
        if kind=="joint":
            return lw[::2],lb[::2],v[::2],lw[1::2],lb[1::2]
        if kind in ("root","other"):
            return None,None,None,lw,lb
        return lw,lb,v,None,None
    if kind=="other":
        h=prefix(net,X,sid)
        return None,None,None,*policy(h[net.board],worker,base)
    if kind!="joint":
        raise ValueError(kind)
    if mode=="separate":
        own_w,own_b,own_v=net(np.ascontiguousarray(X[::2]),sid,np.ascontiguousarray(scalars[::2]))
        ho=prefix(net,np.ascontiguousarray(X[1::2]),sid)
        return own_w,own_b,own_v,*policy(ho[net.board],worker,base)
    if mode!="shared":
        raise ValueError(mode)
    h=prefix(net,X,sid)
    hf=stages(net,h[:,::2].contiguous(),net.blocks[2:])[net.board]
    w,b=policy(hf,net.hw,net.hb)
    return w,b,value(net,hf,scalars[::2]),*policy(h[net.board,1::2,:],worker,base)
