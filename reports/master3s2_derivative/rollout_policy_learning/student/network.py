"""Frozen original three-block trunk plus a sparse circular 3x3 readout for policy sites."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
OBSERVED=Path(__file__).resolve().parents[2]/'rollout_readout'
sys.path.insert(0,str(OBSERVED/'timing'))
from forward import prefix, stages

NEIGHBORS=torch.tensor([[((p//15+dy)%15)*15+(p%15+dx)%15 for dy in (-1,0,1) for dx in (-1,0,1)] for p in range(225)])


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(960,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,8))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer('mean',torch.zeros(960));self.register_buffer('scale',torch.ones(960))

    def forward(self,patch,initial):
        return initial+self.mlp((patch-self.mean)/self.scale)


@torch.inference_mode()
def sites(net,x,sid,crosscheck=False):
    h2=prefix(net,x,sid)
    h3=stages(net,h2,net.blocks[2:3])[net.board].permute(1,0,2).contiguous()
    slots=x[:,[2,4,12]].max(axis=1).reshape(len(x),225)>0
    row,pos=np.nonzero(slots); r=torch.as_tensor(row); p=torch.as_tensor(pos)
    local=h3[r[:,None],NEIGHBORS[p]].reshape(len(row),864)
    z=torch.cat([local,h3.mean(1)[r]],1)
    center=h3[r,p]
    init=torch.cat([center@net.hw[0]+net.hw[1],center@net.hb[0]+net.hb[1]],1)
    shallow=h2[net.board].permute(1,0,2)[r,p] if crosscheck else None
    return row,pos,z,init,shallow


@torch.inference_mode()
def predict(net,student,x,sid):
    row,pos,z,initial,_=sites(net,x,sid)
    v=student(z,initial).numpy()
    worker=np.zeros((len(x),225,5),dtype=np.float32);base=np.zeros((len(x),225,3),dtype=np.float32)
    worker[row,pos]=v[:,:5];base[row,pos]=v[:,5:]
    return worker,base
