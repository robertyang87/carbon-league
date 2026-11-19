"""One shared board encoder, autoregressive legal action policy and two terminal heads."""
from setup import *
import os
import numpy as np
import torch
from torch import nn
F=bridge.F
PLAN=bridge.load(HERE/'protocol.json')
def collect():
    rows=[];eo=[]
    for card in PLAN['sources']:
        path=ROOT/card['path'];assert bridge.sha(path)==card['sha'];g=bridge.load(path);ss=g['candidate_seat'];sim=bridge.FastCarbon(g['game_configuration']).reset(seed=g['seed'])
        assert card['terrain']==hashlib.sha256(np.asarray(sim.carbon,np.uint8).tobytes()).hexdigest()
        ff=[F.Feat(s) for s in (0,1)];sub=[]
        for t,a in enumerate(g['actions']):
            for s in (0,1):
                x=ff[s].observe(sim.obs(s))
                if t%8:continue
                y=ff[s].labels(a[s]);pf=F.Prefix(x,ff[s].aux);moves=np.zeros((12,5,F.MF),np.float32);legal=np.zeros((12,5),bool);uf=np.zeros((12,F.UF),np.float32)
                for j in np.where(y>=0)[0]:
                    uf[j],moves[j],legal[j]=pf.feats(j);assert legal[j,y[j]];pf.advance(j,int(y[j]))
                sub.append((x['maps'].T,x['scal'],x['tok_pos'][:12],x['tok_kind'][:12],uf,moves,legal,y,sim.cash[s]-sim.cash[1-s],s))
            sim.step(*a)
        assert len(g['actions'])==299
        off=bridge.load(path.parent/'official_replay.json')['runs'][path.parent.name]['games'];cert=next(z for z in off if z['seed']==g['seed'] and z['seat']==ss);assert cert['transcript_sha256']==card['sha'] and cert['steps']==299 and cert['statuses']==['DONE','DONE'];scores=cert['official_reward_by_seat']
        assert np.allclose(scores,sim.cash,atol=1e-5,rtol=0)
        for maps,scal,pos,kind,u,m,l,y,cash,s in sub:
            mar=scores[s]-scores[1-s];wdl=int(mar>0)+.5*int(mar==0);util=.5*wdl+.5/(1+np.exp(-mar/1000))
            rows.append((maps,scal,pos,kind,u,m,l,y,cash,util,(mar-cash)/4000.,card['split']=='fit'))
    return [torch.from_numpy(np.array(col)) for col in zip(*rows)]
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.cell=nn.Linear(66,16);self.context=nn.Linear(52,16);self.pol=nn.Linear(16+16+3+5+F.UF+F.MF,24);self.last=nn.Linear(24,1);self.value=nn.Linear(16,2)
    def forward(self,bs):
        maps,sc,pos,kind,unit,moves,legal=bs[:7];z=maps.float()/255
        board=self.cell(torch.cat((z,z[:,nb].mean(2)),dim=-1)).tanh()
        h=self.context(torch.cat((board.mean(1),board.amax(1),sc.float()),-1)).tanh()
        v=self.value(h)
        p=pos.long().clamp(0,224);dest=dst[p];dest[:,10:]=p[:,10:,None].expand(-1,-1,5)
        local=board[torch.arange(len(p))[:,None,None],dest]
        shared=h[:,None,None].expand(-1,12,5,-1)
        kinds=torch.nn.functional.one_hot(kind.long().clamp(0,2),3).float()[:,:,None].expand(-1,-1,5,-1)
        acts=torch.eye(5)[None,None].expand(len(p),12,-1,-1)
        inp=torch.cat((local,shared,kinds,acts,unit.float()[:,:,None].expand(-1,-1,5,-1),moves.float()),-1)
        return self.last(self.pol(inp).tanh()).squeeze(-1).masked_fill(~legal,-1e9),v
nb=torch.from_numpy(F.DEST[:,1:]).long();dst=torch.from_numpy(F.DEST).long()
def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(842551)
    d=collect();net=Net();opt=torch.optim.AdamW(net.parameters(),lr=.002)
    rng=np.random.default_rng(842551);ii=torch.where(d[-1])[0].numpy();hh=torch.where(~d[-1])[0].numpy()
    for epoch in range(12):
        for ix in np.array_split(rng.permutation(ii),max(1,(len(ii)+47)//48)):
            b=[z[ix] for z in d];lg,v=net(b);mask=(b[7]>=0)&(b[6].sum(-1)>1)
            pi=nn.functional.cross_entropy(lg[mask],b[7][mask].long())
            loss=pi+2*(v[:,0].sigmoid()-b[9].float()).square().mean()+nn.functional.smooth_l1_loss(v[:,1],b[10].float())
            opt.zero_grad();loss.backward();opt.step()
    weights={k:p.detach().numpy() for k,p in net.state_dict().items()};np.savez(HERE/'shared.npz',**weights)
    groups={}
    for name,idx in [('fit',ii),('held',hh)]:
        with torch.no_grad():
            aa=[];rr=[];uu=[];counts=[]
            for ix in np.array_split(idx,max(1,len(idx)//48)):
                b=[z[ix] for z in d];lg,v=net(b);mask=(b[7]>=0)&(b[6].sum(-1)>1)
                counts.append([int((lg.argmax(-1)[mask]==b[7][mask]).sum()),int(mask.sum())]);rr.extend(((v[:,1]-b[10])*4000).tolist());uu.extend((v[:,0].sigmoid()-b[9]).tolist())
            groups[name]=dict(roots=len(idx),policy=np.sum(counts,axis=0).tolist(),u_mae=float(np.abs(uu).mean()),future_net_margin_mae=float(np.abs(rr).mean()))
    bridge.save(HERE/'fit.json',dict(protocol=bridge.sha(HERE/'protocol.json'),source=bridge.sha(__file__),weights=bridge.sha(HERE/'shared.npz'),parameters=sum(v.size for v in weights.values()),groups=groups))
if __name__=='__main__':run()
