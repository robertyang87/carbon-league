"""Check inter-unit dependence of a deployed fork and numerical agreement of NumPy board inference."""
from setup import *
import copy,sys,time
import numpy as np
import torch
from train import Net,F
from audit import module
def run():
    torch.set_num_threads(1);p=bridge.load(HERE/'protocol.json');cards=[c for c in p['sources'] if c['split']=='held'];cases=[];merr=0.;private=[]
    pkg=module(HERE/'candidate.py','joint_isolation')
    n=Net()
    with np.load(HERE/'shared.npz') as w:n.load_state_dict({k:torch.from_numpy(w[k]) for k in w.files})
    for card in cards:
        s=card['seat'];raw=bridge.load(ROOT/card['path']);cfg=raw['game_configuration'];sim=bridge.FastCarbon(cfg).reset(seed=card['seed'])
        for t,pair in enumerate(raw['actions'][:226]):
            obs=sim.obs(s)
            if t==0:pkg.agent(obs,cfg)
            elif t%25:pkg._FEAT.observe(obs);pkg._JB_FEAT.observe(pkg.public_enemy(obs,pkg._FEAT.est))
            else:
                snap=(copy.deepcopy(pkg._FEAT),copy.deepcopy(pkg._JB_FEAT))
                a=pkg.agent(obs,cfg);info=copy.deepcopy(pkg._JB_INFO)
                if t==150:
                    pkg._FEAT,pkg._JB_FEAT=copy.deepcopy(snap)
                    duplicate=pkg.agent(dict(obs,oracle_carbon=[80.]*225,full_carbon=[123.]*225,reveal={15:99},enemy_commands={'foo':'RIGHT'}),cfg)
                    same_candidates=info.get('candidates')==pkg._JB_INFO.get('candidates')
                    assert same_candidates and duplicate==a;private.append(dict(seed=card['seed'],seat=s,compared=len(info.get('candidates',[]))))
                x=snap[0].observe(obs);f=snap[0];enc=[]
                for pol in pkg._NETS:
                    tok,ctx=pol.encode(x,pkg._HYBRID['style']);enc.append((pol,tok,ctx,pol.lin(ctx,'h0')))
                _,_,vh=pkg._board_batch([x])
                maps=x['maps'].T.astype(np.float32)/255;p0=torch.from_numpy(maps)
                local=n.cell(torch.cat([p0,p0[torch.from_numpy(F.DEST[:,1:])].mean(1)],-1)).tanh()
                h=n.context(torch.cat([local.mean(0),local.amax(0),torch.from_numpy(x['scal']).float()])).tanh()
                merr=max(merr,float(np.abs(n.value(h).detach().numpy()-vh[0]).max()))
                y0=f.labels(info['base'])
                for kind in info.get('kinds',[]):
                    if kind['kind']!='paired':continue
                    l,r=sorted(kind['forced'])
                    def go(ids):
                        cap={};ys=[pkg._jb_decode(x,f.aux,enc,mode,{j:1 for j in ids},cap) for mode in (False,True)]
                        return pkg._guard_pair(x,*ys),cap
                    ya,_=go([l]);yb,_=go([r]);both,_=go([l,r])
                    assert f.to_actions(both) in info['candidates']
                    joined=y0.copy();joined[ya!=y0]=ya[ya!=y0];joined[yb!=y0]=yb[yb!=y0]
                    ca=pkg._return_conflicts(x,joined)-pkg._return_conflicts(x,y0)
                    # Does the second unit's actual prefix see different features when the first takes its alternative?
                    pold=F.Prefix(x,f.aux);pnew=F.Prefix(x,f.aux)
                    for j in range(r):
                        if y0[j]>=0:pold.advance(j,int(y0[j]));pnew.advance(j,int(both[j]))
                    _,mo,lo=pold.feats(r);_,mn,ln=pnew.feats(r)
                    cases.append(dict(seed=card['seed'],seat=s,t=t,slots=[l,r],uids=[f.own_uid[z] for z in (l,r)],roles=[int(x['tok_kind'][z]) for z in (l,r)],
                                second_prefix_move_features_changed=bool(np.any(mo!=mn)),second_legality_changed=bool(np.any(lo!=ln)),cannot_obtain_by_splicing=bool(np.any(joined!=both)),splice_new_collision=bool(ca)))
            sim.step(*pair)
    assert merr<1e-4
    bridge.save(HERE/'interaction.json.gz',dict(receipt=bridge.sha(HERE/'binding.json'),source=bridge.sha(__file__),paired=cases,num=len(cases),changed_input=sum(z['second_prefix_move_features_changed'] for z in cases),
         splice_diff=sum(z['cannot_obtain_by_splicing'] for z in cases),torch_numpy_max_value_error=merr,public_isolation=private))
if __name__=='__main__':run()
