from pathlib import Path
import sys,inspect,importlib.util,json
import numpy as np
import torch
import collect as C
from collect import HERE,ROOT,P,B,panel
sys.path.insert(0,str(B.ACTOR/'blended_policy_return_dev'))
import policy
sys.path.insert(0,str(B.ACTOR/'master3s2_policy_distill_dev'))
import fit as shared
base=shared.base
sp=importlib.util.spec_from_file_location('serving_repacker',B.ACTOR/'blended_actor_dev/serving/build.py');pack=importlib.util.module_from_spec(sp);sp.loader.exec_module(pack)
def native_teacher_forced(m):
    needle='action = int(np.argmax(np.where(legal, combined, -np.inf)))';src=inspect.getsource(m._return_decode)
    assert src.count(needle)==1
    src=src.replace('encoded, apply_residual):','encoded, apply_residual, forced, fixed):').replace(needle,
             'fixed[slot]=combined-0.25*scores[1]\n        action=int(forced[slot])\n        assert legal[action]')
    ns=dict(vars(m));exec(src,ns);return ns['_return_decode']
def materialize():
    m=policy.load(B.PACK);compiled=native_teacher_forced(m);rows=[];refs=[];cards=[]
    for c in P['cards']:
        if c['split']!='fit':continue
        card=HERE/f"card_{c['id']:02d}.json.gz";data=B.read(card);assert data['protocol']==B.sha(HERE/'protocol.json')
        source=B.read(ROOT/c['source']);feat=panel.F.Feat(c['seat']);sim=panel.FastCarbon(source['game_configuration']).reset(seed=c['seed'])
        for t,cmd in enumerate(source['actions'][:158]):
            obs=sim.obs(c['seat']);x=feat.observe(obs)
            if t>=150:
                z=data['records'][t-150];assert z['t']==t and z['executed']==cmd[c['seat']]
                orig=policy.Root(m,feat,x);a,_=orig.decode(False);b,_=orig.decode(True);y,guarded=policy.guard(m,x,a,b)
                assert feat.to_actions(y)==cmd[c['seat']]
                actual=feat.labels(z['teacher']);assert feat.to_actions(actual)==z['teacher']
                r=panel.projected(feat,x,np.ones((12,5),np.float32),actual,0,c['id'],t)
                repair=np.flatnonzero(r['prefix_repaired']);cut=int(repair[0]) if len(repair) else 12
                r['eligible']=(np.arange(12)<cut)&(actual>=0)&(r['legal'].sum(-1)>1)
                fixed=np.zeros((12,5),np.float32);out=compiled(x,feat.aux,orig.enc,not guarded,r['y'],fixed)
                assert np.array_equal(out,r['y'])
                r['fixed']=fixed;rows.append(r);cards.append([c['id'],t,cut])
            sim.step(*cmd)
    return {k:np.stack([r[k] for r in rows]) for k in set(base.trainer.LOAD)|{'fixed','eligible'}},m,cards
def train():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(785251)
    d,m,cards=materialize();net,meta=base.model(shared.PARENT,shared.PARENT_HASH)
    parent_deq=m._NETS[1].w;net.load_state_dict({k:torch.from_numpy(parent_deq[k].copy()) for k in net.state_dict()});net.eval()
    fitparams=[p for k,p in net.named_parameters() if p.requires_grad_(not k.startswith(shared.NOT_POLICY)).requires_grad]
    opt=torch.optim.AdamW(fitparams,lr=.00008,weight_decay=.0001)
    fixed=torch.from_numpy(d['fixed']);legal=torch.from_numpy(d['legal']);mask=torch.from_numpy(d['eligible'])
    full=np.arange(len(cards));assert len(full)==64
    with torch.no_grad():lg0=(fixed+.25*net(base.batch(d,full))[0]).masked_fill(~legal,-1e9);r0=lg0.softmax(-1);l0=lg0.log_softmax(-1)
    rng=np.random.default_rng(785251);order=np.concatenate([rng.permutation(full) for _ in range(3)]).reshape(24,8)
    errors=[]
    for ix in order:
        b=base.batch(d,ix);logits=(fixed[ix]+.25*net(b)[0]).masked_fill(~legal[ix],-1e9)
        logp=logits.log_softmax(-1);targets=torch.from_numpy(d['y'][ix].astype(np.int64)).clamp_min(0)
        ce=-logp.gather(-1,targets[:,:,None]).squeeze(-1);kl=(r0[ix]*(l0[ix]-logp)).sum(-1);loss=(ce+kl)[mask[ix]].mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(fitparams,1,error_if_nonfinite=True);opt.step();errors.append(float(loss.detach()))
    with np.load(shared.PARENT,allow_pickle=False) as ref:
        assert pack.quantize(shared.PARENT,ref)['_W64']==m._W64
        w={k:ref[k].copy()+(net.state_dict()[k].detach().numpy()-parent_deq[k]) for k in net.state_dict()}
        assert np.isfinite([np.max(np.abs(v)) for v in w.values()]).all()
        out=HERE/'student.npz';np.savez(out,**w,meta=ref['meta']);embedded=pack.quantize(out,ref)
    candidate=HERE/'student.py';candidate.write_text(pack.substitute(B.PACK.read_text(),embedded));new=policy.load(candidate)
    assert new._B1==m._B1 and new._W64!=m._W64 and new._HYBRID==m._HYBRID and np.array_equal(new._RETURN_THETA,m._RETURN_THETA)
    receipt=dict(protocol=B.sha(HERE/'protocol.json'),fit_source=B.sha(Path(__file__)),parent=B.sha(B.PACK),parent_student=shared.PARENT_HASH,
            student=B.sha(candidate),weights=B.sha(out),order=order.tolist(),roots=cards,valid_slots=int(mask.sum()),excluded_illegal_prefix=sum(c[2]!=12 for c in cards),
            unchanged_partner=True,unchanged_guard=True,changed_fp32=sum(np.count_nonzero(v-parent_deq[k])>0 for k,v in w.items()),loss=errors)
    receipt['changed_fp32']=int(sum(np.count_nonzero(net.state_dict()[k].detach().numpy()-parent_deq[k]) for k in w))
    B.live.save(HERE/'fit.json',receipt)
if __name__=='__main__':train()
