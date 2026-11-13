"""62e0dee interfaces only. No optimizer step updates a model; smoke optimizer/loader/loss are stubs.
Set CARBON_REPO to the isolated latest checkout; results only stdout/CARBON_REVIEW_OUT.
"""
import contextlib, io, json, math, os, subprocess, sys, tempfile, types, hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
os.environ['BC_DEVICE']='cpu';os.environ['BC_RAM']='1';os.environ['BC_BS']='1'
repo=Path(os.environ['CARBON_REPO']).resolve();sys.path.insert(0,str(repo/'code'));torch.set_num_threads(1)
BASE='98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c'
import cnet2_model as M
import cnet2_np as NP
import cnet2_feat as F
import cnet2_train as CT
import cnet2_pack as PACK
import rl2_ppo_cnet2 as C
import stage0_smoke_heads as SM
import eval_v53 as E
from fastsim import FastCarbon
oldsrc=subprocess.check_output(['git','-C',str(repo),'show',BASE+':code/cnet2_model.py'],text=True)
oldmod=types.ModuleType('review_old_cnet2');oldmod.__file__=str(repo/'code/cnet2_model.py')
exec(compile(oldsrc,'98ffadfa:code/cnet2_model.py','exec'),oldmod.__dict__)
cfg=dict(C=16,NB=1,D=16,NH=4,L=1,E=8,G=8)
torch.manual_seed(9);old=oldmod.CNet2(**cfg).eval()
oldsd={k:v.detach().cpu().numpy().copy() for k,v in old.state_dict().items()}
oldw=dict(oldsd,meta=np.array(json.dumps({'cfg':old.cfg})))
result={'head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'old_model_source':BASE,'scope':'Real old/new small models and loading/loss/pack interfaces. No real training; smoke main uses nonmutating stub optimizer and injected losses.'}
try:C.Cnet2Adapter().make_model(oldw);result['old_npz_ppo']={'raised':False}
except Exception as e:result['old_npz_ppo']={'raised':True,'exception':type(e).__name__,'message':str(e)}

if os.environ.get('CARBON_OLD_WEIGHTS'):
    realpath=Path(os.environ['CARBON_OLD_WEIGHTS']);realweights=dict(np.load(realpath));realmeta=json.loads(str(realweights['meta']))
    realresult={'basename':realpath.name,'sha256':hashlib.sha256(realpath.read_bytes()).hexdigest(),'cfg':realmeta.get('cfg'),'recorded_epoch':realmeta.get('epoch'),'note':'Existing historical trained b128 artifact from prior runtime review, not a claim about latest online weights.'}
    try:C.Cnet2Adapter().make_model(realweights);realresult['raised']=False
    except Exception as e:realresult.update(raised=True,exception=type(e).__name__,message=str(e))
    result['existing_trained_npz_ppo']=realresult

net=M.CNet2(**cfg).eval();missing=net.load_state_dict(old.state_dict(),strict=False)
result['bc_legacy_load_allowlist']={'missing':list(missing.missing_keys),'unexpected':list(missing.unexpected_keys),'only_heads':all(k.startswith(('utility.','wdl.','margin.')) for k in missing.missing_keys)}
# A migrated model state alone cannot repair the old Adam parameter groups.
oldopt=torch.optim.Adam(old.parameters(),lr=1e-5);newopt=torch.optim.Adam(net.parameters(),lr=1e-5)
try:newopt.load_state_dict(oldopt.state_dict());result['old_optimizer']={'raised':False}
except Exception as e:result['old_optimizer']={'raised':True,'exception':type(e).__name__,'message':str(e),'old_parameters':len(oldopt.param_groups[0]['params']),'new_parameters':len(newopt.param_groups[0]['params'])}
try:net.load_state_dict(old.state_dict());result['old_checkpoint_net']={'raised':False}
except Exception as e:result['old_checkpoint_net']={'raised':True,'exception':type(e).__name__,'message':str(e)}
# Feature/Prefix/teacher-forcing interface with an old-policy-produced record.
sim=FastCarbon(E.CFG).reset(seed=42);obs=sim.obs(0);feat=F.Feat(0);x=feat.observe(obs)
actor=C.Cnet2Actor(oldw);yy,uf,mf,legal,lp,v,lpu=actor._decode(x,feat.aux,np.random.default_rng(11),1.,False)
r={k:np.asarray(x[k]) for k in C.XKEYS};r.update(uf=uf,mf=mf,legal=legal,y=yy,logp=lp,value=v,lpu=lpu)
b=C.Cnet2Buffer([[r]]).batch([0],torch.device('cpu'));b['carbon']=torch.zeros((1,F.N));b['value']=torch.zeros(1)
with torch.no_grad():
    before=old(b);after=net(b);extended=net.forward_heads(b)
valid=b['legal'] & b['tok_mask'][:,:12,None]
result['old_policy_preserved_after_allowed_load']={'max_logit_error':float((before[0]-after[0])[valid].abs().max()),'max_legacy_value_error':float((before[1]-after[1]).abs().max()),'forward_heads_legacy_logit_error':float((after[0]-extended[0])[valid].abs().max()),'new_shapes':[list(t.shape) for t in extended[3:]]}
net.zero_grad();lpi,lv,la,lg=CT.losses(net,b);(lpi+.2*lv+la).backward()
heads=('utility.','wdl.','margin.')
result['normal_bc_gradients']={'new_heads':{k:p.grad is not None for k,p in net.named_parameters() if k.startswith(heads)},'stem_has_gradient':net.stem.weight.grad is not None,'legacy_value_has_gradient':net.value.weight.grad is not None,'loss':float(lpi+.2*lv+la)}
net.zero_grad(set_to_none=True)
for k,p in net.named_parameters():p.requires_grad_(k.startswith(heads))
loss,*_=SM.head_losses(net,b);loss.backward()
result['dedicated_smoke_gradient_path']={'new_heads':{k:p.grad is not None for k,p in net.named_parameters() if k.startswith(heads)},'stem_has_gradient':net.stem.weight.grad is not None,'loss_finite':bool(torch.isfinite(loss))}
neww={k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()};neww['meta']=oldw['meta']
oldpol=NP.Policy(oldw);newpol=NP.Policy(neww)
a0,l0,v0=oldpol.act(x,feat.aux,teacher=1,forced=yy);a1,l1,v1=newpol.act(x,feat.aux,teacher=1,forced=yy)
mut=dict(neww)
for k in list(mut):
    if k.startswith(heads):mut[k]=np.full_like(mut[k],12345.)
a2,l2,v2=NP.Policy(mut).act(x,feat.aux,teacher=1,forced=yy)
mask=np.asarray(legal)&x['tok_mask'][:12,None]
result['numpy_old_new_heads']={'policy_logits_max_error':float(np.max(np.abs(l0[mask]-l1[mask]))),'legacy_value_error':abs(float(v0-v1)),'large_new_head_mutation_changes_logits':bool(np.any(l1[mask]!=l2[mask])),'large_new_head_mutation_changes_value':bool(v1!=v2)}
# Probe actual smoke main without any model training: only leaf test-loss backward, optimizer step is no-op.
class NoUpdate:
    def __init__(self,*a,**kw):pass
    def zero_grad(self,*a,**kw):pass
    def step(self,*a,**kw):pass

def fake_loader(D,idx,bs,**kw):
    yield {'probe_only':True}

def smoke_case(path,kind):
    calls=[0]
    def fake_losses(net,b):
        calls[0]+=1
        bad=kind=='nonfinite_train' or (kind=='nonfinite_validation' and calls[0]>1)
        val=float('nan') if bad else 1.
        return torch.tensor(val,requires_grad=True),val,val,val,.5,val,(.5,.5)
    out=io.StringIO();code=0
    with patch.dict(os.environ,{'BC_INIT':str(path)}),patch.object(sys,'argv',['stage0_smoke_heads.py','stub_ds','1']),patch.object(CT,'load',lambda data:({},np.array([False,True]),{})),patch.object(CT,'loader',fake_loader),patch.object(SM,'head_losses',fake_losses),patch.object(torch.optim,'AdamW',NoUpdate),patch.object(torch.optim.lr_scheduler,'OneCycleLR',NoUpdate),contextlib.redirect_stdout(out):
        try:SM.main()
        except SystemExit as e:code=e.code
    lines=out.getvalue().splitlines();summary=next((json.loads(s[6:]) for s in lines if s.startswith('SMOKE ')),None)
    return {'exit_equivalent':code,'summary':summary,'init_report':json.loads(lines[0]),'scope':'Actual main/control flow; stub loader/loss, no-op optimizer/scheduler; no model parameters updated.'}
with tempfile.TemporaryDirectory(prefix='carbon-latest-heads-') as td:
    td=Path(td);op=td/'old.npz';np.savez(op,**oldw);nw=td/'new.npz';np.savez(nw,**neww)
    result['smoke_constant_loss']=smoke_case(op,'constant')
    result['smoke_nonfinite_train']=smoke_case(op,'nonfinite_train')
    result['smoke_nonfinite_validation']=smoke_case(op,'nonfinite_validation')
    broken=dict(oldw);del broken['stem.weight'];bp=td/'broken.npz';np.savez(bp,**broken)
    result['smoke_missing_old_trunk']=smoke_case(bp,'constant')
    # Actual pack writer; execute only one action, preserving existing sys.modules afterwards.
    packpath=td/'new_pack.py';stdout=io.StringIO()
    with contextlib.redirect_stdout(stdout):PACK.main(str(nw),str(packpath))
    mods={k:sys.modules.get(k) for k in PACK.MODS}
    try:
        scope={'__file__':str(packpath)};exec(compile(packpath.read_text(),str(packpath),'exec'),scope)
        act=scope['agent'](obs,E.CFG)
        result['new_weight_pack']={'writer_ok':True,'one_action_ok':True,'actions':act,'bytes':packpath.stat().st_size}
    finally:
        for k,m in mods.items():
            if m is None:sys.modules.pop(k,None)
            else:sys.modules[k]=m
result['new_heads_main_trainer_status']='Only dedicated forward_heads/smoke uses heads. Existing BC/PPO/NumPy still uses legacy value; interface addition is not an E[U] training migration.'
def clean(x):
    if isinstance(x,float) and not math.isfinite(x):return str(x)
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x
text=json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n';print(text,end='')
if os.environ.get('CARBON_REVIEW_OUT'):
    out=Path(os.environ['CARBON_REVIEW_OUT']);out.mkdir(parents=True,exist_ok=True);(out/'latest-heads-results.json').write_text(text)
