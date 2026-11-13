"""Small local network/PPO contracts; no optimizer, training, remote task or full match.
Set CARBON_REPO; outputs JSON to stdout and optionally CARBON_REVIEW_OUT.
"""
import ast, copy, json, os, queue, sys, tempfile, subprocess, types, warnings
from pathlib import Path
import numpy as np
import torch
os.environ['BC_DEVICE']='cpu'
os.environ['RL2_SERVER_READER']='0'; os.environ['RL2_SERVER_THREADS']='1'; os.environ['RL2_SERVER_MAXB']='1'
repo=Path(os.environ['CARBON_REPO']).resolve(); sys.path.insert(0,str(repo/'code'))
torch.set_num_threads(1); torch.manual_seed(23)
import cnet2_feat as F
import cnet2_np as NP
import rl2_ppo_cnet2 as C
import rl2_ppo_rollout as RO
import rl2_ppo_server as SV
from cnet2_model import CNet2
from fastsim import FastCarbon
from rl2_ppo_train import gae
import eval_v53 as E
cfg=dict(C=16,NB=1,D=16,NH=4,L=1,E=8,G=8,TA=1,TNH=4)
net=CNet2(**cfg).eval()
with torch.no_grad():
    # Exercise the real attention computation, not identity-only inserted TA.
    for layer in net.tattn:
        layer.out.weight.normal_(0,.04); layer.f2.weight.normal_(0,.04)
ad=C.Cnet2Adapter(); weights=ad.export(net); policy=NP.Policy(weights)
C.TEACHER=2; C.TEACHER_LATE=7; C.LATE_STEP=240; C.TEMP_KIND=[.35,.8,.9]
actor=C.Cnet2Actor(weights); actor.temp_kind=True
sim=FastCarbon(E.CFG).reset(seed=42)
features=[F.Feat(0),F.Feat(1)]; records=[]; cases=[]
rng=np.random.default_rng(11)
for step in range(12):
    for side in (0,1):
        x=features[side].observe(sim.obs(side)); y,uf,mf,legal,lp,v,lpu=actor._decode(x,features[side].aux,rng,1.3,False)
        rec={k:np.asarray(x[k]) for k in C.XKEYS}; rec.update(uf=uf,mf=mf,legal=legal,y=y,logp=lp,value=v,lpu=lpu)
        records.append(rec); cases.append({'side':side,'step':step,'active':int(x['tok_mask'][:12].sum())})
    acts=[]
    for side in (0,1):
        obs=sim.obs(side); p=obs['players'][side]
        a={u:'UP' for u in p[2]}
        if step<2:
            a.update({b:'RECCOLLECTOR' for b in p[1]})
        acts.append(a)
    sim.step(*acts)
# Same information snapshot at late-style boundary; these are constructed boundary observations.
for step in (239,240,298):
    f=F.Feat(0); obs=copy.deepcopy(sim.obs(0)); obs['step']=step
    x=f.observe(obs); y,uf,mf,legal,lp,v,lpu=actor._decode(x,f.aux,rng,1.3,False)
    rec={k:np.asarray(x[k]) for k in C.XKEYS};rec.update(uf=uf,mf=mf,legal=legal,y=y,logp=lp,value=v,lpu=lpu)
    records.append(rec);cases.append({'side':0,'step':step,'active':int(x['tok_mask'][:12].sum()),'constructed_boundary':True})
# Constructed maximum own-unit occupancy covers all 12 autoregressive slots.
fmax=F.Feat(0);omax=copy.deepcopy(sim.obs(0))
omax['players'][0][2]={f'max-worker-{i}':[30+i,float(i*10),'COLLECTOR' if i<5 else 'PLANTER'] for i in range(10)}
omax['players'][0][0]=60.
xmax=fmax.observe(omax);yy,uu,mm,ll,lp,v,lpu=actor._decode(xmax,fmax.aux,rng,1.3,False)
rmax={k:np.asarray(xmax[k]) for k in C.XKEYS};rmax.update(uf=uu,mf=mm,legal=ll,y=yy,logp=lp,value=v,lpu=lpu)
records.append(rmax);cases.append({'side':0,'step':omax['step'],'active':12,'constructed_max_units':True})
buf=C.Cnet2Buffer([C.PackedTraj(records[:8]), C.PackedTraj(records[8:])]); b=buf.batch(np.arange(len(records)),torch.device('cpu'))
with torch.no_grad():
    out=ad.evaluate(net,b,1.3); chosen=out['lp'].gather(1,out['act'][:,None]).squeeze(1)
    joint=torch.zeros(len(records)).index_add_(0,out['seg'],chosen)
    loop=net(b);seq=net.forward_seq(b);tok,ctx,aux=net.encode(b);dec=net.decode(b,tok,ctx)
valid=b['legal'] & b['tok_mask'][:,:12,None]
result={'head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'cfg':cfg,'positions':cases,
        'joint_logp_max_error':float(np.max(np.abs(joint.numpy()-buf.lp0))),
        'value_max_error':float(np.max(np.abs(out['value'].numpy()-[r['value'] for r in records]))),
        'loop_seq_logit_max_error':float((loop[0]-seq[0])[valid].abs().max()),
        'forward_decode_logit_max_error':float((loop[0]-dec)[valid].abs().max()),
        'late_teacher_rows':b['teacher'][24:27].tolist(),'variable_packed_rows':len(buf.lp0)}
# Gradients of two equivalent teacher-forced paths; no optimizer or parameter update.
net.zero_grad();logits=net(b)[0]; logits[valid].square().mean().backward()
grad0={k:p.grad.clone() for k,p in net.named_parameters() if p.grad is not None}
net.zero_grad();logits=net.forward_seq(b)[0];logits[valid].square().mean().backward()
result['loop_seq_gradient_max_error']=max(float((p.grad-grad0[k]).abs().max()) for k,p in net.named_parameters() if p.grad is not None and k in grad0)
# Budget-sensitive two-base prefix, exact cash 30; no live match mutation.
x=copy.deepcopy(x); x['cash']=30.;x['slots']=2
pf=F.Prefix(x,f.aux);_,_,legal0=pf.feats(10);pf.advance(10,1);_,_,legal1=pf.feats(11)
result['base_budget']={'first_recruit_legal':bool(legal0[1]),'second_recruit_legal':bool(legal1[1]),'cash_after_first':pf.cash}
# UID histories survive canonical slot reordering by cargo.
fuid=F.Feat(0);o=copy.deepcopy(sim.obs(0));o['players'][0][2]={'uid-A':[30,2000.,'COLLECTOR'],'uid-B':[31,1000.,'COLLECTOR']};a=fuid.observe(o);old_uid=fuid.own_uid[:]
uids=list(o['players'][0][2]);prior={u:fuid.cpos(o['players'][0][2][u][0]) for u in uids}
o2=copy.deepcopy(o);o2['step']+=1
for i,u in enumerate(uids):o2['players'][0][2][u][1]=float((i+1)*1000)
a2=fuid.observe(o2)
result['uid_history']={'slot_order_changed':old_uid!=fuid.own_uid,'all_prior_cells_follow_uid':all(int(a2['hist_cell'][s,-2])==prior[u] for s,u in enumerate(fuid.own_uid[:10]) if u is not None)}
# Local serve loop + real weight loader/encoder, ordinary queue only; no worker processes.
d={k:np.stack([records[4][k]]) for k in C.XKEYS}
with tempfile.TemporaryDirectory(prefix='carbon-r5-net-') as td:
    wp=Path(td)/'small.npz'; np.savez(wp,**weights)
    RO._W_CACHE.clear();RO._REMOTE[0]=None
    learner=RO._actor('cnet2',str(wp),role='learner');snap=RO._actor('cnet2',str(wp),role='snap')
    result['roles']={'separate_objects':learner is not snap,'learner_temp_kind':learner.temp_kind,'snap_temp_kind':snap.temp_kind}
    rq=queue.Queue();responses=[queue.Queue()]
    for rid,teacher in ((1,2),(2,7)):rq.put((0,str(wp),teacher,d,rid))
    rq.put(None);SV._serve('cnet2','cpu',rq,responses,None)
    errs=[]
    for rid,teacher in ((1,2),(2,7)):
        got=responses[0].get_nowait(); direct=policy.encode({k:v[0] for k,v in d.items()},teacher)
        errs.append({'rid':got[0],'teacher':teacher,'tok_max_error':float(np.max(np.abs(got[1][0]-direct[0][:12]))),'ctx_max_error':float(np.max(np.abs(got[2][0]-direct[1])))})
    result['server_numpy_parity']=errs
    # Missing input is fault injection, not evidence of an observed deployment failure.
    rq=queue.Queue();responses=[queue.Queue()];missing=str(Path(td)/'missing.npz')
    rq.put((0,missing,2,d,9));rq.put(None)
    try:SV._serve('cnet2','cpu',rq,responses,None)
    except Exception as e:result['server_missing_weight']={'exception':type(e).__name__,'worker_received_error':not responses[0].empty(),'scope':'injected missing path; no server timeout wait or remote process'}
result['gae']={}
for n in (1,3,9):
    reward=np.zeros(n,np.float32);reward[-1]=-.31;adv,ret=gae(reward,np.linspace(-.1,.1,n,dtype=np.float32),1.,1.)
    result['gae'][str(n)]={'return_max_error':float(np.max(np.abs(ret+.31))),'length':len(ret)}

# R03 resume: execute the exact restore branch, no trainer or optimizer step.
main=next(n for n in ast.parse((repo/'code/rl2_ppo_train.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
restore=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.resume')
restore_code=compile(ast.Module(body=[restore],type_ignores=[]),'actual_resume_branch','exec')
result['resume']=[]
with tempfile.TemporaryDirectory(prefix='carbon-r5-resume-') as td:
    model=torch.nn.Linear(1,1);vm=torch.nn.Linear(1,1);path=Path(td)/'checkpoint.pt'
    op=torch.optim.Adam(model.parameters(),lr=1e-5);vop=torch.optim.Adam(vm.parameters(),lr=3e-4)
    torch.save({'net':model.state_dict(),'vnet':vm.state_dict(),'opt':op.state_dict(),'vopt':vop.state_dict(),'it':24,'kl_coef':.1},path)
    for label,req,expected in [('inherit',(None,None,None),(1e-5,3e-4,.1)),('override_all',(3e-5,8e-4,.025),(3e-5,8e-4,.025)),('zero_kl',(None,None,0.),(1e-5,3e-4,0.)),('lr_only',(2e-5,None,None),(2e-5,3e-4,.1))]:
        args=types.SimpleNamespace(resume=str(path),lr=req[0],vlr=req[1],kl=req[2]);op=torch.optim.Adam(model.parameters(),lr=1e-5);vop=torch.optim.Adam(vm.parameters(),lr=3e-4)
        scope=dict(torch=torch,args=args,net=model,vnet=vm,opt=op,vopt=vop,it0=0,kl_coef=.1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',FutureWarning);exec(restore_code,scope)
        actual=(op.param_groups[0]['lr'],vop.param_groups[0]['lr'],scope['kl_coef'])
        result['resume'].append({'case':label,'actual':actual,'passed':actual==expected})

result['no_training']=True
print(json.dumps(result,ensure_ascii=False,indent=2))
if os.environ.get('CARBON_REVIEW_OUT'):
    out=Path(os.environ['CARBON_REVIEW_OUT']);out.mkdir(parents=True,exist_ok=True);(out/'network-ppo-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
