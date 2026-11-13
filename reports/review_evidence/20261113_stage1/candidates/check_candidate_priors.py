"""Guarded-candidate joint-probability probe; real small network, no training."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
repo=Path(os.environ['CARBON_REPO']).resolve()
out=Path(os.environ.get('CARBON_REVIEW_OUT','/tmp/carbon-s1-independent'));out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(repo/'code'));torch.set_num_threads(1);torch.manual_seed(21)
from cnet2_model import CNet2
from cnet2_search import CachePolicy, BankruptcyGuard
import stage0_teacher as T
import cnet2_feat as F
from fastsim import FastCarbon
from eval_v53 import CFG
cfg=dict(C=16,NB=1,D=16,NH=4,L=1,E=8,G=8,TA=0,TNH=4)
net=CNet2(**cfg).eval()
with torch.no_grad():
    net.bhead.weight.zero_();net.bhead.bias.copy_(torch.tensor([2.,0.,-1.]))
weights={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
weights['meta']=np.array(json.dumps({'cfg':cfg}))
pol=CachePolicy(weights)
sim=FastCarbon(CFG).reset(seed=0)
# One legal recruitment spends 30 from the initial 80, leaving 50 and no collector.
# At the next root the production guard forces a collector at the first base;
# this changes the later base's legal recruitment budget and autoregressive prefix.
base_id=next(iter(sim.obs(0)['players'][0][1]))
sim.step({base_id:'RECPLANTER'}, {})
feat=F.Feat(0);x=feat.observe(sim.obs(0))
guard=BankruptcyGuard(T.TEACHER_GUARD_CFG)
arms,priors=T.gen_candidates(pol,x,feat.aux,teacher=2,K=16,counter=[0],guard=guard,own_uid=feat.own_uid)
true_lps=[]
illegal=[]
for a in arms:
    labels,logits,_=pol.act(x,feat.aux,teacher=2,forced=a)
    assert np.array_equal(labels,a)
    true_lps.append(T.joint_logp(logits,a))
    pf=F.Prefix(x,feat.aux)
    invalid=[]
    for s in range(F.OWN):
        if not x['tok_mask'][s]:continue
        _,_,legal=pf.feats(s)
        if not legal[int(a[s])]:invalid.append({'slot':s,'action':int(a[s])})
        pf.advance(s,int(a[s]))
    illegal.append(invalid)
true_lps=np.array(true_lps);correct=np.exp(true_lps-true_lps.max());correct/=correct.sum()
result={'review_ref':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
 'scope':'Unmodified teacher gen_candidates and real BankruptcyGuard/CachePolicy; deterministic randomly initialized small CNet2, bhead biases set for initial no-recruit preference; no optimizer or real model training.',
 'initial_game_seed':0,'legal_prefix':[{base_id:'RECPLANTER'}],'root_step':sim.step_count,'root_cash':x['cash'],'teacher':2,
 'arms':[a.tolist() for a in arms],'returned_priors':priors.tolist(),'rescored_complete_action_priors':correct.tolist(),
 'max_abs_prior_error':float(np.max(np.abs(priors-correct))),
 'illegal_prefix_actions_by_arm':illegal,'arms_with_illegal_prefix_action':sum(bool(v) for v in illegal),
 'guard_state':{'lostw':guard.lostw,'guard_base':sorted(guard.guard_base)},
 'expected_contract':'P_A is normalized raw-network joint probability of final complete candidate actions; post-guard labels must be rescored under their own autoregressive prefixes.'}
(out/'candidate-priors.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert result['max_abs_prior_error']>1e-4,'Expected current-source prior mismatch was not reproduced'
