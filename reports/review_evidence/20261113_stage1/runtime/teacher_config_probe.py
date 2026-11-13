"""Verify that producer style row2 versus fixed gate row1 is a real model input difference.
Run guard_probe.py first for its untrained tiny-network fixture.
"""
import json,os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
repo=Path(os.environ['CARBON_REPO']).resolve();out=Path(os.environ.get('CARBON_REVIEW_OUT','/tmp/carbon-stage1-runtime')).resolve();sys.path.insert(0,str(repo/'code'))
import numpy as np,cnet2_feat as F,cnet2_np as NP
from fastsim import FastCarbon
from eval_v53 import CFG
w=dict(np.load(out/'tiny_heads.npz'));p=NP.Policy(w);rows=[]
for seed in range(20):
 sim=FastCarbon(CFG).reset(seed=seed)
 for seat in (0,1):
  obs=sim.obs(seat);obs['step']=50;obs['players'][seat][2]={'a':[30+seed,30+seed,'COLLECTOR'],'b':[150+seed,0,'PLANTER']}
  f=F.Feat(seat);x=f.observe(obs);r1=p.act(x,f.aux,teacher=1,want_u=True);r2=p.act(x,f.aux,teacher=2,want_u=True)
  rows.append({'seed':seed,'seat':seat,'label_diff':bool(np.any(r1[0]!=r2[0])),'actions_t1':f.to_actions(r1[0]),'actions_t2':f.to_actions(r2[0]),'max_logits_delta':float(np.abs(r1[1]-r2[1]).max()),'utility_t1':r1[3],'utility_t2':r2[3]})
result={'fixture':'same seed8731 untrained tiny model as guard_probe; no claim about real heads_v2 effect size','producer_teacher':2,'gate_teacher':1,'states':len(rows),'action_different_states':sum(x['label_diff'] for x in rows),'max_logits_delta':max(x['max_logits_delta'] for x in rows),'max_utility_delta':max(abs(x['utility_t1']-x['utility_t2']) for x in rows),'rows':rows}
(out/'teacher-config-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
