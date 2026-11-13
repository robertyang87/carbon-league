#!/usr/bin/env python3
"""Small exact-head-loss mathematical fixtures; no optimizer step or model training.
python check_latest_heads.py --repo /path/to/repo --out /tmp/new-head-evidence
Requires torch and numpy, like the project's training scripts.
"""
import argparse, ast, hashlib, json, math, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ref',default='62e0dee16419725ece6f18a22f095c83bb6fcd0f');a=p.parse_args()
 def read(f):return subprocess.check_output(['git','-C',str(a.repo),'show',f'{a.ref}:{f}'],text=True)
 names=['code/cnet2_model.py','code/stage0_smoke_heads.py','code/stage0_heads_patch.py','code/cnet2_build.py']
 src={f:read(f) for f in names}
 fn=next(n for n in ast.parse(src['code/stage0_smoke_heads.py']).body if isinstance(n,ast.FunctionDef) and n.name=='head_losses')
 ns={'torch':torch,'Fn':Fn};exec(compile(ast.Module(body=[fn],type_ignores=[]),'fixed_ref_head_losses','exec'),ns)
 vals=np.array([0]*9+[math.tanh(10000/3000)],dtype=np.float16).astype(np.float32)
 b={'value':torch.from_numpy(vals)}
 recovered=3000*torch.atanh(b['value'].clamp(-.999,.999))
 class StubNet:
  def __init__(self,k):self.mk=torch.tensor(k,dtype=torch.float32,requires_grad=True)
  def forward_heads(self,b):
   n=len(b['value']);z=torch.zeros(n);return z,z,z,torch.full((n,),.5),torch.zeros(n,3),self.mk.expand(n)
 gradients=[]
 for k in [1/9,float(recovered.mean()/1000)]:
  net=StubNet(k);got=ns['head_losses'](net,b);got[0].backward()
  gradients.append({'prediction_m_over_1000':k,'total_loss':float(got[0].detach()),'margin_smooth_l1':got[3],'total_loss_gradient_wrt_mk':float(net.mk.grad),'margin_MAE':got[5]})
 assert abs(gradients[0]['total_loss_gradient_wrt_mk'])<1e-6
 assert gradients[1]['total_loss_gradient_wrt_mk']>.1
 raw=np.array([0.,.01,100.,1000.,4000.,5000.,10000.,11400.,13320.,20000.])
 stored=np.tanh(raw/3000).astype(np.float16)
 recovered2=3000*torch.atanh(torch.from_numpy(stored.astype(np.float32)).clamp(-.999,.999))
 rows=[]
 for m,v,mr in zip(raw,stored,recovered2.tolist()):
  w=1 if m>0 else .5 if m==0 else 0;wr=1 if mr>0 else .5 if mr==0 else 0
  ut=.5*w+.5/(1+math.exp(-m/1000));ur=.5*wr+.5/(1+math.exp(-mr/1000))
  rows.append({'raw_margin':float(m),'stored_float16':float(v),'recovered_margin':mr,'margin_error':mr-float(m),'normal_U_error':ur-ut})
 result={'review_ref':a.ref,'scope':'Exact extracted head_losses with constant output stub; backward derivatives only, no optimizer step, no dataset training.',
 'source_sha256':{f:hashlib.sha256(s.encode()).hexdigest() for f,s in src.items()},
 'expectation_vs_huber':{'distribution':'90% M=0, 10% M=10000, passed through actual float16 label storage then inverse transform','true_mean_raw_margin':1000.,'mean_recovered_margin':float(recovered.mean()),'smooth_l1_optimum_margin':1000/9,'gradient_probes':gradients},
 'inverse_tanh_float16':rows,'clamp_mathematical_cap':3000*math.atanh(.999),
 'outcome_kind_not_identifiable_from_margin':{'margin':100,'normal_win_U':.5+.5/(1+math.exp(-.1)),'single_fault_win_U':1.0,'comment':'The same margin field cannot determine whether soft margin applies. No claim that ds_m3s2 actually contains abnormal games.'},
 'implemented_and_not_implemented':{'utility_sigmoid_and_U_MSE':True,'WDL_three_class_CE':True,'unbounded_margin_output':True,'margin_MSE_in_smoke':False,'separate_bare_mixed_heads':False,'raw_terminal_margin_label_in_builder':False,'smoke_exports_trained_heads':False}}
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'latest-heads-evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
