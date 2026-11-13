#!/usr/bin/env python3
"""Execute only fixed-ref evaluate/targets with tiny in-memory fixtures, no training.
Usage: python check_latest_calibration.py --repo /path/to/repo --out /tmp/evidence
"""
import argparse, ast, hashlib, json, math, subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ref',default='57557a83a73cd4d6b26b56ebebb4c0df8e13a155');a=p.parse_args()
 def read(f):return subprocess.check_output(['git','-C',str(a.repo),'show',f'{a.ref}:{f}'],text=True)
 path='code/stage0_train_heads.py';src=read(path)
 funcs=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name in ('evaluate','targets')]
 ns={'np':np,'torch':torch,'CT':SimpleNamespace(loader=lambda D,va,bs,aug,shuffle:iter([D]))}
 exec(compile(ast.Module(body=funcs,type_ignores=[]),'fixed_ref_eval_functions','exec'),ns)
 class FixtureNet:
  def eval(self):return self
  def forward_heads(self,b):
   n=len(b['value']);z=torch.zeros(n);return z,z,z,b['prediction'],torch.zeros(n,3),z
 def run(label,raw_margins,pred):
  value=np.tanh(np.array(raw_margins,dtype=np.float64)/3000).astype(np.float16).astype(np.float32)
  b={'value':torch.from_numpy(value),'prediction':torch.tensor(pred,dtype=torch.float32)}
  actual=ns['evaluate'](FixtureNet(),b,np.arange(len(pred)),len(pred))
  target=ns['targets'](b)[1].numpy();prediction=np.array(pred)
  bins=np.clip((prediction*10).astype(int),0,9);ece=0.
  for j in range(10):
   mask=bins==j
   if mask.any():ece+=float(mask.mean()*abs(prediction[mask].mean()-target[mask].mean()))
  return {'name':label,'actual_targets':target.tolist(),'predictions':pred,'actual_repository_evaluate':actual,'prediction_binned_absolute_calibration':ece}
 c1=run('calibrated_constant_mean_false_alarm',[-1000*math.log(4),1000*math.log(4)],[.5,.5])
 c2=run('miscalibrated_predictions_false_pass',[0,0],[.2,.8])
 assert c1['actual_repository_evaluate']['val_u_calib_binned']>.39 and c1['prediction_binned_absolute_calibration']<1e-6
 assert c2['actual_repository_evaluate']['val_u_calib_binned']==0 and c2['prediction_binned_absolute_calibration']>.29
 result={'review_ref':a.ref,'scope':'Only exact evaluate/targets AST functions, in-memory fixture loader/net. No optimizer or training.',
 'source_sha256':{path:hashlib.sha256(src.encode()).hexdigest(),'code/stage0_smoke_heads.py':hashlib.sha256(read('code/stage0_smoke_heads.py').encode()).hexdigest()},
 'fixtures':[c1,c2],
 'formal_target':'Mean calibration requires E[observed U | predicted u] = predicted u; binning on observed U measures a different conditional residual.',
 'training_scope_change':{'imports_smoke_head_losses':'from stage0_smoke_heads import head_losses' in src,'calls_same_loss':'head_losses(net, b)' in src,'writes_weights':'np.savez(out, **sd_out)' in src,'margin_loss_still_smooth_l1':'Fn.smooth_l1_loss(mk, mg / 1000.0)' in read('code/stage0_smoke_heads.py')}}
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'latest575-calibration-evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
