"""Rebuild current and vendored packages; check embedded sources and cache restoration.
No ladder/deployment. CARBON_REPO and CARBON_WEIGHTS can override local inputs.
"""
import ast, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
repo=Path(os.environ['CARBON_REPO']).resolve()
out=Path(os.environ.get('CARBON_REVIEW_OUT',Path(__file__).parent)).resolve();out.mkdir(parents=True,exist_ok=True)
weights=Path(os.environ.get('CARBON_WEIGHTS', str(out/'baseline.npz')))
sys.path.insert(0,str(repo/'code'))
import numpy as np
import eval_v53
from fastsim import FastCarbon
os.environ['CNET2_NO_ROLL']='1'
conf={'SEARCH':False,'ROLL':'self','GUARD':True,'GUARD_NARROW':True,'MODE':'sweep','MIX':False,'TEACHER':2,'TEACHER_LATE':2,'LATE_STEP':240,'BUDGET':0.7,'HARD':0.85,'ABORT_DL':0.95,'START_DL':None}
rows=[];modules={}
for label,folder in [('authoritative',repo/'code'),('vendored',repo/'code/packs')]:
 spec=importlib.util.spec_from_file_location('packer_'+label,folder/'cnet2_search_pack.py');packer=importlib.util.module_from_spec(spec);spec.loader.exec_module(packer)
 path=out/(label+'_standard.py');packer.main(str(path),repr(conf),str(weights))
 g={'__name__':'submission'};exec(compile(path.read_text(),'<submission>','exec'),g);modules[label]=g
 sim=FastCarbon(eval_v53.CFG).reset(seed=970500)
 observed=[]
 for i in range(2):
  a=g['agent'](sim.obs(0),eval_v53.CFG);observed.append(a);sim.step(a,{})
 S=g['_P']['cnet2_search'];NP=g['_P']['cnet2_np'];F=g['_P']['cnet2_feat']
 obs=FastCarbon(eval_v53.CFG).reset(seed=970500).obs(0);f=F.Feat(0);x=f.observe(obs)
 policy=S.CachePolicy(g['_W'][0]);NP.act_tta(policy,x,f.aux,teacher=1,k=2)
 expected=NP.Policy.encode(policy,x,1)
 # Follow each version's actual consumer path: current enc_for vs vendored last_enc.
 got=policy.enc_for(x,1) if hasattr(policy,'enc_for') else policy.last_enc
 delta=max(float(np.abs(u-v).max()) for u,v in zip(got,expected))
 row={'version':label,'package_bytes':path.stat().st_size,'embedded_search_sha256':hashlib.sha256(g['_SRC']['cnet2_search'].encode()).hexdigest(),'source_search_sha256':hashlib.sha256((folder/'cnet2_search.py').read_bytes()).hexdigest(),'standard_conf_two_step_actions':observed,'standard_conf_stats':g['_AG'].stats,'tta_consumer_encoding_max_abs_error':delta,'has_enc_for':hasattr(policy,'enc_for')};rows.append(row)
identity=[]
for name in ('cnet2_feat.py','cnet2_np.py','cnet2_pack.py','cnet2_search.py','cnet2_search_pack.py','fastsim.py'):
 a=hashlib.sha256((repo/'code'/name).read_bytes()).hexdigest();b=hashlib.sha256((repo/'code/packs'/name).read_bytes()).hexdigest();identity.append({'file':name,'code_sha256':a,'packs_sha256':b,'identical':a==b})
result={'git_sha':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'fixture':str(weights),'fixture_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'limitation':'Historical b128 fixture, not latest Echo3/n1r weights; two-step and encoding probes only','copies':identity,'packages':rows}
(out/'pack-source-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
