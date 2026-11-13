"""Real frozen-artifact smoke plus current pack-source gates and build manifest checks.
All generated weights/packages stay in CARBON_REVIEW_OUT, never in the checkout.
"""
import ast,base64,hashlib,importlib.util,json,os,shutil,subprocess,sys,zlib
from pathlib import Path
from common import paths,env
repo,out=paths();os.environ.update(env());sys.path.insert(0,str(repo/'code'))
import numpy as np,cnet2_pack,eval_v53
from fastsim import FastCarbon
sha=lambda b:hashlib.sha256(b).hexdigest()
def literals(p,names):
 d={}
 for n in ast.parse(p.read_bytes()).body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names:d[n.targets[0].id]=ast.literal_eval(n.value)
 return d
rows=[];original_actions={}
for name in ['n1r_int8_reference.py','echo3_int8.py']:
 p=repo/'code/packs'/name;b=p.read_bytes();d=literals(p,{'CONF','_SRC','_M1','_Q1','_B1'})
 g={'__name__':'submission'};exec(compile(b,'<submission>','exec'),g);sim=FastCarbon(eval_v53.CFG).reset(seed=970500);actions=[]
 for _ in range(2):
  obs=sim.obs(0);a=g['agent'](obs,eval_v53.CFG);assert set(a)<=set(obs['players'][0][1])|set(obs['players'][0][2]);actions.append(a);sim.step(a,{})
 original_actions[name]=actions
 rows.append({'pack':name,'bytes':len(b),'md5':hashlib.md5(b).hexdigest(),'sha256':sha(b),'conf':d['CONF'],'exec_without_file':'__file__' not in g,'two_step_actions':actions,'stats':g['_AG'].stats,'embedded_source_sha256':{k:sha(v.encode()) for k,v in d['_SRC'].items()}})
 if name=='n1r_int8_reference.py':
  w=cnet2_pack.dequantize(zlib.decompress(base64.b64decode(d['_B1'])),d['_Q1']);w['meta']=np.array(d['_M1']);np.savez(out/'n1r_dequantized.npz',**w);conf=d['CONF']
base=repo/'code/packs';keys=['cnet2_feat','cnet2_np','fastsim','cnet2_search'];weights=out/'n1r_dequantized.npz';e=env();e['CNET2_NO_ROLL']='1'
def command(packer,args):
 r=subprocess.run([sys.executable,'-B',str(packer),*map(str,args)],env=e,capture_output=True,text=True,timeout=20)
 return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
results={'reviewed_sha':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'frozen_artifacts':rows}
results['clean_check']=command(base/'cnet2_search_pack.py',['--check-source'])
results['current_build']=command(base/'cnet2_search_pack.py',[out/'rebuilt_n1r.py',repr(conf),weights]);assert results['current_build']['exit_code']==0
p=out/'rebuilt_n1r.py';d=literals(p,{'_SRC','_BUILD_HASHES'});results['build_hashes_match_embedded']={k:d['_BUILD_HASHES'][k]==hashlib.md5(d['_SRC'][k].encode()).hexdigest() for k in keys}
g={'__name__':'submission'};exec(compile(p.read_bytes(),'<submission>','exec'),g);sim=FastCarbon(eval_v53.CFG).reset(seed=970500);a=[]
for _ in range(2):
 action=g['agent'](sim.obs(0),eval_v53.CFG);a.append(action);sim.step(action,{})
results['rebuilt_two_step_matches_frozen']=a==original_actions['n1r_int8_reference.py']
S=g['_P']['cnet2_search'];F=g['_P']['cnet2_feat'];NP=g['_P']['cnet2_np'];obs=FastCarbon(eval_v53.CFG).reset(seed=970500).obs(0);f=F.Feat(0);x=f.observe(obs);pol=S.CachePolicy(g['_W'][0]);NP.act_tta(pol,x,f.aux,teacher=2,k=2);got=pol.enc_for(x,2);expected=NP.Policy.encode(pol,x,2);results['current_tta_encoding_max_abs_error']=max(float(np.abs(u-v).max()) for u,v in zip(got,expected))
# A drift inside an otherwise complete repository is rejected.
scratch=out/'source-drift/code';packdir=scratch/'packs';packdir.mkdir(parents=True,exist_ok=True)
for k in keys:
 q=scratch/(k+'.py')
 if not q.exists():q.symlink_to(repo/'code'/(k+'.py'))
for name in [*(k+'.py' for k in keys),'cnet2_pack.py','cnet2_search_pack.py']:shutil.copyfile(base/name,packdir/name)
p=packdir/'cnet2_search.py';p.write_text(p.read_text()+'\n# local review drift injection\n')
results['local_drift_check']=command(packdir/'cnet2_search_pack.py',['--check-source'])
results['local_drift_build']=command(packdir/'cnet2_search_pack.py',[out/'must_not_exist.py',repr(conf),weights]);results['local_drift_output_exists']=(out/'must_not_exist.py').exists()
# The documented standalone build may contain different source; clean-tree
# --check-source accepts no candidate argument, so it cannot certify that pack.
stand=out/'standalone';stand.mkdir(exist_ok=True)
for name in [*(k+'.py' for k in keys),'cnet2_pack.py','cnet2_search_pack.py']:shutil.copyfile(base/name,stand/name)
shutil.copyfile(base/'historical/cnet2_search_v12_frozen.py',stand/'cnet2_search.py')
results['standalone_drift_build']=command(stand/'cnet2_search_pack.py',[out/'standalone_old.py',repr(conf),weights]);d=literals(out/'standalone_old.py',{'_BUILD_HASHES'})
results['standalone_search_hash']=d['_BUILD_HASHES']['cnet2_search'];results['authoritative_search_hash']=hashlib.md5((repo/'code/cnet2_search.py').read_bytes()).hexdigest()
results['clean_repo_recheck_after_standalone_build']=command(base/'cnet2_search_pack.py',['--check-source'])
results['fixture_limitation']='Rebuild uses dequantized n1r embedded q8 weights, not the unavailable pre-quantization training weight file; frozen artifacts never modified.'
(out/'pack-export-results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:v for k,v in results.items() if k not in ['frozen_artifacts','current_build']},indent=2))
