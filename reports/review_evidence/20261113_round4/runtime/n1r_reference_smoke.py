"""Two-step exec/no-__file__ smoke of the tracked n1r reference; no timing certification."""
import hashlib,json,os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
repo=Path(os.environ['CARBON_REPO']).resolve();out=Path(os.environ.get('CARBON_REVIEW_OUT',Path(__file__).parent)).resolve()
sys.path.insert(0,str(repo/'code'))
import eval_v53
from fastsim import FastCarbon
p=repo/'code/packs/n1r_int8_reference.py';raw=p.read_bytes();g={'__name__':'submission'}
exec(compile(raw,str(p),'exec'),g)
sim=FastCarbon(eval_v53.CFG).reset(seed=970500);rows=[]
for _ in range(2):
 obs=sim.obs(0);a=g['agent'](obs,eval_v53.CFG)
 owned=set(obs['players'][0][1])|set(obs['players'][0][2]);assert set(a)<=owned
 assert all(v in {'UP','RIGHT','DOWN','LEFT','RECPLANTER','RECCOLLECTOR'} for v in a.values())
 rows.append({'step':obs['step'],'actions':a,'owned_unit_ids':True});sim.step(a,{})
r={'pack':'code/packs/n1r_int8_reference.py','md5':hashlib.md5(raw).hexdigest(),'bytes':len(raw),'exec_without_file': '__file__' not in g,'rows':rows,'stats':g['_AG'].stats,'limitation':'Only two steps against idle, no performance/resource/strength certification; not Echo3'}
(out/'n1r-reference-smoke.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
