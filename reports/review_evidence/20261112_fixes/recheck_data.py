"""Reproduce split inheritance and dataset-cache behavior using temporary fixtures.

Set CARBON_REPO to the repository root when running outside its tree. Otherwise,
locate the root by walking upward from this script and finding code/fastsim.py.
Results are printed to stdout. Optional CARBON_REVIEW_OUT names an output
directory in which data-results.json is also written. Business source files and
remote services are never modified; all generated datasets live in a temporary
directory that is deleted after the run.

The replay fixture declares 300 steps in its index to pass the normal scanner,
but contains only three decisions, intentionally isolating cache behavior.
"""
import contextlib
import gzip
import io
import json
import os
import pathlib
import sys
import tempfile


def find_repo():
    configured = os.environ.get("CARBON_REPO")
    if configured:
        repo = pathlib.Path(configured).expanduser().resolve()
        if not (repo / "code" / "fastsim.py").is_file():
            raise SystemExit("CARBON_REPO must contain code/fastsim.py")
        return repo
    for directory in pathlib.Path(__file__).resolve().parents:
        if (directory / "code" / "fastsim.py").is_file():
            return directory
    raise SystemExit("Repository root not found; set CARBON_REPO")


os.environ["BUILD_WORKERS"] = "1"
sys.path.insert(0, str(find_repo() / "code"))
import numpy as np
import cnet2_build as B
import merge_dagger as M
from fastsim import FastCarbon
import eval_v53
report={}
with tempfile.TemporaryDirectory(prefix='carbon-fix-data-') as tmp:
 root=pathlib.Path(tmp); rd=root/'corpus'; out=root/'dataset'; (rd/'replays').mkdir(parents=True)
 sim=FastCarbon(eval_v53.CFG).reset(seed=9123); steps=[]
 for t in range(4):
  if t: sim.step({},{})
  steps.append([{'observation':sim.obs(p),'status':'ACTIVE','action':{}} for p in (0,1)])
 with gzip.open(rd/'replays/demo-0.json.gz','wt') as f: json.dump({'steps':steps},f)
 index=[{'file':'replays/demo-0.json.gz','steps':300,'order':[0,1],'strategies':[{'name':B.F.TEACHERS[0]},{'name':'not-a-teacher'}],'scores':[100,0]}]
 (rd/'index.json').write_text(json.dumps(index))
 def build():
  with contextlib.redirect_stdout(io.StringIO()): B.main(str(out),[str(rd)])
  return {'steps':np.load(out/'step.npy').tolist(),'teachers':np.load(out/'teacher.npy').tolist(),'value':np.load(out/'value.npy').tolist()}
 report['initial']=build()
 B.STEP_MIN=2; B.F.TEACHERS[0],B.F.TEACHERS[1]=B.F.TEACHERS[1],B.F.TEACHERS[0]
 report['step_and_teacher_change']=build()
 B.STEP_MIN=100
 report['empty_window_first']=build()
 report['empty_window_second']=build()
 B.STEP_MIN=0; report['metadata_initial']=build()
 index[0]['scores']=[3000,0]; (rd/'index.json').write_text(json.dumps(index))
 report['metadata_changed']=build(); report['metadata_expected_value']=float(np.float16(np.tanh(1)))
 # Actual merge over one-row shape-correct synthetic data, original val game.
 base=root/'base'; base.mkdir()
 arrays={k:np.zeros((1,)+tuple(sh),dt) for k,(dt,sh) in B.FIELDS.items()}
 for k,v in arrays.items(): np.save(base/(k+'.npy'),v)
 np.save(base/'w.npy',np.ones(1,np.float16)); np.save(base/'uw.npy',np.ones((1,B.F.OWN),np.float16))
 orig='0efcfa1227b59fd7f89ee0c9b9ca9268-1.json.gz'
 (base/'meta.json').write_text(json.dumps({'games':[{'gi':0,'key':orig,'val':True,'n':1}],'teachers':B.F.TEACHERS}))
 for mode in ('with_orig','legacy'):
  dg=root/mode; (dg/'shards').mkdir(parents=True)
  arr=dict(arrays,uw=np.ones((1,B.F.OWN),np.float16))
  if mode=='with_orig': arr['orig_id']=np.array(orig)
  np.savez(dg/'shards'/('lb_cp_'+orig.replace('.json.gz','')+'.npz'),**arr)
  dest=root/('merge_'+mode)
  with contextlib.redirect_stdout(io.StringIO()): M.main(str(base),str(dg),str(dest))
  games=json.loads((dest/'meta.json').read_text())['games']
  report['merge_'+mode]=[{'key':g['key'],'val':g['val'],'orig':g.get('orig')} for g in games]
  for g in games:
   if g.get('src')=='dagger': g['val']=False
  report['merge_'+mode+'_after_documented_override']=[{'key':g['key'],'val':g['val']} for g in games]
result = json.dumps(report, indent=2) + "\n"
print(result, end="")
if os.environ.get("CARBON_REVIEW_OUT"):
    destination = pathlib.Path(os.environ["CARBON_REVIEW_OUT"]).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data-results.json").write_text(result, encoding="utf-8")
