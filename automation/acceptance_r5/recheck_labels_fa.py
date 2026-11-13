"""R5 generator semantics with real sim/features and deterministic policy stubs.
No neural-network training/inference or source edits. Original env checks terminal.
CARBON_REPO and CARBON_REVIEW_OUT follow recheck_data.py. JSON stdout by default.
"""
import ast, contextlib, copy, hashlib, io, json, os
from pathlib import Path
import subprocess, sys, tempfile, types
from unittest.mock import patch
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
choices=[Path(os.environ['CARBON_REPO'])] if os.environ.get('CARBON_REPO') else list(Path(__file__).resolve().parents)
repo=next((p.resolve() for p in choices if (p/'code/fastsim.py').is_file()),None)
if repo is None:raise SystemExit('Set CARBON_REPO')
sys.path[:0]=[str(repo/'code'),str(repo/'code/env_src/carbon_challenge-master')]
import numpy as np
import cnet2_agent as CA, cnet2_search as S, cnet2_np as NP, cnet2_feat as F
import dagger_gen as DG, lb_relabel as LB, eval_v53
from fastsim import FastCarbon
from zerosum_env import make
R={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
   'scope':'Real source functions, simulator, features and original environment; deterministic inference stubs; no training'}
def recruit(o,c,different=False):
    if o['step'] not in (0,1):return {}
    bases=list(o['players'][o['player']][1])
    return {bases[o['step']%2 if different else 0]:'RECCOLLECTOR'}
def opponent(o,c):return recruit(o,c,True)
ref=make('carbon',configuration={'episodeSteps':300})
ref.run([recruit,opponent])
expected=[float(s.reward) for s in ref.steps[-1]]
end=int(ref.steps[-1][0].observation.step)
R['reference']={'steps':end,'rewards':expected,'margin':expected[0]-expected[1],
                'value_float16':float(np.float16(np.tanh((expected[0]-expected[1])/3000)))}
class ToySearch:
    def __init__(self):self.cfg={};self.stats={'searched':0,'changed':0,'errors':0}
    def __call__(self,o,c):return recruit(o,c)
class ToyLabel:
    def __init__(self,*a,**kw):pass
    def act(self,x,aux,teacher,forced):return forced,np.zeros((F.OWN,5)),0
with tempfile.TemporaryDirectory(prefix='carbon-r5-labels-') as tmp:
    out=Path(tmp);(out/'shards').mkdir()
    with patch.object(CA,'make_agent',lambda *a,**kw:recruit),patch.object(S,'make_search_agent',lambda *a,**kw:ToySearch()),patch.object(eval_v53,'load_agent_fn',lambda *a:opponent):
        result=LB.play_selfplay((str(out),'stub.npz',2,0,False))
    with np.load(result['shard']) as d:
        R['lb_selfplay']={'result':{k:v for k,v in result.items() if k!='shard'},'step_range':[int(d['step'].min()) if d['step'].size else -1,int(d['step'].max()) if d['step'].size else -1],
                          'unique_values':np.unique(d['value']).tolist(),'post_terminal_rows':int((d['step']>=end).sum())}
    with patch.object(CA,'load_weights',lambda *a:{}),patch.object(NP,'Policy',ToyLabel),patch.object(eval_v53,'load_agent_fn',lambda path:recruit if path.endswith('review_ours.py') else opponent):
        result=DG.play((str(out),'review_ours.py','stub.npz',2,0))
    with np.load(result['shard']) as d:
        R['dagger']={'result':{k:v for k,v in result.items() if k!='shard'},'step_range':[int(d['step'].min()) if d['step'].size else -1,int(d['step'].max()) if d['step'].size else -1],
                     'unique_values':np.unique(d['value']).tolist(),'post_terminal_rows':int((d['step']>=end).sum())}

import json as _j; print(_j.dumps(R, ensure_ascii=False))
