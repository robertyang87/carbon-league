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
        R['lb_selfplay']={'result':{k:v for k,v in result.items() if k!='shard'},'step_range':[int(d['step'].min()),int(d['step'].max())],
                          'unique_values':np.unique(d['value']).tolist(),'post_terminal_rows':int((d['step']>=end).sum())}
    with patch.object(CA,'load_weights',lambda *a:{}),patch.object(NP,'Policy',ToyLabel),patch.object(eval_v53,'load_agent_fn',lambda path:recruit if path.endswith('review_ours.py') else opponent):
        result=DG.play((str(out),'review_ours.py','stub.npz',2,0))
    with np.load(result['shard']) as d:
        R['dagger']={'result':{k:v for k,v in result.items() if k!='shard'},'step_range':[int(d['step'].min()),int(d['step'].max())],
                     'unique_values':np.unique(d['value']).tolist(),'post_terminal_rows':int((d['step']>=end).sum())}

# Same-prefix contract: execute the unchanged prefix-generation loop from DG.play.
sim=FastCarbon(eval_v53.CFG).reset(seed=9123);obs=sim.full_observation()
obs['players'][0][2]={'player-0-worker-0':[111,0,'PLANTER'],'player-0-worker-1':[112,0,'PLANTER']}
feat=F.Feat(0);x=feat.observe(obs);y=feat.labels({});ym=y.copy();ym[0]=F.MOVES.index('UP');ym[1]=F.MOVES.index('LEFT')
source=ast.parse((repo/'code/dagger_gen.py').read_text());play=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='play')
loop=next(n for n in play.body if isinstance(n,ast.For))
prefix_nodes=[n for n in loop.body if n.lineno>=64 and n.end_lineno<=72]
ns={'F':F,'np':np,'x':x,'feat':feat,'y':y,'ym':ym}
exec(compile(ast.Module(body=prefix_nodes,type_ignores=[]),'dagger_gen.py:64-72 unchanged AST','exec'),ns)
uf,mf,legal,labels=LB._prefix_rows(F,x,feat,ym)
R['dagger_prefix']={'worker_uids':feat.own_uid[:2],'actual_prefix_y':y[:2].tolist(),'stored_targets_y':ns['ym'][:2].tolist(),
                    'later_slot_mf_max_abs_difference_from_stored_y_prefix':float(np.max(np.abs(ns['mf'][1]-mf[1]))),
                    'different_coordinates':np.argwhere(ns['mf'][1]!=mf[1]).tolist(),
                    'model_prefix_source':'cnet2_model.py:273,290-291 reads stored b[y], whereas dagger_gen.py:72 advances actual y'}
R['checks']={'reference_extinction_at_2':end==2,
             'lb_post_terminal_and_wrong_target':R['lb_selfplay']['post_terminal_rows']==R['lb_selfplay']['result']['n'] and R['lb_selfplay']['unique_values']!=[R['reference']['value_float16']],
             'dagger_post_terminal_and_wrong_target':R['dagger']['post_terminal_rows']==298 and R['dagger']['unique_values']!=[R['reference']['value_float16']],
             'dagger_prefix_mismatch':R['dagger_prefix']['later_slot_mf_max_abs_difference_from_stored_y_prefix']>0}
R['all_expected_checks_passed']=all(R['checks'].values())
result=json.dumps(R,ensure_ascii=False,indent=2)+'\n';print(result,end='')
if os.environ.get('CARBON_REVIEW_OUT'):
    out=Path(os.environ['CARBON_REVIEW_OUT']);out.mkdir(parents=True,exist_ok=True);(out/'label-results.json').write_text(result)
if not R['all_expected_checks_passed']:raise SystemExit(1)
