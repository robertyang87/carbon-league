"""Read the exact original source; exercise its scheduler and optional-net path without a game."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
FREEZE='d944673f8c6487cb4cd9ff10a001cc61735d157c'
SOURCE='sources/master3s2/master3s2.py'


def run():
    raw=subprocess.check_output(['git','show',f'{FREEZE}:{SOURCE}'],cwd=ROOT)
    tree=ast.parse(raw)
    defs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assigns={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    if '_ROLL' in {n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}:
        raise ValueError('source gains a rollout assignment')
    coeff={s:ast.literal_eval(assigns[s].value) for s in ['_S_K','_S_GEN','_S_M','_S_MMAX','_S_STAGES','_S_D2','_S_ROWS','_S_EPS','_S_ABORT','_S_FILL','_S_SAFE','_S_CFIX','_S_HOMEFIX','_S_ROLL','_S_RTEMP']}
    weight_info={}
    for s in ('_WEIGHTS','_VALUE'):
        with np.load(io.BytesIO(base64.b64decode(ast.literal_eval(assigns[s].value))),allow_pickle=False) as f:
            weight_info[s]= {k:list(f[k].shape) for k in f.files}
    # Stub only the backend construction/forward; original _net controls which
    # payloads are selected, the exact warmup sizes and fallback behaviour.
    calls=[];builds=[]
    class Unit:
        def __call__(self,X,sid,S=None):
            calls.append([len(X),S is not None])
    def build(policy,value):
        builds.append(value is not None);return Unit()
    env=dict(np=np,base64=base64,io=io,_time=time,_POLICY=object(),_VALUE=ast.literal_eval(assigns['_VALUE'].value),
             _build=build,_NET=None,_RNET=None,_SEARCH_OK=True,_S_STATS={},_S_COST=[0.,0.,0.,0.],NPL=54,_SID=0,**coeff)
    exec(compile(ast.Module(body=[defs[k] for k in ('_net','_plan','_terminal')],type_ignores=[]),SOURCE,'exec'),env)
    net=env['_net']()
    assert builds==[True] and net is env['_RNET'] and env['_S_STATS']['rnet']=='the same'
    grid={str(n):list(env['_plan'](n,8)) for n in (2,4,8,16,32)}
    result={'git_freeze':FREEZE,'original_source':SOURCE,'file_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            'blob':subprocess.check_output(['git','rev-parse',f'{FREEZE}:{SOURCE}'],cwd=ROOT,text=True).strip(),
            'policy_value_weight_shapes':weight_info,'native_search_constants':coeff,'rollout_payload_bindings':0,
            'fixture':{'type':'AST-extracted original scheduler and asset selection with a dummy numerical backend, no game, no genuine timing',
                       'actual_builds_with_value_head':builds,'same_policy_object_used_for_rollout':net is env['_RNET'],
                       'logged_rnet_type':env['_S_STATS']['rnet'],'dummy_batch_sizes_and_values_requested':calls,
                       'plan_full_depth8_n_to_rows_steps_passes':grid,'terminal_scores_by_margin':{str(m):env['_terminal'](m) for m in (-15000,-3000,-1,0,1,3000,15000)}},
            'source_line_ranges':{k:[defs[k].lineno,defs[k].end_lineno] for k in ['BeliefTracker','Belief','_Policy','_act','agent','_sample','_deviations','_plan','_net','_search','_turn','_terminal']}}
    (HERE/'source_facts.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(grid)


if __name__=='__main__':run()
