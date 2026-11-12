"""Read-only dynamic checks of c64bbed R01/R02/R07 against original environment."""
import ast, json, os, pathlib, sys, tempfile
def repo_root():
    configured = os.environ.get('CARBON_REPO')
    if configured:
        root = pathlib.Path(configured).expanduser().resolve()
        if not (root / 'code/fastsim.py').is_file():
            raise RuntimeError('CARBON_REPO does not contain code/fastsim.py')
        return root
    for root in pathlib.Path(__file__).resolve().parents:
        if (root / 'code/fastsim.py').is_file():
            return root
    raise RuntimeError('Set CARBON_REPO to the repository root')
ROOT = repo_root()

sys.path.insert(0,str(ROOT/'code/env_src/carbon_challenge-master'))
sys.path.insert(0,str(ROOT/'code'))
from zerosum_env import make
from fastsim import FastCarbon
import eval_v53
out={'commit':'c64bbeda8a00b2f4d79ace589f201bab3cca5163'}

def original(a,b):
    env=make('carbon', configuration={'episodeSteps':300})
    env.run([a,b])
    return {'rewards':[float(s.reward) for s in env.steps[-1]],
            'step':env.steps[-1][0].observation.step,
            'cash':[float(p[0]) for p in env.steps[-1][0].observation.players]}

def load_agent(src):
    ns={}; exec(src,ns); return ns['agent']
noop='def agent(o,c):\n    return {}\n'
last='def agent(o,c):\n    return {next(iter(o["players"][o["player"]][1])):"RECCOLLECTOR"} if o["step"]==299 else {}\n'
suicide='def agent(o,c):\n    return {list(o["players"][o["player"]][1])[0]:"RECCOLLECTOR"} if o["step"]<2 else {}\n'
survive='def agent(o,c):\n    return {list(o["players"][o["player"]][1])[o["step"]]:"RECCOLLECTOR"} if o["step"]<2 else {}\n'
with tempfile.TemporaryDirectory() as td:
    def run(a,b):
        pa=pathlib.Path(td)/'a.py';pb=pathlib.Path(td)/'b.py'
        pa.write_text(a);pb.write_text(b)
        r=eval_v53.play_game(('review',str(pa),str(pb),42,0))
        return {k:r[k] for k in ('my_cash','opp_cash','my_score','opp_score','margin','win','nsteps','extinct','ledger_pol','ledger_extra')}
    out['R01_horizon']={'original':original(load_agent(last),load_agent(noop)), 'local':run(last,noop)}
    out['R02_single_extinction']={'original':original(load_agent(suicide),load_agent(survive)), 'local':run(suicide,survive)}
    out['R02_both_extinction']={'original':original(load_agent(suicide),load_agent(suicide)), 'local':run(suicide,suicide)}
    out['R07_same_step_death']={'original':original(load_agent(suicide),load_agent(noop)), 'local':run(suicide,noop)}
    def trim(x):
        l=x['local']; ex=l.pop('ledger_extra');l['diagnostic_recruit_fields']={k:ex['pol'][k] for k in ['rec60_c','rec60_p','pla_rec','pla_dead','e_prec25','e_clost30']}
    for x in out.values():
        if isinstance(x,dict) and 'local' in x:trim(x)

# Execute the exact nested leaf/extinction helper definitions from current source.
# These early-return cases do not need model weights; no rule source is modified.
src=ast.parse((ROOT/'code/cnet2_search.py').read_text())
factory=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='make_search_agent')
helpers=[n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name in ('_extinct','_terminal','leaf')]
ns={};exec(compile(ast.Module(body=helpers,type_ignores=[]),'cnet2_search_helpers_current','exec'),ns)
s=FastCarbon().reset(full_carbon=[0]*225);o=s.full_observation();o['step']=9
# A has no potential; B survives with a worker. The terminal score must prefer B.
o['players'][0][0]=20;o['players'][1][0]=10
bp=next(iter(o['players'][1][1].values()));o['players'][1][2]={'player-1-worker-0':[bp,0,'COLLECTOR']}
s.load_observation(o).step({},{})
out['R02_search_leaf_single']={'terminal':ns['_terminal'](s),'leaf_side_0':ns['leaf'](s,0,289),
 'cash':s.cash,'expected_original_rewards':[-291,10],'expected_reward_margin':-301}
# Two extinct sides with unequal leftover cash are a draw in the real game.
o['players'][1][2]={};s.load_observation(o).step({},{})
out['R02_search_leaf_both']={'terminal':ns['_terminal'](s),'leaf_side_0':ns['leaf'](s,0,289),
 'cash':s.cash,'expected_original_rewards':[-291,-291],'expected_reward_margin':0}
out['search_horizon_expressions']=[ast.unparse(n) for n in ast.walk(factory)
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='H' for t in n.targets)]
print(json.dumps(out,ensure_ascii=False,indent=2))
