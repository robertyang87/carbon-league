"""R3: equality checks against original env, not the prior bug-probe outputs."""
import ast, copy, json, pathlib, sys, tempfile, types, subprocess
import os

def find_repository():
    explicit = os.environ.get("CARBON_REPO")
    if explicit:
        root = pathlib.Path(explicit).expanduser().resolve()
        if not (root / "code/fastsim.py").is_file():
            raise RuntimeError("CARBON_REPO must name a checkout containing code/fastsim.py")
        return root
    starts = (pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve())
    for start in starts:
        for root in (start, *start.parents):
            if (root / "code/fastsim.py").is_file():
                return root
    raise RuntimeError("Cannot locate code/fastsim.py; set CARBON_REPO to the checkout root")

ROOT = find_repository()
# Every execution writes into a NEW temporary directory, preserving all historical evidence.
review_base = os.environ.get("CARBON_REVIEW_OUT")
if review_base:
    pathlib.Path(review_base).expanduser().mkdir(parents=True, exist_ok=True)
    review_base = str(pathlib.Path(review_base).expanduser().resolve())
OUT = pathlib.Path(tempfile.mkdtemp(prefix="carbon-simulator-r4-", dir=review_base))
sys.path.insert(0,str(ROOT/'code/env_src/carbon_challenge-master'));sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from zerosum_env import make,utils
from zerosum_env.envs.carbon import carbon as rules,idgen
from fastsim import FastCarbon,DEFAULT_CONFIG
import eval_v53, cnet2_search as S, cnet2_feat as F
out={'tested_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(), 'reference_commit':'14d53320b20d58bdb82676f7fa866b8e11a55062','eval_cases':{},'leaf_cases':{}}
NOOP='def agent(o,c):\n return {}\n'
def recruit(steps,kind='RECCOLLECTOR',different=False):
 return 'def agent(o,c):\n return {list(o["players"][o["player"]][1])['+('o["step"]%2' if different else '0')+']:'+repr(kind)+'} if o["step"] in '+repr(steps)+' else {}\n'
def load(src):
 d={};exec(src,d);return d['agent']
def orig(a,b):
 e=make('carbon',configuration={'episodeSteps':300});e.run([load(a),load(b)])
 return {'scores':[float(s.reward) for s in e.steps[-1]],'end_step':e.steps[-1][0].observation.step}
cases={
 'noop_natural_horizon':(NOOP,NOOP),
 'unreachable_action_299':(recruit((299,)),NOOP),
 'last_real_step_spend':(NOOP,recruit((298,))),
 'extinction_survivor_spends':(recruit((0,1)),recruit((0,1),different=True)),
 'both_extinct':(recruit((0,1)),recruit((0,1))),
 'extinction_last_real_step':(recruit((297,298)),recruit((297,298),different=True)),
 'planter_same_step_death':(recruit((0,1),'RECPLANTER'),NOOP),
 'collector_same_step_death':(recruit((0,1)),NOOP),
}
with tempfile.TemporaryDirectory() as td:
 for name,(a,b) in cases.items():
  pa=pathlib.Path(td)/'a.py';pb=pathlib.Path(td)/'b.py';pa.write_text(a);pb.write_text(b)
  ref=orig(a,b);r=eval_v53.play_game(('r3',str(pa),str(pb),42,0));got={'scores':[r['my_score'],r['opp_score']],'end_step':r['nsteps']}
  out['eval_cases'][name]={'reference':ref,'got':got,'pass':got==ref}
  assert got==ref,(name,got,ref)
  if name.endswith('same_step_death'):
   ex=r['ledger_extra']['pol'];keys=['pla_rec','pla_dead','pla_dead0','e_prec25','e_pdead30','e_pdead0_30','rec60_p','rec60_c','e_clost30','pd_other','pd_clash_move','pd_clash_stay','pd_owncar','pd_ecar','pd_unknown']
   out['eval_cases'][name]['ledger']=r['ledger_pol'];out['eval_cases'][name]['typed']={k:ex[k] for k in keys}
   assert r['ledger_pol']['recruit_count']==2 and r['ledger_pol']['deaths']==2 and r['ledger_pol']['deposits']==0
   if name.startswith('planter'):
    assert sum(ex[k] for k in ex if k.startswith('pd_'))==ex['pla_dead']
    assert ex['pd_unknown']==1 and ex['pd_clash_stay']==1
    for k in ['pla_rec','pla_dead','pla_dead0','e_prec25','e_pdead30','e_pdead0_30','rec60_p']:assert ex[k]==2
   else:
    assert ex['rec60_c']==2 and ex['e_clost30']==2

src=ast.parse((ROOT/'code/cnet2_search.py').read_text());factory=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='make_search_agent')
helpers=[n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name in ('_extinct','_terminal','leaf')]
ns={'cfg':dict(S.DEFAULTS),'st':{},'np':np,'F':F};exec(compile(ast.Module(body=helpers,type_ignores=[]),'current_search_helpers','exec'),ns)
def micro(name,step,cash,worker_b=False,tree_b=False,act_b=False):
 sim=FastCarbon().reset(full_carbon=[100]*225);o=sim.full_observation();o['step']=step
 for p in range(2):o['players'][p][0]=cash[p]
 if worker_b:
  bp=next(iter(o['players'][1][1].values()));o['players'][1][2]={'player-1-worker-0':[bp,0,'COLLECTOR']}
 if tree_b:
  o['players'][1][3]={'player-1-tree-0':[112,10]};o['trees']={'player-1-tree-0':[None,0]};o['carbon'][112]=0
 ab={list(o['players'][1][1])[1]:'RECCOLLECTOR'} if act_b else {}
 sim.load_observation(o);pc=list(sim.cash);states=[]
 for p in range(2):
  obs=copy.deepcopy(o);obs['full_carbon']=list(obs['carbon']);obs['player']=p
  states.append(utils.structify(dict(observation=obs,action=ab if p else {},status='ACTIVE',reward=pc[p])))
 idgen.reset()
 for p in (0,1):
  for _ in range(sim.worker_counters[p]):idgen.new_worker_id(p)
  for _ in range(sim.tree_counters[p]):idgen.new_tree_id(p)
 r=rules.compute_next_board(states,types.SimpleNamespace(configuration=utils.structify(DEFAULT_CONFIG)));sim.step({},ab)
 expected=float(r[0].reward-r[1].reward);left=299-sim.step_count
 got=float(ns['leaf'](sim,0,left,pc))
 # Nonterminal leaves include assets; only compare their natural-horizon result.
 out['leaf_cases'][name]={'reference_rewards':[float(x.reward) for x in r],'pre_cash':pc,'post_cash':sim.cash,'terminal':ns['_terminal'](sim),'got_leaf':got,'expected':expected,'pass':got==expected}
 assert got==expected,(name,got,expected)
micro('root_extinction_survivor_recruit',9,[20,50],worker_b=True,act_b=True)
micro('root_extinction_survivor_tree_income',9,[20,50],tree_b=True)
micro('both_extinct_unequal_cash',9,[20,10])
micro('first_transition_extinction',0,[20,80],act_b=True)
micro('natural_horizon_survivor_spend',298,[80,80],act_b=True)
micro('extinction_on_last_transition',298,[20,80],act_b=True)
# Every production search advance path must snapshot pre-cash and pass a per-branch slot to leaf.
out['production_sites']={'root_capture':[1305,1518],'later_terminal_capture':[1367,1567], 'leaf_pass':[1377,1582], 'horizon':[1240,1526]}
payload = json.dumps(out, ensure_ascii=False, indent=2)
(OUT / "results.json").write_text(payload + "\n", encoding="utf-8")
print(payload)
print(f"Fresh evidence: {OUT / 'results.json'}", file=sys.stderr)
