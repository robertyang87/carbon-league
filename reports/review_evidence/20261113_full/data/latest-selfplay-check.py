"""62e selfplay producer review; temp synthetic records, no training/remote shell.
CARBON_REPO points at reviewed checkout; optional CARBON_REVIEW_OUT stores JSON.
Production generator and builder functions execute, policy inference is stubbed.
CLI cases use the producer's documented SPGEN_STUB=1 local validation mode.
"""
import ast, contextlib, gzip, hashlib, importlib.util, io, json, math, os
from pathlib import Path
import subprocess, sys, tempfile
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
choices=[Path(os.environ['CARBON_REPO'])] if os.environ.get('CARBON_REPO') else list(Path(__file__).resolve().parents)
repo=next((p.resolve() for p in choices if (p/'code/producer/selfplay_gen.py').is_file()),None)
if repo is None:raise SystemExit('Set CARBON_REPO to reviewed checkout')
sys.path.insert(0,str(repo/'code'))
import numpy as np
import cnet2_build as B
from fastsim import FastCarbon
spec=importlib.util.spec_from_file_location('review_selfplay',repo/'code/producer/selfplay_gen.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
R={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
   'source_sha256':hashlib.sha256((repo/'code/producer/selfplay_gen.py').read_bytes()).hexdigest(),
   'source_md5':hashlib.md5((repo/'code/producer/selfplay_gen.py').read_bytes()).hexdigest(),
   'training_was_launched':False, 'checks':{}}
def recruit(o,c,different=False):
    if o['step'] not in (0,1):return {}
    return {list(o['players'][o['player']][1])[o['step']%2 if different else 0]:'RECCOLLECTOR'}
def opp(o,c):return recruit(o,c,True)
def noop(o,c):return {}
with tempfile.TemporaryDirectory(prefix='carbon-latest-selfplay-') as tmp:
 root=Path(tmp)
 def clean(x):
    if isinstance(x,str):return x.replace(str(root),'<TMP>')
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
 def play(name,agents,seed=0):
    rd=root/name;gen=G._Gen('stub.npz',dict(G.CONF),'review-teacher','review-teacher');gen.agents=agents
    entry,n,margin,error=gen.play(seed,str(rd))
    data=json.load(gzip.open(rd/entry['file'])) if entry else None
    return rd,entry,n,margin,error,data
 rd,e,n,m,err,d=play('terminal',[recruit,opp])
 (rd/'index.json').write_text(json.dumps([e]));R['terminal']={'entry':e,'nsteps':n,'margin_a':m,'error':err,
    'result':d['result'],'frames':len(d['steps']),'first_done':next(i for i,s in enumerate(d['steps']) if s[0]['status']=='DONE'),
    'first_action_is_empty':d['steps'][0][0]['action']=={},'last_obs_step':d['steps'][-1][0]['observation']['step']}
 R['checks']['terminal_exact']=n==2 and d['result']['scores']==[-299.0,50.0] and m==-349.0 and R['terminal']['first_done']==2
 R['checks']['utility_correct']=abs(sum(d['result']['u'])-1)<1e-12 and abs(d['result']['u'][0]-(0.5/(1+math.exp(.349))))<1e-12
 B.F.TEACHERS[:]=['review-teacher']+[f'_slot{i}' for i in range(1,B.F.N_TEACH)]+['other']
 out=root/'built';cap=io.StringIO()
 with contextlib.redirect_stdout(cap):B.main(str(out),[str(rd)])
 meta=json.loads((out/'meta.json').read_text());R['builder_default_early_terminal']={'stdout':cap.getvalue(),'meta':meta}
 R['checks']['default_builder_drops_early_terminal']=meta['total']==0 and meta['valid_games']==0 and meta['filtered_games']=={'min_steps':1}
 # Direct per-game extraction isolates the status/action convention from scanner filtering.
 shard=root/'terminal-direct.npz';gi,rows,error=B.build_game((str(rd/e['file']),[0,1],0,[0,0],[-349.,349.],str(shard)))
 with np.load(shard) as a:
    R['direct_builder']={'rows':rows,'error':error,'steps':a['step'].tolist(),'values':a['value'].tolist(),'labels':a['y'].tolist()}
    truth=FastCarbon(G.CFG).reset(seed=0).carbon
    expected_carbon=B.F.q8(np.asarray(truth)[B.F.GRID]/100.)
    R['auxiliary_carbon']={'record_has_full_carbon':'full_carbon' in d['steps'][0][0]['observation'], 'builder_step0_nonzero_cells':int(np.count_nonzero(a['carbon'][0])), 'true_step0_nonzero_cells':int(np.count_nonzero(expected_carbon)), 'mismatched_cells':int(np.count_nonzero(a['carbon'][0]!=expected_carbon))}
    R['checks']['missing_full_carbon_mislabels_aux']=R['auxiliary_carbon']['builder_step0_nonzero_cells']==0 and R['auxiliary_carbon']['mismatched_cells']>0
 R['checks']['no_postterminal_rows_if_admitted']=rows==4 and R['direct_builder']['steps']==[0,1,0,1]
 # Exact current cnet2_search final fallback wrapper catches an inner fault.
 tree=ast.parse((repo/'code/cnet2_search.py').read_text());factory=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='make_search_agent')
 wrapper=[x for x in factory.body if isinstance(x,ast.FunctionDef) and x.name=='agent'][-1]
 def inner_error(*a,**kw):raise ValueError('review injected policy inference failure')
 ns={'_inner':inner_error,'stats':{},'cfg':{}}
 exec(compile(ast.Module(body=[wrapper],type_ignores=[]),'cnet2_search final fallback unchanged AST','exec'),ns)
 rd2,e2,n2,m2,er2,d2=play('hidden_errors',[ns['agent'],recruit])
 R['hidden_errors']={'generator_error':er2,'wrapper_stats':ns['stats'],'replay_written':bool(e2),'nsteps':n2,'scores':d2['result']['scores']}
 R['checks']['fallback_errors_not_rejected']=er2 is None and ns['stats']['root_errors']==2 and bool(e2)
 # Controlled natural end: all NONE still requires exactly 299 transitions.
 rd3,e3,n3,m3,er3,d3=play('natural',[noop,noop],seed=2)
 R['natural']={'nsteps':n3,'frames':len(d3['steps']),'final_statuses':[x['status'] for x in d3['steps'][-1]],'margin':m3}
 R['checks']['natural_horizon']=n3==299 and len(d3['steps'])==300 and all(x['status']=='DONE' for x in d3['steps'][-1])
 # Existing replay but no index emulates interruption between replay and index publish.
 cli_env={k:v for k,v in os.environ.items() if not k.startswith(('CNET2_','SPGEN_'))}
 cli_env.update(SPGEN_STUB='1',PYTHONPATH=str(repo/'code'),PYTHONDONTWRITEBYTECODE='1',BUILD_WORKERS='1')
 def cli(name,out,seed0,seed1,workers=1,maxgames=1):
    args=[sys.executable,str(repo/'code/producer/selfplay_gen.py'),'--weights','stub.npz','--out',str(out),'--seed0',str(seed0),'--seed1',str(seed1),'--workers',str(workers),'--warmup','0','--max-games',str(maxgames),'--max-seconds','10','--name','review-teacher']
    p=subprocess.run(args,env=cli_env,capture_output=True,text=True,timeout=30)
    ip=out/'index.json';result={'exit_code':p.returncode,'stdout':clean(p.stdout),'stderr':clean(p.stderr),'index':json.loads(ip.read_text()) if ip.exists() else None}
    summaries=[s.split(' ',1)[1] for s in p.stdout.splitlines() if s.startswith('SPGEN-RESULT {')]
    if summaries:result['summary']=clean(json.loads(summaries[-1]))
    R[name]=clean(result);return result
 r=cli('orphan_replay_retry',rd3,2,3)
 R['checks']['orphan_is_unrecoverable']=r['exit_code']==1 and r['index']==[] and r['summary']['errors']==[{'seed':2,'err':'exists'}]
 # Existing index skips work even when the promised replay no longer exists.
 (rd3/'index.json').write_text(json.dumps([e3]));(rd3/e3['file']).unlink();r=cli('indexed_missing_replay',rd3,2,3)
 R['checks']['indexed_missing_returns_success']=r['exit_code']==0 and r['summary']['games']==0 and r['summary']['errors']==[] and not (rd3/e3['file']).exists()
 # No real workers/net/external resources: local STUB workers expose max-games overshoot.
 r=cli('max_games_overshoot',root/'batch',10,15,workers=3,maxgames=1)
 R['checks']['batch_budget_overshoot']=r['exit_code']==0 and r['summary']['games']==3 and len(r['index'])==3
 # Direct documented script invocation cannot resolve parent code modules; fleet copy differs.
 import_env=dict(cli_env);import_env.pop('PYTHONPATH',None)
 p=subprocess.run([sys.executable,str(repo/'code/producer/selfplay_gen.py'),'--help'],cwd=root,env=import_env,capture_output=True,text=True)
 R['direct_cli_import']={'exit_code':p.returncode,'stderr':clean(p.stderr)}
 R['checks']['direct_import_failure']=p.returncode!=0 and "No module named 'eval_v53'" in p.stderr
 R['all_expected_checks_passed']=all(R['checks'].values())
result=json.dumps(clean(R),ensure_ascii=False,indent=2)+'\n';print(result,end='')
if os.environ.get('CARBON_REVIEW_OUT'):
 out=Path(os.environ['CARBON_REVIEW_OUT']);out.mkdir(parents=True,exist_ok=True);(out/'latest-selfplay-results.json').write_text(result)
if not R['all_expected_checks_passed']:raise SystemExit(1)
