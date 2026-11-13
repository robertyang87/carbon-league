"""S1 gate input/identity/CLI review. Real main/sampler; only expensive workers stubbed.
CARBON_REPO selects checkout; optional CARBON_REVIEW_OUT stores JSON; stdout default.
Synthetic length-only replay shells must not be used for policy-quality claims.
No real model, long match, training, remote action, or business-source edits.
"""
import ast, contextlib, gzip, hashlib, importlib.util, io, json, os
from pathlib import Path
import subprocess, sys, tempfile
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
choices=[Path(os.environ['CARBON_REPO'])] if os.environ.get('CARBON_REPO') else list(Path(__file__).resolve().parents)
repo=next((p.resolve() for p in choices if (p/'code/stage1_teacher_gate.py').is_file()),None)
if repo is None:raise SystemExit('Set CARBON_REPO')
sys.path.insert(0,str(repo/'code'))
import numpy as np
import stage1_teacher_gate as G

def fake_tasks(fn,tasks,workers):
    mode=os.environ.get('REVIEW_GATE_CASE','positive')
    if mode=='worker_error':raise RuntimeError('review injected worker failure before report publication')
    rows=[]
    for i,(s,cfg) in enumerate(tasks):
        base={k:s[k] for k in ('game','step','me')}
        if fn.__name__=='_q_scene':
            gain=0.0 if mode=='negative_q' else 0.02
            rows.append(dict(base,gain=gain,improved=gain>0,soft_gain=gain,soft_gain_behavior=gain,entropy=.2,agree=gain==0,best_arm=[0]*12,wall_ms=1.,guard={}))
        elif fn.__name__=='_anchor_task':
            rows.append(dict(base,du=.05,u_q=.55,u_g=.5))
        elif fn.__name__=='_du_scene':
            du=(.1 if i%2==0 else .5) if mode=='large_sigma' else .05+(i%3)*.001
            rows.append(dict(base,du=du,u_guided=.7+du/2,u_greedy=.7-du/2,teacher_calls=1,teacher_ms=1.))
        else:raise AssertionError(fn.__name__)
    return rows

if len(sys.argv)>1 and sys.argv[1]=='--child':
    config=json.loads(Path(sys.argv[2]).read_text());G.run_tasks=fake_tasks
    sys.argv=[str(repo/'code/stage1_teacher_gate.py')]+config['argv']
    os.environ['REVIEW_GATE_CASE']=config['case']
    raise SystemExit(G.main())

R={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
   'scope':'Actual CLI main and sampler in subprocess; expensive run_tasks replaced with named deterministic records; no teacher-quality conclusion',
   'source_sha256':hashlib.sha256((repo/'code/stage1_teacher_gate.py').read_bytes()).hexdigest(),
   'checks':{}}
with tempfile.TemporaryDirectory(prefix='carbon-s1-inputs-') as tmp:
 root=Path(tmp);replays=root/'replays';replays.mkdir();weights=root/'header-only.npz';weights.write_bytes(b'review header-only fixture, never passed to NP.Policy')
 def clean(v):
    if isinstance(v,str):return v.replace(str(root),'<TMP>')
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    return v
 for prefix in ('q','d'):
    for i in range(50):
        with gzip.open(replays/f'{prefix}{i}.json.gz','wt') as f:
            json.dump({'fixture':'length-only; never evaluated by policy','steps':[[]]*300},f)
 default={'q':[f'q{i}' for i in range(50)],'du':[f'd{i}' for i in range(50)]}
 def call(name,split=None,mode='q',extra=(),ds=None,case='positive',existing=None):
    sp=root/(name+'-split.json');sp.write_text(json.dumps(split if split is not None else default));out=root/(name+'-result.json')
    if existing is not None:out.write_text(existing)
    args=['--weights',str(weights),'--replays',str(replays),'--split-manifest',str(sp),'--mode',mode,'--out',str(out)]
    if ds is not None:
        dp=root/(name+'-ds.txt');dp.write_text('\n'.join(ds)+'\n');args+=['--ds-games-file',str(dp)]
    args+=list(extra);cp=root/(name+'-args.json');cp.write_text(json.dumps({'case':case,'argv':args}))
    p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',str(cp)],env=dict(os.environ,CARBON_REPO=str(repo)),capture_output=True,text=True,timeout=20)
    result={'exit_code':p.returncode,'stderr':clean(p.stderr),'prereg_printed':'PREREG ' in p.stdout,'result_exists':out.exists()}
    if out.exists():
        raw=out.read_text();result['out']=json.loads(raw);result['contains_nonstandard_NaN']='NaN' in raw
        result['old_output_unchanged']=existing is not None and raw==existing
    result['stdout_tail']=clean('\n'.join(p.stdout.splitlines()[-1:]))[:1800]
    R[name]=clean(result);return result
 # Formal PASS despite absent/empty training-corpus list.
 r=call('without_training_ids');R['checks']['missing_ds_formal_pass']=r['exit_code']==0 and r['out']['prereg']['ds_overlap_check']=='NOT RUN' and r['out']['q_gate']['verdict']=='PASS'
 r=call('empty_training_ids',ds=[]);R['checks']['empty_ds_formal_pass']=r['out']['prereg']['ds_overlap_check']=='RUN, disjoint (ds n=0)' and r['out']['q_gate']['verdict']=='PASS'
 # Positive checks: exact aliases per declared stem contract are rejected.
 r=call('exact_training_overlap',ds=['q0']);R['checks']['exact_training_overlap_refused']=r['exit_code']!=0 and not r['result_exists']
 overlap=dict(default,du=['q0']);r=call('q_du_overlap',split=overlap);R['checks']['q_du_overlap_refused']=r['exit_code']!=0 and not r['result_exists']
 r=call('missing_replay',split={'q':['not-found'],'du':['d0']});R['checks']['missing_replay_refused']=r['exit_code']!=0 and not r['result_exists']
 # Common raw meta.keys include .json.gz, but the input silently accepts wrong namespace.
 r=call('training_ids_with_suffix',ds=[f'q{i}.json.gz' for i in range(50)]);R['checks']['unvalidated_id_namespace_observed']=r['exit_code']==0 and r['out']['q_gate']['verdict']=='PASS'
 # Duplicate IDs are not normalized/deduped before sampling, inflating sample count.
 r=call('duplicate_manifest_ids',split={'q':['q0']*50,'du':['d0']},ds=['other-training-game'])
 rr=r['out']['rows'];R['duplicate_manifest_ids']['unique_games']=len({x['game'] for x in rr});R['duplicate_manifest_ids']['unique_roots']=len({(x['game'],x['step'],x['me']) for x in rr})
 R['checks']['duplicates_formal_pass']=r['out']['q_gate']['n']==150 and R['duplicate_manifest_ids']['unique_games']==1 and r['out']['q_gate']['verdict']=='PASS'
 # Declared diagnostic flag can still return formal PASS if its cap is not hit.
 r=call('nonbinding_diagnostic_cap',extra=['--max-scenes','999'],ds=['other']);R['checks']['nonbinding_cap_formal_pass']=r['out']['prereg']['canonical_config'] and r['out']['q_gate']['verdict']=='PASS'
 # Stage B floor is caller-controlled, with no Stage A proof input.
 r=call('tiny_stage_b',split={'q':['q0'],'du':['d0']},mode='du',extra=['--n-required','2'],ds=['other'])
 R['checks']['tiny_stage_b_formal_pass']=r['out']['du_gate']['n']<150 and r['out']['du_gate']['verdict']=='PASS'
 a=call('stage_a_all_scenes',mode='du',extra=['--pilot'],ds=['other'],case='large_sigma')
 b=call('stage_b_without_locked_n',mode='du',ds=['other'],case='large_sigma')
 aroots=[(r['game'],r['step'],r['me']) for r in a['out']['rows']];broots=[(r['game'],r['step'],r['me']) for r in b['out']['rows']]
 R['stage_a_b_binding']={'same_roots':aroots==broots,'stage_a_n':a['out']['du_gate']['n'],'stage_b_recommended_n':a['out']['du_gate']['stageB_N_formula'], 'stage_b_actual_n':b['out']['du_gate']['n'],'stage_b_verdict':b['out']['du_gate']['verdict']}
 R['checks']['stage_a_b_unbound']=aroots==broots and a['out']['du_gate']['n']>30 and b['out']['du_gate']['n']<a['out']['du_gate']['stageB_N_formula'] and b['out']['du_gate']['verdict']=='PASS'
 # Statistical FAIL is a successful process exit; worker/input errors remain nonzero.
 r=call('statistical_fail_exit',case='negative_q',ds=['other']);R['checks']['fail_verdict_exit_zero']=r['exit_code']==0 and r['out']['q_gate']['verdict']=='FAIL'
 old='{"q_gate":{"verdict":"PASS"},"old_run":true}\n';r=call('worker_failure_old_artifact',case='worker_error',ds=['other'],existing=old)
 R['checks']['worker_error_nonzero_old_artifact_remains']=r['exit_code']!=0 and r['old_output_unchanged']
 # Empty list is refused; short/empty replays can yield empty successful diagnostics.
 r=call('empty_mode_games',split={'q':[],'du':['d0']});R['checks']['empty_mode_refused']=r['exit_code']!=0
 with gzip.open(replays/'empty.json.gz','wt') as f:json.dump({'steps':[]},f)
 r=call('empty_replay_diagnostic',split={'q':['empty'],'du':['d0']},ds=['other'])
 R['checks']['empty_replay_nan_report']=r['exit_code']==0 and r['out']['q_gate']['n']==0 and r['contains_nonstandard_NaN'] and r['out']['q_gate']['verdict']=='DIAGNOSTIC ONLY'
 # Actual continuation evidence: terminal result never calls the teacher's learned leaf.
 sim=G.FastCarbon(G.CFG).reset(seed=7);full=sim.full_observation(0);full['step']=298
 full['players'][0][0]=80;full['players'][1][0]=60;sim.load_observation(full)
 fm=G.F.Feat(0);fo=G.F.Feat(1);xm=fm.observe(sim.obs(0));xo=fo.observe(sim.obs(1))
 scene={'me':0,'step':298,'full':full,'fm':fm,'fo':fo,'x_me':xm,'x_opp':xo}
 class Policy:
    def act(self,x,aux,teacher=1):
        # Base-only root: legal NONE on bases, no actions for absent workers.
        labels=np.full(G.F.OWN,-1,np.int64);labels[10:12]=0
        return labels,np.zeros((G.F.OWN,5)),0
 original_leaf=G.T._leaf_u
 def forbidden_leaf(*a,**kw):raise AssertionError('learned leaf must not score independent continuation')
 G.T._leaf_u=forbidden_leaf
 try:
    u1=G.continue_game(Policy(),scene,None,100);u2=G.continue_game(Policy(),scene,None,999)
 finally:G.T._leaf_u=original_leaf
 R['independent_continuation']={'u_seed100':u1,'u_seed999':u2,'expected_terminal_u':.5+.5/(1+np.exp(-.02)), 'learned_leaf_forbidden':True,'policy_only_received_features':True}
 R['checks']['terminal_anchor_independent_of_learned_leaf']=abs(u1-R['independent_continuation']['expected_terminal_u'])<1e-12 and u1==u2
 R['all_expected_checks_passed']=all(R['checks'].values())
result=json.dumps(R,ensure_ascii=False,indent=2)+'\n';print(result,end='')
if os.environ.get('CARBON_REVIEW_OUT'):
 out=Path(os.environ['CARBON_REVIEW_OUT']);out.mkdir(parents=True,exist_ok=True);(out/'gate-input-results.json').write_text(result)
if not R['all_expected_checks_passed']:raise SystemExit(1)
