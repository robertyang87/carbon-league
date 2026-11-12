"""Read-only business-code review: all datasets and injected failures are temporary.
CARBON_REPO selects repo; otherwise discover code/fastsim.py above the script.
CARBON_REVIEW_OUT optionally names a directory for JSON evidence; default stdout.
Synthetic replay index length is 300 to pass scanning; only three decisions
are stored to isolate cache behavior. The actual builder and merger execute;
no training, remote work, or business-source writes are performed.
"""
import contextlib,gzip,hashlib,io,json,os,pathlib,subprocess,sys,tempfile,zlib
os.environ['BUILD_WORKERS']='1'
def find_repo():
    configured = os.environ.get('CARBON_REPO')
    if configured:
        candidate = pathlib.Path(configured).expanduser().resolve()
        if not (candidate / 'code/fastsim.py').is_file():
            raise SystemExit('CARBON_REPO must contain code/fastsim.py')
        return candidate
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / 'code/fastsim.py').is_file():
            return candidate
    raise SystemExit('Repository not found; set CARBON_REPO')

repo=find_repo()
sys.path.insert(0,str(repo/'code'))
import numpy as np
import cnet2_build as B
import merge_dagger as M
from fastsim import FastCarbon
import eval_v53
report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
with tempfile.TemporaryDirectory(prefix='carbon-rereview-data-') as tmp:
 root=pathlib.Path(tmp); rd=root/'corpus'; out=root/'dataset'; (rd/'replays').mkdir(parents=True)
 sim=FastCarbon(eval_v53.CFG).reset(seed=9123); steps=[]
 for t in range(4):
  if t: sim.step({},{})
  steps.append([{'observation':sim.obs(p),'status':'ACTIVE','action':{}} for p in (0,1)])
 rp=rd/'replays/demo-0.json.gz'
 with gzip.open(rp,'wt') as f: json.dump({'steps':steps},f)
 good_bytes=rp.read_bytes()
 teachers=list(B.F.TEACHERS)
 index=[{'file':'replays/demo-0.json.gz','steps':300,'order':[0,1],'strategies':[{'name':teachers[0]},{'name':'not-a-teacher'}],'scores':[100,0]}]
 def put_index(): (rd/'index.json').write_text(json.dumps(index))
 put_index()
 def snapshot(log=''):
  meta=json.loads((out/'meta.json').read_text()); manifest=json.loads((out/'shards/_cfp.json').read_text())
  return {'steps':np.load(out/'step.npy').tolist(),'teachers':np.load(out/'teacher.npy').tolist(),'value':np.load(out/'value.npy').tolist(),'game_rows':[g['n'] for g in meta['games']], 'manifest_entries':manifest['shards'],'shard_names':sorted(p.name for p in (out/'shards').glob('*.npz')),'error_lines':[s for s in log.splitlines() if s.startswith('err ')]}
 def build():
  capture=io.StringIO()
  with contextlib.redirect_stdout(capture): B.main(str(out),[str(rd)])
  return snapshot(capture.getvalue())
 report['initial']=build()
 B.STEP_MIN=2; B.F.TEACHERS[0],B.F.TEACHERS[1]=B.F.TEACHERS[1],B.F.TEACHERS[0]
 report['step_and_teacher_change']=build()
 B.STEP_MIN=100
 report['empty_window_first']=build(); report['empty_window_second']=build()
 B.STEP_MIN=0; B.F.TEACHERS[:]=teachers
 report['metadata_initial']=build()
 index[0]['scores']=[3000,0]; put_index(); report['metadata_score_change']=build()
 report['metadata_expected_value']=float(np.float16(np.tanh(1)))
 index[0]['strategies'][0]['name']=teachers[1]; put_index(); report['metadata_teacher_change']=build()
 index[0]['order']=[1,0]; put_index(); report['metadata_order_change']=build()
 # CLI failure injection: preserve one valid game, corrupt a second, then restore and retry.
 rp2=rd/'replays/demo-1.json.gz'; rp2.write_bytes(good_bytes)
 index.append(dict(index[0],file='replays/demo-1.json.gz')); put_index(); report['two_games_good']=build()
 rp.write_bytes(b'not a gzip archive')
 def cli_build():
  p=subprocess.run([sys.executable,str(repo/'code/cnet2_build.py'),str(out),str(rd)],capture_output=True,text=True)
  x=snapshot(p.stdout); x['exit_code']=p.returncode; x['stdout']=p.stdout; x['stderr']=p.stderr
  return x
 report['failed_partial_build']=cli_build(); report['failed_retry']=cli_build()
 # Execute only the real replica-chain prelaunch stages; NEVER run its shell or train.
 top_env=dict(os.environ,CNET2_TOP_TEACHERS=teachers[1]+'@2')
 bt=subprocess.run([sys.executable,str(repo/'code/build_top.py'),str(out),str(rd)],env=top_env,capture_output=True,text=True)
 uw=subprocess.run([sys.executable,str(repo/'code/set_uw.py'),str(out),'1','1','1'],capture_output=True,text=True)
 actual_meta=json.loads((out/'meta.json').read_text())
 report['replica_prelaunch_gates']={'build_top_exit':bt.returncode,'set_uw_exit':uw.returncode,'metadata_games':len(actual_meta['games']),'index_games':len(index),'game_rows':[g['n'] for g in actual_meta['games']],'replica_count_gate_passes':len(actual_meta['games'])==len(index),'build_stdout':bt.stdout,'weights_stdout':uw.stdout,'training_was_launched':False}
 rp.write_bytes(good_bytes); report['restored_retry']=cli_build()
 # Original-ID split test including refusal of legacy and empty orig IDs.
 base=root/'base'; base.mkdir()
 arrays={k:np.zeros((1,)+tuple(sh),dt) for k,(dt,sh) in B.FIELDS.items()}
 for k,v in arrays.items(): np.save(base/(k+'.npy'),v)
 np.save(base/'w.npy',np.ones(1,np.float16)); np.save(base/'uw.npy',np.ones((1,B.F.OWN),np.float16))
 orig='0efcfa1227b59fd7f89ee0c9b9ca9268-1.json.gz'
 (base/'meta.json').write_text(json.dumps({'games':[{'gi':0,'key':orig,'val':True,'n':1}],'teachers':B.F.TEACHERS}))
 for mode in ('with_orig','legacy','empty_orig','new_selfplay'):
  dg=root/mode; (dg/'shards').mkdir(parents=True); arr=dict(arrays,uw=np.ones((1,B.F.OWN),np.float16))
  if mode=='with_orig': arr['orig_id']=np.array(orig)
  if mode=='empty_orig': arr['orig_id']=np.array('')
  name='lb_sp_510000.npz' if mode=='new_selfplay' else 'lb_cp_'+orig.replace('.json.gz','')+'.npz'
  np.savez(dg/'shards'/name,**arr); dest=root/('merge_'+mode); cap=io.StringIO(); err=io.StringIO(); code=0
  try:
   with contextlib.redirect_stdout(cap),contextlib.redirect_stderr(err): M.main(str(base),str(dg),str(dest))
  except SystemExit as e: code=e.code
  r={'exit_code':code,'output_exists':dest.exists(),'stderr':err.getvalue().replace(str(root),'<TMP>')}
  if (dest/'meta.json').exists():
   games=json.loads((dest/'meta.json').read_text())['games']
   r['games']=[{'key':g['key'],'val':g['val'],'orig':g.get('orig')} for g in games]
  report['merge_'+mode]=r
readme=(repo/'README.md').read_text(); handoff=(repo/'reports/HANDOFF_nn.md').read_text(); current=(repo/'reports/live/CURRENT.md').read_text(); prep=(repo/'reports/live/lb_expert_iteration_prep.md').read_text()
report['docs']={'readme_points_CURRENT':'reports/live/CURRENT.md' in readme,'handoff_points_CURRENT':'reports/live/CURRENT.md' in handoff,'CURRENT_documents_outside_tree':'不在本仓 Git tree 内' in current,'CURRENT_marks_hash_unverified':'待核' in current,'CURRENT_eval_hash_matches':hashlib.md5((repo/'code/eval_v53.py').read_bytes()).hexdigest() in current,'eval_hash':hashlib.md5((repo/'code/eval_v53.py').read_bytes()).hexdigest(),'prep_split_override_prohibited':'禁止任何 val 覆盖/改写步骤' in prep,'prep_dynamic_eval_hash':'当期完整 MD5' in prep}
result=json.dumps(report,indent=2,ensure_ascii=False)+'\n'; print(result,end='')
if os.environ.get('CARBON_REVIEW_OUT'):
 d=pathlib.Path(os.environ['CARBON_REVIEW_OUT']); d.mkdir(parents=True,exist_ok=True); (d/'data-round3-results.json').write_text(result)
