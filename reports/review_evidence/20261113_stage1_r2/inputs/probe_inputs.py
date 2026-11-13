"""S1-r2 gate input/pilot/I-O control-flow review; real main/sampler, stub workers.
Set CARBON_REPO; optional CARBON_REVIEW_OUT. No real teacher-quality claim.
Writes only temporary fixtures and evidence outputs, never business source.
"""
import contextlib,gzip,hashlib,io,json,os,subprocess,sys,tempfile
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
repo=Path(os.environ['CARBON_REPO']).resolve();sys.path.insert(0,str(repo/'code'))
import stage1_teacher_gate as G
CASE='positive';calls=[]
def workers(fn,tasks,n):
 calls.append({'fn':fn.__name__,'n':len(tasks)})
 if CASE=='worker_error':raise RuntimeError('review worker exception before publication')
 rows=[]
 for i,(s,cfg) in enumerate(tasks):
  b={k:s[k] for k in ('game','step','me')}
  gi=int(''.join(ch for ch in s['game'] if ch.isdigit()) or 0)
  if fn.__name__=='_q_scene':
   gain=0.0 if CASE=='fail_q' else .03
   rows.append(dict(b,gain=gain,soft_gain=gain,soft_gain_behavior=gain,improved=gain>0,entropy=.2,agree=False,best_arm=[0]*12,wall_ms=1.,guard={}))
  elif fn.__name__=='_anchor_task':rows.append(dict(b,du=.04,du_soft=.02,u_q=.54,u_g=.5,u_soft=.52,n_arms=2))
  elif fn.__name__=='_du_scene':
   du=.125 if CASE=='constant' else .05+gi*.0001
   rows.append(dict(b,du=du,u_guided=.5+du,u_greedy=.5,teacher_calls=1,teacher_ms=1.))
  else:raise AssertionError(fn.__name__)
 return rows
G.run_tasks=workers
R={'commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
 'scope':'Original main, sampler, artifact verification and output code; only expensive run_tasks is replaced. Synthetic length-only replays and header-only weights, no neural evaluation or performance conclusion.',
 'source_sha256':hashlib.sha256((repo/'code/stage1_teacher_gate.py').read_bytes()).hexdigest(),'observations':{},'checks':{}}
with tempfile.TemporaryDirectory(prefix='carbon-s1-r2-input-') as temp:
 root=Path(temp);rd=root/'replays';rd.mkdir();w=root/'header-only.npz';w.write_bytes(b'header hash fixture; never loaded by model')
 for prefix,n in [('q',50),('d',50),('p',20)]:
  for i in range(n):
   with gzip.open(rd/f'{prefix}{i}.json.gz','wt') as f:json.dump({'fixture':'length only','steps':[[]]*300},f)
 qids=[f'q{i}' for i in range(50)];dids=[f'd{i}' for i in range(50)];pids=[f'p{i}' for i in range(10)]
 def norm(x):
  if isinstance(x,str):return x.replace(str(root),'<TMP>')
  if isinstance(x,dict):return {k:norm(v) for k,v in x.items()}
  if isinstance(x,list):return [norm(v) for v in x]
  return x
 def summary(data):
  if data is None:return None
  out={k:v for k,v in data.items() if k not in ('rows','anchor_rows')}
  for field in ('rows','anchor_rows'):
   if field in data:
    rows=data[field];keys=[(v['game'],v['step'],v['me']) for v in rows]
    out[field+'_summary']={'n':len(rows),'games':sorted({x[0] for x in keys}),'unique_roots':len(set(keys)),'identity_sha256':hashlib.sha256(json.dumps(keys).encode()).hexdigest()}
  return norm(out)
 def call(name,mode='q',games=None,ds=('training',),extra=(),case='positive',out_path=None,existing=None):
  global CASE
  CASE=case;calls.clear();sp=root/(name+'-split.json');sp.write_text(json.dumps({'q':qids if mode=='q' and games is None else games if mode=='q' else qids,'du':dids if mode=='du' and games is None else games if mode=='du' else dids}))
  out=out_path or root/(name+'.json')
  if existing is not None:out.write_text(existing)
  args=['gate','--weights',str(w),'--replays',str(rd),'--split-manifest',str(sp),'--mode',mode,'--out',str(out)]
  if ds is not None:
   dp=root/(name+'-ds.txt');dp.write_text('\n'.join(ds)+'\n');args+=['--ds-games-file',str(dp)]
  args+=list(extra);saved=sys.argv;sys.argv=args
  stdout=io.StringIO();stderr=io.StringIO();error=None;status=None
  try:
   with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):status=G.main()
  except SystemExit as e:status=e.code if isinstance(e.code,int) else 1;error=str(e)
  except Exception as e:status=1;error=repr(e)
  finally:sys.argv=saved
  text=out.read_text() if out.exists() else None
  data=json.loads(text) if text is not None else None
  artpath=Path(str(out)+'.pilot.json');art=json.loads(artpath.read_text()) if artpath.exists() else None
  obs={'exit_code':status,'error':norm(error),'workers':list(calls),'output':summary(data),'strict_json':not text or 'NaN' not in text and 'Infinity' not in text,'old_output_unchanged':existing is not None and text==existing,'pilot_file_exists':artpath.exists(),'pilot_file':art,'stderr':stderr.getvalue()[-300:]}
  if data and data.get('prereg'):obs['stdout_run_id_matches_output']=data['prereg']['run_id'] in stdout.getvalue()
  R['observations'][name]=norm(obs)
  return data,artpath,art,obs
 def verdict(data):return (data.get('q_gate') or data.get('du_gate'))['verdict']
 for name,ds in [('missing_ds',None),('empty_ds',[])]:
  d,_,_,o=call(name,ds=ds);R['checks'][name+'_diagnostic']=verdict(d)=='DIAGNOSTIC ONLY' and not d['prereg']['formal']
 for name,ds in [('suffix_ds',['q0.json.gz']),('overlap_ds',['q0']),('duplicate_ds',['training']*2)]:
  d,_,_,o=call(name,ds=ds);R['checks'][name+'_refused']=o['exit_code']!=0 and d is None and not o['workers']
 d,_,_,o=call('duplicate_split',games=['q0']*50);R['checks']['duplicate_split_refused']=o['exit_code']!=0 and d is None and not o['workers']
 d,_,_,o=call('q_without_train_version');R['checks']['missing_train_version_formal_pass']=verdict(d)=='PASS' and d['prereg']['train_data_version'] is None
 d,_,_,o=call('tiny_du_no_artifact',mode='du',games=['d0'],extra=['--n-required','2']);R['checks']['tiny_no_artifact_diagnostic']=verdict(d)=='DIAGNOSTIC ONLY'
 d,_,_,o=call('ordinary_du_no_artifact',mode='du');R['checks']['no_artifact_diagnostic']=verdict(d)=='DIAGNOSTIC ONLY'
 p,pa,art,o=call('pilot_valid',mode='du',games=pids,extra=['--pilot','--train-data-version','v1']);assert art
 b,_,_,_=call('stage_b_valid',mode='du',extra=['--pilot-artifact',str(pa),'--train-data-version','v1']);R['checks']['valid_bound_stage_b_pass']=verdict(b)=='PASS' and b['du_gate']['n_games']>=art['locked_N_games']
 d,_,_,o=call('pilot_games_reuse',mode='du',games=pids+dids,extra=['--pilot-artifact',str(pa)]);R['checks']['pilot_game_reuse_diagnostic']=verdict(d)=='DIAGNOSTIC ONLY' and 'overlap' in d['du_gate']['pilot_check']
 # Genuine pilot generated by the driver from a diagnostic capped run; no artifact editing.
 p,cap_pa,cap_art,o=call('pilot_capped_two_roots',mode='du',games=pids,extra=['--pilot','--max-scenes','2']);assert cap_art
 b,_,_,_=call('stage_b_from_capped_pilot',mode='du',extra=['--pilot-artifact',str(cap_pa)]);R['checks']['diagnostic_two_root_pilot_accepted']=not p['prereg']['formal'] and p['du_gate']['n']==2 and verdict(b)=='PASS'
 # Pilot source eligibility is dropped when the artifact is emitted.
 p,no_pa,no_art,o=call('pilot_without_ds',mode='du',games=pids,ds=None,extra=['--pilot']);assert no_art
 b,_,_,_=call('stage_b_from_no_ds_pilot',mode='du',ds=pids,extra=['--pilot-artifact',str(no_pa)]);R['checks']['unchecked_training_pilot_accepted']=not p['prereg']['formal'] and verdict(b)=='PASS'
 R['pilot_training_overlap']={'pilot_games':no_art['pilot_games'],'stage_b_training_ids':pids,'overlap':sorted(set(no_art['pilot_games'])&set(pids)),'stage_b_verdict':verdict(b)}
 # No rehashing or editing: normal data manifest/version change is not bound by artifact.
 b,_,_,_=call('stage_b_other_dataset_version',mode='du',ds=['different-training'],extra=['--pilot-artifact',str(pa),'--train-data-version','v2']);R['checks']['changed_training_identity_accepted']=verdict(b)=='PASS' and 'ds_manifest_md5' not in art and 'train_data_version' not in art
 # The honest pilot file is checked for accidental corruption, weights and config.
 bad=root/'corrupt-art.json';bad.write_text(pa.read_text().replace('stage1_du_pilot_artifact','corrupted_pilot_artifact'))
 d,_,_,_=call('artifact_corruption',mode='du',extra=['--pilot-artifact',str(bad)]);R['checks']['corrupt_artifact_diagnostic']=verdict(d)=='DIAGNOSTIC ONLY' and 'hash mismatch' in d['du_gate']['pilot_check']
 d,_,_,_=call('artifact_style_mismatch',mode='du',extra=['--pilot-artifact',str(pa),'--teacher-style','1']);R['checks']['style_mismatch_diagnostic']=verdict(d)=='DIAGNOSTIC ONLY' and 'teacher_style' in d['du_gate']['pilot_check']
 # Reuse --out normally; a later zero-variance pilot never republishes/deletes its old sidecar.
 reused=root/'reused-pilot.json'
 old,old_pa,old_art,_=call('pilot_before_zero_variance',mode='du',games=pids,extra=['--pilot'],out_path=reused)
 old_bytes=old_pa.read_bytes()
 new,_,new_art,_=call('pilot_zero_variance_same_out',mode='du',games=[f'p{i}' for i in range(10,20)],extra=['--pilot'],case='constant',out_path=reused)
 b,_,_,_=call('stage_b_with_latest_pilot_games',mode='du',games=[f'p{i}' for i in range(10,20)]+dids[:40],extra=['--pilot-artifact',str(old_pa)])
 R['stale_pilot_link']={'old_sidecar_unchanged':old_pa.read_bytes()==old_bytes,'old_run_id':old['prereg']['run_id'],'new_run_id':new['prereg']['run_id'],'new_pilot_games':[f'p{i}' for i in range(10,20)],'sidecar_games':new_art['pilot_games'],'sidecar_has_run_id':'run_id' in new_art,'new_main_has_artifact':'pilot_artifact' in new['du_gate'],'stage_b_verdict':verdict(b),'stage_b_includes_latest_pilot_games':set([f'p{i}' for i in range(10,20)]).issubset(b['prereg']['games'])}
 R['checks']['zero_variance_reuses_stale_pilot_sidecar']=R['stale_pilot_link']['old_sidecar_unchanged'] and not R['stale_pilot_link']['new_main_has_artifact'] and verdict(b)=='PASS'
 d,_,_,o=call('fail_exit_code',case='fail_q');R['checks']['fail_exits_one']=verdict(d)=='FAIL' and o['exit_code']==1
 previous=json.dumps({'prereg':{'run_id':'old-run'},'q_gate':{'verdict':'PASS'}})+'\n'
 d,_,_,o=call('worker_error_preserves_old',case='worker_error',existing=previous);R['checks']['worker_error_preserves_old_nonzero']=o['exit_code']!=0 and o['old_output_unchanged']
 # Strict JSON with empty sampled data, still only diagnostic.
 with gzip.open(rd/'empty.json.gz','wt') as f:json.dump({'steps':[]},f)
 d,_,_,o=call('empty_scene_strict_json',games=['empty']);R['checks']['empty_scene_is_strict_diagnostic']=o['strict_json'] and verdict(d)=='DIAGNOSTIC ONLY' and d['q_gate']['q_gain_mean'] is None
 d,_,_,o=call('nonbinding_cap',extra=['--max-scenes','999']);R['checks']['nonbinding_cap_matches_help']=verdict(d)=='PASS' and d['prereg']['canonical_config']
 # Output atomicity mechanics are exercised above; no multi-writer safety claim.
 R['all_expected_observations']=all(R['checks'].values())
def make_compact(full,raw):
 keep_prereg=['run_id','formal','canonical_config','diagnostic_reasons','n_scenes','ds_overlap_check','ds_manifest_md5','train_data_version','n_required','teacher_style','behavior_temp']
 res={k:full[k] for k in ['commit','scope','source_sha256','checks','all_expected_observations','pilot_training_overlap','stale_pilot_link']}
 res['derived']={'source_file':'results.json','source_sha256':hashlib.sha256(raw).hexdigest(),'rule':'Omit repeated CLI headers, stdout and per-row data; retain verdicts, eligibility, counts, root identity hash and unique artifact bodies. Full evidence reproducible with probe_inputs.py; timestamps and paths can vary.'}
 res['artifacts']={};res['observations']={}
 for name,obs in full['observations'].items():
  out={k:obs[k] for k in ['exit_code','error','workers','strict_json','old_output_unchanged','pilot_file_exists']}
  d=obs['output']
  if d:
   p=d.get('prereg',{});out['prereg']={k:p[k] for k in keep_prereg if k in p}
   for key in ['q_gate','du_gate']:
    if key not in d:continue
    gate=dict(d[key]);art=gate.pop('pilot_artifact',None)
    if art:res['artifacts'][art['artifact_id']]=art;gate['pilot_artifact_id']=art['artifact_id']
    # Full cluster figures kept where available, compact summaries omit no verdict terms.
    out[key]=gate
   for key in ['rows_summary','anchor_rows_summary']:
    if key in d:
     r=dict(d[key]);r['n_games']=len(r.pop('games'));out[key]=r
  if obs['pilot_file']:
   art=obs['pilot_file'];res['artifacts'][art['artifact_id']]=art;out['pilot_file_artifact_id']=art['artifact_id']
  res['observations'][name]=out
 # Store shared source/weight/config identity once; artifact IDs retain their exact original body hash.
 first=next(iter(res['artifacts'].values()));res['shared_artifact_fields']={k:first[k] for k in ('weights_md5','code_md5','guard_cfg','K','M','H','teacher_style','behavior_temp')}
 for key,art in list(res['artifacts'].items()):
  assert all(art[k]==v for k,v in res['shared_artifact_fields'].items())
  res['artifacts'][key]={k:v for k,v in art.items() if k not in res['shared_artifact_fields']}
 return res

out=Path(os.environ.get('CARBON_REVIEW_OUT','/tmp/carbon-s1-r2-inputs'));out.mkdir(parents=True,exist_ok=True)
(out/'results.json').write_text(json.dumps(R,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
compact=make_compact(R,(out/'results.json').read_bytes())
(out/'results.compact.json').write_text(json.dumps(compact,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
print(json.dumps({'checks':R['checks'],'all_expected_observations':R['all_expected_observations'],'output':str(out/'results.json')},ensure_ascii=False,indent=2))
if not R['all_expected_observations']:raise SystemExit(1)
