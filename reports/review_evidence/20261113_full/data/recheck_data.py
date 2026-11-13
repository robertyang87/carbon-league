"""Read-only R5 data review; no training, deployment, or remote writes.
CARBON_REPO selects checkout (otherwise discover code/fastsim.py above script).
CARBON_REVIEW_OUT optionally saves JSON; default stdout. Temporary data only.
Synthetic scanner fixture has 3 decisions but index.steps=300 to pass min_steps.
One committed real replay is read with git show; no full corpus checkout needed.
Fault injection patches only in-memory os.replace, never business source.
"""
import ast, contextlib, copy, gzip, hashlib, io, json, os
from pathlib import Path
import subprocess, sys, tempfile, zlib
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['BUILD_WORKERS'] = '1'
for key in list(os.environ):
    if key.startswith('CNET2_'):
        del os.environ[key]

def find_repo():
    choices = [Path(os.environ['CARBON_REPO']).expanduser()] if os.environ.get('CARBON_REPO') else list(Path(__file__).resolve().parents)
    for p in choices:
        if (p / 'code/fastsim.py').is_file():
            return p.resolve()
    raise SystemExit('Set CARBON_REPO to a checkout containing code/fastsim.py')
repo = find_repo()
sys.path.insert(0, str(repo / 'code'))
import numpy as np
import cnet2_build as B
import merge_dagger as M
from fastsim import FastCarbon
import eval_v53
B.F.TEACHERS[:] = ['_slot0', '_slot1', 'review-teacher'] + [f'_slot{i}' for i in range(3, B.F.N_TEACH)] + ['other']
# Exact loader function, without importing/running the training module.
tree = ast.parse((repo / 'code/cnet2_train.py').read_text())
loader_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('load', 'fetch')]
loader_ns = {'np': np, 'os': os, 'json': json, 'FIELDS': B.FIELDS,
             'LOAD': [k for k in B.FIELDS if k not in ('gid', 'step')],
             'AUXK': ('aux_die','aux_die_m','aux_dep','aux_dep_m','aux_dw','aux_dw_m')}
exec(compile(ast.Module(body=loader_nodes, type_ignores=[]), 'cnet2_train.py:load/fetch (unchanged AST)', 'exec'), loader_ns)
R = {'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
     'scope': 'Real builder/merger and exact extracted loader only; no trainer, remote writes, or production rebuild',
     'training_was_launched': False, 'checks': {}}

def run():
  with tempfile.TemporaryDirectory(prefix='carbon-r5-data-') as tmp:
    root = Path(tmp)
    def clean(x):
        if isinstance(x,str): return x.replace(str(root), '<TMP>')
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list): return [clean(v) for v in x]
        return x
    sim = FastCarbon(eval_v53.CFG).reset(seed=9123)
    steps=[]
    for t in range(4):
        if t: sim.step({}, {})
        steps.append([{'observation':sim.obs(p),'status':'ACTIVE','action':{}} for p in (0,1)])
    fixture = {'steps':steps}
    def replay(p, data=fixture):
        with gzip.open(p,'wt') as f: json.dump(data,f)
    def prepare(name, both=False):
        rd=root/name/'corpus'; out=root/name/'dataset'; (rd/'replays').mkdir(parents=True)
        idx=[]
        for i in range(2):
            f=f'replays/demo-{i}.json.gz'; replay(rd/f)
            idx.append({'file':f,'steps':300,'order':['a','b'], 'scores':{'a':100,'b':0},
                        'strategies':{'a':{'name':'review-teacher'},'b':{'name':'review-teacher' if both else 'non-teacher'}}})
        (rd/'index.json').write_text(json.dumps(idx))
        return rd,out
    def snapshot(out):
        arrays={}
        for p in out.glob('*.npy'):
            a=np.load(p)
            arrays[p.stem]={'shape':list(a.shape),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        meta=json.loads((out/'meta.json').read_text()) if (out/'meta.json').exists() else None
        man=json.loads((out/'shards/_cfp.json').read_text()) if (out/'shards/_cfp.json').exists() else None
        return {'meta':clean(meta),'arrays':arrays,'manifest':man,
                'shards':sorted(p.name for p in (out/'shards').glob('*.npz')),
                'temp_dirs':sorted(p.name for p in out.glob('.tmp_publish_*'))}
    def build(name, out, dirs):
        cap=io.StringIO(); rc=0; error=None
        try:
            with contextlib.redirect_stdout(cap): B.main(str(out),list(map(str,dirs)))
        except SystemExit as e: rc=e.code
        except Exception as e: rc=1; error=repr(e)
        res={'exit_code':rc,'error':error,'stdout':clean(cap.getvalue()),'dataset':snapshot(out)}
        R[name]=res
        return res
    def loaded(out):
        D,val,meta=loader_ns['load'](str(out))
        return {'rows_by_field':{k:len(v) for k,v in D.items()},'meta_total':meta['total'],
                'loaded_weights':D['w'].tolist() if 'w' in D else None,'validation_rows':int(val.sum())}

    # R4 N03: eligible missing default fails; partial declares; restore rebuilds.
    rd,out=prepare('missing')
    build('missing_seed',out,[rd]); (rd/'replays/demo-0.json.gz').unlink()
    r=build('missing_default',out,[rd]); R['checks']['N03_default_fixed']=r['exit_code']==1 and r['dataset']['meta'] is None
    os.environ['CNET2_ALLOW_PARTIAL']='1'; B.ALLOW_PARTIAL=True
    r=build('missing_partial',out,[rd]); m=r['dataset']['meta']
    R['checks']['N03_partial_fixed']=r['exit_code']==0 and m['partial'] and m['valid_games']==1 and m['failed_games'][0]['key']=='demo-0.json.gz' and m['failed_games'][0]['reason']=='missing_replay'
    del os.environ['CNET2_ALLOW_PARTIAL']; B.ALLOW_PARTIAL=False
    replay(rd/'replays/demo-0.json.gz'); r=build('missing_restored',out,[rd])
    R['checks']['N03_retry_fixed']=r['exit_code']==0 and r['dataset']['meta']['valid_games']==2 and r['dataset']['meta']['failed_games']==[]
    # Legitimate missing-but-filtered records must not become failures.
    idx=json.loads((rd/'index.json').read_text()); short=copy.deepcopy(idx[0]); short.update(file='replays/short.json.gz',steps=2)
    nonteacher=copy.deepcopy(idx[0]); nonteacher['file']='replays/no-teacher.json.gz'; nonteacher['strategies']['a']['name']='not-listed'
    (rd/'index.json').write_text(json.dumps(idx+[short,nonteacher]))
    r=build('eligible_filter_accounting',out,[rd]);m=r['dataset']['meta']
    R['checks']['filter_accounting']=r['exit_code']==0 and m['failed_games']==[] and m['filtered_games']=={'min_steps':1,'no_teacher_side':1}

    # R4 P3 original failure: a cached shard lacks maps, while y passes cache.
    rd,out=prepare('mergefailure'); before=build('merge_seed',out,[rd])
    shard=out/'shards/demo-0.npz'
    with np.load(shard) as d: arr={k:d[k] for k in d.files if k!='maps'}
    np.savez(shard,**arr); r=build('mid_merge_failure',out,[rd])
    R['checks']['P3_mid_merge_preserves_old_product']=r['exit_code']==1 and r['dataset']['meta']==before['dataset']['meta'] and r['dataset']['arrays']==before['dataset']['arrays'] and not r['dataset']['temp_dirs']
    # Scanner parse failure still nonzero; intact previous product remains.
    (rd/'index.json').write_text('{bad json');r=build('scan_parse_failure',out,[rd])
    R['checks']['scan_failure_preserves_old_product']=r['exit_code']==1 and r['dataset']['arrays']==before['dataset']['arrays']

    # Ordinary same-directory publication is not a dataset-wide atomic swap.
    rd,out=prepare('publish'); before=build('publish_seed',out,[rd]); B.STEP_MIN=2
    actual_replace=B.os.replace; observed=[]
    def intercepted_replace(src,dst):
        if Path(dst).name=='scal.npy':
            raise OSError('review injected second-file publication failure')
        actual_replace(src,dst)
        if Path(dst).name=='maps.npy': observed.append(loaded(out))
    B.os.replace=intercepted_replace
    try:r=build('publication_replace_failure',out,[rd])
    finally:B.os.replace=actual_replace; B.STEP_MIN=0
    R['publication_replace_failure']['reader_after_first_replace']=observed
    R['publication_replace_failure']['post_failure_loader']=loaded(out)
    R['checks']['publication_mixed_version_observed']=r['exit_code']==1 and observed[0]['rows_by_field']['maps']==2 and observed[0]['rows_by_field']['y']==6 and observed[0]['meta_total']==6 and r['dataset']['meta']==before['dataset']['meta']

    # New scan ordering regression: first missing duplicate suppresses a good copy.
    rd1,out=prepare('duplicates'); rd2=root/'duplicate_good'; (rd2/'replays').mkdir(parents=True)
    idx=json.loads((rd1/'index.json').read_text())[:1]
    (rd1/'index.json').write_text(json.dumps(idx)); (rd2/'index.json').write_text(json.dumps(idx))
    (rd1/'replays/demo-0.json.gz').unlink(); replay(rd2/'replays/demo-0.json.gz')
    bad=build('duplicate_missing_first',out,[rd1,rd2]); good=build('duplicate_good_first',out,[rd2,rd1])
    R['checks']['duplicate_order_regression_observed']=bad['exit_code']==1 and good['exit_code']==0 and good['dataset']['meta']['valid_games']==1

    # Stale generated weight survives an explicit weighted -> unweighted rebuild.
    rd,out=prepare('weights',both=True); os.environ['CNET2_WEIGHTED']='1'
    build('weighted_seed',out,[rd]); original_w=np.load(out/'w.npy').tolist()
    del os.environ['CNET2_WEIGHTED']; r=build('unweighted_rebuild',out,[rd])
    R['unweighted_rebuild']['loader']=loaded(out); R['unweighted_rebuild']['old_weights']=original_w
    R['checks']['stale_weight_observed']=r['exit_code']==0 and R['unweighted_rebuild']['loader']['loaded_weights']==original_w and len(set(original_w))==2

    # Positive schema/split boundary: original-ID inheritance + old-shard refusal.
    base=root/'splitbase';base.mkdir();orig='0efcfa1227b59fd7f89ee0c9b9ca9268-1.json.gz'
    arrays={k:np.zeros((1,)+tuple(sh),dt) for k,(dt,sh) in B.FIELDS.items()}
    for k,v in arrays.items(): np.save(base/(k+'.npy'),v)
    np.save(base/'w.npy',np.ones(1,np.float16));np.save(base/'uw.npy',np.ones((1,B.F.OWN),np.float16))
    (base/'meta.json').write_text(json.dumps({'games':[{'gi':0,'key':orig,'val':True,'n':1}],'total':1,'teachers':B.F.TEACHERS}))
    for mode in ('with_orig','missing_orig'):
        dg=root/mode;(dg/'shards').mkdir(parents=True);arr=dict(arrays,uw=np.ones((1,B.F.OWN),np.float16))
        if mode=='with_orig':arr['orig_id']=np.array(orig)
        np.savez(dg/'shards/lb_cp_example.npz',**arr);dest=root/('merged_'+mode);cap=io.StringIO();err=io.StringIO();rc=0
        try:
            with contextlib.redirect_stdout(cap),contextlib.redirect_stderr(err): M.main(str(base),str(dg),str(dest))
        except SystemExit as e:rc=e.code
        res={'exit_code':rc,'output_exists':dest.exists(),'stdout':clean(cap.getvalue()),'stderr':clean(err.getvalue())}
        if (dest/'meta.json').exists():res['games']=json.loads((dest/'meta.json').read_text())['games']
        R['split_'+mode]=clean(res)
    R['checks']['split_inherited']=all(g['val'] for g in R['split_with_orig']['games'])
    R['checks']['legacy_derived_rejected']=R['split_missing_orig']['exit_code']==1 and not R['split_missing_orig']['output_exists']

    # One actual newly committed replay; reconstruct exactly ladder_corpus index schema.
    rel='data/m9xj460_corpus/0bebaa4d73009b498af099841c4163d2-1.json.gz'
    raw=subprocess.check_output(['git','show',R['commit']+':'+rel],cwd=repo)
    matches=json.loads(subprocess.check_output(['git','show',R['commit']+':data/m9xj460_matches.json'],cwd=repo))
    match=next(m for m in matches if m['id']=='0bebaa4d73009b498af099841c4163d2' and m['game']==1)
    d=json.loads(gzip.decompress(raw));rd=root/'real_corpus';(rd/'replays').mkdir(parents=True);out=root/'real_ds'
    fn='replays/'+Path(rel).name;(rd/fn).write_bytes(raw)
    entry=dict(match,file=fn,steps=len(d['steps'])-1,strategies={'a':{'name':match['name_a']},'b':{'name':match['name_b']}})
    (rd/'index.json').write_text(json.dumps([entry]));B.F.TEACHERS[2]=match['name_a']
    r=build('actual_corpus_smoke',out,[rd]);meta=r['dataset']['meta'];teacher_side=match['order'].index('a')
    expected=float(np.float16(np.tanh((match['scores']['a']-match['scores']['b'])/3000)))
    r.update(source_path=rel,replay_sha256=hashlib.sha256(raw).hexdigest(),order=match['order'],teacher_side=teacher_side,
             replay_frames=len(d['steps']),final_statuses=[s.get('status') for s in d['steps'][-1]],
             expected_value=expected,unique_values=np.unique(np.load(out/'value.npy')).tolist())
    R['checks']['actual_schema_smoke']=r['exit_code']==0 and meta['valid_games']==1 and meta['games'][0]['sides']==[teacher_side] and r['unique_values']==[expected]
    R['all_expected_checks_passed']=all(R['checks'].values())
  result=json.dumps(R,ensure_ascii=False,indent=2)+'\n';print(result,end='')
  if os.environ.get('CARBON_REVIEW_OUT'):
    p=Path(os.environ['CARBON_REVIEW_OUT']);p.mkdir(parents=True,exist_ok=True);(p/'data-results.json').write_text(result)
  if not R['all_expected_checks_passed']:raise SystemExit(1)
if __name__=='__main__':run()
