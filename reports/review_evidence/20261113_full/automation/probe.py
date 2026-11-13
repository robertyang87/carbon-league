#!/usr/bin/env python3
"""Read-only-source automation probes; no game, training, network or cloud call.

Execute only extracted shell helpers/marker branches in temporary directories,
and bigeval with an in-process fake executor. Never run a cloud queue script.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import types

REPO = Path(os.environ.get('CARBON_REPO', '/workspace/carbon-league')).resolve()
OUT = Path(__file__).resolve().parent
RESULT = {'commit': subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip(),
          'scope': 'local extracted-helper/AST/fake-worker probes only; no real evaluation, training, network, workflow dispatch or cloud operations'}

def sh(code, cwd, env=None):
    p = subprocess.run(['bash','-c',code], cwd=cwd, env=dict(os.environ, **(env or {})), text=True, capture_output=True, timeout=10)
    return {'exit': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

source = (REPO/'code/cloud/q_h14.sh').read_text()
miss = source[source.index('miss()'):source.index('# --- N02')]
helpers = source[source.index('count_seeds()'):source.index('SX_PIDS=')]
health = helpers[:helpers.index('collect_group()')]
marker = source[source.index('if [ "$SX_RC" -ne 0 ]'):source.index('export AWR_US_WEIGHTS=', source.index('echo SX_DONE'))]

with tempfile.TemporaryDirectory(prefix='automation-review-') as td:
    root = Path(td); code = root/'code'; data = root/'data'; live = data/'bigeval_live'
    code.mkdir(); live.mkdir(parents=True)
    tag = 'sx_S20_probe'
    p = live/(tag+'.jsonl')
    # Failed old record must be retried, but the real helper excludes it.
    p.write_text(json.dumps({'seed':994000,'err':'RuntimeError(probe)','serr':1,'margin':None})+'\n')
    r = sh(miss+f'\nmiss {tag} 994000', code)
    seeds = [int(x) for x in r['stdout'].strip().split(',')]
    (data/'bigeval.jsonl').write_text(json.dumps({'tag':tag,'FAIL':False,'n':len(seeds), 'games':[[s,s%2,1] for s in seeds]})+'\n')
    checked = sh(health+f'\ncheck_bigeval_health {tag} {len(seeds)} ""', code)
    RESULT['failed_old_seed_omitted_then_resumed_batch_accepted'] = {
        'old_error_seed':994000, 'missing_count':len(seeds), 'old_error_seed_retried':994000 in seeds,
        'new_summary_health_check':checked, 'total_expected':100, 'healthy_expected_after_resume':99}
    # Even fully healthy records take the unconditional empty-list fail branch.
    p.write_text(''.join(json.dumps({'seed':s,'err':None,'serr':0,'margin':1,'config_hash':'same_config'})+'\n' for s in range(994000,994100)))
    empty=sh(miss+f'\nmiss {tag} 994000',code)
    RESULT['complete_healthy_missing_list']={'missing_stdout':empty['stdout'], 'missing_exit':empty['exit'],
        'source_empty_branch_sets_failure': 'SX_RC=1' in source[source.index('if [ "$N" -eq 0 ]'):source.index('echo "N02: sx_S20_$O')]}
    # The tag/count-only health predicate accepts the wrong identity and seed set.
    wrong={'tag':tag,'FAIL':False,'n':2,'a':'old_actor.py','b':'wrong_opponent.py','games':[[7,1,1],[7,1,1]]}
    (data/'bigeval.jsonl').write_text(json.dumps(wrong)+'\n')
    RESULT['wrong_identity_duplicate_seed_summary_accepted']={'summary':wrong,'health_check':sh(health+f'\ncheck_bigeval_health {tag} 2 ""',code)}
    (data/'bigeval.jsonl').unlink()
    RESULT['no_summary_no_output_accepted']=sh(health+f'\ncheck_bigeval_health {tag} 2 ""',code)
    # Existing success marker remains visible after a real extracted failure branch.
    (data/'sx.done').write_text('SX_DONE\n')
    failed=sh('SX_RC=1; COLLECT_RC=0\n'+marker,code)
    RESULT['stale_done_after_failure']={'branch':failed,'done_still_exists':(data/'sx.done').exists(),'fail_exists':(data/'sx.fail').exists()}
    # Per-PID collection now propagates failure: fake local child, no game worker.
    RESULT['per_pid_failure_propagates']=sh(helpers+'\n(false) &\nPIDS=($!); TAGS=(probe); WANTS=(2); OUTS=("")\ncollect_group PIDS TAGS WANTS OUTS\nexit "$COLLECT_RC"',code)
    RESULT['count_contracts']={}
    for name in ('q_h14d.sh','q_h15.sh'):
        s=(REPO/'code/cloud'/name).read_text()
        vals={k:v.split(',') for k,v in re.findall(r"^(SEEDS_\w+)='([^']*)'",s,re.M)}
        RESULT['count_contracts'][name]={k:{'count':len(v),'unique_count':len(set(v))} for k,v in vals.items()}
    # Guard only: cut off the real dgx script before its first cd/wait/copy/train.
    dgx=(REPO/'code/cloud/dgx_full.sh').read_text().split('cd ~/junyang/carbon/code')[0]
    RESULT['dgx_unconfirmed_prefix']=sh(dgx,code,{'DGX_FULL_CONFIRM':''})
    RESULT['dgx_confirmed_prefix_only']=sh(dgx+'\necho GUARD_PASSED_NO_OPERATION',code,{'DGX_FULL_CONFIRM':'1'})
    # Historical post-opt-in control flow: local copy fixture, fake env() makes
    # every would-be trainer call fail before any Python/process is launched.
    logs=data/'ex1g_logs'; logs.mkdir(); (logs/'run.log').write_text('SWEEP_DONE\n')
    (code/'cnet2_train.py').write_text('original_stub_trainer\n')
    dgx_tail=(REPO/'code/cloud/dgx_full.sh').read_text().split('cd ~/junyang/carbon/code\n',1)[1]
    dr=sh('env() { return 7; }\n'+dgx_tail,code)
    RESULT['dgx_confirmed_tail_failures_still_done']={'fake_env_intercepts_all_training':True,'branch':dr,
        'unversioned_copy_exists':(code/'cnet2_train.py.d0dd809').exists(),
        'trainer_still_original':(code/'cnet2_train.py').read_text()=='original_stub_trainer\n',
        'full_done_written':'FULL_DONE' in (logs/'full.log').read_text()}
    # Real bigeval.main, with all game evaluation and multiprocessing replaced.
    tree=ast.parse((REPO/'code/bigeval.py').read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and "__name__" in ast.unparse(n.test))]
    ns={'__file__':str(code/'bigeval.py'),'__name__':'review_bigeval'}
    exec(compile(tree,str(REPO/'code/bigeval.py'),'exec'),ns)
    class FakeFuture:
        def __init__(self,v): self.v=v
        def result(self): return self.v
    class FakeExecutor:
        def __init__(self,*a,**kw): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def submit(self,fn,t):
            ledger={k:0 for k in ('deposits','death_cargo','tree_income','plant_count')}
            return FakeFuture({'seed':t[3],'policy_side':t[4],'margin':1,'ledger_pol':dict(ledger),'ledger_opp':dict(ledger),'a_errors':0})
    ns['ProcessPoolExecutor']=FakeExecutor; ns['as_completed']=lambda f:f
    prev=os.environ.get('EVO_SEEDS'); os.environ['EVO_SEEDS']='17,17'
    output=io.StringIO(); exit_code=0
    try:
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            ns['main']('A.py','B.py',0,2,'duplicate_probe')
    except SystemExit as e: exit_code=e.code
    finally:
        if prev is None: os.environ.pop('EVO_SEEDS',None)
        else: os.environ['EVO_SEEDS']=prev
    summary=json.loads((data/'bigeval.jsonl').read_text().splitlines()[-1])
    RESULT['bigeval_duplicate_explicit_seeds']={'exit':exit_code,'summary':summary,'unique_seeds':len({g[0] for g in summary['games']}),'fake_workers':True}
    # Existing client retries POST connection failures too. Model accepted request
    # followed by a lost response; this illustrates ambiguity, not a live duplicate.
    prevcookie=os.environ.get('LADDER_COOKIES'); os.environ['LADDER_COOKIES']=str(root/'nonexistent-cookie')
    spec=importlib.util.spec_from_file_location('review_ladder',REPO/'code/ladder.py'); ladder=importlib.util.module_from_spec(spec); spec.loader.exec_module(ladder)
    if prevcookie is None: os.environ.pop('LADDER_COOKIES',None)
    else: os.environ['LADDER_COOKIES']=prevcookie
    import http.client
    class AcceptedThenDisconnect:
        def __init__(self):self.attempts=[]
        def open(self,req,timeout):
            self.attempts.append(req.get_method())
            if len(self.attempts)==1: raise http.client.RemoteDisconnected('stub: request accepted but response lost')
            return 'stub_success'
    op=AcceptedThenDisconnect(); ladder._opener=op
    ladder.time=types.SimpleNamespace(sleep=lambda _:None)
    req=ladder.urllib.request.Request('https://local-stub.invalid/api/submit',data=b'stub',method='POST')
    returned=ladder._open_retry(req,1)
    RESULT['ladder_post_ambiguous_retry']={'attempt_methods':op.attempts,'return':returned,'network_calls':0,'scope':'client retry behavior only; server acceptance simulated'}
    # Reproduce the workflow's string interpolation with benign values and a
    # fake python3 function. The intended remote env reference is expanded on
    # the runner before it reaches the runner tool. No credentials are read.
    cloud=(REPO/'.github/workflows/cloud-assistant.yml').read_text()
    line=next(l.strip() for l in cloud.splitlines() if 'acs_runcommand.py' in l and 'inputs.command }}' in l)
    rendered=line.replace('${{ inputs.instance_id }}','i-stub').replace('${{ inputs.command }}','echo "$REVIEW_REMOTE_ONLY"')
    RESULT['cloud_command_local_expansion']=sh('python3() { printf "%s\\n" "$@"; }\n'+rendered,code,{'REVIEW_REMOTE_ONLY':'LOCAL_RUNNER_VALUE'})

pack=(REPO/'code/packs/echo3_int8.py').read_bytes()
RESULT['ladder_submit_static']={'trigger':'workflow_dispatch_only','bytes':len(pack),'md5':hashlib.md5(pack).hexdigest(),'workflow_executed':False,'client_executed_live':False}
(OUT/'results.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(RESULT,ensure_ascii=False,indent=2))
