"""Two-step bigeval boundary checks; records only newly written summaries."""
from pathlib import Path
import json,os,subprocess,sys
repo=Path(os.environ['CARBON_REPO']).resolve()
out=Path(os.environ.get('CARBON_REVIEW_OUT',Path(__file__).parent)).resolve()
script=repo/'reports/review_evidence/20261113_fixes/runtime/gates.py'
(out/'always_error.py').write_text('def agent(observation, configuration):\n raise RuntimeError("controlled all-fail error")\n')
env=dict(os.environ,CARBON_REPO=str(repo),CARBON_REVIEW_OUT=str(out),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
records=[];saved=out/'exam_output/data/bigeval.jsonl'
for tag,kind,n,override in [('zero_valid','always_error',2,None),('one_valid','conditional_error',2,None),('one_internal','invalid',1,None),('short_seed_list','valid',3,'970500,970501')]:
 old=saved.read_text().splitlines() if saved.exists() else []
 if override:env['EVO_SEEDS']=override
 else:env.pop('EVO_SEEDS',None)
 p=subprocess.run([sys.executable,'-B',str(script),'--worker',kind,'970500',str(n)],env=env,capture_output=True,text=True)
 new=saved.read_text().splitlines() if saved.exists() else []
 record={'test':tag,'exit_code':p.returncode,'new_records':[json.loads(x) for x in new[len(old):]],'stdout':p.stdout,'stderr':p.stderr};records.append(record)
(out/'bigeval-boundaries.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
