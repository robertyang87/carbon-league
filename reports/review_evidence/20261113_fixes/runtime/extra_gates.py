import os,sys,subprocess,json
from pathlib import Path
from common import setup,emit

def main():
 _,out=setup()
 (out/'always_error.py').write_text('def agent(observation, configuration):\n raise RuntimeError("controlled all-fail error")\n')
 rows=[]
 for tag,kind,seed,n,override in [('zero_valid','always_error',970500,2,None),('one_valid','conditional_error',970500,2,None),('one_internal','invalid',970500,1,None),('short_seed_list','valid',970500,3,'970500,970501')]:
  env=dict(os.environ)
  if override:env['EVO_SEEDS']=override
  else:env.pop('EVO_SEEDS',None)
  p=subprocess.run([sys.executable,str(Path(__file__).resolve().with_name('gates.py')),'--worker',kind,str(seed),str(n)],capture_output=True,text=True,env=env)
  saved=json.loads((out/'exam_output/data/bigeval.jsonl').read_text().splitlines()[-1])
  rows.append({'test':tag,'exit_code':p.returncode,'saved':saved,'stdout':p.stdout,'stderr':p.stderr})
 emit('extra_gates.jsonl',rows)
if __name__=='__main__':main()
