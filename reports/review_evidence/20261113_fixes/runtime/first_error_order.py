import os,json,time
from common import setup
repo,out=setup()
os.environ['EVAL_STEPS']='2';os.environ['EVO_WORKERS']='4';os.environ.pop('EVO_SEEDS',None)
import bigeval
original_play=bigeval._play

def labeled_task(task):
 seed=task[3];os.environ['CARBON_TEST_SEED']=str(seed)
 if seed==970500:time.sleep(.4)
 return original_play(task)

if __name__=='__main__':
 (out/'seed_error.py').write_text('import os\ndef agent(observation, configuration):\n seed=int(os.environ["CARBON_TEST_SEED"])\n if seed%2==0: raise RuntimeError("error-seed="+str(seed))\n return {}\n')
 bigeval._play=labeled_task;bigeval.HERE=str(out/'deterministic_output/code');(out/'deterministic_output/code').mkdir(parents=True,exist_ok=True)
 try:bigeval.main(str(out/'seed_error.py'),str(out/'idle.py'),970500,4,'first_error_order')
 except SystemExit as e:code=e.code
 summary=json.loads((out/'deterministic_output/data/bigeval.jsonl').read_text().splitlines()[-1])
 completion=[json.loads(line)['seed'] for line in (out/'deterministic_output/data/bigeval_live/first_error_order.jsonl').read_text().splitlines()]
 result={'exit_code':code,'completion_order':completion,'first_error_has_lowest_seed':'error-seed=970500' in summary['first_error'],'saved':summary}
 (out/'first_error_order.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
