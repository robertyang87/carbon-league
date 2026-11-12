import os,sys,json,io,contextlib,copy
import pathlib
from pathlib import Path
def repo_root():
    configured = os.environ.get('CARBON_REPO')
    if configured:
        root = pathlib.Path(configured).expanduser().resolve()
        if not (root / 'code/fastsim.py').is_file():
            raise RuntimeError('CARBON_REPO does not contain code/fastsim.py')
        return root
    for root in pathlib.Path(__file__).resolve().parents:
        if (root / 'code/fastsim.py').is_file():
            return root
    raise RuntimeError('Set CARBON_REPO to the repository root')
ROOT = repo_root()
SOURCE=ROOT/'tools/ecs_release.py'
os.environ['ALIYUN_ACCESS_KEY_ID']='review-placeholder'
os.environ['ALIYUN_ACCESS_KEY_SECRET']='review-placeholder'
sys.argv=[str(SOURCE),'i-review','192.0.2.1','REVIEW']
inst={'InstanceId':'i-review','PublicIpAddress':{'IpAddress':['192.0.2.1']},'Status':'Running'}
class Clock:
 def __init__(self):self.t=0
 def time(self):return self.t
 def sleep(self,n):self.t+=n

def run(label,polls):
 ns={'__name__':'review_module','__file__':str(SOURCE)}
 exec(compile(SOURCE.read_text(),str(SOURCE),'exec'),ns)
 calls=[]; state=[0]
 def call(action,**params):
  calls.append(action)
  if len(calls)==1:return 200,{'Instances':{'Instance':[copy.deepcopy(inst)]},'TotalCount':1}
  if action=='DeleteInstance':return 200,{'RequestId':'mock-delete'}
  i=state[0];state[0]+=1
  return polls[min(i,len(polls)-1)]
 ns['call']=call;ns['time']=Clock()
 output=io.StringIO();code=0
 with contextlib.redirect_stdout(output):
  try:ns['main']()
  except SystemExit as e:code=e.code
 lines=output.getvalue().splitlines()
 return {'case':label,'exit_code':code,'verified':any('RELEASED_VERIFIED' in x for x in lines),'last_line':lines[-1],'verify_queries':state[0]}
results=[
 run('503_then_valid_empty',[(503,{'Code':'ServiceUnavailable'}),(200,{'Instances':{'Instance':[]},'TotalCount':0})]),
 run('503_until_deadline',[(503,{'Code':'ServiceUnavailable'})]),
 run('200_still_present_until_deadline',[(200,{'Instances':{'Instance':[inst]},'TotalCount':1})]),
 run('200_malformed_empty_container',[(200,{'Instances':{}})])]
print(json.dumps(results,ensure_ascii=False,indent=2))
