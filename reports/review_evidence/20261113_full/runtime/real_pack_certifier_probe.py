"""Short certificates for real frozen packs and rebuilt error-injection packages.
Requires pack_export_probe.py first. These are smoke results, not cgroup qualification.
"""
import concurrent.futures,json,os,subprocess,sys
from common import paths,env
repo,out=paths();e=env();e['CNET2_NO_ROLL']='1';e.pop('CNET2_ENS',None)
conf={'SEARCH':False,'ROLL':'self','GUARD':True,'GUARD_NARROW':True,'MODE':'sweep','MIX':False,'TEACHER':2,'TEACHER_LATE':2,'LATE_STEP':240,'BUDGET':0.7,'HARD':0.85,'ABORT_DL':0.95,'START_DL':None}
cases=[('n1r',repo/'code/packs/n1r_int8_reference.py'),('echo3',repo/'code/packs/echo3_int8.py')]
for name,changes in [('rule_error',{'TG_MAX':'invalid'}),('root_error',{'TEACHER':999})]:
 p=out/(name+'_pack.py');r=subprocess.run([sys.executable,'-B',str(repo/'code/packs/cnet2_search_pack.py'),str(p),repr(dict(conf,**changes)),str(out/'n1r_dequantized.npz')],env=e,capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr;cases.append((name,p))
def run(item):
 name,p=item;r=subprocess.run([sys.executable,'-B',str(repo/'code/certify_pack.py'),str(p),'--steps-per-game','2','--mem-limit-mib','512','--out',str(out/(name+'.short-certificate.json'))],env=e,capture_output=True,text=True,timeout=20)
 return {'case':name,'exit_code':r.returncode,'certificate':json.loads(r.stdout),'stderr':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,cases))
result={'limitation':'Two measured steps after certifier warmup; unconstrained host, not 1.5 CPU/512MiB cgroup qualification, no strength test. Original n1r/Echo3 unchanged. Error packs rebuilt from dequantized n1r q8 fixture.','cases':rows}
(out/'real-pack-certifier-results.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:print(r['case'],r['exit_code'],r['certificate']['overall'],r['certificate']['aggregate']['a_errors'])
