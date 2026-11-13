"""Small real-process probes for certification gates; not platform qualification."""
import concurrent.futures,json,subprocess,sys
from common import paths,env
repo,out=paths()
cases={
 'legal':('def agent(obs,cfg): return {}\n',['--steps-per-game','2']),
 'cold_3050':('import time\ncalled=False\ndef agent(obs,cfg):\n global called\n if not called: time.sleep(3.05);called=True\n return {}\n',['--steps-per-game','2']),
 'warm_3050':('import time\ndef agent(obs,cfg):\n time.sleep(3.05)\n return {}\n',['--steps-per-game','1']),
 'two_games_two_slow_each':('import time\ndef agent(obs,cfg):\n time.sleep(1.02)\n return {}\n',['--steps-per-game','2','--games','2']),
 'invalid_opponent':('def agent(obs,cfg): return {}\n',['--steps-per-game','2','--opp',str(out/'nonexistent-opponent.py')]),
 'wrong_action_key':('def agent(obs,cfg): return {"not-my-unit":"UP"}\n',['--steps-per-game','2']),
}
def run(item):
 name,(source,extra)=item;p=out/(name+'.py');p.write_text(source)
 command=[sys.executable,'-B',str(repo/'code/certify_pack.py'),str(p),*extra,'--out',str(out/(name+'.certificate.json'))]
 r=subprocess.run(command,env=env(),capture_output=True,text=True,timeout=30)
 try:certificate=json.loads(r.stdout)
 except ValueError:certificate=None
 return {'case':name,'command':command,'exit_code':r.returncode,'certificate':certificate,'stdout_nonjson':r.stdout if certificate is None else None,'stderr':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,cases.items()))
result={'reviewed_sha':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'method':'real subprocesses; source unmodified; short workloads only; default time gates','cases':rows}
(out/'certifier-probes.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
 c=r['certificate'];print(r['case'],r['exit_code'],c['overall'] if c else None,[(k,v) for k,v in c['checks'].items() if v['pass'] is False] if c else r['stderr'])
