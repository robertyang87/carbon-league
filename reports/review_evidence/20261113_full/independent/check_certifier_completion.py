"""Local resource-contract probes. No training, network or deployment.
Uses the unmodified certifier with a short explicitly overridden game budget.
"""
import hashlib,json,os,pathlib,subprocess,sys,tempfile,time
repo=pathlib.Path(os.environ.get('CARBON_REPO',str(pathlib.Path.cwd()))).resolve()
out=pathlib.Path(os.environ.get('CARBON_REVIEW_OUT','/tmp/carbon-review-round5/independent')).resolve();out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
rows=[]
with tempfile.TemporaryDirectory(prefix='carbon-r5-cert-') as td:
 t=pathlib.Path(td)
 (t/'idle.py').write_text('def agent(obs, cfg):\n return {}\n')
 (t/'slow_opp.py').write_text('import time\ndef agent(obs, cfg):\n time.sleep(3.05)\n return {}\n')
 for kind,opp in [('idle_control',t/'idle.py'),('slow_last_advance',t/'slow_opp.py')]:
  report=t/(kind+'.json')
  argv=[sys.executable,str(repo/'code/certify_pack.py'),str(t/'idle.py'),'--opp',str(opp),'--steps-per-game','1','--games','1','--game-limit-s','2','--out',str(report)]
  start=time.perf_counter();p=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=30);elapsed=time.perf_counter()-start
  cert=json.loads(report.read_text()) if report.exists() else None
  rows.append({'case':kind,'exit_code':p.returncode,'actual_outer_elapsed_s':elapsed,'certificate':cert,'stderr':p.stderr,'scope':'real child/opponent execution; 1 step, game budget overridden to 2 s; no platform performance assertion'})
 result={'reviewed_sha':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip(),'source_sha256':hashlib.sha256((repo/'code/certify_pack.py').read_bytes()).hexdigest(),'results':rows}
 (out/'certifier-completion.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps([{'case':r['case'],'exit_code':r['exit_code'],'actual_outer_elapsed_s':round(r['actual_outer_elapsed_s'],3),'reported_game_wall_s':r['certificate']['aggregate']['max_game_wall_s'],'overall':r['certificate']['overall']} for r in rows],indent=2))
