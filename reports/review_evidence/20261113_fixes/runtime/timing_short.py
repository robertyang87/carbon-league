"""Execute the real timing tool with only its 299-step loop shortened in memory to 2.
This checks health reporting/exit codes, not performance certification.
"""
import ast,os,sys,subprocess,json
from pathlib import Path
from common import setup,emit

def worker(name):
 repo,out=setup()
 path=repo/'code/cnet2_search_timing.py'
 tree=ast.parse(path.read_text());count=0
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==299:
   node.args[0].value=2;count+=1
 assert count==1,('expected one timing loop',count)
 sys.argv=[str(path),str(out/(name+'.py')),str(out/'idle.py'),'970500','1']
 exec(compile(tree,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})

def main():
 repo,out=setup()
 import cnet2_search_pack
 os.environ['CNET2_NO_ROLL']='1'
 cnet2_search_pack.main(str(out/'rule_invalid.py'),repr({'SEARCH':False,'ROLL':'self','TEACHER':1,'TG_MAX':'invalid'}),str(out/'baseline.npz'))
 rows=[]
 for name in ('valid','invalid','rule_invalid'):
  p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name],capture_output=True,text=True)
  rows.append({'kind':name,'step_loop_shortened_in_memory_to':2,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 # Show the actual category-specific error counter hidden by the timing summary.
 import eval_v53
 from fastsim import FastCarbon
 g={'__name__':'submission'};exec(compile((out/'rule_invalid.py').read_text(),'<submission>','exec'),g)
 sim=FastCarbon(eval_v53.CFG).reset(seed=970500)
 for _ in range(2):sim.step(g['agent'](sim.obs(0),eval_v53.CFG),{})
 rows.append({'kind':'rule_invalid_direct_stats','stats':g['_AG'].stats})
 emit('timing_short.jsonl',rows)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2])
 else:main()
