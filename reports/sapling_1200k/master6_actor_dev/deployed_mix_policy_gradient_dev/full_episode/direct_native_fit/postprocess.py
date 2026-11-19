"""Consume each already-complete game once, without competing with the independent original examiner's two cores."""
import subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
def main():
    done=set()
    while len(done)<8:
        for p in sorted(here.glob('complete_*.json')):
            if p.name in done:continue
            subprocess.run([sys.executable,'cash_and_actions.py','--case',str(p)],check=True,cwd=here)
            done.add(p.name);print('cash verified',len(done),p.name,flush=True)
        if len(done)<8:time.sleep(3)
    while not (here/'scores.json').exists():time.sleep(1)
    subprocess.run([sys.executable,'decide.py'],cwd=here,check=True)
    subprocess.run([sys.executable,'cash_and_actions.py'],cwd=here,check=True)
if __name__=='__main__':main()
