import sys,fcntl,subprocess,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
sys.path.insert(0,str(SOURCE.parent/'tree_response_training'))
import select_roots as sr
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('two_turn_runner',SOURCE/'trial.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def prepare():
    selected=[]
    for entry in sr.load(SOURCE/'pure_screen.json'):
        seat=entry['seat'];root=entry['proposals'][-1]
        old=sr.load(SOURCE/f'seat{seat}'/'live_protocol.json')
        dest=HERE/f'seat{seat}';dest.mkdir(exist_ok=True)
        q=dict(old,root=dict(**root,tree=''),root_t=root['t'],source_dir=str(dest.relative_to(sr.ROOT)),
               screen_sha256=sr.sha(SOURCE/'pure_screen.json'))
        sr.retain(dest/'live_protocol.json',q)
        sr.retain(dest/'official_replay.json',sr.load(SOURCE/f'seat{seat}'/'official_replay.json'))
        runner.live.prepare(dest);selected.append(q['root'])
    sr.retain(HERE/'selection.json',selected)

if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif sys.argv[1:]==['verify']:
        import ledger
        ledger.summarize(HERE)
    elif len(sys.argv)>1:
        # The pre-frozen action and its continuation use the upstream route implementation verbatim.
        seat=int(sys.argv[1]);arm=sys.argv[2];runner.HERE=HERE
        runner.play(seat,arm)
    else:
        with (runner.runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for seat in (0,1):
                jobs=[subprocess.Popen(['taskset','-c',str((seat+i)%2),sys.executable,__file__,str(seat),arm]) for i,arm in enumerate(('parent','approach'))]
                codes=[j.wait() for j in jobs]
                if any(codes):raise RuntimeError(codes)
