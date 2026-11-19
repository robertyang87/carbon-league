"""A third paired natural stream for two already registered training decisions."""
import fcntl
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import retest
import verify_retest

def prepare():
    for seat in (0,1):
        source=HERE.parent/f'seat{seat}'
        dest=HERE/f'seat{seat}'
        dest.mkdir(exist_ok=True)
        retest.sr.retain(dest/'live_protocol.json',retest.sr.load(source/'live_protocol.json'))
        retest.live.prepare(dest)
        hist=retest.sr.load(source/'live_protocol.json')
        for p in (source,retest.TREE/hist['original_case']):
            assert retest.sr.load(dest/'live_shared_init.json')['mt_keys']!=retest.sr.load(p/'live_shared_init.json')['mt_keys']

def verify():
    verify_retest.HERE=HERE
    results=[verify_retest.check(s) for s in (0,1)]
    retest.sr.retain(HERE/'result.json',results)

if __name__=='__main__':
    if sys.argv[1:]==['prepare']:
        prepare()
    elif sys.argv[1:]==['verify']:
        verify()
    elif len(sys.argv)>1:
        retest.HERE=HERE
        retest.play(int(sys.argv[1]),sys.argv[2])
    else:
        with (retest.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for seat in (0,1):
                jobs=[subprocess.Popen(['taskset','-c',str(1-(seat+i)%2),sys.executable,__file__,str(seat),arm])
                      for i,arm in enumerate(('parent','alternative'))]
                codes=[j.wait() for j in jobs]
                if any(codes):raise RuntimeError(codes)
        verify()
