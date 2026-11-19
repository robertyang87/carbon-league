import ctypes
import os
import resource
import sys
import json
import hashlib
from pathlib import Path
from time import perf_counter
from types import ModuleType

pack, cap_mib=sys.argv[1:]
os.sched_setaffinity(0,{2})
if int(cap_mib):
    cap=int(cap_mib)*1024**2
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
assert 'numpy' not in sys.modules
inherited=os.environ['OPENBLAS_NUM_THREADS']
t0=perf_counter()
source=Path(pack).read_bytes()
sha=hashlib.sha256(source).hexdigest()
module=ModuleType('ladder_agent')
module.__file__=pack
sys.modules[module.__name__]=module
exec(compile(source,pack,'exec'),module.__dict__)
del source
elapsed=perf_counter()-t0
import numpy
lib=next(p for p in Path(numpy.__file__).resolve().parents[1].joinpath('numpy.libs').iterdir() if 'openblas' in p.name)
threads=ctypes.CDLL(str(lib)).openblas_get_num_threads64_()
def emit(obj):
    print(json.dumps(obj,separators=(',',':'),allow_nan=False),flush=True)
emit(dict(ready=True,inherited=inherited,actual_threads=threads,as_limit=resource.getrlimit(resource.RLIMIT_AS),
          affinity=sorted(os.sched_getaffinity(0)),runtime=os.environ['OPENBLAS_NUM_THREADS'],pid=os.getpid(),sha256=sha,
          rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,package_init_s=elapsed,python=sys.version,numpy=numpy.__version__))
for line in sys.stdin:
    req=json.loads(line)
    a=module.agent(req['observation'],req['configuration'])
    emit(dict(action=a,step=req['observation']['step'],rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
