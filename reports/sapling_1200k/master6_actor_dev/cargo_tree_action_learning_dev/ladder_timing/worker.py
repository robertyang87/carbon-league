"""Persistent JSON-lines public entry. No other game inputs are mounted."""
import os
import sys
import resource
import json
import hashlib
import time

pack, cpu, memory = sys.argv[1:]
os.sched_setaffinity(0, {int(cpu)})
if int(memory):
    cap = int(memory) * 1024 ** 2
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
start = time.perf_counter()
with open(pack, 'rb') as f:
    source = f.read()
sha = hashlib.sha256(source).hexdigest()
from types import ModuleType
mod = ModuleType('ladder_agent')
mod.__file__ = pack
sys.modules[mod.__name__] = mod
exec(compile(source, pack, 'exec'), mod.__dict__)
del source
import numpy

def emit(value):
    print(json.dumps(value, separators=(',', ':'), allow_nan=False), flush=True)

emit(dict(ready=True, sha256=sha, python=sys.version, numpy=numpy.__version__,
          import_seconds=time.perf_counter()-start, affinity=sorted(os.sched_getaffinity(0)),
          rlimit_as=resource.getrlimit(resource.RLIMIT_AS),
          cgroup={k:open('/sys/fs/cgroup/'+k).read().strip()
                  for k in ('cpu.max','memory.max','cpuset.cpus.effective')},
          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
for line in sys.stdin:
    req = json.loads(line)
    action = mod.agent(req['observation'], req['configuration'])
    emit(dict(action=action, step=req['observation']['step'],
              peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
