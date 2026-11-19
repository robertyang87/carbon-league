"""Preflight before any agent code, then replace the preflight process with a cold Python runner."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def emit(value):
    print(json.dumps(value, separators=(',', ':'), allow_nan=False), flush=True)


def resource_sample():
    import resource
    root = Path('/sys/fs/cgroup')
    peak = root/'memory.peak'
    cpu_stat = root/'cpu.stat'
    cpu = dict(line.split() for line in cpu_stat.read_text().splitlines()) if cpu_stat.is_file() else {}
    return {'rss_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cgroup_peak_bytes': int(peak.read_text()) if peak.is_file() else None,
            'cgroup_cpu': {k: int(cpu[k]) for k in ('usage_usec', 'nr_periods', 'nr_throttled', 'throttled_usec') if k in cpu}}


def probe():
    from importlib import metadata
    import importlib.util
    import socket

    root = Path('/sys/fs/cgroup')
    files = ('cpu.max', 'memory.max', 'memory.swap.max', 'cpuset.cpus.effective')
    groups = {key: (root/key).read_text().strip() if (root/key).is_file() else None for key in files}
    # Some lean images install NumPy without the dist-info metadata. Read its source
    # version from the actual import; exec below gives each agent a fresh interpreter.
    import numpy
    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    spec = importlib.util.find_spec('zerosum_env')
    helper = Path(spec.origin).parent/'envs/carbon/helpers.py' if spec and spec.origin else None
    return dict(cgroup=groups, cgroup_inode=root.stat().st_ino, affinity=sorted(os.sched_getaffinity(0)),
                interfaces=sorted(x[1] for x in socket.if_nameindex()), python=list(sys.version_info[:3]),
                numpy=numpy.__version__, torch=version('torch'),
                helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest() if helper and helper.is_file() else None)


def run(role):
    # Same public entry loading and JSON contract as ladder_timing/worker.py. This
    # lane leaves the CPU set and native torch threading entirely to the real cgroup.
    from types import ModuleType
    source = Path('/submission.py').read_bytes()
    mod = ModuleType('ladder_agent')
    mod.__file__ = '/submission.py'
    sys.modules[mod.__name__] = mod
    exec(compile(source, mod.__file__, 'exec'), mod.__dict__)
    identity = hashlib.sha256(source).hexdigest()
    del source
    ready = dict(ready=True, sha256=identity, **resource_sample())
    if role == 'opponent':
        ready['original'] = {k:getattr(mod,k,None) for k in ('_TEMP','_SEED','_S_ABORT','_SEARCH_OK')}
    emit(ready)
    for line in sys.stdin:
        request = json.loads(line)
        old_single = mod._S_STATS['single'] if role == 'opponent' else None
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        action = mod.agent(request['observation'], request['configuration'])
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        response = dict(action=action, step=request['observation']['step'], public_wall_seconds=wall,
                        public_cpu_seconds=cpu, **resource_sample())
        if role == 'opponent':
            stats = mod._S_STATS
            response['search_counts'] = {key: stats[key] for key in ('searched','single','aborted','changed')}
            if stats['single'] > old_single:
                response['search_last'] = dict(candidates=1, depth=0, rolls=0, search_seconds=None)
            elif 'last' in stats:
                candidates, depth, rolls, elapsed, _ = stats['last']
                response['search_last'] = dict(candidates=int(candidates), depth=int(depth), rolls=int(rolls), search_seconds=float(elapsed))
        emit(response)


if __name__ == '__main__':
    role = sys.argv[1]
    if len(sys.argv)==3 and sys.argv[2]=='loaded':
        run(role)
    else:
        emit(probe())
        if sys.stdin.buffer.readline()==b'{"load":true}\n':
            os.execv(sys.executable, [sys.executable, '-u', __file__, role, 'loaded'])
