"""Use the incumbent's same external Python 3.12 runner on the learned direction bundle."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import os
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[2]
OLD=EXP/'ladder_timing'
spec=importlib.util.spec_from_file_location('sapling_timing',OLD/'replay.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def main():
    assert os.sched_getaffinity(0)=={2}
    reg=json.loads((HERE/'protocol.json').read_bytes())
    assert bench.digest(HERE.parent/'candidate.py')==reg['candidate']
    assert bench.digest(HERE.parent/'candidate.npz')==reg['weights']
    p=json.loads((OLD/'protocol.json').read_bytes())
    assert bench.digest(OLD/'replay.py')==p['client'] and bench.digest(OLD/'worker.py')==p['worker']
    p['candidate']=reg['candidate']
    records=[]
    for file in reg['tapes']:
        source=EXP/'direct_original/games/candidate'/file
        dst=HERE/file.replace('raw_','timing_')
        if dst.exists():
            rec=json.loads(gzip.decompress(dst.read_bytes()))
            assert rec['ready']['sha256']==reg['candidate'] and rec['n']==299 and rec['transcript']==p['raw'][file]
        else:
            with tempfile.TemporaryDirectory(prefix='sapling-timing-') as tmp:
                root=Path(tmp)
                (root/'candidate.py').symlink_to(HERE.parent/'candidate.py')
                work=root/'worker';work.mkdir()
                (work/'worker.py').symlink_to(OLD/'worker.py')
                bench.HERE=work
                rec=bench.collect(source,p)
            rec.update(transcript=p['raw'][file],weights=reg['weights'])
            dst.write_bytes(gzip.compress(json.dumps(rec,allow_nan=False,separators=(',',':')).encode(),mtime=0))
        records.append(rec)
    times=[r['external_s'] for rec in records for r in rec['steps']]
    result=dict(candidate=reg['candidate'],weights=reg['weights'],games=len(records),steps=len(times),
        p95_s=float(np.percentile(times,95)),p99_s=float(np.percentile(times,99)),max_s=max(times),slow_steps=sum(t>1 for t in times),
        max_cold_start_s=max(r['init_external_s']for r in records),max_rss_mib=max(r['peak_rss_kib']for r in records)/1024,
        passed_all_local=all(r['local_thresholds_pass']for r in records),actual_cgroup=records[0]['ready']['cgroup'],
        formal_resource_certification=False)
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
