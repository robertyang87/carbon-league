"""Matched serial timing on the original four signed histories, with independently decoded candidate actions."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
OLD=EXP/'ladder_timing'
spec=importlib.util.spec_from_file_location('parent_timing_replay',OLD/'replay.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

def encode(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def write(path,data):
    b=encode(data)
    tmp=path.with_suffix('.incomplete')
    tmp.write_bytes(gzip.compress(b,mtime=0) if path.suffix=='.gz' else b)
    tmp.replace(path)

def main():
    reg=json.loads((HERE/'protocol.json').read_text());p=reg['baseline']
    assert bench.digest(OLD/'replay.py')==p['client'] and bench.digest(OLD/'worker.py')==p['worker']
    assert bench.digest(EXP/'candidate.npz')==reg['parent_weights'] and bench.digest(HERE.parent/'candidate.npz')==reg['candidate_weights']
    assert os.sched_getaffinity(0)=={2}
    old_receive=bench.receive
    steps=[]
    def receive(proc,timeout):
        msg=old_receive(proc,timeout)
        if 'step' in msg:steps.append(msg['action'])
        return msg
    bench.receive=receive
    for i,fn in enumerate(sorted(p['raw'])):
        for arm in (('parent','candidate') if i%2==0 else ('candidate','parent')):
            out=HERE/(arm+'_'+fn.replace('raw_','timing_'));pkg=reg['candidate'] if arm=='candidate' else p['candidate']
            if out.exists():
                r=json.loads(gzip.decompress(out.read_bytes()))
                assert r['ready']['sha256']==pkg and r['transcript_sha256']==p['raw'][fn] and r['n']==len(r['public_actions'])==299
                continue
            bench.HERE=OLD if arm=='parent' else HERE
            src=EXP/'direct_original/games/candidate'/fn;steps.clear()
            r=bench.collect(src,dict(p,candidate=pkg))
            assert r['n']==len(steps)==299
            if arm=='parent':assert r['matches_tape']==r['n']
            assert all(isinstance(a,dict) for a in steps)
            r.update(public_actions=list(steps),actions_sha256=hashlib.sha256(encode(steps)).hexdigest(),
                     seat_weights_sha256=reg[arm+'_weights'],transcript_sha256=p['raw'][fn],arm=arm)
            write(out,r)
            print(arm,r['seed'],r['seat'],r['n'],r['matches_tape'],r['slow_steps'],r['max_s'],flush=True)
if __name__=='__main__':main()
