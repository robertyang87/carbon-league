"""Compare a fixed H4/H8 terminal ranking blend against native H8."""
import argparse
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
spec=importlib.util.spec_from_file_location('signed_v2_probe',FAMILY/'single_worker_priority/probe.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
check=base.check


def build():
    raw=base.SOURCE.read_bytes()
    if check.sha(raw)!=base.PARENT_SHA:
        raise ValueError('Parent package changed')
    a=raw.index(b'def _search(')
    b=raw.index(b'\r\n_net()',a)
    block=raw[a:b]
    patches=(
        ('    dep = {}', '    four = {}\r\n    dep = {}'),
        ('        depth = t\r\n        if t == D2 or not n:',
         '        depth = t\r\n        if t == 4:\r\n            four = val.copy()\r\n        if t == D2 or not n:'),
        ('    best = max(alive, key=lambda c: mean[c])\r\n    if best == 0',
         '    # Preserve exact terminal values and scores from incomplete searches.\r\n'
         '    if depth == 8 and step + depth <= LAST_TURN and rolls == M:\r\n'
         '        mean = {c: 0.75 * mean[c] + 0.25 * sum(four[c,m] for m in range(M)) / M for c in alive}\r\n'
         '    best = max(alive, key=lambda c: mean[c])\r\n    if best == 0')
    )
    for before,after in patches:
        key=before.encode()
        if block.count(key)!=1:
            raise ValueError('Expected a unique edit location')
        block=block.replace(key,after.encode())
    output=raw[:a]+block+raw[b:]
    path=HERE/'candidate.py'
    if path.exists() and path.read_bytes()!=output:
        raise ValueError('Candidate already frozen with different bytes')
    path.write_bytes(output)
    return check.sha(output)


def registration():
    p=base.registration()
    p['schema']='native-depth4-8-blend/1'
    p['design']='At a complete nonterminal H8 only, choose using 0.75 H8 + 0.25 H4. Unchanged earlier pruning, exact terminal evaluation, source policy, sample count and budget. Use matched-work public root probes on four previously played development maps.'
    for c in p['tickets']:
        c['rng']=int(check.sha(f"depth-blend/1/{c['seed']}/{c['seat']}/{c['step']}".encode())[:8],16)
    return p


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--cpu',type=int)
    args=parser.parse_args()
    check.HERE,check.PARENT,check.CANDIDATE=HERE,base.SOURCE,HERE/'candidate.py'
    check.DIGESTS=(base.PARENT_SHA,build())
    check.registration=registration
    if args.register:
        dst=HERE/'protocol.json'
        raw=check.encode(registration())
        assert not dst.exists() or dst.read_bytes()==raw
        dst.write_bytes(raw)
        print(check.DIGESTS[1],check.sha(raw))
    else:
        check.execute(args.cpu)
