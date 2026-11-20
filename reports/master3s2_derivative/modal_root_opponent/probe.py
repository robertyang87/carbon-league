"""Use a modal opponent proposal at the root and retain natural rollout sampling."""
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
        ('    opp = {}',
         '    opp = {}\r\n\r\n'
         '    class ModalDraw:\r\n'
         '        def choice(self, size, p):\r\n'
         '            return int(np.argmax(p))'),
        ('_sample(players, op, lwo[0], lbo[0], np.random.RandomState(seed0 + 7 + 31 * m), step)',
         '_sample(players, op, lwo[0], lbo[0], ModalDraw(), step)'),
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
    p['schema']='modal-root-opponent/1'
    p['design']='Use each entity’s modal predicted root action with the original occupancy and cash decoder. Later rollout moves remain naturally sampled. Same V2 model, root menu generation, and wall budget. At the root record overall exact physical command match and per-worker executed action; freeze root ids before reading holds.'
    for c in p['tickets']:
        c['rng']=int(check.sha(f"modal-root/1/{c['seed']}/{c['seat']}/{c['step']}".encode())[:8],16)
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
