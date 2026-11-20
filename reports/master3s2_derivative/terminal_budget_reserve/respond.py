"""Replay the reserved-width command and V2 command against one shared fresh original reply."""
import argparse
import gzip
import importlib.util
import json

from build import FAMILY, HERE, V2_SHA, sha

spec=importlib.util.spec_from_file_location('paired_reservation',FAMILY/'single_worker_priority/response/run.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
OUT=HERE/'response'


def schedule():
    raw=(HERE/'results.json.gz').read_bytes()
    data=json.loads(gzip.decompress(raw))
    proto=(HERE/'protocol.json').read_bytes()
    p=json.loads(proto)
    assert data['protocol']==sha(proto) and p['packages']['v2']==V2_SHA
    cases=[]
    for row in sorted(data['rows'],key=lambda r:(-r['ticket']['step'],r['ticket']['seed'],r['ticket']['seat'])):
        t=row['ticket']
        if not row['stable_completed_change'] or any(c['seed']==t['seed'] for c in cases):
            continue
        a,b=[row['replies'][name][0]['issued'] for name in ('v2','reserve')]
        cases.append(t|dict(orders={'parent':a,'reserved':b},history_source_package=V2_SHA,
                            own_public_sha256=row['public_sha256'],initial_cost=row['cost'],
                            rival_rng=t['rng']^0x1D7AC43F,
                            branch_order=['parent','reserved'] if len(cases)%2==0 else ['reserved','parent']))
    return dict(schema='reserved-depth-original-replies/1',source_probe_sha256=sha(raw),parent=V2_SHA,
                candidate=p['packages']['reserve'],examiner=runner.ORIGINAL_SHA,tickets=cases,
                selection='Take the latest reproducible completed change on each eligible board; allow width change as part of the package.')


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    cli.add_argument('--seed',type=int)
    a=cli.parse_args()
    p=schedule()
    raw=runner.check.encode(p)
    OUT.mkdir(exist_ok=True)
    dst=OUT/'protocol.json'
    if a.register:
        assert not dst.exists() or dst.read_bytes()==raw
        dst.write_bytes(raw)
        print(sha(raw),[t['seed'] for t in p['tickets']])
    else:
        assert dst.read_bytes()==raw
        p=p|dict(tickets=[t for t in p['tickets'] if t['seed']==a.seed])
        assert len(p['tickets'])==1
        runner.HERE=OUT/str(a.seed)
        runner.HERE.mkdir(exist_ok=True)
        local=runner.HERE/'protocol.json'
        frozen=runner.check.encode(p)
        assert not local.exists() or local.read_bytes()==frozen
        local.write_bytes(frozen)
        runner.execute(p,a.cpu)
