"""Replay stable deployed action changes against the unmodified original through settlement."""
import argparse
import gzip
import importlib.util
import json

from build import FAMILY, HERE, PARENT_SHA, sha

spec=importlib.util.spec_from_file_location('native_endgame_continuations',FAMILY/'single_worker_priority/response/run.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def plan(location):
    raw=(location/'results.json.gz').read_bytes()
    data=json.loads(gzip.decompress(raw))
    proto=(location/'protocol.json').read_bytes()
    assert data['protocol']==sha(proto)
    parent,candidate=json.loads(proto)['packages']
    assert parent==PARENT_SHA
    cases=[]
    for row in sorted(data['rows'],key=lambda r:(-r['ticket']['step'],r['ticket']['seed'],r['ticket']['seat'])):
        t=row['ticket']
        if not row['changed'] or any(c['seed']==t['seed'] for c in cases):
            continue
        a,b=[row['replies'][name][0]['issued'] for name in ('parent','extension')]
        cases.append(t|dict(orders={'parent':a,'extension':b}, history_source_package=parent,
                            own_public_sha256=row['public_sha256'],initial_cost=row['cost'],
                            rival_rng=t['rng']^0x7D4A1F6B,
                            branch_order=['parent','extension'] if not cases else ['extension','parent']))
        if len(cases)==2:
            break
    return dict(schema='terminal-extension-original-replies/1',source_probe_sha256=sha(raw),parent=parent,
                candidate=candidate,examiner=runner.ORIGINAL_SHA,tickets=cases,
                selection='Up to two distinct boards at fixed stable-change roots, ordered later turns then smaller seed/seat.')


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    cli.add_argument('--replication',action='store_true')
    a=cli.parse_args()
    location=HERE/'replication' if a.replication else HERE
    p=plan(location)
    out=location/'response'
    raw=runner.check.encode(p)
    out.mkdir(exist_ok=True)
    dst=out/'protocol.json'
    if a.register:
        assert not dst.exists() or dst.read_bytes()==raw
        dst.write_bytes(raw)
        print(sha(raw),len(p['tickets']))
    else:
        assert dst.read_bytes()==raw and p['tickets']
        runner.HERE=out
        runner.execute(p,a.cpu)
