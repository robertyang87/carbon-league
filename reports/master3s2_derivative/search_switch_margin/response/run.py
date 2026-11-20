"""Check two conservative first actions against naturally responding original Master3s2."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parents[1]
ENTRY = FAMILY/'single_worker_priority/response/run.py'
spec = importlib.util.spec_from_file_location('verified_suffix_runner',ENTRY)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.HERE = HERE
check=runner.check


def plan():
    raw=(HERE.parent/'results.json.gz').read_bytes()
    rows=json.loads(gzip.decompress(raw))
    proto=(HERE.parent/'protocol.json').read_bytes()
    assert check.sha(proto)==rows['protocol_sha256']
    parent,candidate=json.loads(proto)['packages']
    assert parent==runner.PARENT_SHA
    selected={}
    for row in sorted(rows['rows'],key=lambda r:(r['ticket']['seed'],r['ticket']['seat'])):
        c=row['ticket']
        x,y=[row['live'][n][0] for n in ('v4','critic')]
        if not row['reliable_change'] or c['step']!=247:
            continue
        selected.setdefault(c['seed'],row)
    tickets=[]
    for row in list(selected.values())[:2]:
        c=row['ticket']
        a,b=[row['live'][n][0]['issued'] for n in ('v4','critic')]
        assert a!=b
        tickets.append(c|dict(orders={'parent':a,'guarded':b},history_source_package=parent,
                              own_public_sha256=row['public_sha256'],initial_cost=row['cost'],
                              rival_rng=c['rng']^0x7D4A1F6B,
                              branch_order=['parent','guarded'] if not tickets else ['guarded','parent']))
    return dict(schema='switch-margin-action-response/1',source_probe_sha256=check.sha(raw),
                parent=parent,candidate=candidate,examiner=runner.ORIGINAL_SHA,
                design='Choose first two distinct boards at t247 with a stable changed native command also reproduced at matched work. Change only our root order, use actual public V2 and original thereafter, and record official outcomes for all four branches.',
                tickets=tickets)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--register',action='store_true')
    ap.add_argument('--cpu',type=int)
    args=ap.parse_args()
    p=plan()
    raw=check.encode(p)
    dst=HERE/'protocol.json'
    if args.register:
        assert not dst.exists() or dst.read_bytes()==raw
        dst.write_bytes(raw)
        print(check.sha(raw),[(c['seed'],c['seat']) for c in p['tickets']])
    else:
        assert dst.read_bytes()==raw and p['tickets']
        runner.execute(p,args.cpu)
