"""Check two wider-menu first actions against naturally responding original Master3s2."""
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
    raw=(HERE.parent/'self_calibrated.json.gz').read_bytes()
    rows=json.loads(gzip.decompress(raw))
    proto=(HERE.parent/'protocol.json').read_bytes()
    assert check.sha(proto)==rows['source_protocol_sha256']
    parent,candidate=json.loads(proto)['packages']
    assert parent==runner.PARENT_SHA
    selected={}
    for row in sorted(rows['rows'],key=lambda r:(r['ticket']['seed'],r['ticket']['seat'])):
        c=row['ticket']
        x,y=[row['answers'][n][0] for n in ('parent','faster')]
        if not row['stable_change'] or c['step']!=247 or x['depth']!=8 or y['depth']!=8 or y['issued'] in x['menu']:
            continue
        selected.setdefault(c['seed'],row)
    tickets=[]
    for row in list(selected.values())[:2]:
        c=row['ticket']
        a,b=[row['answers'][n][0]['issued'] for n in ('parent','faster')]
        assert a!=b
        tickets.append(c|dict(orders={'parent':a,'wider_menu':b},history_source_package=parent,
                              own_public_sha256=row['public_sha256'],initial_cost=row['cost']['parent'],
                              rival_rng=c['rng']^0x7D4A1F6B,
                              branch_order=['parent','wider_menu'] if not tickets else ['wider_menu','parent']))
    assert len(tickets)==2
    return dict(schema='native-speed-action-response/1',source_probe_sha256=check.sha(raw),
                parent=parent,candidate=candidate,examiner=runner.ORIGINAL_SHA,
                design='Choose first two distinct boards at t247 with repeatable H8 on the first call of both packages and a winning command absent from the parent menu. Change only our root order, use actual public V2 and original thereafter, and record official outcomes for all four branches.',
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
        assert dst.read_bytes()==raw
        runner.execute(p,args.cpu)
