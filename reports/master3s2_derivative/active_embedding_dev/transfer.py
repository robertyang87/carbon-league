"""Fetch the verified dev run's raw GitHub artifact ZIPs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import zipfile


HERE=Path(__file__).resolve().parent
API='repos/robertyang87/carbon-league/actions/'


def get(path):
    return subprocess.check_output(['gh','api',API+path])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=int,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--protocol',required=True)
    args=p.parse_args()
    run=json.loads(get(f'runs/{args.run}'))
    assert run['path']=='.github/workflows/master3s2-active-embedding-dev.yml'
    assert run['head_sha']==args.commit and run['run_attempt']==1 and run['status']=='completed'
    raw=(HERE/'protocol.json').read_bytes()
    assert sha256(raw).hexdigest()==args.protocol
    names={f"strategic-dev-{t['seed']}-{t['seat']}-{args.run}" for t in json.loads(raw)['schedule']}
    index=json.loads(get(f'runs/{args.run}/artifacts?per_page=100'))
    assert index['total_count']==len(index['artifacts'])
    selected=[a for a in index['artifacts'] if a['name'] in names]
    assert {a['name'] for a in selected}==names and len(selected)==len(names)
    dest=HERE/'hosted'
    dest.mkdir(exist_ok=True)
    for a in selected:
        assert a['workflow_run']['id']==args.run and not a['expired']
        path=dest/f"{a['id']}.zip"
        payload=path.read_bytes() if path.exists() else get(f"artifacts/{a['id']}/zip")
        assert len(payload)==a['size_in_bytes'] and a['digest']=='sha256:'+sha256(payload).hexdigest()
        if not path.exists():
            stage=path.with_suffix('.partial')
            stage.write_bytes(payload);stage.replace(path)
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
        print(a['name'],a['id'],a['digest'],flush=True)


if __name__=='__main__':
    main()
