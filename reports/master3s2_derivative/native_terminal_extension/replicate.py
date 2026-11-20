"""Repeat the live-clock test on four different archived physical maps."""
import argparse
import importlib.util
import json
import zipfile

from build import FAMILY, HERE, PARENT_SHA, build, sha

spec=importlib.util.spec_from_file_location('terminal_extension_root_repeat',HERE/'probe.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OUT=HERE/'replication'


def registration():
    before=json.loads((HERE/'protocol.json').read_bytes())
    old={r['terrain'] for r in before['tickets']}
    source=FAMILY/'switch_margin_dev/hosted'
    tickets=[]
    for file in sorted(source.glob('*.zip')):
        payload=file.read_bytes()
        with zipfile.ZipFile(file) as z:
            reg=z.read('parent/registration.json')
        info=json.loads(reg)
        seed,seat=info['seed'],info['candidate_seat']
        terrain=info['physical_map_sha256']
        assert seed in range(621782470,621782474) and terrain not in old
        assert [info['files'][k]['sha256'] for k in ('candidate','opponent')]==[PARENT_SHA,base.OPPONENT]
        for step in base.STEPS:
            tickets.append(dict(seed=seed,seat=seat,terrain=terrain,step=step,
                                archive=str(file.relative_to(FAMILY)),registration_sha256=sha(reg),
                                archive_sha256=sha(payload),arm='parent',
                                rng=int(sha(f'terminal-extension-repeat/1/{seed}/{seat}/{step}'.encode())[:8],16)))
    assert len(tickets)==16 and len({r['terrain'] for r in tickets})==4
    return dict(schema='native-terminal-extension-replication/1',
                packages=[PARENT_SHA,build()],original=base.OPPONENT,tickets=tickets,
                first_panel_sha256=sha((HERE/'protocol.json').read_bytes()),
                plan=before['plan'])


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    args=cli.parse_args()
    plan=registration()
    raw=base.encode(plan)
    OUT.mkdir(exist_ok=True)
    path=OUT/'protocol.json'
    if args.register:
        assert not path.exists() or path.read_bytes()==raw
        path.write_bytes(raw)
        print(sha(raw))
    else:
        assert path.read_bytes()==raw
        base.run(args.cpu,plan,OUT)
