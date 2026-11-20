"""Run the return-route correction through the V2 native public entry."""
import argparse
import importlib.util
import json
import zipfile

from build import FAMILY, HERE, PARENT, PARENT_SHA, build, sha

spec=importlib.util.spec_from_file_location('native_deposit_root_check',FAMILY/'native_critic_update/public_search/check.py')
check=importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
STEPS=(145,238)


def registration():
    tickets=[]
    sel={621782461,621782463,621782471,621782473}
    for panel in ('native_speed_dev','switch_margin_dev'):
        for path in sorted((FAMILY/panel/'hosted').glob('*.zip')):
            with zipfile.ZipFile(path) as z:
                reg=z.read('parent/registration.json')
            data=json.loads(reg)
            seed,seat=data['seed'],data['candidate_seat']
            if seed not in sel:
                continue
            assert [data['files'][k]['sha256'] for k in ('candidate','opponent')]==[PARENT_SHA,check.ORIGINAL]
            for step in STEPS:
                rng=int(sha(f'deposit-leaf/1/{seed}/{seat}/{step}'.encode())[:8],16)
                tickets.append(dict(archive=str(path.relative_to(FAMILY)),archive_sha256=sha(path.read_bytes()),
                                    registration_sha256=sha(reg),arm='parent',seed=seed,seat=seat,step=step,rng=rng,
                                    terrain=data['physical_map_sha256']))
    assert len(tickets)==16
    return dict(schema='bilateral-deposit-leaf/1', packages=check.DIGESTS,
                selection='Every second held board across two panels, both seats, fixed t145 and t238.',
                target='Compare native H8 ranking plus half the difference of route-aware versus total-load forecast.',
                tickets=tickets)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    a=cli.parse_args()
    check.PARENT,check.CANDIDATE= PARENT,HERE/'candidate.py'
    check.DIGESTS,check.STEPS,check.registration=(PARENT_SHA,build()),STEPS,registration
    check.HERE=HERE/'search'
    check.HERE.mkdir(exist_ok=True)
    dst=check.HERE/'protocol.json'
    if a.register:
        raw=check.encode(registration())
        assert not dst.exists() or dst.read_bytes()==raw
        dst.write_bytes(raw)
        print(sha(raw))
    else:
        check.execute(a.cpu)
