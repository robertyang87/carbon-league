"""Test a opponent rollout calibration on signed historical public roots."""
import argparse
import importlib.util
import json
import zipfile

from build import CALIBRATION, HERE, PARENT, PARENT_SHA, build, sha
FAMILY = HERE.parent

spec=importlib.util.spec_from_file_location('temperature_v2_private_entry',FAMILY/'native_critic_update/public_search/check.py')
check=importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
STEPS=(245,285)


def protocol():
    tickets=[]
    for path in sorted((FAMILY/'native_speed_dev/hosted').glob('*.zip')):
        raw=path.read_bytes()
        with zipfile.ZipFile(path) as z:
            reg=z.read('parent/registration.json')
        info=json.loads(reg)
        seed,seat=info['seed'],info['candidate_seat']
        assert seed in range(621782460,621782464)
        assert [info['files'][k]['sha256'] for k in ('candidate','opponent')]==[PARENT_SHA,check.ORIGINAL]
        for step in STEPS:
            tickets.append(dict(seed=seed,seat=seat,terrain=info['physical_map_sha256'],step=step,arm='parent',
                                archive=str(path.relative_to(FAMILY)),archive_sha256=sha(raw),registration_sha256=sha(reg),
                                rng=int(sha(f'rollout-temperature-v2/1/{seed}/{seat}/{step}'.encode())[:8],16)))
    assert len(tickets)==16
    return dict(schema='rollout-temperature-v2-search/1',packages=check.DIGESTS,tickets=tickets,
                design='Same four existing terrain IDs and signed V2 public histories. t245/t285, both seats and two native repeats '
                       'with an equal-work follow-up repeat. Only enemy simulated action logits after the root are scaled.')


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--cpu',type=int)
    cli.add_argument('--register',action='store_true')
    a=cli.parse_args()
    check.PARENT,check.CANDIDATE=PARENT,HERE/'candidate.py'
    raw = build()
    assert (HERE / "candidate.py").read_bytes() == raw
    check.DIGESTS,check.STEPS,check.registration=(PARENT_SHA,sha(raw)),STEPS,protocol
    check.HERE=HERE/'search'
    check.HERE.mkdir(exist_ok=True)
    f=check.HERE/'protocol.json'
    if a.register:
        raw=check.encode(protocol())
        assert not f.exists() or f.read_bytes()==raw
        f.write_bytes(raw)
        print(sha(raw))
    else:
        check.execute(a.cpu)
