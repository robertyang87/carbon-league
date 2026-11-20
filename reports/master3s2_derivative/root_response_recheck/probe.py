"""Test a second first-step recheck on signed historical public roots."""
import argparse
import sys
import importlib.util
import json
import zipfile

from build import FAMILY, HERE, PARENT, PARENT_SHA, build, sha

spec=importlib.util.spec_from_file_location('first_response_private_entry',FAMILY/'native_critic_update/public_search/check.py')
check=importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
STEPS=(254,278)


def protocol():
    tickets=[]
    for path in sorted((FAMILY/'rollout_temperature_v2/development/hosted').glob('*.zip')):
        raw=path.read_bytes()
        with zipfile.ZipFile(path) as z:
            reg=z.read('parent/registration.json')
        info=json.loads(reg)
        seed,seat=info['seed'],info['candidate_seat']
        assert seed in range(621782492,621782496)
        assert [info['files'][k]['sha256'] for k in ('candidate','opponent')]==[PARENT_SHA,check.ORIGINAL]
        for step in STEPS:
            tickets.append(dict(seed=seed,seat=seat,terrain=info['physical_map_sha256'],step=step,arm='parent',
                                archive=str(path.relative_to(FAMILY)),archive_sha256=sha(raw),registration_sha256=sha(reg),
                                rng=int(sha(f'root-response-recheck/1/{seed}/{seat}/{step}'.encode())[:8],16)))
    assert len(tickets)==16
    return dict(schema='root-response-recheck-search/1',packages=check.DIGESTS,tickets=tickets,
                design='Same four existing terrain IDs and signed V2 public histories. t254/t278, both seats and two native repeats '
                       'with an equal-work follow-up repeat. A half-weight alternative-response H1 gap correction can veto a native H8 switch to a search move.')


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--cpu',type=int)
    cli.add_argument('--register',action='store_true')
    a=cli.parse_args()
    check.PARENT,check.CANDIDATE=PARENT,HERE/'candidate.py'
    check.DIGESTS,check.STEPS,check.registration=(PARENT_SHA,build()),STEPS,protocol
    check.HERE=HERE/'search'
    check.HERE.mkdir(exist_ok=True)
    f=check.HERE/'protocol.json'
    if a.register:
        raw=check.encode(protocol())
        assert not f.exists() or f.read_bytes()==raw
        f.write_bytes(raw)
        print(sha(raw))
    else:
        helper = FAMILY/'opponent_policy/search_probe/full_search_check.py'
        reader = importlib.util.spec_from_file_location('full_search_check',helper)
        base = importlib.util.module_from_spec(reader)
        source = helper.read_text()
        assert source.count('len(enemy_samples) <= 1') == 1
        exec(compile(source.replace('len(enemy_samples) <= 1','len(enemy_samples) <= 2'),str(helper),'exec'),base.__dict__)
        original = base.root_call

        def record(mod,*args,**kwargs):
            mod._S_STATS.pop('response_recheck',None)
            result = original(mod,*args,**kwargs)
            result['recheck'] = mod._S_STATS.get('response_recheck')
            result['secondary_samples'] = result['opponent_samples'][1:]
            result['opponent_samples'] = result['opponent_samples'][:1]
            return result

        base.root_call = record
        sys.modules['full_search_check'] = base
        check.execute(a.cpu)
