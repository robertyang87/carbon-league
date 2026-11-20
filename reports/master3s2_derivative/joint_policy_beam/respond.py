"""Test action differences that specifically use a new joint beam order."""
import argparse
import gzip
import importlib.util
import json

from build import FAMILY, HERE, PARENT_SHA, sha

spec=importlib.util.spec_from_file_location('joint_policy_beam_respond',FAMILY/'single_worker_priority/response/run.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
OUT=HERE/'response'


def plan():
    raw=(HERE/'search/results.json.gz').read_bytes()
    found=json.loads(gzip.decompress(raw))
    assert found['protocol_sha256']==sha((HERE/'search/protocol.json').read_bytes())
    baseline,candidate=json.loads((HERE/'search/protocol.json').read_bytes())['packages']
    assert baseline==PARENT_SHA
    cases = []
    ordered = sorted(found['rows'], key=lambda r: (r['ticket']['seed'], r['ticket']['seat']))
    for row in ordered:
        ticket = row['ticket']
        if ticket['step'] != 246 or not row['reliable_change']:
            continue
        accepted = [row['live'][k] for k in ('v4', 'critic')]
        if any(min(q['depth'] for q in group) < 4 for group in accepted):
            continue
        if len({json.dumps(q['opponent_samples'], sort_keys=True) for group in accepted for q in group}) != 1:
            continue
        if any(t['seed'] == ticket['seed'] for t in cases):
            continue
        parent, trial = [group[0]['issued'] for group in accepted]
        assert parent != trial
        if trial in accepted[0][0]['menu'] or trial not in accepted[1][0]['menu'][-3:]:
            continue
        cases.append(ticket | dict(
            orders={'parent': parent, 'beam': trial},
            own_public_sha256=row['public_sha256'], history_source_package=baseline,
            initial_cost=row['cost'], rival_rng=ticket['rng'] ^ 0x7D4A1F6B,
            branch_order=['parent','beam'] if len(cases)%2==0 else ['beam','parent'],
        ))
        if len(cases) == 2:
            break
    return dict(schema='joint-policy-beam-native-reply/1', source_probe_sha256=sha(raw), parent=baseline,
                candidate=candidate, examiner=runner.ORIGINAL_SHA, tickets=cases,
                selection='First two physical maps at t246 where a new beam order was chosen stably under native and matching-work calls. At least H4 and the same simulated foe root response. One forced candidate action, then V2 and the complete original decide until terminal.')



if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    cli.add_argument('--seed',type=int)
    a=cli.parse_args()
    cases=plan()
    raw=runner.check.encode(cases)
    OUT.mkdir(exist_ok=True)
    path=OUT/'protocol.json'
    runner.HERE=OUT
    if a.register:
        assert not path.exists() or path.read_bytes()==raw
        path.write_bytes(raw)
        print(sha(raw),len(cases['tickets']))
    else:
        assert path.read_bytes()==raw and cases['tickets']
        if a.seed is not None:
            cases = cases | {'tickets':[t for t in cases['tickets'] if t['seed']==a.seed]}
            assert len(cases['tickets'])==1
            runner.HERE = OUT/str(a.seed)
            runner.HERE.mkdir(exist_ok=True)
            shard = runner.HERE/'protocol.json'
            assert not shard.exists() or shard.read_bytes()==runner.check.encode(cases)
            shard.write_bytes(runner.check.encode(cases))
        runner.execute(cases,a.cpu)
