"""Measure the added joint order with the original responding to official terminal."""
import argparse
import gzip
import importlib.util
import json

from build import FAMILY, HERE, PARENT_SHA, sha

spec=importlib.util.spec_from_file_location('coordinated_return_respond',FAMILY/'single_worker_priority/response/run.py')
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
    for row in found['rows']:
        parent = row['equal_work']['v4'][0]
        trial = row['equal_work']['critic'][0]
        if not row['same_work'] or parent['menu'] == trial['menu']:
            continue
        ticket = row['ticket']
        base_order, coordinated = parent['issued'], trial['menu'][1]
        assert coordinated not in parent['menu'] and coordinated != base_order
        cases.append(ticket | dict(
            orders={'parent': base_order, 'coordinated': coordinated},
            own_public_sha256=row['public_sha256'], history_source_package=baseline,
            initial_cost=row['cost'], rival_rng=ticket['rng'] ^ 0x7D4A1F6B,
            branch_order=['parent', 'coordinated'],
        ))
    return dict(schema='coordinated-return-native-reply/1', source_probe_sha256=sha(raw), parent=baseline,
                candidate=candidate, examiner=runner.ORIGINAL_SHA, tickets=cases,
                selection='Exploratory replay of every same-work root with a distinct menu. Compare the added joint order with the parent actual choice. Then both arms return to V2 and the complete original uses native search until official terminal.')



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
