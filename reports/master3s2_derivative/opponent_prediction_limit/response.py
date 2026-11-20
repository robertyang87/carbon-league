"""Compare changed commands under a fresh shared response from the unmodified original."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTRY = HERE.parent / 'single_worker_priority/response/run.py'
spec = importlib.util.spec_from_file_location('original_response_runner', ENTRY)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.HERE = HERE


def plan():
    raw = (HERE / 'roots.json.gz').read_bytes()
    roots = json.loads(gzip.decompress(raw))
    assert roots['parent'] == runner.PARENT_SHA
    tickets = []
    selected_maps = set()
    ordered = sorted(roots['rows'], key=lambda row: (row['ticket']['seed'], row['ticket']['seat']))
    for row in ordered:
        ticket = row['ticket']
        seed = ticket['seed']
        if not row['changed'] or ticket['step'] != 247 or seed in selected_maps:
            continue
        parent = row['reference']['issued']
        alternate = row['answers']['actual'][0]['issued']
        assert parent != alternate
        branch_order = ['parent', 'alternate'] if not tickets else ['alternate', 'parent']
        tickets.append(ticket | dict(orders={'parent': parent, 'alternate': alternate},
                                     history_source_package=roots['parent'], own_public_sha256=row['public_sha256'],
                                     initial_cost=row['cost'], rival_rng=ticket['rng'] ^ 0x7D4A1F6B,
                                     branch_order=branch_order))
        selected_maps.add(seed)
        if len(tickets) == 2:
            break
    return dict(schema='actual-first-opponent-response/1', source_probe_sha256=runner.check.sha(raw),
                parent=roots['parent'], candidate=runner.check.sha((HERE / 'probe.py').read_bytes()),
                examiner=runner.ORIGINAL_SHA,
                design='The first two distinct boards at t247 where the retrospective command changed our issued action '
                       'under matching native work. Candidate identity is the diagnostic source hash, not a bot. '
                       'The original samples a fresh shared root action, then both sides continue naturally.',
                tickets=tickets)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--register', action='store_true')
    cli.add_argument('--cpu', type=int)
    args = cli.parse_args()
    p = plan()
    raw = runner.check.encode(p)
    destination = HERE / 'responses_protocol.json'
    if args.register:
        assert not destination.exists() or destination.read_bytes() == raw
        destination.write_bytes(raw)
        print(runner.check.sha(raw), [(c['seed'], c['seat']) for c in p['tickets']])
    else:
        assert destination.read_bytes() == raw and p['tickets']
        runner.execute(p, args.cpu)
