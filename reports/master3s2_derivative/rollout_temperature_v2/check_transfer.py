"""Score the frozen tempering on archived V2 matches, without refitting."""
import json
import sys

from build import CALIBRATION, HERE, PARENT, PARENT_SHA, WEIGHT_SHA

sys.path.insert(0, str(CALIBRATION))
import study


def main():
    family = HERE.parent
    host = family / 'active_embedding_dev/hosted'
    results = host / 'settlement.json'
    settlement = json.loads(results.read_bytes())
    assert settlement['records_reconciled'] and settlement['formal_exam_games'] == 0
    archive = {(item['seed'], item['seat']): item for item in settlement['archives']}
    tickets = []
    for game in settlement['games']:
        if game['arm'] != 'parent':
            continue
        source = archive[game['seed'], game['seat']]
        tickets.append(dict(seed=game['seed'], seat=game['seat'], arm='parent', map=game['terrain'],
                            rel=str((host / f"{source['id']}.zip").relative_to(family)),
                            archive_sha256=source['sha256']))
    assert len(tickets) == 8 and len({c['map'] for c in tickets}) == 4
    original_split = json.loads((CALIBRATION / 'protocol.json').read_bytes())['tickets']
    assert not {c['map'] for c in tickets} & {c['map'] for c in original_split}
    raw = PARENT.read_bytes()
    assert study.sha(raw) == PARENT_SHA
    raw_weights = (CALIBRATION / 'weights.json').read_bytes()
    assert study.sha(raw_weights) == WEIGHT_SHA
    frozen = json.loads(raw_weights)['betas']
    study.PARENT = PARENT_SHA
    import types
    agent = types.ModuleType('frozen_v2_rollout')
    exec(compile(raw, str(PARENT), 'exec'), agent.__dict__)
    assert agent._SEARCH_OK and agent._TEMP == 1 and agent._S_ABORT == .58
    agent._torch.set_num_threads(1)
    roots, skipped = study.collect(agent, tickets)
    scores = study.evaluate(agent, roots, {
        'parent': dict.fromkeys(study.ROLES, 1.0),
        'shared': dict.fromkeys(study.ROLES, frozen['shared']),
        'roles': frozen['roles'],
    })
    by_map, summary = study.aggregate(scores)
    output = dict(schema='frozen-opponent-calibration-transfer/1',
                  settlement_sha256=study.sha(results.read_bytes()), weights_sha256=WEIGHT_SHA,
                  parent_sha256=PARENT_SHA, scorer_sha256=study.sha((CALIBRATION/'study.py').read_bytes()),
                  roots=len(scores), skipped=skipped, draws=study.DRAWS, tickets=tickets,
                  by_map=by_map, summary=summary, rows=scores)
    study.put(HERE/'transfer_check.json.gz', output)
    print(len(scores), len(skipped), {k: (v['group']['all'],v['joint_brier'],v['exact']) for k,v in summary.items()})


if __name__ == '__main__':
    main()
