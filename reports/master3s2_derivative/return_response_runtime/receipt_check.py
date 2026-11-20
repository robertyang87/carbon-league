"""Replay real captured native decision inputs through the exact export selector and neural scorer."""
import time
import numpy as np
import build
import probe


def main():
    parent = probe.load('bound_parent_v2', probe.PARENT)
    parent._torch.set_num_threads(1)
    training = probe.load('fit_geometry', build.LEARN/'model.py')
    weights = build.read(build.LEARN/'weights.json')
    namespace = dict(np=np, _OPP_STEPS=parent._OPP_STEPS, FEAT_DIST=parent.FEAT_DIST, _S_ABORT=parent._S_ABORT)
    exec(compile(build.export_model(), 'model_with_native_tables', 'exec'), namespace)
    exec(compile((probe.HERE/'selector.py').read_bytes(), 'bound_selector', 'exec'), namespace)
    for name in ('_RR_MOVE','_RR_DIST'):
        assert np.array_equal(getattr(training,name),namespace[name])
    plan = build.read(build.LEARN/'protocol.json')
    rows = []
    times = []
    for held in build.read(build.LEARN/'held.json')['pairs']:
        card = next(t for t in plan['pairs'] if t['split']=='held' and t['seed']==held['seed'])
        seed, side, step = (card[k] for k in ('seed','seat','step'))
        path = probe.D/'action_rank_data'/'held'/f'{seed}_{side}_{step}_root.json.gz'
        root = probe.read(path)
        vals = {int(i):v for i,v in root['all_final_values'].items()}
        arms = root['arms']
        indices = [a['candidate_index'] for a in arms]
        original = root['menu'].index(root['issued'])
        assert original == indices[0]
        players = root['public_root']['players']
        commands = [a['order'] for a in arms]
        # Apart from aliasing the original v2 lookup tables the exported evaluator is the fitted code.
        trained_scores = training.rr_scores(players,side,step,commands,weights)
        def score(players_, side_, step_, orders):
            start = time.perf_counter()
            result = namespace['rr_scores'](players_,side_,step_,orders,weights)
            times.append((time.perf_counter()-start)*1000.)
            np.testing.assert_array_equal(result, trained_scores)
            return result
        boards = []
        for a in arms:
            leaf, = a['leaves']
            active = not leaf['exact_terminal'] and namespace['_rr_in_play'](leaf['players'],30)
            # Extinction is absorbing under registered costs (no workers/trees and cannot recruit).
            boards.append([a['candidate_index'], leaf['roll_index'], {'step':leaf['leaf_step']}, None, active])
        # Use archived elapsed time without claiming a new live deadline or sampled action.
        corrected, reason, detail = namespace['_rr_select'](original, list(vals), vals, {i:root['depth'] for i in vals}, boards,
                root['menu'], step, side, players, root['depth'], root['rolls'], False, 0.,
                lambda:root['clock']['wall'], score, weights['pair_scale'], parent._S_EPS)
        delta = held['margins'][indices.index(corrected)] - held['margins'][0]
        rows.append(dict(seed=seed,seat=side,step=step,archive_sha=probe.sha(path.read_bytes()),
                         native=original,matched_pair=indices,corrected=corrected,reason=reason,detail=detail,
                         elapsed_at_archived_public_return=root['clock']['wall'],both_native_arms_won=all(held['wins']),
                         observed_difference_of_archived_terminal_margins=delta))
    assert sum(r['corrected']!=r['native'] for r in rows)==1
    assert rows[0]['seed']==621758323 and rows[0]['corrected']==5
    extra_pairs = 0
    for row in probe.read(probe.HERE/'native_roots.json.gz')['rows']:
        rec=row['record']
        if not rec['eligible']:continue
        extra_pairs += 1
        commands=[a['command'] for a in rec['arms']]
        args=(row['players'],row['seat'],row['t'],commands,weights)
        np.testing.assert_array_equal(namespace['rr_scores'](*args),training.rr_scores(*args))
    report=dict(weight_sha256=probe.sha((build.LEARN/'weights.json').read_bytes()),parent_sha256=probe.EXPECTED,
                move_table_entries=int(parent._OPP_STEPS.size),distance_table_entries=int(parent.FEAT_DIST.size),
                exported_pair_numerical_parity_count=extra_pairs+len(rows),export_inference_observed_ms=times,
                elapsed_uses_archive_clock=True,source='action_rank_data held finalists; historic 7cf5, source of paired returns',
                result=rows)
    probe.save(probe.HERE/'historical_selector.json', report)
    print([(r['seed'],r['native'],r['corrected'],r['reason']) for r in rows])


if __name__=='__main__':
    main()
