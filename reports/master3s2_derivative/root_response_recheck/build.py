"""Build a bounded first-response recheck for the final two original-derived V2 choices."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
PARENT=FAMILY/'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA='8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    raw = PARENT.read_bytes()
    assert sha(raw) == PARENT_SHA
    text = raw.decode().replace('\r\n', '\n')

    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    mark = 'def _search('
    replace(mark, (HERE/'runtime.py').read_text() + '\n\n' + mark)
    mark = '    boards = begin(alive, range(M))'
    replace(mark, '    first_values = {}\n' + mark)
    mark = '        depth = t\n        if t == D2 or not n:'
    replace(mark, '        if t == 1:\n            first_values = dict(val)\n' + mark)
    mark = '    if _S_REC is not None:\n        _S_REC.append'
    replace(mark, """    _S_STATS['response_recheck'] = ('outside',)
    if best and depth == D2 and rolls == 1 and 250 <= step <= 289 and not late(2, 2):
        original_advantage = mean[best] - mean[0]
        alternative = _sample(players, op, lwo[0], lbo[0], _AntitheticResponse(seed0 + 7), step)
        if alternative == opp[0]:
            _S_STATS['response_recheck'] = ('same_enemy_order',)
        else:
            pairs = []
            for c in (0, best):
                acts = [cands[c], alternative] if me == 0 else [alternative, cands[c]]
                pairs.append([c, 0, _turn(root, configuration, acts), players])
            if not late(2, 0):
                prior_cost = _S_COST[:]
                new_value, _ = look(pairs, False)
                _S_COST[:] = prior_cost
                elapsed = _time.perf_counter() - t0
                _S_STATS['last'] = (len(cands), depth, rolls, elapsed, tuple(_S_COST))
                _S_STATS['time'][-1] = elapsed
                base_gap = first_values[best, 0] - first_values[0, 0]
                added_gap = new_value[1] - new_value[0]
                revised = original_advantage + 0.5 * (added_gap - base_gap)
                applied = elapsed <= _S_ABORT and revised <= _S_EPS
                _S_STATS['response_recheck'] = ('evaluated', best, original_advantage, base_gap, added_gap,
                                                 revised, applied, elapsed, alternative)
                if applied:
                    best = 0
""" + mark)
    result = text.replace('\n', '\r\n').encode()
    dest = HERE/'candidate.py'
    compile(result, str(dest), 'exec')
    if dest.exists():
        assert dest.read_bytes() == result
    else:
        dest.write_bytes(result)
    return sha(result)


if __name__ == '__main__':
    print(build())
