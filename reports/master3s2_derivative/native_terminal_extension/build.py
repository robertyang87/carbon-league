"""Continue only two finalists beyond native H8 when both can reach settlement."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
PARENT = FAMILY/'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


EXTENSION = '''    # Reuse the completed H8 states and their own policy logits.  This is an
    # optional paired continuation; a partial run is never compared to H8.
    if not cut and depth == D2 and rolls == 1 and 287 <= step <= 290 and len(alive) >= 2 and n:
        order = sorted(alive, key=lambda c: -val[c, 0])
        if val[order[0], 0] - val[0, 0] <= _S_EPS and order[0] != 0:
            order.remove(0)
            order.insert(0, 0)
        left = LAST_TURN + 1 - step - D2
        pair = [b for b in run if b[0] in order[:2]]
        nlive = sum(not over(b) for b in pair)
        unit_net = max(_S_COST[0], _S_COST[2])
        unit_step = max(_S_COST[1], _S_COST[3])
        need = _S_SAFE * (left * _S_CFIX + left * (2 * unit_net + unit_step) * nlive)
        if nlive and _time.perf_counter() - t0 + need < _S_ABORT:
            # Stage-local dictionaries only; no partial value is merged.
            positions = {b[0]: i for i, b in enumerate(live)}
            forecast = {b[0]: pol[positions[b[0]]] for b in pair if not over(b)}
            prior_cost = _S_COST[:]
            extension_completed = True
            try:
                for j in range(left):
                    open_pair = [b for b in pair if not over(b)]
                    if not open_pair:
                        break
                    if late((1 if j == 0 else 2) * len(open_pair), len(open_pair)):
                        extension_completed = False
                        break
                    if j == 0:
                        directions = other(open_pair, [forecast[b[0]] for b in open_pair])
                    else:
                        _, directions = look(open_pair, True)
                    play(open_pair, directions, D2 + j)
            finally:
                _S_COST[:] = prior_cost
            if extension_completed and all(over(b) for b in pair):
                values = {b[0]: result(b) for b in pair}
                for c in order[:2]:
                    val[c, 0] = values[c]
                    dep[c] = D2 + left
                _S_STATS['terminal_extension'] = dict(length=left, finalists=order[:2], values=values)
                alive = order[:2]
                depth = D2 + left
'''


def build():
    raw=PARENT.read_bytes()
    assert sha(raw)==PARENT_SHA
    index=raw.index(b'def _search(')
    first, data=raw[:index],raw[index:].decode().replace('\r\n','\n')
    old='''    def over(b):
        return b[2]['step'] > LAST_TURN

    def result(b):
        c = [float(pl[0]) for pl in b[2]['players']]
        return _terminal(c[me] - c[op])'''
    new='''    settling = 287 <= step <= 290
    recruit_floor = min(configuration.get('recCollectorCost', 30),
                        configuration.get('recPlanterCost', 30))

    def extinct(pl):
        return not pl[2] and not pl[3] and (not pl[1] or pl[0] < recruit_floor)

    def over(b):
        state = b[2]
        return state['step'] > LAST_TURN or settling and any(extinct(pl) for pl in state['players'])

    def result(b):
        state = b[2]
        teams = state['players']
        gone = [extinct(pl) for pl in teams] if settling else [False, False]
        if any(gone):
            # The survivor keeps the score at the start of this transition.
            c = [state['step'] - configuration.get('episodeSteps', 300) - 1
                 if gone[s] else b[3][s][0] for s in (0, 1)]
        else:
            c = [float(pl[0]) for pl in teams]
        return _terminal(c[me] - c[op])'''
    assert data.count(old)==1
    data=data.replace(old,new)
    marker='    dt = _time.perf_counter() - t0\n'
    assert data.count(marker)==1
    # Keep the native policy's tiebreak; the alternative unfinished H8 values
    # cannot come back into the final ranking after a two-candidate extension.
    final='''    if best == 0 or mean[best] - mean[0] <= _S_EPS:
        best = 0'''
    assert data.count(final)==1
    data=data.replace(final,'''    if 0 in mean and (best == 0 or mean[best] - mean[0] <= _S_EPS):
        best = 0''')
    data=data.replace(marker,EXTENSION+marker)
    result=first+data.replace('\n','\r\n').encode()
    target=HERE/'candidate.py'
    compile(result,str(target),'exec')
    if target.exists():
        assert target.read_bytes()==result
    else:
        target.write_bytes(result)
    return sha(result)


if __name__=='__main__':
    print(build())
