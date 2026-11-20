"""Use up to eight shared opponent packets for exact final-turn settlement."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
PARENT=FAMILY/'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA='8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


REPLAY = '''    if step == LAST_TURN and not cut and depth == 1 and M == rolls == 1:
        # Each additional enemy packet is shared across the entire menu. Do
        # not merge a partial set of candidate values after the deadline.
        for sample in range(1, 8):
            cost_hint = max(.002, len(alive) * 1.3 * max(_S_COST[1], _S_COST[3]))
            if _time.perf_counter() - t0 + cost_hint >= _S_ABORT:
                break
            states = begin(alive, (sample,))
            if _time.perf_counter() - t0 >= _S_ABORT:
                break
            assert len(states) == len(alive) and all(over(b) for b in states)
            for b in states:
                val[b[0], sample] = result(b)
            rolls += 1
'''


def build():
    raw=PARENT.read_bytes()
    assert sha(raw)==PARENT_SHA
    first=raw.index(b'def _search(')
    code=raw[first:].decode().replace('\r\n','\n')
    old='''    def result(b):
        c = [float(pl[0]) for pl in b[2]['players']]
        return _terminal(c[me] - c[op])'''
    new='''    def result(b):
        state = b[2]
        c = [float(pl[0]) for pl in state['players']]
        if step == LAST_TURN:
            floor = min(configuration.get('recCollectorCost', 30), configuration.get('recPlanterCost', 30))
            gone = [not pl[2] and not pl[3] and (not pl[1] or pl[0] < floor) for pl in state['players']]
            if any(gone):
                c = [state['step'] - configuration.get('episodeSteps', 300) - 1
                     if gone[s] else b[3][s][0] for s in (0, 1)]
        return _terminal(c[me] - c[op])'''
    marker='    dt = _time.perf_counter() - t0\n'
    assert code.count(old)==code.count(marker)==1
    edits=code.replace(old,new).replace(marker,REPLAY+marker)
    result=raw[:first]+edits.replace('\n','\r\n').encode()
    f=HERE/'candidate.py'
    compile(result,str(f),'exec')
    if f.exists():
        assert f.read_bytes()==result
    else:
        f.write_bytes(result)
    return sha(result)


if __name__=='__main__':
    print(build())
