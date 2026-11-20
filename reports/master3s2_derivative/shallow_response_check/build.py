"""Build a finalist sensitivity check in the existing V2 search."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
PARENT = FAMILY / 'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build():
    raw = PARENT.read_bytes()
    assert sha(raw) == PARENT_SHA
    start, stop = raw.index(b'def _search('), raw.index(b'\r\n_net()', raw.index(b'def _search('))
    block = raw[start:stop].decode().replace('\r\n', '\n')
    edits = (
        ('    dep = {}', '    first_values = {}\n    _S_STATS.pop("response_check", None)\n    dep = {}'),
        ('        depth = t\n        if t == D2 or not n:',
         '        if t == 1:\n            first_values = val.copy()\n        depth = t\n        if t == D2 or not n:'),
        ('    if best == 0 or mean[best] - mean[0] <= _S_EPS:\n        best = 0',
         '''    if best == 0 or mean[best] - mean[0] <= _S_EPS:
        best = 0
    # Spend one short pass on the finalist only when the original work finished
    # and the live cost model leaves room. Preserve exact terminal values.
    if (best and not cut and rolls == 1 and step + depth <= LAST_TURN
            and not late(2, 2) and _time.perf_counter() - t0 < _S_ABORT - 0.030):
        alternate = _sample(players, op, lwo[0], lbo[0], np.random.RandomState(seed0 + 38), step)
        if alternate != opp[0]:
            boards2 = []
            for index in (0, best):
                acts = [cands[index], alternate] if me == 0 else [alternate, cands[index]]
                boards2.append([index, 1, _turn(root, configuration, acts), players])
            original_cost = _S_COST[:]
            try:
                short, _ = look(boards2, False)
            finally:
                _S_COST[:] = original_cost
            h1_primary = first_values[best, 0] - first_values[0, 0]
            h1_alternate = short[1] - short[0]
            deep_diff = mean[best] - mean[0]
            corrected = deep_diff + 0.5 * (h1_alternate - h1_primary)
            veto = bool(np.isfinite(corrected) and h1_alternate < 0 and corrected < 0)
            _S_STATS['response_check'] = dict(primary=h1_primary, alternate=h1_alternate,
                                               depth_gain=deep_diff, corrected=corrected, veto=veto)
            if veto:
                best = 0'''),
    )
    for before, after in edits:
        assert block.count(before) == 1
        block = block.replace(before, after)
    result = raw[:start] + block.replace('\n', '\r\n').encode() + raw[stop:]
    dst = HERE / 'candidate.py'
    if dst.exists():
        assert dst.read_bytes() == result
    else:
        compile(result, str(dst), 'exec')
        dst.write_bytes(result)
    return sha(result)


if __name__ == '__main__':
    print(build())
