"""Rebuild a one-slot V2 menu intervention from the frozen income head."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
PARENT = FAMILY/'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    raw = PARENT.read_bytes()
    assert sha(raw) == PARENT_SHA
    source = FAMILY/'live_tree_income'
    feature = (source/'features.py').read_text()
    feature = feature[feature.index('HORIZON = 16'):].replace('HORIZON', '_RENT_HORIZON')
    feature = feature.replace('def extract(bot,', 'def _existing_tree_features(').replace('bot.', '')
    weights = (source/'weights.json').read_bytes()
    assert json.loads((source/'held.json').read_bytes())['weights_sha256'] == sha(weights)
    extra = (HERE/'runtime.py').read_text().split('\n\n\n',1)[1]
    text = feature + f'\n_RENT_FIT = {json.loads(weights)!r}\n\n' + extra
    cut = raw.index(b'def _search(')
    before, block = raw[:cut], raw[cut:]
    old = b"    if len(cands) < 2:\r\n        _S_STATS['single']"
    new = '''    _S_STATS.pop('rent_site', None)
    if len(cands) >= 2 and 20 <= step <= 92:
        # Preserve the original width and all candidates except the final alternate.
        proposal = _tree_sites(players, step, est, _ST['tr'].bel.known, me, act0, configuration)
        if proposal is not None and proposal not in cands:
            cands[-1] = proposal
            _S_STATS['rent_site'] = proposal
    if len(cands) < 2:
        _S_STATS['single']'''.replace('\n','\r\n').encode()
    assert block.count(old) == 1
    result = before + text.replace('\n','\r\n').encode() + b'\r\n\r\n' + block.replace(old,new)
    target = HERE/'candidate.py'
    compile(result, str(target), 'exec')
    if target.exists():
        assert target.read_bytes() == result
    else:
        target.write_bytes(result)
    return sha(result)


if __name__ == '__main__':
    print(build())
