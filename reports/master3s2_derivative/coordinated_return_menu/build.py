"""Build a v2 candidate offering one additional coordinated return order."""
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
    source = raw.decode().replace('\r\n', '\n')
    insert = source.index('def _search(')
    helper = (HERE / 'menu.py').read_text()
    source = source[:insert] + helper + '\n\n' + source[insert:]
    marker = '    for T, cap in _S_GEN:\n'
    assert source.count(marker) == 1
    source = source.replace(marker, '''    coordinated = _coordinated_return(players, me, lw0, act0, fixed)
    if coordinated is not None and len(cands) < most:
        cands.append(coordinated)
        keys.add(tuple(sorted(coordinated.items())))
''' + marker)
    result = source.replace('\n', '\r\n').encode()
    dest = HERE / 'candidate.py'
    compile(result, str(dest), 'exec')
    dest.write_bytes(result)
    return sha(result)


if __name__ == '__main__':
    print(build())
