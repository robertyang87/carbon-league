"""Build a v2 candidate offering one additional one-base recruitment alternative."""
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
    marker = '    if len(cands) < 2:\n'
    assert source.count(marker) == 1
    source = source.replace(marker, """    if step <= 80:
        alternative = _recruitment_alternative(players, me, lb0, act0, keys)
        if alternative is not None:
            if len(cands) == most:
                cands[-1] = alternative
            else:
                cands.append(alternative)
""" + marker)
    result = source.replace('\n', '\r\n').encode()
    dest = HERE / 'candidate.py'
    compile(result, str(dest), 'exec')
    dest.write_bytes(result)
    return sha(result)


if __name__ == '__main__':
    print(build())
