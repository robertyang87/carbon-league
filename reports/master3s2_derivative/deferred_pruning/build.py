"""Build a v2 candidate whose first elimination waits for four simulated steps."""
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
    marker = b"_S_STAGES = ((1, 0.5), (4, 0.25))"
    revised = b"_S_STAGES = ((4, 0.25),)"
    assert raw.count(marker) == 1 and revised not in raw
    result = raw.replace(marker, revised)
    dest = HERE / 'candidate.py'
    compile(result, str(dest), 'exec')
    if dest.exists():
        assert dest.read_bytes() == result
    else:
        dest.write_bytes(result)
    return sha(result)


if __name__ == '__main__':
    print(build())
