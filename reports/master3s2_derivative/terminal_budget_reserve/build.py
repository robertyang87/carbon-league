"""Set the initial menu size after reserving cost for paired terminal continuation."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
OPTIONAL=FAMILY/'native_terminal_extension/candidate.py'
OPTIONAL_SHA='ab39ab471c1de49abc790ec8a083ae5e7393cbbe201e2a10109e61b7ecc5ed6c'
V2=FAMILY/'ladder/prediction_feedback/package/candidate.py'
V2_SHA='8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    raw=OPTIONAL.read_bytes()
    assert sha(raw)==OPTIONAL_SHA
    old="        left = _S_FILL * (_S_ABORT - (_time.perf_counter() - t0))"
    new='''        left = _S_FILL * (_S_ABORT - (_time.perf_counter() - t0))
        if 287 <= step <= 290:
            terminal_steps = 291 - step
            per_pass = 4 * max(_S_COST[0], _S_COST[2]) + 2 * max(_S_COST[1], _S_COST[3]) + _S_CFIX
            left -= 0.010 + _S_SAFE * terminal_steps * per_pass'''
    src=raw.decode().replace('\r\n','\n')
    assert src.count(old)==1
    result=src.replace(old,new).replace('\n','\r\n').encode()
    dest=HERE/'candidate.py'
    compile(result,str(dest),'exec')
    if dest.exists():
        assert dest.read_bytes()==result
    else:
        dest.write_bytes(result)
    return sha(result)


if __name__=='__main__':
    print(build())
