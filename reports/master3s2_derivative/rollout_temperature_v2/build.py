"""Embed the frozen opponent rollout temperatures at the enemy continuation decoder only."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
CALIBRATION = HERE.parent / "opponent_rollout_calibration"
PARENT=HERE.parent/"ladder/prediction_feedback/package/candidate.py"
PARENT_SHA="8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43"
WEIGHT_SHA="84a50f0f6d6c4cd3651877680ef7240463b752daa5bdde598185fd8cec0fc0a7"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def build():
    parent=PARENT.read_bytes();w=(CALIBRATION/'weights.json').read_bytes()
    assert sha(parent)==PARENT_SHA and sha(w)==WEIGHT_SHA
    b=json.loads(w)['betas']['roles']
    code=(CALIBRATION/'runtime.py').read_text()
    code=code[code.index('def adjust('):].replace('def adjust(','def _rollout_opp_adjust(')
    block=("_ROLLOUT_OPP_BETA = "+repr(b)+"\n\n"+code+"\n\n").replace('\n','\r\n').encode()
    header=b"def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):"
    original=b"            ao = _sample(o['players'], op, lwp, lbp, r2, o['step'])\r\n"
    call=b"            lwp, lbp = _rollout_opp_adjust(o['players'], op, lwp, lbp, _ROLLOUT_OPP_BETA)\r\n"
    assert parent.count(header)==1 and parent.count(original)==1
    candidate=parent.replace(header,block+header).replace(original,call+original)
    assert candidate.replace(block,b'').replace(call,b'')==parent and len(candidate)<=10_000_000
    compile(candidate,'candidate.py','exec')
    return candidate
if __name__=='__main__':
    result=build();(HERE/'candidate.py').write_bytes(result);print(sha(result),len(result))
