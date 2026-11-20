"""Patch only two unique entry points in the frozen deployed v2 single file. No weight or inference changes."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/"ladder/prediction_feedback/package/candidate.py"
IDENTITY="8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43"
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    parent=PARENT.read_bytes();assert sha(parent)==IDENTITY
    runtime=(HERE/"runtime.py").read_bytes().replace(b"\n",b"\r\n")
    first=b"def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):"
    second=b"    opp = {}\r\n\r\n    def begin(cs, ms):"
    addition='''    opp = {}
    # Reuse the native CRN draw for rollout zero. No added forward pass, no enemy command input.
    if _time.perf_counter()-t0 < _S_ABORT-.08:
        opp[0]=_sample(players,op,lwo[0],lbo[0],np.random.RandomState(seed0+7),step)
        proposed=_v3_intercept(players,me,prev,step,act0,home,lwo[0],opp[0])
        if proposed is not None:
            choice,rec=proposed
            idx=None;removed=None
            if all(a.get(rec["worker"],"NONE")!=rec["direction"] for a in cands):
                if len(cands)<most:
                    idx=len(cands);cands.append(choice)
                elif len(cands)>=3:
                    idx=len(cands)-1;removed=cands[idx];cands[idx]=choice
            if _V3_AUDIT is not None:
                _V3_AUDIT.append(dict(rec,step=step,index=idx,dropped=removed,added=choice,original=act0,
                                      opponent_prediction=opp[0],limit=most,already_in_native_or_no_room=idx is None))

    def begin(cs, ms):'''.replace('\n','\r\n').encode()
    assert parent.count(first)==parent.count(second)==1
    out=parent.replace(first,runtime+b'\r\n'+first).replace(second,addition)
    assert out.replace(addition,second).replace(runtime+b'\r\n'+first,first)==parent
    assert len(out)<=10_000_000
    compile(out,"submission.py","exec")
    (HERE/"submission.py").write_bytes(out)
    ledger=dict(candidate_sha256=sha(out),parent_sha256=sha(parent),builder_sha256=sha(Path(__file__).read_bytes()),
                runtime_sha256=sha((HERE/"runtime.py").read_bytes()),bytes=len(out),new_weights=0,base_file=str(PARENT.relative_to(HERE.parents[2])))
    (HERE/"manifest.json").write_text(json.dumps(ledger,indent=2)+"\n")
    print(ledger)
if __name__=="__main__":main()
