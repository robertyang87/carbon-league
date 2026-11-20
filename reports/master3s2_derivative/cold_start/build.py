"""Reduce startup-only shape warmup in the already-frozen original-opponent derivative."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/"opponent_policy"/"candidate.py"
EXPECTED="245b5814d1855c5449435ee6bd4a53560017a8cc7a46d08729fa9dfdc13e9654"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    blob=PARENT.read_bytes()
    if sha(blob)!=EXPECTED:raise ValueError("frozen V1 changed")
    old=b"        for n in range(1, _S_ROWS + 1):          # every batch a turn can need: the first pass of a new shape is slow"
    new=b"        for n in (1, 2, 4, 8, 16, _S_ROWS):     # prime common sizes and the calibration batch"
    if blob.count(old)!=1:raise ValueError("unexpected warmup")
    result=blob.replace(old,new)
    if result.replace(new,old)!=blob:raise ValueError("more than one warmup change")
    dest=HERE/"candidate.py"
    if dest.exists() and dest.read_bytes()!=result:raise ValueError("conflicting V2")
    dest.write_bytes(result)
    manifest=dict(parent=EXPECTED,candidate=sha(result),parent_bytes=len(blob),candidate_bytes=len(result),
        old_line=old.decode(),new_line=new.decode(),warm_batch_rows=[1,2,4,8,16,32],warm_profiles=6,
        old_total_warm_rows=sum(range(1,33)),new_total_warm_rows=sum([1,2,4,8,16,32]),calibration_batch_size=32,
        all_other_bytes_identical=True,weight_bytes_identical=True)
    (HERE/"release.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(manifest["candidate"])
if __name__=="__main__":main()
