"""Generate a single-file hidden-carbon estimate correction against the unaltered ds-pro-v1 source."""
from pathlib import Path
import hashlib
import json
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"cold_start/candidate.py"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def build():
    source=BASE.read_bytes()
    plan=json.loads((HERE/"belief_protocol.json").read_bytes())
    if sha(source)!=plan["base_sha256"]:raise ValueError("Unexpected parent")
    old='''                out[i] = apply_ops(pr, self.hist[i])
        return out
'''
    new='''                if _MIXED_BELIEF:
                    history = tuple(self.hist[i])
                    if history not in schedules:
                        at_zero = apply_ops(0.0, history)
                        positive_mean = sum(apply_ops(v, history) for v in _POSITIVE_PRIOR) * 0.125
                        schedules[history] = at_zero + (pr / 22.0) * (positive_mean - at_zero)
                    out[i] = schedules[history]
                else:
                    out[i] = apply_ops(pr, self.hist[i])
        return out
'''
    def line(s):return s.replace("\n","\r\n").encode()
    changed=source
    for before,after in (
        ("class Belief:\n","_MIXED_BELIEF = True\n_POSITIVE_PRIOR = (2.75, 8.25, 13.75, 19.25, 24.75, 30.25, 35.75, 41.25)\n\n\nclass Belief:\n"),
        ("        pr = self.prior()\n        out = [0.0] * N","        pr = self.prior()\n        schedules = {}\n        out = [0.0] * N"),
        (old,new)):
        b,a=line(before),line(after)
        if changed.count(b)!=1:raise ValueError("Source anchor is not unique")
        changed=changed.replace(b,a)
    return changed
if __name__=="__main__":
    raw=build()
    p=HERE/"candidate.py";p.write_bytes(raw)
    m=dict(candidate=sha(raw),base=sha(BASE.read_bytes()),bytes=len(raw),protocol=sha((HERE/"belief_protocol.json").read_bytes()),builder=sha(Path(__file__).read_bytes()))
    (HERE/"manifest.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
    print(json.dumps(m))
