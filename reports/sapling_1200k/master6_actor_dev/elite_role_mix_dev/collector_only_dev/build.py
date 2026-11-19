"""Keep the frozen parent's full production joint order as a fallback; condition only collector actions."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
A=HERE.parents[1]
P=A/"actor_partner_dev/candidate.py"
C=HERE.parent/"conditional/candidate.py"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    if sha(P)!="c5ad3f43959a4812e84e89574ee0c6e69765ebb06a8b3ec93f839b0dde128dd3" or sha(C)!="72ad985152e628b63fbd99e42dd82948365422e2a62d02e87073660d80cb64e0":
        raise ValueError("source identity changed")
    src=P.read_text();learned=C.read_text().split("\n_ROLE_THETA = ",1)[1]
    decoder=src.split("def _return_decode(",1)[1].split("def _return_basis(",1)[0]
    decoder="def _collector_decode("+decoder
    decoder=decoder.replace("encoded, apply_residual):","encoded, apply_residual, original):")
    point='        action = int(np.argmax(np.where(legal, combined, -np.inf)))'
    if decoder.count(point)!=1:raise ValueError("decoder changed")
    decoder=decoder.replace(point,'''        if int(x["tok_kind"][slot]) == 0:
            combined = combined + _role_residual(x, slot, unit, move, legal, scores[0], scores[1])
            action = int(np.argmax(np.where(legal, combined, -np.inf)))
        else:
            action = int(original[slot])
            if not legal[action]:
                return None''')
    # Original implementation's counters only describe the actual frozen-parent decode.
    left=decoder.index('        if not apply_residual:\n')
    right=decoder.index('        labels[slot]',left)
    decoder=decoder[:left]+decoder[right:]
    at=src.rindex('    baseline = _return_decode(x, aux, enc, False)')
    guard=src[at:]
    if not guard.rstrip().endswith('return candidate'):raise ValueError("return producer changed")
    src=src[:at]+'''    original = _guarded_parent(x, aux, enc)
    baseline = _collector_decode(x, aux, enc, False, original)
    if baseline is None:
        return original
    chosen = baseline
    adjusted = _collector_decode(x, aux, enc, True, original)
    if adjusted is not None and not (_return_conflicts(x, adjusted) - _return_conflicts(x, baseline)):
        chosen = adjusted
    if _return_conflicts(x, chosen) - _return_conflicts(x, original):
        return original
    return chosen


def _guarded_parent(x, aux, enc):
    F = sys.modules["cnet2_feat"]
'''+guard+'\n_ROLE_THETA = '+learned+'\n'+decoder
    out=HERE/'candidate.py'
    if out.exists() and out.read_text()!=src:raise ValueError("would replace frozen candidate")
    if not out.exists():out.write_text(src)
    print(sha(out),out.stat().st_size)


if __name__=="__main__":build()
