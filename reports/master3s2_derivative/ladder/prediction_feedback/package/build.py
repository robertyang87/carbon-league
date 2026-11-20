"""Materialize the two frozen action-logit offsets in the exact V2 opponent-root call."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PARENT = HERE.parents[2] / "cold_start/candidate.py"
WEIGHTS = "d5ebacffa7fb291b5d2fb2f09c034687f7943a1bd017d77f7a5c81ad75f0f304"
HELD = "5f51e097677564c4c8e43364f11d12ac85318f0db46e3708a589841aa9dca570"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze(name, value):
    dest = HERE/name
    if dest.exists():
        if dest.read_bytes()!=value:
            raise ValueError("Conflicting frozen file: "+name)
        return
    partial = dest.with_suffix(".partial")
    partial.write_bytes(value)
    partial.replace(dest)


def build():
    coeffs = (STUDY/"weights.json").read_bytes()
    result = (STUDY/"held_result.json").read_bytes()
    protocol = (STUDY/"protocol.json").read_bytes()
    runtime = (STUDY/"runtime.py").read_bytes()
    parent = PARENT.read_bytes()
    w, held = json.loads(coeffs), json.loads(result)
    assert sha(coeffs)==held["weights_sha256"]==WEIGHTS and sha(result)==HELD
    assert sha(protocol)==held["protocol_sha256"]==w["protocol_sha256"]
    assert sha(parent)==held["parent"]==w["parent"]=="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
    assert sha(runtime)==w["runtime_sha256"] and w["dtype"]=="float32" and len(w["coefficients"])==2
    assert held["verdict"]=="PASS" and all(held["checks"].values())
    text=runtime.decode()
    names=("features","adjust")
    fn=[ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
    assert [v.split("(",1)[0] for v in fn]==["def "+n for n in names]
    # Retain the validated NumPy equations verbatim, changing names only to avoid collisions.
    helpers=("\n\n".join(fn)+"\n").replace("features(", "_feedback_features(").replace("adjust(", "_feedback_adjust(")
    extra=("".join(["_FEEDBACK_WEIGHTS = np.array(",repr(w["coefficients"]),", dtype=np.float32)\n\n"])+helpers+"\n").replace("\n","\r\n").encode()
    marker=b"def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):"
    call=b"    lwo, lbo = _opponent_root_readout(xo, lwo, lbo, players, op)\r\n"
    edit=b"    lwo = _feedback_adjust(lwo, players, op, _FEEDBACK_WEIGHTS, _OPP_STEPS, FEAT_DIST)\r\n"
    if parent.count(marker)!=1 or parent.count(call)!=1 or b"_feedback_" in parent:
        raise ValueError("Unexpected root site or name collision")
    package=parent.replace(marker,extra+marker).replace(call,call+edit)
    assert package.replace(extra,b'').replace(edit,b'')==parent and len(package)<=10_000_000
    compile(package,'candidate.py','exec')
    manifest=dict(candidate_sha256=sha(package),bytes=len(package),parent_sha256=sha(parent),
                  parent_path="reports/master3s2_derivative/cold_start/candidate.py",
                  weights_sha256=sha(coeffs),runtime_sha256=sha(runtime),coefficients=w["coefficients"],
                  held_result_sha256=sha(result),protocol_sha256=sha(protocol),builder_sha256=sha(Path(__file__).read_bytes()),
                  changed_site="Only the simulated enemy collectors' native worker logits, immediately after V2 opponent root readout. Actor, critic, base logits, rollout policy, temperature and native deadline remain as V2.")
    return package,manifest


if __name__=="__main__":
    code,manifest=build()
    freeze('candidate.py',code)
    freeze('manifest.json',(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    print(manifest["candidate_sha256"],manifest["bytes"])
