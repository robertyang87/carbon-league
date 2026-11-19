"""Construct an offset head over the frozen v1 autoregressive base scores."""
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
BASE=DEV.parent
ROOT=BASE.parents[1]
SOURCE=ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    b=path.read_bytes()
    return json.loads(gzip.decompress(b) if path.suffix==".gz" else b)


def write(path,obj):
    raw=(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
    if path.suffix==".gz":raw=gzip.compress(raw,mtime=0)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError(f"frozen artifact differs {path.name}")
    else:
        t=path.with_suffix(path.suffix+".tmp");t.write_bytes(raw);t.replace(path)


def parts():
    plan=read(HERE/"protocol.json")
    assert sha(SOURCE)==plan["parent_sha256"]
    src=SOURCE.read_text()
    lo=src.index("def _return_decode(");hi=src.index("\n    return labels",lo)+len("\n    return labels")
    d=src[lo:hi].replace("_return_decode(x, aux, encoded, apply_residual)","_wf_decode(x,aux,encoded,apply_residual,teacher=None,forced_slot=-1,forced=-1,record=True)")
    d=d.replace("    previous = 8","    rows=[]\n    previous = 8")
    a=d.index("        if not apply_residual:");b=d.index("        labels[slot] = action",a)
    d=d[:a]+'''        if slot>=10 and record:
            v,info=_workforce_features(x,aux,F,slot,prefix,labels)
            rows.append(dict(slot=slot, features=v, info=info, scores=np.asarray(combined[:3],dtype=np.float64),legal=legal[:3].copy()))
        if teacher is not None:
            action=int(teacher[slot])
        if slot==forced_slot:
            action=forced
        if action<0 or not legal[action]:
            return None,rows
'''+d[b:]
    d=d.replace("    return labels","    return labels,rows")
    return src,d


def training_module():
    base,decoder=parts()
    ns={"__file__":"workforce_teacher.py","__name__":"workforce_teacher"}
    exec(compile(base+"\n"+(HERE/"features.py").read_text()+"\n"+decoder,"workforce_teacher.py","exec"),ns)
    return ns


def emit():
    src,decoder=parts();i=src.rindex("\ndef _mixed_actions(");select=src[i:]
    assert select.count("return baseline")==3 and select.count("return candidate")==1
    select=select.replace("return baseline","return _workforce(x,aux,enc,baseline,False)").replace("return candidate","return _workforce(x,aux,enc,candidate,True)")
    payload="\n".join([src[:i],select,(HERE/"features.py").read_text(),decoder,
        "_WF_MODEL="+repr(read(HERE/"weights.json")),(HERE/"runtime.py").read_text()]).encode()
    compile(payload,"workforce_agent.py","exec")
    assert len(payload)<=10_000_000
    out=HERE/"agent.py"
    if out.exists():assert out.read_bytes()==payload
    else:out.write_bytes(payload)
    write(HERE/"manifest.json",{"parent_sha256":sha(SOURCE),"candidate_sha256":sha(out),"bytes":len(payload),
        "inputs":{n:sha(HERE/n) for n in ["protocol.json","observational_gate.json","weights.json","build.py","features.py","runtime.py"]}})
    return out


if __name__=="__main__":emit()
