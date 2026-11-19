import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parents[1]
PARENT = ROOT/"data/weights/sapling_1200k/master6_actor_hybrid.py"
PLAN = json.loads((HERE/"protocol.json").read_bytes())


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def atomic(path,data):
    tmp = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def save(path,obj):
    atomic(path,(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode())


def source(theta):
    import math
    if len(theta)!=4 or not all(math.isfinite(v) and abs(v)<=.75 for v in theta):
        raise ValueError("residual must have four bounded finite coordinates")
    raw = PARENT.read_bytes()
    if digest(raw)!=PLAN["parent"]:
        raise ValueError("wrong frozen dual-network base")
    base = raw.decode()
    d = base[base.index("def _mixed_actions"):base.index("\n\ndef agent(",base.index("def _mixed_actions"))]
    start=d.index("    states = []")
    end=d.index("    prefix = ",start)
    d=d[:start]+"    states = [list(entry) for entry in encoded]\n"+d[end:]
    d=d.replace("def _mixed_actions(x, aux):","def _return_decode(x, aux, encoded, apply_residual):",1)
    hook="        action = int(np.argmax("
    if d.count(hook)!=1:
        raise ValueError("unrecognized parent decoder")
    d=d.replace(hook,'''        if apply_residual and slot < 10 and x["tok_kind"][slot] == 0:
            combined = combined + _return_basis(x, aux, slot, move) @ _RETURN_THETA
'''+hook)
    # The original counters refer to both base networks' actual parent disagreement and
    # must not be incremented twice when the altered decoder is also evaluated.
    i,j=d.index("        expert, student ="), d.index("        labels[slot] =")
    d=d[:i]+"        if not apply_residual:\n"+''.join('    '+l for l in d[i:j].splitlines(True))+d[j:]
    theta_src="_RETURN_THETA = np.array("+repr([float(x) for x in theta])+",dtype=np.float32)\n"
    packed=(base+'\n'+theta_src+d+'\n'+(HERE/'runtime.py').read_text()).encode()
    compile(packed,'single_file_residual_agent.py','exec')
    return packed


def prepare(arm, theta):
    pack=source(theta)
    if len(pack)>10_000_000:
        raise ValueError("data exceed package cap")
    ident={"arm":arm,"theta":[float(v) for v in theta],"parent_sha256":PLAN["parent"],
           "pack_sha256":digest(pack),"bytes":len(pack),"source_sha256":{
               f:digest((HERE/f).read_bytes()) for f in ('runtime.py','build.py','protocol.json')}}
    return pack,ident


def emit(arm, theta, directory):
    pack,row=prepare(arm,theta)
    directory.mkdir(parents=True,exist_ok=True)
    p=directory/f"{arm}.py";q=directory/f"{arm}.json"
    if (p.exists() and p.read_bytes()!=pack) or (q.exists() and json.loads(q.read_bytes())!=row):
        raise ValueError(f"existing {arm} package has another identity")
    if not p.exists():atomic(p,pack)
    if not q.exists():save(q,row)
    return p,q,row
