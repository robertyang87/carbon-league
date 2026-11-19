"""Append the fitted risk veto to the exact signed A without changing its network or selector."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import zlib

HERE=Path(__file__).resolve().parent
A=HERE.parent/"tree_tenure_dev/pay_policy"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path,data):
    temp=path.with_name(f".{path.name}.{os.getpid()}.part")
    temp.write_bytes(data)
    temp.replace(path)


def compose(weights):
    if len(weights["risk"])!=18 or any(type(p) not in (float,int) or not math.isfinite(p) or not 0<=p<=1 for p in weights["risk"]):
        raise ValueError("Expected 18 finite probabilities")
    if weights.get("threshold",.50)!=.50:
        raise ValueError("Unexpected veto threshold")
    src=(A/"candidate.py").read_bytes();receipt=json.loads((A/"package.json").read_bytes())
    risk=(HERE/"risk.py").read_bytes();glue=(HERE/"overlay_B.py").read_bytes()
    protocol=(HERE/"protocol.json").read_bytes()
    plan=json.loads(protocol)
    if sha(src)!=receipt["candidate_sha256"] or sha(src)!=plan["A_sha256"] or receipt["protocol_sha256"]!=plan["A_protocol_sha256"]:
        raise ValueError("Frozen A source or protocol changed")
    code=base64.b85encode(zlib.compress(risk,9))
    # Embed only the inferential, immutable probability table. Archive complete training provenance separately.
    pre=(f"\n_B_RISK=types.ModuleType('sapling_planter_stay_risk')\n"
         f"exec(zlib.decompress(base64.b85decode({code!r})),_B_RISK.__dict__)\n"
         f"_B_WEIGHTS={{'risk':{weights['risk']!r}}}\n").encode()
    binary=src+pre+glue
    compile(binary,"candidate_B.py","exec")
    if len(binary)>10_000_000:
        raise ValueError("single-file limit")
    return binary,{"B_sha256":sha(binary),"bytes":len(binary),"A_sha256":sha(src),
                   "v1_parent_sha256":receipt["parent_sha256"],"A_protocol_sha256":receipt["protocol_sha256"],
                   "risk_sha256":sha(risk),"overlay_B_sha256":sha(glue),"protocol_sha256":sha(protocol),
                   "builder_sha256":sha(Path(__file__).read_bytes())}


def main():
    raw=(HERE/"risk_weights.json").read_bytes()
    weights=json.loads(raw);binary,receipt=compose(weights)
    if weights["protocol_sha256"]!=receipt["protocol_sha256"]:
        raise ValueError("Model trained under a different registration")
    receipt.update(risk_weights_sha256=sha(raw),purpose="separate B development diagnostic")
    write(HERE/"candidate_B.py",binary)
    write(HERE/"B_package.json",(json.dumps(receipt,indent=2)+"\n").encode())
    print(receipt["B_sha256"],receipt["bytes"])


if __name__=="__main__":
    main()
