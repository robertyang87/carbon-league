"""Replace only the trained Sapling payload in the frozen two-network policy."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ACTOR = STUDY.parent
ROOT = ACTOR.parents[2]
PARENT = ROOT / "data/weights/sapling_1200k/terminal_return_fit_v1.py"
PLAN = "553f899c47df00482d3fba1f1e41d09ef3860026034deff3db2005c425a084fc"
sys.path[:0] = [str(ROOT / "code"), str(ACTOR)]
import cnet2_pack
from hybrid_build import substitute


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def seal(path, data):
    if not isinstance(data, bytes):
        data = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"attempt to replace frozen artifact {path.name}")
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def encoded_student(asset):
    with np.load(asset, allow_pickle=False) as arrays:
        weights = dict(arrays)
    meta, data, restored = cnet2_pack.quantize(weights)
    if any(not np.isfinite(array).all() for array in restored.values()):
        raise ValueError("nonfinite quantized weight")
    config = str(weights["meta"])
    fields = ("_W64 = " + repr(base64.b64encode(zlib.compress(data, 9)).decode()),
              "_QMETA = " + json.dumps(meta), "_WMETA = " + repr(config))
    restored["meta"] = np.array(config)
    return "\n".join(fields) + "\n", restored


def build():
    plan = read(HERE/"protocol.json")
    train = read(STUDY/"result.json")
    if (sha(HERE/"protocol.json") != PLAN or sha(PARENT) != plan["parent_pack_sha256"]
            or train["protocol_sha256"] != plan["training_protocol_sha256"]
            or train["eligible_for_native_comparison"] is not True or not all(train["checks"].values())
            or cnet2_pack.QMODE != "int8"):
        raise ValueError("training result or frozen parent changed")
    source = PARENT.read_text()
    result={"protocol_sha256":PLAN, "training_result_sha256":sha(STUDY/"result.json"),
            "parent_pack_sha256":sha(PARENT), "parent_weight_manifest_sha256":
                "8a5544ce41cf7fef04424eb735d43c7a659411f872bbd8d90f0046843bad33a2",
            "build_sha256":sha(Path(__file__)), "student_weight":.25, "expert_weight":.75, "arms":{}}
    for arm in ("control", "candidate"):
        path=STUDY/(arm+".npz")
        if sha(path) != plan["student_actor_sha256"][arm] or sha(path)!=train["weights"][arm]["actor_sha256"]:
            raise ValueError("FP32 checkpoint differs")
        fields, arrays=encoded_student(path)
        packet=substitute(source,fields)
        if substitute(packet,source)!=source or len(packet.encode())>10_000_000:
            raise ValueError("a nonstudent field changed or package exceeds limit")
        output=HERE/(arm+".py")
        seal(output,packet.encode())
        result[arm+"_pack_sha256"]=sha(output)
        result["arms"][arm]={"fp32_sha256":sha(path),"pack_sha256":sha(output),"bytes":output.stat().st_size,
                             "student_parameter_tensors":len(arrays)-1}
    seal(HERE/"release.json",result)
    return result


if __name__=="__main__":
    print(json.dumps(build()))
