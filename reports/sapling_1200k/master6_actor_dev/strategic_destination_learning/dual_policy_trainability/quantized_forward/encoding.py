"""Count changed stored int8 codes, per-row scales and unquantized FP32 parameters."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[2]/"cargo_tree_action_learning_dev/candidate.py"


def count(src, directory, patch):
    rawpatch = (directory/patch["file"]).read_bytes()
    if hashlib.sha256(rawpatch).hexdigest()!=patch["sha256"]:
        raise ValueError("changed patch")
    towers = {}
    for (blob,layout),row in zip((("_B1","_Q1"),("_W64","_QMETA")),patch["towers"],strict=True):
        orig = zlib.decompress(base64.b64decode(src[blob]))
        diff = zlib.decompress(rawpatch[row["start"]:row["start"]+row["length"]])
        new = np.bitwise_xor(np.frombuffer(orig,np.uint8),np.frombuffer(diff,np.uint8)).tobytes()
        for buf,key in ((orig,"original_raw_sha256"),(new,"raw_sha256")):
            if hashlib.sha256(buf).hexdigest()!=row[key]:
                raise ValueError("tower mismatch")
        counts = {k:{"total":0,"changed":0} for k in ("int8_codes","row_scales","float32")}
        off = 0
        for _,shape,storage,bs in src[layout]:
            n = int(np.prod(shape))
            if storage == "int8":
                fields = (("int8_codes",np.int8,n,off), ("row_scales","<f4",shape[0],off+n))
            elif storage == "f32":
                fields = (("float32","<f4",n,off),)
            else:
                raise ValueError(storage)
            for name,typ,length,start in fields:
                a,b = [np.frombuffer(v,typ,length,start) for v in (orig,new)]
                counts[name]["total"] += length
                counts[name]["changed"] += int(np.count_nonzero(a!=b))
            off += bs
        if off != len(orig):
            raise ValueError("unread storage")
        towers[row["tower"]] = counts
    return towers


def run():
    ast_nodes = ast.parse(PARENT.read_bytes()).body
    src={n.targets[0].id:ast.literal_eval(n.value) for n in ast_nodes if isinstance(n,ast.Assign)
         and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("_B1","_Q1","_W64","_QMETA")}
    r=json.loads((HERE/"result.json").read_bytes())
    out = {"one_step":count(src,HERE.parent,json.loads((HERE.parent/"result.json").read_bytes())["patch"])}
    for k,arm in r["arms"].items():
        out[k]=count(src,HERE,arm["patch"])
    (HERE/"encoding.json").write_text(json.dumps(out,indent=2)+"\n")
    print(out)


if __name__=="__main__":
    run()
