"""Pack and validate the NB12 experimental capacity arm from its exact training receipt."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zlib

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"code"))
import cnet2_pack
from cnet2_model import CNet2
from sapling_capacity_ab import ARCH

WEIGHTS=ROOT/"data/weights/sapling_1200k"
RAW=WEIGHTS/"soft_it50_deep214.npz"
PUB=WEIGHTS/"soft_it50_deep214_agent.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt=json.loads((HERE/"capacity_ab_214_training.json").read_text())
    if receipt["trained_sha256"]!=sha(RAW):
        raise ValueError("raw weight differs from training receipt")
    with np.load(RAW,allow_pickle=False) as z:
        meta=json.loads(str(z["meta"]))
        cfg=dict(C=93,NB=12,D=96,NH=4,L=1,E=32,G=3,TA=0)
        if (meta["cfg"]!=cfg or meta["architecture_id"]!=ARCH or meta["model_name"]!="SaplingExperimental"
                or meta["params"]!=2143916 or "preset" in meta):
            raise ValueError("architecture mislabeled")
        shape_ref=CNet2(**cfg).state_dict()
        if set(shape_ref)!=set(z.files)-{"meta"}:
            raise ValueError("missing or extra tensors")
        for k,v in shape_ref.items():
            if z[k].shape!=tuple(v.shape) or not np.isfinite(z[k]).all():
                raise ValueError("shape or finiteness failure: "+k)
    with tempfile.TemporaryDirectory(dir=WEIGHTS) as tmp:
        cp=Path(tmp)/RAW.name
        out=Path(tmp)/"agent.py"
        shutil.copyfile(RAW,cp)
        cnet2_pack.QMODE="int8"
        cnet2_pack.main(str(cp),str(out),teacher=2)
        spec=importlib.util.spec_from_file_location("validate_deep_public",out)
        pack=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pack)
        deq=pack._dequantize(zlib.decompress(base64.b64decode(pack._W64)),pack._QMETA)
        if json.loads(pack._WMETA)!=meta or set(deq)!=set(shape_ref):
            raise ValueError("mismatched packed model identity")
        for k,v in deq.items():
            if tuple(v.shape)!=tuple(shape_ref[k].shape) or not np.isfinite(v).all():
                raise ValueError("packed tensor changed shape or is non-finite")
        if PUB.exists():
            if sha(PUB)!=sha(out):
                raise FileExistsError("different frozen candidate")
        else:
            os.link(out,PUB)
    print("FP32",sha(RAW),"int8",sha(PUB),PUB.stat().st_size,flush=True)


if __name__=="__main__":
    main()
