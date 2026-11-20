"""Fixed NumPy inference from public observation history; imports no fit data."""
import hashlib
import json
from pathlib import Path

import numpy as np

from features import FREIGHT_PATH, NAMES, vector

HERE=Path(__file__).resolve().parent
ARMS=("geometry","neural")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outputs(weights, x, arm):
    d=np.asarray(x,np.float64)
    scaled=np.clip((d-weights["mean"])/weights["scale"],-6,6)*weights[arm+"_mask"]
    hid=np.tanh(scaled@weights[arm+"_hidden"].T+weights[arm+"_hidden_bias"])
    out=hid@weights[arm+"_output"].T+weights[arm+"_output_bias"]
    p=1/(1+np.exp(-np.clip(out[...,0],-60,60)))
    log_amt=weights["positive_mean"]+weights["positive_scale"]*out[...,1]
    conditional=np.expm1(np.clip(log_amt,0,weights["max_log_payment"]))
    return p,p*conditional


class PublicReturn:
    def __init__(self):
        release=json.loads((HERE/"release.json").read_bytes())
        bindings={"weights":HERE/"weights.npz","features":HERE/"features.py","predict":Path(__file__).resolve(),
                  "freight_features":FREIGHT_PATH,"protocol":HERE/"protocol.json"}
        if any(sha(path)!=release["sha256"][k] for k,path in bindings.items()):
            raise ValueError("a frozen feature, weights or inference artifact changed")
        with np.load(bindings["weights"],allow_pickle=False) as f:
            self.weights={k:f[k].copy() for k in f}
        if tuple(self.weights["names"])!=tuple(NAMES):
            raise ValueError("feature order differs from fitted model")

    def predict(self, history, uid):
        x=vector(history,uid)
        forecasts={}
        for arm in ARMS:
            p,a=outputs(self.weights,x,arm)
            forecasts[arm]=float(p)
            forecasts[arm+"_cash"]=float(a)
        return forecasts
