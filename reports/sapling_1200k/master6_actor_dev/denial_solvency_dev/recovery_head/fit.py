"""Fit the two fixed logistic heads on training maps only; write the plain-JSON deployment weights."""
from collections import Counter
import gzip
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from dataset import HERE, fetch
from head import FEATURES, BASE_FEATURES, predict

TARGET="no_collector_in_every_next_30_states_given_trade"


def read_rows():
    data=(HERE/"events.jsonl.gz").read_bytes()
    if fetch.sha(data)!=fetch.load(HERE/"split.json")["events_sha256"]: raise ValueError("event receipt mismatch")
    return [json.loads(l) for l in gzip.decompress(data).splitlines()]


def map_weights(rows):
    by=Counter(r["map"] for r in rows)
    return np.array([1/(len(by)*by[r["map"]]) for r in rows])


def cell(r):
    return f"{int(r['public']['rival_has_tree'])}:{int(r['public']['rival_cash_shortfall_in_fees']>.5)}"


def train(rows, columns):
    w=map_weights(rows)
    y=np.array([r["label_30"] for r in rows])
    x=np.array([[r["public"][f] for f in columns] for r in rows])
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    scale[scale<1e-8]=1
    design=np.column_stack([np.ones(len(x)),np.clip((x-mean)/scale,-3,3)])
    def objective(b):
        z=design@b
        obj=w@(np.logaddexp(0,z)-y*z)+.04*np.dot(b[1:],b[1:])
        deriv=design.T@(w*(expit(z)-y))
        deriv[1:]+=.08*b[1:]
        return obj,deriv
    result=minimize(objective,np.zeros(design.shape[1]),jac=True,method="L-BFGS-B",
                    options={"ftol":1e-14,"gtol":1e-9,"maxiter":600})
    if not result.success:raise ValueError(result.message)
    obj,deriv=objective(result.x)
    if max(abs(deriv))>1e-6:raise ValueError("gradient did not converge")
    out={"target":TARGET,"features":list(columns),"mean":mean.tolist(),"scale":scale.tolist(),
         "intercept":result.x[0],"coef":result.x[1:].tolist(),"penalty":.08,
         "objective_at_fit":obj,"max_abs_gradient":float(max(abs(deriv)))}
    if not np.allclose([predict(r["public"],out) for r in rows],expit(design@result.x),atol=1e-12,rtol=0):
        raise ValueError("standalone inference differs")
    return out


def main():
    train_rows=[r for r in read_rows() if r["split"]=="train" and r["label_30"] is not None]
    w=map_weights(train_rows)
    rate=float(w@np.array([r["label_30"] for r in train_rows]))
    effective=w*len({r["map"] for r in train_rows})
    cells={}
    for key in ("0:0","0:1","1:0","1:1"):
        idx=[i for i,r in enumerate(train_rows) if cell(r)==key]
        cells[key]=(4*rate + sum(effective[i]*train_rows[i]["label_30"] for i in idx))/(4+sum(effective[i] for i in idx))
    model={"fit_events":len(train_rows),"fit_maps":len({r["map"] for r in train_rows}),"constant":rate,
           "tree_gap_table":cells,"tree_gap":train(train_rows,BASE_FEATURES),"state":train(train_rows,FEATURES),
           "protocol_sha256":fetch.sha((HERE/"protocol.json").read_bytes()),"split_sha256":fetch.sha((HERE/"split.json").read_bytes())}
    fetch.dump(HERE/"weights.json",model)

if __name__=="__main__":main()
