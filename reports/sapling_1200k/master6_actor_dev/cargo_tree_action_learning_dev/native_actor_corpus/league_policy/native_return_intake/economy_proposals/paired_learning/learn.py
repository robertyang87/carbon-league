"""One registered neural readout update. Fit/foreign prediction/label in that order."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

import model


HERE = Path(__file__).resolve().parent
PLAN = "8486ff0b4c087bb168a5523db3828159046b32d5bb4befd65a1e2b1a1fefbb8e"
FIT, HOLDOUT = (621758462, 0), (621758463, 1)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(doc):
    return (json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def seal(path, raw):
    """Publish complete bytes exactly once, tolerating concurrent identical work."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as temp:
        temp.write(raw)
        tmp = Path(temp.name)
    try:
        try:
            os.link(tmp,path)
        except FileExistsError:
            if path.read_bytes()!=raw:
                raise ValueError(f"different evidence already sealed at {path}")
    finally:
        tmp.unlink()


def arrays(case):
    menu = case["catalog"]
    proposed = [r["command"] for r in menu["orders"]]
    p = np.asarray([r["probability"] for r in menu["orders"]],dtype=float)
    phi = np.asarray(case["phi"],dtype=float)
    a,b = model.require_pair(menu,*case["options"])
    if not np.array_equal(model.probabilities(p,phi,model.zeros()),p):
        raise ValueError("zero readout did not recover original physical menu")
    return proposed,p,phi,a,b


def update(difference, delta_u):
    if difference.shape != (8,96) or not np.all(np.isfinite(difference)) or (
            not math.isfinite(delta_u) or abs(delta_u)>1):
        raise ValueError("invalid feature or branch utility difference")
    gradient = -.5*delta_u*difference
    raw = -.25*gradient
    return gradient, raw/max(1.,float(np.linalg.norm(raw))/model.MAX_SHIFT)


def checks(diff,d):
    grad,theta = update(diff,d)
    reverse_g,reverse_theta = update(diff,-d)
    swapped_g,swapped_theta = update(-diff,-d)
    _,zero = update(diff,0.)
    _,identical = update(np.zeros_like(diff),d)
    if not (np.array_equal(grad,-reverse_g) and np.array_equal(theta,-reverse_theta) and
            np.array_equal(grad,swapped_g) and np.array_equal(theta,swapped_theta) and
            not np.any(zero) and not np.any(identical)):
        raise ValueError("zero, reversed outcome, or swapped arm check failed")
    direction = np.sign(d)
    epsilon = 1e-5
    numerical = abs(d)*(np.logaddexp(0,-direction*epsilon*diff)-
                        np.logaddexp(0,direction*epsilon*diff))/(2*epsilon)
    error = float(np.max(np.abs(numerical-grad)))
    if error > 1e-9:
        raise ValueError("numerical gradient disagrees with one-step update")
    return {"zero_label_no_update":True,"opposite_outcome_reverses_update":True,
            "swapping_both_arms_and_labels_preserves_update":True,
            "identical_features_no_update":True,"compared_coordinates":grad.size,
            "max_abs_gradient_error":error, "raw_gradient_norm":float(np.linalg.norm(grad))}


def utility(scores,seat):
    margin = float(scores[seat])-float(scores[1-seat])
    if not math.isfinite(margin):
        raise ValueError("nonfinite official margin")
    wdl = .5 if margin==0 else float(margin>0)
    return .5*wdl+.5*math.exp(-float(np.logaddexp(0.,-margin/1000)))


def difference(label,seat):
    if not label["eligible"]:
        raise ValueError("unqualified terminal label")
    first,second = label["official_scores_by_arm"]
    return utility(second,seat)-utility(first,seat)


def prediction(case,theta):
    commands,p,phi,a,b = arrays(case)
    q = model.probabilities(p,phi,theta)
    ranks = lambda v:[1+int(np.sum(v>item)) for item in v]
    old = np.log(p[b])-np.log(p[a])
    new = np.log(q[b])-np.log(q[a])
    shift = float(np.sum((phi[b]-phi[a])*theta))
    if abs(new-old-shift)>1e-12:
        raise ValueError("full support normalization disagrees with physical pair odds")
    return {"case_files":case["files"], "A_index":a,"B_index":b,
            "physical_orders":commands,"old_probabilities":p.tolist(),"new_probabilities":q.tolist(),
            "old_ranks":ranks(p),"new_ranks":ranks(q),
            "old_log_odds_B_over_A":float(old),"new_log_odds_B_over_A":float(new),
            "change_log_odds_B_over_A":shift,"max_abs_residual":float(np.max(np.abs(np.einsum("nad,ad->n",phi,theta)))),
            "public_features_sha256":sha(np.asarray(phi,dtype="<f8").tobytes()),
            "observation_sha256":sha(encoded(case["observation"]))}


def train(out):
    import evidence
    def pending(case, stage):
        return {"status":case["status"], "stage":stage, "reason":case.get("reason")}
    if sha((HERE/"protocol.json").read_bytes())!=PLAN:
        raise ValueError("paired-learning registration changed")
    versions = {file:sha((HERE/file).read_bytes()) for file in ("learn.py","model.py","evidence.py")}
    fit_public = evidence.public_case(*FIT)
    if fit_public["status"]!="ready":
        return pending(fit_public,"fit_public")
    fit_label = evidence.label_case(*FIT)
    if not fit_label["eligible"]:
        return pending(fit_label,"fit_label")
    _,_,phi,a,b = arrays(fit_public)
    delta = difference(fit_label,FIT[1])
    diff = phi[b]-phi[a]
    law = checks(diff,delta)
    _,theta = update(diff,delta)
    if not np.any(theta):
        status = {"status":"no_update","delta_U_B_minus_A":delta,"checks":law,
                  "reason":"zero label" if delta==0 else "identical representations"}
        seal(out/"no_update.json",encoded(status))
        return status
    stream=io.BytesIO()
    np.save(stream,theta,allow_pickle=False)
    weights = stream.getvalue()
    fit_report={"sources_sha256":versions, "protocol_sha256":PLAN, "seed":FIT[0], "seat":FIT[1],
                "label":fit_label, "delta_U_B_minus_A":delta, "weights_sha256":sha(weights),
                "norm":float(np.linalg.norm(theta)),"checks":law,"prediction":prediction(fit_public,theta)}
    seal(out/"readout.npy",weights)
    seal(out/"fit.json",encoded(fit_report))
    held_public = evidence.public_case(*HOLDOUT)
    if held_public["status"]!="ready":
        return pending(held_public,"model_frozen_waiting_for_holdout_root")
    pred={"protocol_sha256":PLAN,"sources_sha256":versions,"seed":HOLDOUT[0],"seat":HOLDOUT[1],
          "fit_sha256":sha(encoded(fit_report)),"weights_sha256":sha(weights),**prediction(held_public,theta)}
    pred_path=out/"holdout_prediction.json"
    seal(pred_path,encoded(pred))
    held_label=evidence.label_case(*HOLDOUT,sealed_transfer=pred_path)
    if not held_label["eligible"]:
        return pending(held_label,"prediction_frozen_waiting_for_holdout_label")
    measured=difference(held_label,HOLDOUT[1])
    shift=pred["change_log_odds_B_over_A"]
    direction="tie" if shift==0 or measured==0 else ("agrees" if shift*measured>0 else "disagrees")
    result={"status":"complete_diagnostic","holdout_prediction_sha256":sha(encoded(pred)),
            "holdout_label":held_label,"holdout_delta_U_B_minus_A":measured,
            "direction_of_adjustment":direction,"production_weights_changed":False,
            "promotion_eligible":False}
    seal(out/"result.json",encoded(result))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"artifacts")
    print(json.dumps(train(parser.parse_args().out),ensure_ascii=False,indent=2,allow_nan=False))
