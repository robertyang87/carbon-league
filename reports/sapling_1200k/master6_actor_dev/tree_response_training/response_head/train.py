"""Fit and evaluate a decision on a paid tree and one exact physical defer move."""
import argparse
from collections import Counter, defaultdict
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import t as student_t
import policy

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       indent=2) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fp:
        temp = Path(fp.name)
        try:
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def derive(records):
    rows = []
    for record in records:
        r, q = record["root"], record["label"]
        is_v1 = q["pay_is_v1"]
        defer = r["new"] if is_v1 else r["old"]
        if (r["old"] if is_v1 else r["new"]) != "NONE":
            raise ValueError("payer did not issue the stay command")
        uid = record["uid"]
        obs = r["public_at_root"]
        x = policy.features(obs, record["cfg"], uid, "NONE", defer, r["v1"])
        pay, deferred = q["pay"], q["defer"]
        delta = float(q["target_delta_official_margin"])
        if not math.isfinite(delta) or abs(pay["official_margin"] - deferred["official_margin"] - delta) > 1e-5:
            raise ValueError("pay-minus-defer official points disagree")
        rows.append(dict(board=record["board"], seed=record["seed"], seat=record["seat"],
                         t=record["t"], uid=uid, batch=record["batch"],
                         root_hash=record["root_hash"], x=list(x), y=delta,
                         pay_is_v1=is_v1, defer_cmd=defer, filled=q["pay_intent_filled"],
                         pay_debit=pay["root_debit"], defer_debit=deferred["root_debit"],
                         pay_margin=pay["official_margin"], defer_margin=deferred["official_margin"],
                         pay_tape=pay["actual_action_tape_sha256"],
                         defer_tape=deferred["actual_action_tape_sha256"]))
    return sorted(rows, key=lambda r: (r["board"], r["seat"], r["t"], r["uid"]))


def support(rows):
    paid = [r for r in rows if r["filled"]]
    sign = lambda predicate: len({r["board"] for r in paid if predicate(r["y"])})
    counts = dict(paid_terrains=len({r["board"] for r in paid}), paid_pairs=len(paid),
                  paid_positive_terrains=sign(lambda y: y > 0),
                  paid_negative_terrains=sign(lambda y: y < 0))
    minima = PROTOCOL["minimum"]
    missing = {k: max(0, n-counts[k]) for k,n in minima.items()}
    return dict(counts=counts, minimum=minima, missing=missing, sufficient=not any(missing.values()),
                intent_terrains=len({r["board"] for r in rows}), intent_pairs=len(rows),
                failed_pay_intents=len(rows)-len(paid), paid_zero_targets=sum(r["y"] == 0 for r in paid))


def ridge(rows):
    """Caller supplies paid samples from TRAIN terrains only."""
    if not rows or any(not r["filled"] for r in rows):
        raise ValueError("empty training group or unfilled expense")
    per_map = Counter(r["board"] for r in rows)
    w = np.array([1/(len(per_map)*per_map[r["board"]]) for r in rows])
    x = np.asarray([r["x"] for r in rows], dtype=float)
    y = np.asarray([r["y"] for r in rows], dtype=float)
    if x.shape != (len(y), len(policy.FEATURE_NAMES)) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("invalid design matrix or effect")
    design = np.column_stack([np.ones(len(y)), x])
    spec = PROTOCOL["ridge"]
    regularization = np.diag([spec["intercept_penalty"]] + [spec["slope_penalty"]]*x.shape[1])
    beta = np.linalg.solve((design.T * w) @ design + regularization, design.T @ (w*y))
    def model(coef):
        return dict(features=list(policy.FEATURE_NAMES), beta=coef, low=x.min(axis=0).tolist(),
                    high=x.max(axis=0).tolist(), support_pad=spec["support_pad"],
                    minimum_predicted_departure=spec["minimum_predicted_departure"],
                    training_terrains=sorted(per_map), training_paid_pairs=len(rows),
                    parent_pack_sha256=PROTOCOL["parent_pack_sha256"])
    return model(beta.tolist()), model([float(w @ y)/(1+spec["intercept_penalty"])] + [0.0]*x.shape[1])


def signed_gain(row, side):
    if side not in ("pay", "defer"):
        raise ValueError("unknown physical decision")
    if (side == "pay") == row["pay_is_v1"]:
        return 0.0
    return row["y"] if side == "pay" else -row["y"]


def mean(xs):
    return math.fsum(xs)/len(xs) if xs else None


def subset_mean(rows, value, test):
    return mean([r[value] for r in rows if test(r)])


def leave_one_map_out(rows):
    eligibility = support(rows)
    if not eligibility["sufficient"]:
        raise ValueError("registered support for real-data fitting not met")
    by_map = defaultdict(list)
    for row in rows:
        by_map[row["board"]].append(row)
    folds = []
    for board in sorted(by_map):
        train_rows = [r for r in rows if r["board"] != board and r["filled"]]
        model, constant = ridge(train_rows)
        roots = []
        for r in by_map[board]:
            chosen = policy.choose(model, r["x"], r["pay_is_v1"])
            simple = policy.choose(constant, r["x"], r["pay_is_v1"])
            parent = "pay" if r["pay_is_v1"] else "defer"
            estimate = policy.predict(model, r["x"])
            roots.append(dict(seed=r["seed"], seat=r["seat"], t=r["t"], uid=r["uid"], root_hash=r["root_hash"],
                              filled=r["filled"], y=r["y"], pay_debit=r["pay_debit"], defer_cmd=r["defer_cmd"],
                              parent=parent, choice=chosen, issued="NONE" if chosen=="pay" else r["defer_cmd"],
                              changed=chosen!=parent, gain=signed_gain(r,chosen),
                              constant_choice=simple, constant_gain=signed_gain(r,simple),
                              prediction=estimate, absolute_error=abs(r["y"]-estimate),
                              constant_absolute_error=abs(r["y"]-policy.predict(constant,r["x"])),
                              zero_absolute_error=abs(r["y"])))
        paid = lambda r: r["filled"]
        folds.append(dict(board=board, roots=roots, fold_weights=model, constant=constant,
                          sampled_intents=len(roots), realized_pays=sum(r["filled"] for r in roots),
                          changed=sum(r["changed"] for r in roots), gain=mean([r["gain"] for r in roots]),
                          constant_gain=mean([r["constant_gain"] for r in roots]),
                          paid_gain=subset_mean(roots,"gain",paid),
                          unfilled_gain=subset_mean(roots,"gain",lambda r:not r["filled"]),
                          paid_mae=subset_mean(roots,"absolute_error",paid),
                          constant_paid_mae=subset_mean(roots,"constant_absolute_error",paid),
                          zero_paid_mae=subset_mean(roots,"zero_absolute_error",paid)))
    gains = [f["gain"] for f in folds]
    avg = mean(gains)
    lower = float(avg - student_t.ppf(.95, len(folds)-1)*np.std(gains,ddof=1)/math.sqrt(len(gains)))
    agg = {k: mean([f[k] for f in folds if f[k] is not None])
           for k in ("gain","constant_gain","paid_gain","unfilled_gain","paid_mae","constant_paid_mae","zero_paid_mae")}
    limits = PROTOCOL["development_gate"]
    checks = dict(mean_gain=avg>=limits["minimum_all_intent_mean_gain"],
                  lower=lower>limits["one_sided_t_lower_must_exceed"],
                  changed_maps=sum(f["changed"]>0 for f in folds)>=limits["minimum_changed_terrains"],
                  positive_map_fraction=sum(g>0 for g in gains)/len(gains)>=limits["minimum_positive_terrain_fraction"],
                  beat_constant=avg>agg["constant_gain"])
    return dict(schema="tree-head-lomo/1", folds=folds, equal_map_means=agg,
                lower_one_sided_t95=lower, positive_terrains=sum(g>0 for g in gains),
                checks=checks, permits_mechanism_check=all(checks.values()))


def calculate(intake_root):
    from evidence import collect
    raw, pending = collect(Path(intake_root))
    rows = derive(raw)
    info = support(rows)
    status = dict(schema="tree-response-readiness/1", parent_pack_sha256=PROTOCOL["parent_pack_sha256"],
                  code_sha256={name:digest((HERE/name).read_bytes()) for name in
                               ("protocol.json","policy.py","train.py","evidence.py")},
                  schema_features=list(policy.FEATURE_NAMES), sealed_batches=sorted({r["batch"] for r in rows}),
                  pending_batches=pending, support=info, samples_sha256=digest(json_bytes(rows)),
                  eligible_for_development=False, sampled_maps=[])
    for board in sorted({r["board"] for r in rows}):
        group = [r for r in rows if r["board"]==board]
        status["sampled_maps"].append(dict(terrain=board, seed=group[0]["seed"], intent=len(group),
                                            paid=sum(r["filled"] for r in group),
                                            paid_positive=sum(r["filled"] and r["y"]>0 for r in group),
                                            paid_negative=sum(r["filled"] and r["y"]<0 for r in group),
                                            failed_positive=sum(not r["filled"] and r["y"]>0 for r in group),
                                            failed_negative=sum(not r["filled"] and r["y"]<0 for r in group)))
    if not info["sufficient"]:
        status["stage"]="insufficient_realized_support"
        return status, rows, None, None
    evaluation = leave_one_map_out(rows)
    model, _ = ridge([r for r in rows if r["filled"]])
    status["evaluation_sha256"] = digest(json_bytes(evaluation))
    status["weights_sha256"] = digest(json_bytes(model))
    status["eligible_for_development"] = evaluation["permits_mechanism_check"]
    status["stage"] = "ready_for_mechanism_verification" if status["eligible_for_development"] else "heldout_gate_failed"
    return status, rows, evaluation, model


def publish(intake_root, output):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    with (HERE/"protocol.json").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status, samples, evaluation, model = calculate(intake_root)
        atomic(output/"samples.json.gz", gzip.compress(json_bytes(samples),mtime=0))
        if model is not None:
            atomic(output/"evaluation.json",json_bytes(evaluation))
            atomic(output/"weights.json",json_bytes(model))
        atomic(output/"status.json",json_bytes(status))
        return status


def rebuild_release(intake_root=HERE.parent):
    """Recount and validate every exact terminal source before granting the pack builder access."""
    with (HERE/"protocol.json").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        status, rows, evaluation, model = calculate(intake_root)
        if not status["eligible_for_development"]:
            raise ValueError("no eligible release: " + status["stage"])
        exact = {"status.json":status,"evaluation.json":evaluation,"weights.json":model}
        for fname,value in exact.items():
            if (HERE/fname).read_bytes() != json_bytes(value):
                raise ValueError("research artifacts diverged from independently rebuilt " + fname)
        if gzip.decompress((HERE/"samples.json.gz").read_bytes()) != json_bytes(rows):
            raise ValueError("signed sample receipts differ from verified live inputs")
        return model


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-root",type=Path,default=HERE.parent)
    parser.add_argument("--out",type=Path,default=HERE)
    args=parser.parse_args()
    result=publish(args.intake_root,args.out)
    print(json.dumps({k:result[k] for k in ("stage","support","sealed_batches","pending_batches")},ensure_ascii=False))
