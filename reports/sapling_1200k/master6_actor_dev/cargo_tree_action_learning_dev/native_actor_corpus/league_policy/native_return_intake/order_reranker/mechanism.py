"""Decision-level numeric checks; cross-fitted trajectory returns are reported in ope.json."""
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np

from features import _order_rows, _order_logit_delta
from rerank import HERE


def summarize(rows):
    stats = []
    for seat in (0, 1):
        active = [r for r in rows if r["seat"] == seat]
        if not active:
            raise ValueError("Missing seat")
        stats.append(dict(seat=seat, supported=len(active),
                          same_inputs_preferred_order_flip=sum((r["old"] >= .5) != (r["new"] >= .5) for r in active),
                          changed_probability=sum(abs(r["old"] - r["new"]) >= 1e-6 for r in active),
                          mean_abs_delta_probability=np.mean([abs(r["old"]-r["new"]) for r in active]),
                          maximum_abs_delta_logodds=max(abs(r["delta"]) for r in active),
                          mean_full_physical_kl=np.mean([r["kl"] for r in active])))
    return stats


def run():
    m = json.loads((HERE.parent / 'hosted/training_manifest.json').read_text())
    w = json.loads((HERE / 'weights.json').read_text())
    p = json.loads((HERE / 'fold_predictions.json').read_text())
    oof = {(e["seed"], e["seat"], e["arm"]): e["q0"] for e in p["episodes"]}
    records = defaultdict(list)
    parameters = sum(np.asarray(v).size for v in w["full"].values())
    for ep in m["episodes"]:
        for e in ep["events"]:
            if not e["supported"]:
                continue
            x = _order_rows(e["public_observation"], e["uid"], e["options"], e["native_logits"])
            old = e["probabilities"][0]
            gap = e["native_logits"][0] - e["native_logits"][1]
            for scope, weights in [("held", w["folds"][str(ep["seed"])]), ("all_six", w["full"])]:
                arrays = {k:np.asarray(v) for k,v in weights.items()}
                delta = _order_logit_delta(x, arrays)
                new = 1 / (1 + math.exp(-gap-delta))
                if scope == "held" and abs(new-oof[(ep["seed"], ep["seat"], ep["arm"])][str(e["step"])])>1e-14:
                    raise ValueError("Held-out probability mismatch")
                records[scope].append(dict(seat=ep["seat"], old=old, new=new, delta=delta,
                    kl=old*math.log(old/new)+(1-old)*math.log((1-old)/(1-new))))
    out = dict(added_trainable_parameters=parameters, learned_weights_dtype="float64",
               ordinary_public_transitions=24*299-288, scheduled_singletons=13,
               by_scope={k:summarize(v) for k,v in records.items()},
               all_six_candidate_sha256=p['full_package_sha256'])
    (HERE/'mechanism.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    run()
