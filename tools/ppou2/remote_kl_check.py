#!/usr/bin/env python3
"""Print each arm's latest formal-round KL reading as JSON lines.
Runs on L20. Used by the serial driver's KL stop-line check.
Output per arm: {"arm": "m", "it": 20, "kl": 0.1288, "kl_coef": 0.05}
(it/kl are null when the arm has no formal round yet.)
"""
import json

for arm in ("m", "c"):
    rec = {"arm": arm, "it": None, "kl": None, "kl_coef": None}
    try:
        lines = [json.loads(l) for l in
                 open(f"/root/carbon/data/rl3/ppou2_{arm}/train.jsonl")
                 if l.strip()]
        formal = [l for l in lines if not l.get("warm")]
        if formal:
            last = formal[-1]
            rec.update(it=last["it"], kl=last["kl"],
                       kl_coef=last.get("kl_coef"))
    except FileNotFoundError:
        pass
    print(json.dumps(rec))
