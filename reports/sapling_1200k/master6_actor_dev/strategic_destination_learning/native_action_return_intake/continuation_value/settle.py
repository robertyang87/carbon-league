"""Resolve the already-frozen estimates against official cents. Do not use for fitting."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from statistics import mean

here = Path(__file__).resolve().parent
frozen = here / "frozen.json"
hashfile = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if hashfile(frozen) != "2da9623b99d9916f899d42b61c317d6212b6936a755592ff7dc14d791dca44c1":
    raise ValueError("Changed predictions")
real = []
grades = {540:"exposed retrospective", 541:"analyst-blind retrospective", 542:"analyst-blind retrospective", 543:"prospective before first observed arm"}
for r in json.loads(frozen.read_bytes())["rows"]:
    path = here.parent / "tickets" / str(r["seed"]) / "summary.json"
    label = json.loads(path.read_bytes(),parse_float=Decimal)
    if (label["terrain"],label["seat"],label["root_public"]) != (r["terrain"],r["seat"],r["root_public_sha256"]):
        raise ValueError("Root identity mismatch")
    values = {}
    for name in ("parent","alternative"):
        x = label["arms"][name]
        if x["root_order"] != r["arms"][name]["team_order"] or x["terminal"]["step"] != 299:
            raise ValueError("Action or completion mismatch")
        scores = x["terminal"]["scores"]
        margin = scores[r["seat"]] - scores[1-r["seat"]]
        values[name]=margin.quantize(Decimal("0.01"))
    signed = float(values["alternative"]-values["parent"])
    pick = "parent" if signed<0 else ("alternative" if signed>0 else "tie")
    pred = r["delta"]
    crit = r["existing_action_critic"]["pick"]
    real.append(dict(seed=r["seed"],seat=r["seat"],grade=grades[r["seed"]%1000],receipt_sha256=hashfile(path),
        cents={k:int(v*100) for k,v in values.items()},realized_delta=signed,predicted_delta=pred,choice=r["pick"],realized_choice=pick,
        sign_correct=r["pick"]==pick,absolute_delta_error=abs(signed-pred),cash_absolute_delta_error=abs(signed-r["cash_delta"]),
        old_critic_correct=None if crit is None else crit==pick,
        realized_gain_over_parent_if_followed=signed if r["pick"]=="alternative" else 0.,
    ))
report=dict(forecast=hashfile(frozen),readback_utc="03:04:42.327Z",prospective_first_result_utc="03:05:08.413Z",
    prospective_pair_done_utc="03:05:30.023Z",rows=real,correct=sum(r["sign_correct"] for r in real),prospective_correct=real[-1]["sign_correct"],
    model_delta_mae=mean(r["absolute_delta_error"] for r in real),cash_delta_mae=mean(r["cash_absolute_delta_error"] for r in real),
    realized_mean_gain_if_followed=mean(r["realized_gain_over_parent_if_followed"] for r in real))
print(json.dumps(report,indent=2))
out=here/"settled.json"
serialized=json.dumps(report,sort_keys=True,allow_nan=False,indent=2)+"\n"
if out.exists() and out.read_text()!=serialized:
    raise ValueError("Conflicting settlement")
out.write_text(serialized)
