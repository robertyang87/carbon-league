"""Independent paired terminal-return scoring. This module opens only the registered ticket list."""
import hashlib
import json
import math
from pathlib import Path

MODELS = ("native", "full", "one", "cash")
HERE = Path(__file__).resolve().parent


def terminal(margin):
    return min(2.5, max(-2.5, margin / 3000.0)) + 0.5 * sign(margin)


def sign(x):
    return (x > 0) - (x < 0)


def registered_tickets():
    plan = json.loads((HERE / "protocol.json").read_bytes())
    manifest = (HERE.parent / "action_rank_data" / "protocol.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != plan["selection_sha256"]:
        raise ValueError("Paired-action ticket manifest changed")
    return json.loads(manifest)["tickets"]


def identity(row):
    return (row["terrain"], row["seed"], row["seat"], row["step"])


def finite_pair(data, label):
    pair = [float(x) for x in data]
    if len(pair) != 2 or not all(math.isfinite(x) for x in pair):
        raise ValueError("Expected two finite numbers in " + label)
    return pair


def evaluated(row):
    margin = finite_pair(row["margins"], "official margins")
    values = finite_pair(row["terminal_values"], "original terminal values")
    if any(not math.isclose(t, terminal(m), rel_tol=0, abs_tol=1e-7) for m, t in zip(margin, values)):
        raise ValueError("Per-arm native terminal values disagree with official margins")
    # Use full-precision values from official margins including clipping and the sign jump.
    values = [terminal(m) for m in margin]
    target = values[0] - values[1]
    data = {k: row[k] for k in ("terrain", "seed", "seat", "step")}
    data.update(margins=margin, terminal_values=values, target=target, raw_difference=margin[0]-margin[1],
                outcomes=[("loss", "draw", "win")[sign(x)+1] for x in margin],
                saturated=[abs(x)>=7500.0 for x in margin], raw_true_tie=margin[0]==margin[1],
                terminal_true_tie=target==0, flat_despite_raw_difference=target==0 and margin[0]!=margin[1],
                sign_jump=sign(margin[0])!=sign(margin[1]))
    methods = {}
    for name in MODELS:
        prediction = float(row["preds"][name])
        if not math.isfinite(prediction):
            raise ValueError("Nonfinite paired prediction")
        delta = prediction-target
        arm = 0 if prediction>=0 else 1
        methods[name]=dict(prediction=prediction, rank_sign=sign(prediction), prediction_tie=prediction==0,
                           chosen=arm, error=delta, absolute_error=abs(delta), squared_error=delta*delta,
                           rank_correct=None if target==0 or prediction==0 else sign(target)==sign(prediction),
                           selected_correct=None if target==0 else arm==int(target<0),
                           chosen_margin=margin[arm], chosen_terminal=values[arm], chosen_win=margin[arm]>0)
    native = methods["native"]
    for record in methods.values():
        record.update(choice_changed=record["chosen"]!=native["chosen"],
                      fixed=native["rank_correct"] is False and record["rank_correct"] is True,
                      newly_wrong=native["rank_correct"] is True and record["rank_correct"] is False,
                      choice_fixed=native["selected_correct"] is False and record["selected_correct"] is True,
                      choice_harmed=native["selected_correct"] is True and record["selected_correct"] is False,
                      terminal_effect=record["chosen_terminal"]-native["chosen_terminal"],
                      margin_effect=record["chosen_margin"]-native["chosen_margin"],
                      win_effect=int(record["chosen_win"])-int(native["chosen_win"]))
    data["models"]=methods
    return data


def scored_panel(rows, tickets):
    expected = {identity(x) for x in tickets if x["split"]=="held"}
    all_ids = {identity(x) for x in tickets}
    if len(expected)!=4 or len({x[0] for x in expected})!=4:
        raise ValueError("Four unique held physical maps required")
    actual = [identity(x) for x in rows]
    if len({x[0] for x in actual})!=len(actual) or not set(actual).issubset(all_ids):
        raise ValueError("One registered ticket per terrain required")
    panel = [evaluated(row) for row in rows]
    n = len(panel)
    metrics = {}
    def mean(items):
        return math.fsum(items)/n if n else None
    for name in MODELS:
        predictions = [r["models"][name] for r in panel]
        metrics[name]={
            "mse": mean(p["squared_error"] for p in predictions), "mae": mean(p["absolute_error"] for p in predictions),
            "terminal_effect":mean(p["terminal_effect"] for p in predictions),
            "margin_effect":mean(p["margin_effect"] for p in predictions),
            "win_effect":sum(p["win_effect"] for p in predictions),
            "win_fixes":sum(p["win_effect"]>0 for p in predictions),
            "wins_lost":sum(p["win_effect"]<0 for p in predictions),
            "unchanged_selected":sum(not p["choice_changed"] for p in predictions),
            "second_selected":sum(p["chosen"]==1 for p in predictions),
            "strict_right":sum(p["rank_correct"] is True for p in predictions),
            "strict_wrong":sum(p["rank_correct"] is False for p in predictions),
            **{key: sum(p[key] for p in predictions) for key in
               ("prediction_tie", "choice_changed", "fixed", "newly_wrong", "choice_fixed", "choice_harmed")},
        }
    full, baseline = metrics['full'],metrics['native']
    checks = dict(all_four_registered_held=set(actual)==expected,
                  strict_lower_mse=bool(n and full['mse']<baseline['mse']),
                  corrects_native_without_new_misrank=full['fixed']>=1 and full['newly_wrong']==0,
                  improves_terminal_and_margin=bool(n and full['terminal_effect']>0 and full['margin_effect']>0),
                  selects_alternative=full['second_selected']>=1)
    tied = sum(r["terminal_true_tie"] for r in panel)
    passed = all(checks.values())
    verdict = "PASS" if passed else ("INDETERMINATE" if not checks["all_four_registered_held"] or n==tied else "FAIL")
    missing = [dict(terrain=t, seed=seed, seat=seat, step=step)
               for t, seed, seat, step in sorted(expected-set(actual))]
    return dict(n=n, expected_held=4, observed_registered_held=len(expected.intersection(actual)),
                missing_registered_held=missing, true_terminal_ties=tied, raw_ties=sum(r["raw_true_tie"] for r in panel),
                plateaus_with_different_margins=sum(r["flat_despite_raw_difference"] for r in panel),
                native_strictly_disagrees_with_terminal=sum(r["models"]["native"]["rank_correct"] is False for r in panel),
                all_predictions_equal_native=all(r["models"]["native"]["prediction"]==r["models"]["full"]["prediction"] for r in panel),
                checks=checks, pass_gate=passed, verdict=verdict, by_model=metrics, rows=panel)


def compare(rows):
    """Score a single historical split. Favorable status is possible only on the signed four held tickets."""
    return scored_panel(rows, registered_tickets())


def selftest():
    """Synthetic oracles only; do not load any measured outcomes or predictions."""
    tickets=[dict(terrain=str(i),seat=i%2,seed=100+i,step=279 if i%2==0 else 285,split='held') for i in range(4)]
    def row(i,margins,old,new):
        return {**tickets[i], "margins":margins,"terminal_values":[terminal(x) for x in margins],
                "preds":dict(native=old,full=new,one=0.,cash=old)}
    def run(rows):
        return scored_panel(rows,tickets)
    # Different official cash on the same clipped plateau. No true native ranking can be corrected.
    plateau=run([row(i,(7501,9000),.2,-.1) for i in range(4)])
    assert plateau["verdict"]=="INDETERMINATE" and plateau["raw_ties"]==0 and plateau["true_terminal_ties"]==4
    f=plateau["by_model"]["full"]
    assert f['margin_effect']==1499 and f['fixed']==0 and f['terminal_effect']==0
    # An improvement in MSE alone can leave every chosen move and every outcome identical.
    unchanged=run([row(i,(300,-300),1.7,1.2) for i in range(4)])
    assert unchanged["verdict"]=="FAIL" and unchanged["checks"]["strict_lower_mse"] and unchanged["by_model"]["full"]["unchanged_selected"]==4
    # Three corrected ranks with one newly wrong rank fail, even if both realized averages improve.
    mixed=[row(i,(-300,300),1.0,-1.0) for i in range(3)]+[row(3,(3,0),.5,-.01)]
    s=run(mixed)
    assert s['checks']['improves_terminal_and_margin'] and not s['pass_gate']
    assert s['by_model']['full']['fixed']==3 and s['by_model']['full']['newly_wrong']==1
    # A genuine improvement, a true margin tie, and an exact predicted tie exercise the success path.
    good=mixed[:3]+[row(3,(0,0),0,0)]
    s=run(good)
    assert s['pass_gate'] and s['rows'][3]['models']['full']['chosen']==0 and s['by_model']['full']['win_fixes']==3
    incomplete=run(good[:3])
    assert not incomplete['pass_gate'] and incomplete['missing_registered_held'][0]['terrain']=='3'
    assert not run([])['pass_gate']


if __name__=="__main__":
    selftest()
    print("Synthetic terminal pair oracles passed")
