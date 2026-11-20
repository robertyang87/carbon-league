"""Forecast from the signed training-only head first; a separate command opens both official returns."""
import argparse
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
LEARNING=HERE.parents[1]/"terminal_value_pairs"
sys.path.insert(0,str(LEARNING))
import train
from metrics import sign,terminal
from physical import public_features

read,require,sha=train.read,train.require,train.sha


def forecasts(head_path):
    plan=read(HERE/"protocol.json")
    model=read(head_path)
    require(model["contract_sha256"]==plan["learning_protocol"] and isinstance(model["train_eligible"],bool),"Invalid training freeze")
    require(not model["train_eligible"] or model["heads"],"Missing coefficients")
    require(not set(r["terrain"] for r in model["training"]).intersection(t["terrain"] for t in plan["tickets"]),"Study roots were used in fit")
    _,m=train.setup();net=m._net()
    forecast=[]
    for t in plan["tickets"]:
        s=t["seat"];base=f"seat{s}"
        p=HERE/(base+"_root.json.gz");r=read(p)
        require(r["ticket"]==t and r["diagnostic_protocol"]==sha((HERE/"protocol.json").read_bytes()),"Different capture")
        if r["status"]=="skipped":
            forecast.append(dict(seat=s,status="skipped",native_depth=r["depth"],reason=r["reason"],root_hash=sha(p.read_bytes())))
            continue
        raw=HERE/(base+"_leaf.npz")
        vals=[];x=[];cash=[];errs=[]
        with np.load(raw,allow_pickle=False) as z:
            require(r["status"]=="paired_eligible" and (z["depth"]==8).all() and z["X"].shape==(2,1,54,15,15),"Depth or dimension")
            require([a["name"] for a in r["arms"]]==["first","second"] and r["arms"][0]["order"]!=r["arms"][1]["order"],"Root interventions")
            for a,arm in enumerate(r["arms"]):
                v=arm["original_value"];leaf=arm["leaves"][0];q=leaf["leaf_step"];ros=leaf["players"]
                require(z["candidate_index"][a]==arm["candidate_index"] and z["roll_index"].tolist()==[leaf["roll_index"]] and
                        q==t["step"]+8==z["leaf_step"][a,0] and z["critic"][a,0]==v,"Root-to-leaf binding")
                feat=np.zeros(2) if leaf["exact_terminal"] else public_features(ros,q,s)
                require(np.array_equal(feat,leaf["physical_features"]),"Wrong feature")
                cells=z["X"][a,0].astype(np.float32)*np.float32(.01)
                expected=m._scalars(ros,q,s)
                require(sha(cells.tobytes())==leaf["x_float32_sha256"] and np.allclose(z["scalars"][a,0],expected,rtol=0,atol=1e-6),"Value inputs")
                cash_now=m._terminal(float(ros[s][0])-float(ros[1-s][0]))
                pred=cash_now if leaf["exact_terminal"] else float(net(cells[None],m._SID,expected[None])[2][0])
                require(np.isclose(pred,v,rtol=.0003,atol=.0003),"Native prediction does not reproduce")
                errs.append(abs(pred-v))
                vals.append(v);x.append(feat);cash.append(v if leaf["exact_terminal"] else cash_now)
        row=dict(seat=s,status="predicted",x=(x[0]-x[1]).tolist(),native=vals[0]-vals[1],cash=cash[0]-cash[1],
                 forward_replay_max_error=max(errs),root_hash=sha(p.read_bytes()),leaf_hash=sha(raw.read_bytes()))
        # The training contract bounds each of the two coefficients to [-1,1].
        maximum_correction = sum(abs(z) for z in row["x"])
        row["possible_fitted_gap"] = [row["native"]-maximum_correction, row["native"]+maximum_correction]
        row["predictions"]=train.forecast(row,model["heads"]) if model["train_eligible"] else {"native":row["native"],"cash":row["cash"]}
        forecast.append(row)
    return dict(precommitted_training_heads_sha256=sha(head_path.read_bytes()),training_eligible=model["train_eligible"],
                coefficient_source=str(head_path.relative_to(HERE.parents[3])),rows=forecast)


def settle(head_path):
    f=HERE/"blind_predictions.json"
    prediction=read(f)
    require(prediction["precommitted_training_heads_sha256"]==sha(head_path.read_bytes()),"Frozen training weights changed")
    rows=[]
    for row in prediction["rows"]:
        s=row["seat"];result=read(HERE/f"seat{s}_status.json")
        require(result["root_sha256"]==row["root_hash"],"Changed observed root")
        if row["status"]=="skipped":
            require(result["status"]=="skipped" and result["reason"]==row["reason"],"Invalid root omission")
            rows.append(row);continue
        require(result["status"]=="complete" and result["leaf_sha256"]==row["leaf_hash"],"Incomplete")
        p=HERE/f"sealed/seat{s}_pair.json";require(sha(p.read_bytes())==result["sealed_pair_sha256"],"Changed outcome")
        y=read(p);v=[];margins=[];arm_rows=[];root_actions=[];opponent_orders=[]
        for label in ("first","second"):
            a=y["arms"][label]
            evidence=HERE/f"sealed/{result['ticket']['seed']}_{s}_{result['ticket']['step']}_{label}.json.gz"
            require(sha(evidence.read_bytes())==a["evidence_sha256"],"Changed official action evidence")
            full=read(evidence);root_actions.append(full["rows"][0]["orders"][s]);opponent_orders.append(full["rows"][0]["orders"][1-s])
            require(all(st=="DONE" for st in a["statuses"]) and a["scores"][s]-a["scores"][1-s]==a["margin"],"Invalid official outcome")
            require(np.isclose(terminal(a["margin"]),a["terminal_value"]),"Authoritative terminal scale")
            v.append(a["terminal_value"]);margins.append(a["margin"])
            arm_rows.append(dict(label=label,scores=a["scores"],margin=a["margin"],transitions=a["transitions"],
                                 max_action_seconds=[max(r["clocks"][k]["wall"] for r in full["rows"]) for k in (0,1)],
                                 root_opponent_clock=full["rows"][0]["clocks"][1-s]))
        require(root_actions[0]!=root_actions[1] and opponent_orders[0]==opponent_orders[1],"Different simultaneous opponent response")
        gap=v[0]-v[1];mg=margins[0]-margins[1]
        require(np.isclose(gap,y["target_first_minus_second"]),"Changed comparison")
        tested={name:dict(predicted_gap=g,signed_error=g-gap,agrees_with_target_sign=sign(g)==sign(gap),
                         choice="second" if g<0 else "first")
                for name,g in row["predictions"].items()}
        rows.append(dict(seat=s,status="complete",arms=arm_rows,realized_first_minus_second_terminal=gap,
                         first_minus_second_official_margin=mg,models=tested,source_pair_sha256=result["sealed_pair_sha256"]))
    return dict(forecasts_sha256=sha(f.read_bytes()),protocol_sha256=sha((HERE/"protocol.json").read_bytes()),
                training_eligible=prediction["training_eligible"],rows=rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("operation",choices=["forecast","unseal"]);p.add_argument("heads",type=Path)
    args=p.parse_args();heads=args.heads.resolve()
    value=forecasts(heads) if args.operation=="forecast" else settle(heads)
    name="blind_predictions.json" if args.operation=="forecast" else "results.json"
    print(name,train.immutable(HERE/name,value))
