"""Fit one preregistered scalar on training contrasts, then optionally score the sealed holdout."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "action_rank_data"
PLAN = HERE / "protocol.json"
SHA = "55ed66838f26a4c0a34e7bf122e0799af1e833513a9ba07442cc3e7694e2ee6c"
ARMS = ("first","second")
TOL = 1e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def publish(name, obj):
    path = HERE / name
    obj = {**obj,"protocol_sha256":sha(PLAN),"analysis_sha256":sha(Path(__file__))}
    encoded=(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if path.exists():
        require(path.read_bytes() == encoded, "Frozen output differs; retain the existing result")
    else:
        with path.open("xb") as f:
            f.write(encoded)
    return path


def nat(margin):
    return .5*((margin>0)-(margin<0)) + max(-2.5,min(2.5,margin/3000.))


def choice(difference):
    return 0 if difference >= -TOL else 1


def load_split(split):
    plan = read(PLAN)
    require(sha(PLAN)==SHA and sha(DATA/"protocol.json")==plan["source_ticket_registration_sha256"],"Changed preregistration")
    tickets = [t for t in plan["tickets"] if t["split"]==split]
    feature_path = HERE.parent / split / "features.json"
    extracted = read(feature_path)
    require(extracted["root_registration_sha256"]==plan["source_ticket_registration_sha256"] and
            extracted["source_sha256"]==read(DATA/"protocol.json")["base"],"Different source features")
    features = extracted["roots"]
    audit_path = DATA / (split+"_replay_check.json")
    audit = read(audit_path)
    require(audit["protocol_sha256"]==plan["source_ticket_registration_sha256"],"Different official replay check")
    signed = {(item["seed"],item["seat"],item["step"]):item for item in audit["tickets"]}
    require(len(signed)==len(tickets)==len(audit["tickets"]),"Incomplete official replay ledger")
    by_id = {f["ordinal"]:f for f in features}
    require(len(by_id)==len(features)==len(tickets) and set(by_id)=={t["ordinal"] for t in tickets}, "Incomplete feature ledger")
    rows,missing = [],[]
    for t in tickets:
        feature = by_id[t["ordinal"]]
        require(all(feature[k]==t[k] for k in ("seed","seat","step","terrain","split")), "Different source root")
        base = DATA / split
        prefix = f'{t["seed"]}_{t["seat"]}_{t["step"]}_'
        root_file, pair_file = base/(prefix+"root.json.gz"), base/(prefix+"pair.json")
        pair,root = read(pair_file),read(root_file)
        checked = signed[t["seed"],t["seat"],t["step"]]
        require(checked["status"] == pair["status"], "Official replay status differs")
        require(pair["protocol_sha256"] == plan["source_ticket_registration_sha256"] and
                all(pair["ticket"][k]==v for k,v in t.items()) and feature["root_sha256"] == pair["root_sha256"] == sha(root_file),
                "Mismatched archived root")
        if pair["status"] != "complete" or feature["status"] != "verified":
            require(not (pair["status"]=="complete" and feature["status"]!="verified"), "Eligible pair failed feature verification")
            missing.append(dict(ordinal=t["ordinal"],status=feature["status"],pair_status=pair["status"]))
            continue
        require(feature["leaf_sha256"]==pair["leaf_sha256"]==sha(base/(prefix+"leaf.npz")), "Wrong final leaf")
        vals,depths,labels = [],[],[]
        common,identities = [],{}
        for idx,name in enumerate(ARMS):
            f=feature["arms"][idx]
            a=root["arms"][idx]
            require(f["name"]==a["name"]==name and f["candidate_index"]==a["candidate_index"] and f["order"]==a["order"],"Different finalists")
            depth=f["depths"]
            v=[float(depth[h]["value"]) for h in ("1","4","8")]
            require(all(math.isfinite(x) for x in v) and abs(v[2]-a["original_value"])<=1e-5,"Unreproduced H8")
            require(all(depth[h]["x_sha256"] and depth[h]["scalars_sha256"] for h in ("1","4","8")),"Unidentified operands")
            vals.append(v);depths.append(depth)
            path=base/(prefix+name+".json.gz")
            evidence=read(path); label=evidence["outcome"]
            identities[name]=sha(path)
            require(identities[name]==pair["arms"][name]["evidence_sha256"] and
                    checked["arms"][idx]["file"]==path.name and checked["arms"][idx]["sha256"]==identities[name] and
                    checked["arms"][idx]["official_scores"]==label["scores"] and
                    evidence["training_root"]==pair["ticket"] and evidence["choices"][name]==a["order"] and
                    label["statuses"]==["DONE","DONE"] and label["scores"]==pair["arms"][name]["scores"] and
                    abs(label["margin"] - (label["scores"][t["seat"]]-label["scores"][1-t["seat"]]))<1e-7, "Unverified outcome")
            first=evidence["rows"][0]
            require(first["t"]==t["step"] and first["orders"][t["seat"]]==a["order"]
                    and first["orders"][1-t["seat"]]==evidence["root_shared_original_order"],"Different simultaneous root move")
            common.append(tuple(evidence[k] for k in ("native_incoming_h","registered_v2_root_rng_sha","root_shared_original_order","public_pair")))
            labels.append(float(label["margin"]))
        require(common[0]==common[1] and feature["arms"][0]["order"]!=feature["arms"][1]["order"],"Unpaired common random state or identical interventions")
        d1,d4,d8=(first-second for first,second in zip(*vals))
        require(choice(d8)==0, "Native deployed ranking differs")
        identical=all(depths[0]["8"][k]==depths[1]["8"][k] for k in ("x_sha256","scalars_sha256"))
        require(not identical or abs(d8)<1e-6, "Identical H8 inputs changed value")
        x=(d1+d4)/2 - d8
        rows.append(dict(ordinal=t["ordinal"],seed=t["seed"],seat=t["seat"],step=t["step"],terrain=t["terrain"],
                         margins=labels,targets=[nat(v) for v in labels],identical_H8=identical,d1=d1,d4=d4,d8=d8,x=x,
                         pair_sha256=sha(pair_file),branches_sha256=identities))
    return rows,missing,sha(feature_path)


def fit():
    rows,missing,source=load_split("train")
    fused=[r for r in rows if r["identical_H8"]]
    visible=[r["ordinal"] for r in fused if abs((r["d1"]+r["d4"])/2)>1e-6]
    xx=math.fsum(r["x"]**2 for r in rows)
    eligible=(len(rows)>=7 and len(fused)==3 and visible and xx>1e-12)
    coefficient=None
    if eligible:
        xy=math.fsum(r["x"]*((r["targets"][0]-r["targets"][1])-r["d8"]) for r in rows)
        coefficient=max(0.,min(1.,xy/(xx+xx/len(rows))))
    return dict(split="train",status=("INDETERMINATE" if not eligible else "NO_CHANGE" if coefficient==0 else "FROZEN"),
                feature_sha256=source,replay_check_sha256=sha(DATA/"train_replay_check.json"),rows=rows,missing=missing,coalesced_ordinals=[r["ordinal"] for r in fused],
                early_distinguishable=visible,ridge=xx/len(rows) if rows else None,coefficient=coefficient)


def evaluate():
    frozen=HERE/"fit.json"
    state=read(frozen)
    require(state["protocol_sha256"]==SHA and state["analysis_sha256"]==sha(Path(__file__)) and
            state["status"]=="FROZEN" and 0<state["coefficient"]<=1 and
            sha(HERE.parent/"train/features.json")==state["feature_sha256"],"Positive coefficient must be frozen on training maps first")
    rows,missing,source=load_split("held")
    coefficient=state["coefficient"]
    for r in rows:
        y=r["targets"][0]-r["targets"][1]
        a=r["d8"];b=a+coefficient*r["x"]
        i,j=choice(a),choice(b);mi,mj=r["margins"][i],r["margins"][j]
        r.update(target_difference=y,baseline_difference=a,blend_difference=b,native_index=i,blend_index=j,
                 error_baseline=(y-a)**2,error_blend=(y-b)**2,terminal_effect=r["targets"][j]-r["targets"][i],
                 margin_effect=mj-mi,outcome_downgrade=((mj>0)-(mj<0))<((mi>0)-(mi<0)))
    checks=dict(all_four_maps=(len(rows)==4 and not missing),lower_mse=sum(r["error_blend"]-r["error_baseline"] for r in rows)<0,
                changed_to_better_terminal=any(r["terminal_effect"]>0 for r in rows),
                positive_mean_margin_effect=sum(r["margin_effect"] for r in rows)>0,
                no_wdl_downgrade=not any(r["outcome_downgrade"] for r in rows))
    summary={key:math.fsum(r[key] for r in rows)/len(rows) if rows else None
             for key in ("error_baseline","error_blend","terminal_effect","margin_effect")}
    summary["saturated_equal_value_different_margin"] = sum(r["targets"][0]==r["targets"][1] and r["margins"][0]!=r["margins"][1] for r in rows)
    summary["raw_margin_ties"] = sum(r["margins"][0]==r["margins"][1] for r in rows)
    return dict(split="held",summary=summary,replay_check_sha256=sha(DATA/"held_replay_check.json"),status="INDETERMINATE" if not checks["all_four_maps"] else "PASS" if all(checks.values()) else "FAIL",
                fit_sha256=sha(frozen),coefficient=coefficient,feature_sha256=source,checks=checks,rows=rows,missing=missing)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("fit","held"))
    mode=parser.parse_args().mode
    function=fit if mode=="fit" else evaluate
    try:
        result=function()
    except (OSError,KeyError,ValueError,TypeError) as error:
        result=dict(split=mode,status="INDETERMINATE",reason=str(error),coefficient=None)
    target=publish("fit.json" if mode=="fit" else "held.json",result)
    print(result["status"],sha(target))
