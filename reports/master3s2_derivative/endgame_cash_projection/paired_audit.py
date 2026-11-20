"""Retrospective action audit. Freeze forecasts from archived H8 states before loading either suffix."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
import runtime

HERE=Path(__file__).resolve().parent
ROOTS=HERE.parent/"action_rank_data"
MULTI=HERE.parent/"multidepth_value"
WEIGHTS_SHA="ceed9c721d2ff70bcd2a47bb47dacb5ca88b51073e8a7fb9beefe3598e679cd6"
MODELS=("native","cash","aggregate","enhanced")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix==".gz" else raw)


def save(path,value):
    b=(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    if path.suffix==".gz":
        b=gzip.compress(b,mtime=0)
    if path.exists():
        require(path.read_bytes()==b,"Conflicting signed output "+str(path))
    else:
        temp=path.with_suffix(".tmp");temp.write_bytes(b);temp.replace(path)
    print(path.name,sha(b))


def generate():
    weights=read(HERE/"weights.json")
    require(sha((HERE/"weights.json").read_bytes())==WEIGHTS_SHA and weights["runtime_sha256"]==sha((HERE/"runtime.py").read_bytes()),"Different model")
    require(sha((HERE/"maps.json").read_bytes())==weights["maps_sha256"],"Different training maps")
    require(weights["aggregate"]["features"]==list(runtime.BASE) and weights["enhanced"]["features"]==list(runtime.FULL),"Wrong feature order")
    primary=read(ROOTS/"protocol.json");tickets=primary["tickets"]
    markers=read(HERE/"maps.json")["tickets"]
    for k in ("seed","terrain"):
        require(not {t[k] for t in tickets}.intersection(t[k] for t in markers),"Training or validation exposure")
    rows=[];source={}
    for split in ("train","held"):
        proof_path=MULTI/split/"features.json";data=read(proof_path)
        source[split]=sha(proof_path.read_bytes())
        require(data["root_registration_sha256"]==sha((ROOTS/"protocol.json").read_bytes()) and data["source_sha256"]==primary["base"] and
                data["extractor"]==sha((MULTI/"extract.py").read_bytes()),"Native recovery identity")
        for proof in data["roots"]:
            t=tickets[proof["ordinal"]];s=t["seat"];st=t["step"]
            stem=ROOTS/split/f"{t['seed']}_{s}_{st}"
            rpath=stem.with_name(stem.name+"_root.json.gz");r=read(rpath)
            require(t==r["ticket"] and t["split"]==split and sha(rpath.read_bytes())==proof["root_sha256"],"Wrong root")
            row={k:t[k] for k in ("ordinal","split","terrain","seed","step","seat")}
            row["root_sha256"]=proof["root_sha256"]
            if proof["status"]=="original_skip":
                require(r["status"]=="skipped" and r["reason"]==proof["reason"],"Missing paired root without reason")
                rows.append(dict(**row,status="original_skip",reason=proof["reason"]));continue
            leaf=stem.with_name(stem.name+"_leaf.npz")
            require(proof["status"]=="verified" and r["status"]=="paired_eligible" and len(proof["arms"])==len(r["arms"])==2 and
                    r["issued"]==r["arms"][0]["order"] and sha(leaf.read_bytes())==proof["leaf_sha256"],"Native finalist identity")
            arms=[]
            with np.load(leaf,allow_pickle=False) as z:
                require(z["X"].shape==(2,1,54,15,15) and z["roll_index"].tolist()==[0] and (z["depth"]==8).all() and not z["exact_terminal"].any(),"Wrong branch depth or terminal board")
                for i,(arm,recovered) in enumerate(zip(r["arms"],proof["arms"],strict=True)):
                    require(arm["order"]==recovered["order"] and arm["candidate_index"]==recovered["candidate_index"]==z["candidate_index"][i],"Changed order")
                    l=arm["leaves"][0]
                    require(len(arm["leaves"])==1 and l["roll_index"]==0 and l["leaf_step"]==st+8==z["leaf_step"][i,0] and not l["exact_terminal"],"Wrong successor")
                    x8=recovered["depths"]["8"];critic=arm["original_value"]
                    require(critic==z["critic"][i,0]==recovered["archived_H8"] and abs(x8["value"]-critic)<=1e-5 and
                            sha((z["X"][i,0].astype(np.float32)*np.float32(.01)).tobytes())==x8["x_sha256"]==l["x_float32_sha256"] and
                            sha(z["scalars"][i,0].tobytes())==x8["scalars_sha256"],"Recovery disagrees with signed H8 inputs")
                    players=l["players"];cash=float(players[s][0])-float(players[1-s][0])
                    f=runtime.features(players,s,299-l["leaf_step"])
                    forecast=dict(native=critic,cash=cash)
                    for name in ("aggregate","enhanced"):
                        forecast[name]=cash+runtime.predict_residual(f,weights[name])
                    arms.append(dict(name=arm["name"],candidate_index=arm["candidate_index"],encoded_h8_sha256=x8["x_sha256"],
                                     roster_sha256=sha(json.dumps(players,sort_keys=True,separators=(",",":")).encode()),
                                     feature_row=f,forecast=forecast))
            require(arms[0]["name"]=="first" and arms[1]["name"]=="second" and arms[0]["forecast"]["native"]>=arms[1]["forecast"]["native"] and
                    r["arms"][0]["order"]!=r["arms"][1]["order"],"Actual incumbent order")
            choices={n:int(arms[1]["forecast"][n]>arms[0]["forecast"][n]) for n in MODELS}
            rows.append(dict(**row,status="paired",leaf_sha256=proof["leaf_sha256"],arms=arms,choice=choices))
    require(len(rows)==12 and sum(r["status"]=="paired" for r in rows)==11 and len({r["terrain"] for r in rows})==12,"Changed domain")
    return dict(model_sha256=WEIGHTS_SHA,feature_names=runtime.FULL,multidepth_sources=source,scorer_sha256=sha(Path(__file__).read_bytes()),
                protocol_sha256=sha((ROOTS/"protocol.json").read_bytes()),rows=rows)


def summary(rows):
    out={}
    for name in MODELS:
        r=[x[name] for x in rows]
        out[name]=dict(maps=len(r),changed=sum(x["chosen"]=="second" for x in r),
                       improved=sum(x["official_margin_change"]>0 for x in r),worsened=sum(x["official_margin_change"]<0 for x in r),
                       mean_official_margin=mean(x["chosen_margin"] for x in r),
                       mean_official_margin_change=mean(x["official_margin_change"] for x in r),
                       mean_native_terminal_value_change=mean(x["terminal_value_change"] for x in r),
                       wins=sum(x["chosen_margin"]>0 for x in r),draws=sum(x["chosen_margin"]==0 for x in r),
                       higher_realized_margin_selected=sum(x["oracle_gap"]==0 for x in r))
    return out


def score():
    path=HERE/"paired_inputs.json.gz";frozen=read(path)
    require(frozen["model_sha256"]==WEIGHTS_SHA and frozen["scorer_sha256"]==sha(Path(__file__).read_bytes()),"Forecast code changed")
    rows=[]
    for r in frozen["rows"]:
        row={k:r[k] for k in ("ordinal","terrain","split","seed","seat","step","status")}
        if r["status"]!="paired":
            rows.append(dict(**row,reason=r["reason"]));continue
        stem=ROOTS/r["split"]/f"{r['seed']}_{r['seat']}_{r['step']}"
        file=stem.with_name(stem.name+"_pair.json");target=read(file)
        require(target["status"]=="complete" and target["root_sha256"]==r["root_sha256"] and target["leaf_sha256"]==r["leaf_sha256"] and
                target["ticket"]["terrain"]==r["terrain"],"Target binding")
        arms=[target["arms"][n] for n in ("first","second")];margins=[];term=[]
        for name,arm in zip(("first","second"),arms):
            require(sha(stem.with_name(stem.name+"_"+name+".json.gz").read_bytes())==arm["evidence_sha256"] and
                    arm["statuses"]==["DONE","DONE"],"Independent response suffix")
            s=r["seat"];v=arm["scores"][s]-arm["scores"][1-s]
            require(abs(v-arm["margin"])<1e-6,"Official reward mismatch")
            margins.append(float(v));term.append(arm["terminal_value"])
        for name,i in r["choice"].items():
            row[name]=dict(chosen=("first","second")[i],chosen_margin=margins[i],official_margin_change=margins[i]-margins[0],
                           terminal_value_change=term[i]-term[0],oracle_gap=max(margins)-margins[i],
                           estimator_gap_second_minus_first=r["arms"][1]["forecast"][name]-r["arms"][0]["forecast"][name])
        rows.append(dict(**row,first_and_second_official_margins=margins,official_pair_sha256=sha(file.read_bytes())))
    return dict(audit="Retrospective realized rank of two frozen original-opponent continuations per selected native H8 root. Pair labels were historically known. New projection weights used disjoint maps. Each arm has one response sample. No opening matches or win-rate claim.",
                forecast_sha256=sha(path.read_bytes()),model_sha256=WEIGHTS_SHA,rows=rows,
                summaries={subset:summary([r for r in rows if r["status"]=="paired" and (subset=="all" or r["split"]==subset)]) for subset in ("all","train","held")})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["prepare","score"]);a=p.parse_args()
    if a.action=="prepare":save(HERE/"paired_inputs.json.gz",generate())
    else:save(HERE/"paired.json",score())
