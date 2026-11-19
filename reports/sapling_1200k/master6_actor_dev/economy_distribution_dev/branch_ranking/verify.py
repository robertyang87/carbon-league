"""Independent pairing, map-level sensitivity, and payoff rank checks over sealed logs."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def main():
    p=read(HERE/"results.json"); ranks=read(HERE/"rankings.json")
    assert sha(HERE/"protocol.json")==p["protocol_sha256"] and sha(HERE/"run.py")==p["source_sha256"]
    assert ranks["source_sha256"]==sha(HERE/"results.json") and len(p["rows"])==20 and not p["excluded"]
    maps={row["seed"] for row in p["rows"]}
    assert len(maps)==6
    assert all(v["predict_non_ties"]==20 for k,v in ranks["comparisons"].items() if k.startswith("all/") and "/gross/" in k)
    out={}
    for scenario in ("public_estimate","actual_successor"):
        for h in (10,30):
            x={};control={};y={};cost={}
            for g in p["rows"]:
                wd=ROOT/g["source"]; a=g["actual"];f=g["forecast"]
                for arm in ("parent","alternative"):
                    evidence=read(wd/("live_"+arm+".json"))
                    assert evidence["margin"]==a[arm]["official_margin"]
                    assert evidence["action_sha256"]==sha(wd/evidence["action_tape"])
                key=(g["seed"],g["seat"],g["step"])
                def ev(arm, model):
                    v=f[arm][scenario]["readouts"][model][str(h)]["channels"]
                    return sum(v[:2])-sum(v[2:])
                x[key]=ev("alternative","candidate")-ev("parent","candidate")
                control[key]=ev("alternative","control")-ev("parent","control")
                y[key]=a["alternative"]["official_margin"]-a["parent"]["official_margin"]
                cost[key]=max(y[key],0)-(y[key] if x[key]>.01 else 0 if x[key]<-.01 else .5*y[key])
            sub=ranks["comparisons"][f"all/{scenario}/candidate/{h}/gross/official_margin"]
            correct=sum(np.sign(x[k])==np.sign(y[k]) for k in y)
            assert correct==sub["hits_on_real_nonties"]
            regrets=np.array([np.mean([v for k,v in cost.items() if k[0]==m]) for m in sorted(maps)])
            const=np.array([.5*np.mean([abs(v) for k,v in y.items() if k[0]==m]) for m in sorted(maps)])
            assert np.isclose(regrets.mean(),sub["map_equal_regret"])
            rng=np.random.default_rng(561191)
            idx=rng.integers(len(maps),size=(5000,len(maps)))
            diff=regrets-const
            out[f"{scenario}_{h}"]={"nontie_terminal_correct":int(correct),
                "same_order_as_detached_control":int(sum(np.sign(x[k])==np.sign(control[k]) for k in x)),
                "mean_regret_minus_coinflip":float(diff.mean()),
                "six_map_descriptive_bootstrap95":np.quantile(diff[idx].mean(1),[.025,.975]).tolist(),
                "regret_toward_oracle_lower_than_coinflip_maps":int((diff<0).sum())}
    result={"authority_score_and_tapes_readback":40,"maps":6,"roots":20,"comparisons":out}
    (HERE/"verification.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
