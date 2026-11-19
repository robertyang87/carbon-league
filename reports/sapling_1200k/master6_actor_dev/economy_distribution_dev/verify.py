"""Recompute register disjointness, raw freight scores, quantized-control identity and limits."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    fit = read(HERE/"fit_receipt.json")
    assert fit["protocol_sha256"] == sha(HERE/"protocol.json")
    assert fit["fit_source_sha256"] == sha(HERE/"fit_distribution.py")
    assert fit["support_sha256"] == sha(HERE/"support.npz")
    assert len(fit["train_sources"]) == 216
    assert fit["weights"]["control"]["changed_parameters_vs_previous_detached"] == 0
    s = np.load(HERE/"support.npz")
    caps = s["means"].max(-1)
    checks = {}
    groups = {}
    for arm in ("control","candidate"):
        assert fit["weights"][arm]["actor_parameters"] == 1206476
        for suffix, field in ((".npz","actor_sha256"),("_readout.npz","readout_sha256")):
            assert sha(HERE/(arm+suffix)) == fit["weights"][arm][field]
    for panel in ("holdout","transfer","fresh"):
        p=HERE/(panel+"_predictions.npz")
        d=read(HERE/(panel+"_result.json"))
        assert sha(p) == d["predictions_sha256"]
        assert d["implementation_sha256"] == sha(HERE/"score.py")
        assert d["fit_receipt_sha256"] == sha(HERE/"fit_receipt.json")
        z=np.load(p);raw=z["cash"];mask=z["full_horizon"][:,6]
        eligible=z["legal_choice"]; gids=z["gid"]
        assert np.array_equal(z["dist_control_action"],z["log_control_action"])
        assert np.array_equal(z["dist_control_nll"],z["log_control_nll"])
        n=len(set(zip(gids,z["step"])))
        assert n==len(gids)
        groups[panel] = {k:{c[k] for c in d["cards"]} for k in ("seed","id","terrain_sha256")}
        assert set(d["source_replays"]).isdisjoint(fit["train_sources"])
        item={}
        for name in ("dist_control","dist_candidate","log_candidate","log_control"):
            model=z[name+"_cash"]
            assert np.isfinite(model).all() and (model>=0).all()
            if name.startswith("dist"):
                assert (model <= caps+.001).all()
                assert ((z[name+"_p500"]>=0)&(z[name+"_p500"]<=1.00001)).all()
            y=raw[:,6]; v=model[:,6]
            prob = d["by_stratum"]["all"]["scores"][name]
            fp=int(np.sum(mask & (y < 500) & (v >= 500)))
            fn=int(np.sum(mask & (y >= 500) & (v < 500)))
            assert (fp,fn)==tuple(prob["rival_delivery_30_pooled"][k] for k in ("fp","fn"))
            map_mae=np.mean([abs(v[mask&(gids==i)]-y[mask&(gids==i)]).mean() for i in np.unique(gids)])
            assert np.isclose(map_mae,prob["map_equal_rival_30"]["mae"])
            item[name]={"verified_map_mae":float(map_mae),"false_alarms":fp,"missed_high_cash":fn}
        item["conditional_changes"]=int(((z["dist_candidate_action"]!=z["dist_control_action"])&eligible).sum())
        item["conditional_choices"]=int(eligible.sum())
        checks[panel]=item
    for k in groups["fresh"]:
        assert groups["fresh"][k].isdisjoint(groups["holdout"][k] | groups["transfer"][k])
    prev=np.load(HERE.parent/"economy_aux_dev/ladder_transfer/predictions.npz")
    rerun=np.load(HERE/"transfer_predictions.npz")
    assert np.array_equal(prev["gid"],rerun["gid"]) and np.array_equal(prev["step"],rerun["step"])
    assert np.allclose(np.expm1(prev["target_log1p"].astype(float)),rerun["cash"])
    for a in ("candidate","control"):
        assert np.array_equal(prev[a+"_action_probs"].argmax(-1),rerun["log_"+a+"_action"])
        precash=np.expm1(np.maximum(0,prev[a+"_income_standardized"]*prev["training_std"]+prev["training_mean"]))
        assert np.allclose(precash,rerun["log_"+a+"_cash"],rtol=2e-6,atol=.002)
    result={"diagnostics":checks,"prior_21_rerun":True,"support_caps":caps.tolist(),
            "fresh_holdout_transfer_id_seed_layout_disjoint":True,"fit_hashes_match":True}
    (HERE/"verification.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    run()
