"""Fixed net-cash and holding-time readout on all 32 already-exposed placements."""
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
LADDER=HERE.parents[1]
HEADS=LADDER.parent/"tree_control_response"
sys.path.insert(0,str(LADDER.parents[2]/"reports/sapling_1200k/all_top5"))
from context import Game


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project(x,w):
    z=[(v-m)/s for v,m,s in zip(x,w["feature_mean"],w["feature_scale"])]
    return [mu+sum(v*coef[j] for v,coef in zip(z,w["coef"])) for j,mu in enumerate(w["means"])]


def metrics(rows,means):
    measured=np.array([[r["net"],r["duration"]] for r in rows])
    result={"n":len(rows)}
    for method in ("model","constant"):
        hats=np.array([r["estimated"] if method=="model" else means for r in rows])
        rejected=hats[:,0]<0
        bad=measured[:,0]<0
        tp=rejected&bad
        fp=rejected&~bad
        result[method]={"mae":np.abs(hats-measured).mean(axis=0).tolist(),
            "negative_predictions":int(rejected.sum()),"negative_trees":int(bad.sum()),
            "true_negative_net_flagged":int(tp.sum()),"nonnegative_trees_flagged":int(fp.sum()),
            "precision":float(tp.sum()/rejected.sum()) if rejected.any() else None,
            "recall":float(tp.sum()/bad.sum()) if bad.any() else None,
            "unrecovered_outlay_flagged":float(-measured[tp,0].sum()),
            "positive_net_flagged":float(measured[fp,0].sum()),
            "all_flagged_signed_net":float(measured[rejected,0].sum())}
    return result


def produce():
    wr=(HEADS/"weights.json").read_bytes()
    assert sha(wr)=="133a3b77a76a1f160e2fc333feea538c49150f6094f54546783590583c047741"
    w=json.loads(wr)
    runt=load("frozen_tree_feature",LADDER/"tree_payback/runtime.py")
    tree=load("archived_tree_menu_history",HEADS/"runtime.py")
    parent=(HEADS/"candidate.py").read_text()
    assert (HEADS/"runtime.py").read_text() in parent
    expr=repr((LADDER/"tree_payback/runtime.py").read_text())
    assert "exec("+expr+",_TREE_LIB)" in parent
    trees_raw=(HERE/"trees.json.gz").read_bytes()
    assert sha(trees_raw)=="f528d834782df3b3acb7cded3bc58dc37cbe2eedd4af4b710033cbfdf1ec9a56"
    ledger=json.loads(gzip.decompress((HERE.parent/"ledger.json.gz").read_bytes()))
    by_root=defaultdict(list)
    for r in json.loads(gzip.decompress(trees_raw))["rows"]:
        by_root[r["match"],r["t"]].append(r)
    assert not (set(r["terrain_sha256"] for r in ledger["registered_games"]) & set(w["train_maps"]+w["held_maps"]))
    out=[]
    calls=0
    for c in ledger["registered_games"]:
        match,seat=c["match_id"],c["seat"]
        raw=(LADDER/"replays"/f"{match}-{c['game']}.json.gz").read_bytes()
        assert sha(raw)==c["replay_sha256"]
        env=json.loads(gzip.decompress(raw)); game=Game(env);cfg=env["configuration"]
        tree._TREE_HIST={}
        for t in range(game.turns):
            obs=game._obs(t,seat)
            # The deployed hook writes every incoming step before even testing its candidate budget.
            # most=2 bypasses all search choices and isolates the unmodified history writer.
            tree._tree_menu_impl(obs,cfg,{},[],2)
            calls+=1
            for plant in by_root[match,t]:
                uid=plant["planter"]
                saved=tree._TREE_HIST[seat]
                carried={k:saved[t-k] for k in (3,5) if t-k in saved}
                full={k:game._obs(t-k,seat)["players"] for k in (3,5) if t>=k}
                x=runt.features(obs,uid,cfg,carried)
                offline=runt.features(obs,uid,cfg,full)
                if carried!=full or x!=offline:
                    raise ValueError(f"History differs on {match},{t},{uid}")
                hat=project(x,w)
                missing=project(runt.features(obs,uid,cfg,{}),w)
                out.append({key:plant[key] for key in ("match","seat","opponent","outcome","t","tree","cell","planter","cost","gross","net","end_t","exit")}|
                    dict(duration=plant["end_t"]-t,estimated=hat,no_history_estimated=missing,available_lags=sorted(carried),
                         x_sha256=sha(np.array(x,dtype='<f8').tobytes())))
    assert len(out)==1104 and len(set((r["match"],r["tree"]) for r in out))==1104
    groups={}
    games={}
    for c in ledger["registered_games"]:
        rs=[r for r in out if r["match"]==c["match_id"]]
        games[c["match_id"]]=dict(opponent=c["opponent"],outcome=c["outcome"],**metrics(rs,w["means"]))
    for k in ("all","W","L"):
        rs=[r for r in out if k=="all" or r["outcome"]==k]
        g=[d for d in games.values() if k=="all" or d["outcome"]==k]
        result=metrics(rs,w["means"])
        result["games"]=len(g)
        mae={m:np.array([d[m]["mae"] for d in g]) for m in ("model","constant")}
        diff=mae["model"]-mae["constant"]
        rng=np.random.default_rng(734291)
        ix=rng.integers(0,len(g),size=(4000,len(g)))
        result["equal_map_MAE"]={m:a.mean(axis=0).tolist() for m,a in mae.items()}
        result["map_delta_MAE_95"]=np.quantile(diff[ix].mean(axis=1),[.025,.975],axis=0).tolist()
        result["improved_MAE_maps"]=(diff<0).sum(axis=0).tolist()
        groups[k]=result
    no_hx=np.array([r["no_history_estimated"] for r in out]);pred=np.array([r["estimated"] for r in out])
    repro={"weight":sha(wr),"planted_rows_gzip":sha(trees_raw),"runtime":sha((HEADS/"runtime.py").read_bytes()),
           "model_features":sha((LADDER/"tree_payback/runtime.py").read_bytes()),"hook_calls":calls,"complete_roster_feature_parity":len(out),
           "missing_both_lags_sensitivity":{"mean_absolute_delta":np.abs(no_hx-pred).mean(axis=0).tolist(),
               "max_absolute_delta":np.abs(no_hx-pred).max(axis=0).tolist(),"zero_cash_sign_switches":int(np.sum((pred[:,0]<0)!=(no_hx[:,0]<0)))}}
    dst=HERE/"frozen_heads_results.json.gz"
    raw=(json.dumps(dict(repro=repro,means=w["means"],groups=groups,by_match=games,trees=out),separators=(",",":"),sort_keys=True,allow_nan=False,ensure_ascii=False)+"\n").encode()
    dst.write_bytes(gzip.compress(raw,mtime=0))
    print(json.dumps({k:{"n":v["n"],"mae":v["equal_map_MAE"],"screen":v["model"]} for k,v in groups.items()},ensure_ascii=False))
    print(repro,sha(dst.read_bytes()))


if __name__=="__main__":
    produce()
