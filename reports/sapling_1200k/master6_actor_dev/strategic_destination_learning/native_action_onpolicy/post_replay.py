"""Verify the frozen new weights through their public agent on each completed training trajectory."""
import gzip
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import zipfile
import numpy as np
from update import HERE, INTAKE, load, need, physical, save, sha


sys.path.insert(0,str(INTAKE))
import extract


def keys(options):
    return [tuple(sorted(o["action"].items())) for o in options]


def main():
    data=load(INTAKE/"training_rows.json.gz")
    fit=load(HERE/"fit.json")
    if "trained" not in fit:
        save(HERE/"deployment_check.json",dict(eligible_for_original_development=False,reason=fit.get("reason","No candidate")))
        return
    need(fit["data"]==sha(INTAKE/"training_rows.json.gz") and fit["trained"]==sha(HERE/"trained.py") and
         fit["head"]==sha(HERE/"trained.npz"),"Unregistered trained head or data")
    hosted=INTAKE/"hosted"
    settlement=load(hosted/"settlement.json")
    need(sha(hosted/"settlement.json")==data["settlement_sha256"],"Different settlement")
    archives={(a["seed"],a["seat"]): a for a in settlement["archives"]}
    with np.load(HERE/"trained.npz",allow_pickle=False) as h:
        weight=h["W1"].ravel()
    need(np.isfinite(weight).all() and np.any(weight),"Nonfinite or unchanged trained head")
    by_map=[]
    for game in data["games"]:
        if not game["gradient_eligible"]: continue
        seed,seat=game["seed"],game["seat"]
        card=archives[seed,seat]
        pack=hosted/card["blob_filename"]
        need(sha(pack)==game["archive_sha256"]==card["sha256"],"Modified source archive")
        actor_old=extract.collect.Player(HERE/"sampler.py")
        actor_new=extract.collect.Player(HERE/"trained.py")
        old,new=actor_old.mod,actor_new.mod
        old._NATIVE_RNG=np.random.default_rng(seed^19917)
        new._NATIVE_RNG=np.random.default_rng(seed^19917)
        need(np.array_equal(old._NATIVE_W0,new._NATIVE_W0) and np.array_equal(old._NATIVE_B0,new._NATIVE_B0)
             and not np.any(old._NATIVE_W1) and np.array_equal(weight,new._NATIVE_W1.ravel()),"Checkpoint byte mismatch")
        with zipfile.ZipFile(pack) as f:
            views=gzip.decompress(f.read("collector/observations.jsonl.gz")).splitlines()
            reg=json.loads(f.read("collector/registration.json"))
        # Same public settings as the runner, excluding the map RNG and native clocks.
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"registration.json"
            path.write_text(json.dumps(reg))
            live=extract.audit_menu.open_live(path)
            _,cfg=live.official()
        different=sampled=ordinary=0
        max_gap=0.
        for row,view in zip(game["steps"],views,strict=True):
            obs=json.loads(view)["observations"][seat]
            first=actor_old.act(obs,cfg)
            second=actor_new.act(obs,cfg)
            before,after=old._LAST_POLICY,new._LAST_POLICY
            lookup=keys(row["orders"])
            need(before["issued"]==first and after["issued"]==second and
                 keys(before["orders"])==keys(after["orders"])==lookup,"Changed physical menu or sampled command")
            p,_=physical(row["policy_features"],len(lookup),weight)
            emitted=np.asarray([o["probability"] for o in after["orders"]])
            old_probs=np.asarray([o["probability"] for o in before["orders"]])
            gap=float(max(np.max(abs(p-emitted)),np.max(abs(old_probs-[o["probability"] for o in row["orders"]]))))
            max_gap=max(max_gap,gap)
            need(gap<5e-6 and max(abs(np.log(emitted[lookup.index(tuple(sorted(second.items())))])-after["logp"]),
                                  abs(np.log(old_probs[lookup.index(tuple(sorted(first.items())))])-before["logp"]))<5e-6,
                 "Deployed probability, actual sample or serialization changed")
            if len(lookup)==1:
                ordinary+=1
                need(first==second==row["orders"][0]["action"],"Changed deterministic command")
            else:
                sampled+=1
                different+=int(first!=second)
        actor_old.diagnostics()
        actor_new.diagnostics()
        by_map.append(dict(seed=seed,seat=seat,terrain=game["terrain"],matched_common_rng_changed=different,
                           stochastic=sampled,ordinary_agree=ordinary,max_probability_gap=max_gap))
    changed=sum(r["matched_common_rng_changed"]>0 for r in by_map)
    save(HERE/"deployment_check.json",dict(candidate=fit["trained"],sampler=fit["sampler"],weights=fit["head"],
        fit=sha(HERE/"fit.json"),source=sha(INTAKE/"training_rows.json.gz"),by_map=by_map,ordinary_unchanged=True,
        changed_maps_public_replay=changed,eligible_for_original_development=bool(fit["pre_replay_gate"] and changed>=2),
        competition_qualified=False,scope="Eight original games use initial exploration weights. Paid candidate evaluation requires a new independent strength and timing batch."))

if __name__=="__main__":
    main()
