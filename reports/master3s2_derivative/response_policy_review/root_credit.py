"""A two-coordinate root policy, fit to archived differences in official return, not to search visit counts."""
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DER = HERE.parent
DATA = DER / "action_rank_data"
sys.path.insert(0, str(DER / "opponent_policy"))
from original_policy import PROTOCOL, free_entry, module, need, sha


def read(path):
    b = path.read_bytes()
    return json.loads(gzip.decompress(b) if path.suffix == ".gz" else b)


def pre_search(model, card, ticket, root):
    seat, step = ticket["seat"], ticket["step"]
    tracker = model.BeliefTracker(model._Bel, seat)
    previous = None
    captured = {}

    def capture(obs, cfg, est, act0, lw, lb, prev, t0, home):
        captured.update(base=act0, workers=lw, bases=lb)
        return act0

    search = model._search
    model._search = capture
    try:
        for t, obs, cfg, emitted in free_entry.public_history(card):
            if t != step:
                tracker.step(obs["players"], t, obs["carbon"])
                tracker.set_actions(obs["players"], emitted)
                previous = obs["players"]
                continue
            need(all(obs[k] == root["public_root"][k] for k in ("step", "player", "players", "carbon", "trees")), "Wrong public observation")
            model._ST = dict(tr=tracker, prev=previous, rng=np.random.RandomState(ticket["policy_rng"]))
            obs["remainingOverageTime"] = root["public_root"]["remainingOverageTime"]
            base = model.agent(obs, {k:v for k,v in cfg.items() if k != "randomSeed"})
            need(base == captured["base"] == root["menu"][0], "Sampled policy order differs from old actual actor")
            break
        else:
            raise ValueError("Missing registered root")
    finally:
        model._search = search
    return captured


def extract(model):
    cards = {c["sha256"]:c for c in PROTOCOL["games"]}
    raw_registration = (DATA / "protocol.json").read_bytes()
    rows = []
    for tk in json.loads(raw_registration)["tickets"]:
        stem = DATA / tk["split"] / f'{tk["seed"]}_{tk["seat"]}_{tk["step"]}'
        root_path = Path(str(stem) + "_root.json.gz")
        r = read(root_path)
        row = {k:tk[k] for k in ("seed","terrain","seat","step","split")}
        row.update(source_archive_sha256=tk["source_sha256"], root_sha256=sha(root_path.read_bytes()), status=r["status"])
        need(r["ticket"] == tk and r["protocol_sha256"] == sha(raw_registration), "Root protocol mismatch")
        if r["status"] != "paired_eligible":
            row.update(reason=r["reason"]); rows.append(row); continue
        cap = pre_search(model, cards[tk["source_sha256"]], tk, r)
        me = r["public_root"]["players"][tk["seat"]]
        def features(action):
            total = 0.
            mismatches = 0
            for table, logits, ids, workers in ((me[2],cap["workers"],model.WIDX,True), (me[1],cap["bases"],model.BIDX,False)):
                for unit, obj in table.items():
                    pos = obj[0] if workers else obj
                    a = action.get(unit, "NONE")
                    total += float(logits[pos, ids[a]])
                    mismatches += a != cap["base"].get(unit, "NONE")
            return np.array([total, -mismatches])
        arms = r["arms"]
        need(arms[0]["name"] == "first" and arms[1]["name"] == "second" and r["issued"] == arms[0]["order"], "Wrong physical commands")
        dif = features(arms[1]["order"]) - features(arms[0]["order"])
        labels_path = Path(str(stem) + "_pair.json")
        labels = read(labels_path)
        need(labels["root_sha256"] == row["root_sha256"] and labels["ticket"] == tk and labels["status"] == "complete", "Mismatched paired terminal returns")
        a,b = [labels["arms"][k] for k in ("first","second")]
        row.update(delta_root_features=dif.tolist(), first_official_margin=a["margin"], second_official_margin=b["margin"],
                   second_minus_first=round(b["margin"]-a["margin"],2), wins=[a["win"],b["win"]], labels_sha256=sha(labels_path.read_bytes()))
        rows.append(row)
    return rows


def fit(rows):
    x = np.array([r["delta_root_features"] for r in rows])
    y = np.array([r["second_minus_first"] for r in rows]) / 1000.
    scale = np.sqrt((x*x).mean(axis=0))
    scale[scale < 1.e-8] = 1.
    x = x/scale
    # Symmetric difference removes an action-index intercept. Ridge is fixed at 1 per training map.
    w = np.linalg.solve(x.T @ x + len(rows)*np.eye(x.shape[1]), x.T @ y)
    return w/scale


def main():
    import torch
    os.environ["CARBON_DEBUG"]="1";torch.set_num_threads(1);torch.set_num_interop_threads(1)
    raw = (DER/"cold_start/candidate.py").read_bytes()
    need(sha(raw)=="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed","Unexpected parent")
    rows=extract(module(raw))
    train=[r for r in rows if "wins" in r and r["split"]=="train"]
    held=[r for r in rows if "wins" in r and r["split"]=="held"]
    need(len({r["terrain"] for r in rows})==len(rows)==12 and len(train)==7 and len(held)==4,"Map split")
    w=fit(train)
    subs=[fit(train[:i]+train[i+1:]) for i in range(len(train))]
    for r in train+held:
        x=np.array(r["delta_root_features"])
        r.update(predicted_second_minus_first_score_units=float(x@w*1000), choose_second=bool(x@w>0),
                 leave_one_train_map_out_second_votes=sum(bool(x@z>0) for z in subs))
    def aggregate(items):
        labels=np.array([r["second_minus_first"] for r in items])
        choice=np.array([r["choose_second"] for r in items])
        return dict(roots=len(items),choose_second=int(choice.sum()),correct=int(((labels>0)==choice).sum()),
                    frozen_rule_margin_gain_per_game=float(np.mean(labels*choice)),
                    incumbent_margin_gain_per_game=0,runner_up_margin_gain_per_game=float(np.mean(labels)),
                    hindsight_best_of_pair_margin_gain_per_game=float(np.mean(np.maximum(labels,0))),
                    changes_to_winner=sum(r["wins"][0] != r["wins"][1] for r in items))
    out=dict(model_sha256=sha(raw),code_sha256=sha(Path(__file__).read_bytes()),registration_sha256=sha((DATA/"protocol.json").read_bytes()),
             target="official margin(second) - official margin(first), measured from independent natural opponent responses; same root enemy physical action; one sample per option",
             design="ridge, no intercept, scale root deltas by train-only RMS, lambda=1 per train map, return divided by 1000; incumbent retained at predicted tie; retrospective",
             feature_names=["sum unnormalized native actor logits", "negative Hamming distance to original sampled actor joint order"],
             weights_original_feature_units=w.tolist(),train=aggregate(train),held=aggregate(held),rows=rows)
    data=(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    dest=HERE/"root_credit.json"
    need(not dest.exists() or dest.read_bytes()==data,"Prior result differs");dest.write_bytes(data)
    print(out["train"],out["held"],flush=True)


if __name__=="__main__":
    main()
