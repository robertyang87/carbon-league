"""Frozen independent Top Five mutual match check, using both exact logged prefixes."""
import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch

from elite_3s2_agreement import F, CNet2, HERE, REVISION, STYLE_SLOT, TICKS, WEIGHTS
from elite_3s2_agreement import digest, git, resolve_preset_metadata, sample_rows
from build_sapling_top5_h2h import DESCRIPTIONS, prior_exclusions, slot_context
from recover_public import canonical, identity, key_of, validate

KEYS = ("36e68802e4465ab013073813a646602e-1", "67dd2b66f44a1897249a0081517826bd-1",
        "82f195473526f6b251d88d1c8c894988-1", "a7e7a131ac7d151fc2cccf6e4daf6d38-1",
        "bc551c365258a92404b9d73888061070-1", "f8028deec3901b3fd25bf870a1a20d56-1",
        "f975d32b62d02b7cb49a185f918254b5-1")
CONTEXTS = ("attack", "cargo_defense", "tree_defense")


def archived_cards():
    rows = {}
    for family in ("master5s", "master6", "it720preview", "m9xj630", "fablev7"):
        cards = json.loads(git("show", f"{REVISION}:data/{family}_matches.json"))
        for row in cards:
            if row["state"] != "done" or key_of(row) not in KEYS:
                continue
            key = key_of(row)
            if key in rows and identity(rows[key]) != identity(row):
                raise ValueError("conflicting scorecard for " + key)
            rows[key] = row
    if set(rows) != set(KEYS):
        raise ValueError("missing official scorecard")
    return rows


def label_contexts(row, steps, seat, tick, cfg):
    """Return eligibility and response by canonical worker action using the training definition."""
    players = steps[tick][0]["observation"]["players"]
    own, foe = players[seat], players[1-seat]
    y, tok = row["y"].copy(), row["tok_pos"]
    raw = steps[tick+1][seat]["action"] or {}
    cfg_args = (own, foe, cfg, cfg["episodeSteps"]-1-tick, tok, seat)
    observed = {(s, k):respond for s, k, respond in slot_context(*cfg_args, y, raw)}
    outcome = np.zeros((F.OWN, len(CONTEXTS), 5), bool)
    available = np.zeros((F.OWN, len(CONTEXTS)), bool)
    for s, k in observed:
        available[s, CONTEXTS.index(k)] = True
    occupants = {values[0]: wid for wid, values in own[2].items()}
    for command in range(5):
        alt_raw, alt_y = {}, y.copy()
        for s in np.flatnonzero(y[:10] >= 0):
            cell = int(F.CANON[tok[s]]) if seat else int(tok[s])
            alt_y[s] = command
            alt_raw[occupants[cell]] = F.MOVES[F.CANON_MOVE[command] if seat else command]
        answers = {(s,k):b for s,k,b in slot_context(*cfg_args, alt_y, alt_raw)}
        # Context eligibility is independent of the hypothetical commands.
        if answers.keys() != observed.keys():
            raise ValueError("action-dependent eligibility")
        for (s,k), yes in answers.items():
            outcome[s, CONTEXTS.index(k), command] = yes
    if any(outcome[s, CONTEXTS.index(k), y[s]] != v for (s,k), v in observed.items()):
        raise ValueError("mismatched logged response")
    return available, outcome


def collect(replays):
    ids, seeds, excluded_docs = prior_exclusions()
    manifest = HERE / "top5_h2h_data.json"
    v2 = json.loads(manifest.read_bytes())
    ids.update(r["id"].split("-")[0] for r in v2["games"])
    seeds.update(r["seed"] for r in v2["games"])
    seeds.update(v2["extra_excluded_seeds"])
    seeds.update([620050000, 620050001, 620081731, *range(620089740,620089744)])
    archived = HERE.parents[1] / "data/sapling_top5_h2h_corpus/later7.json"
    archived_blob = archived.read_bytes()
    if digest(archived_blob) != "d132c41d4b93de18f6db8ee7393426fd83a564bc34cf33c789c2a63df2bbaa7f":
        raise ValueError("changed independent late-set archive receipt")
    receipts = {g["id"]:g for g in json.loads(archived_blob)["games"]}
    if set(receipts) != set(KEYS):
        raise ValueError("expected exactly the seven late matches")
    excluded_docs[archived.name] = digest(archived_blob)
    cards = archived_cards()
    data, records, groups, opportunities, responses, excluded = [], [], [], [], [], []
    for key in KEYS:
        blob = (replays / (key+".json.gz")).read_bytes()
        if digest(blob) != receipts[key]["sha256"]:
            raise ValueError("late-set replay digest differs: "+key)
        env = json.loads(gzip.decompress(blob))
        check = validate(env, cards[key])
        seed = check["random_seed"]
        if not isinstance(seed,int) or check["statuses"] != ["DONE", "DONE"]:
            raise ValueError("ineligible recorded run")
        if key.split("-")[0] in ids or seed in seeds or 20291101 <= seed <= 20292100:
            excluded.append(dict(id=key, seed=seed, reason="previously used map or game"))
            continue
        ids.add(key.split("-")[0])
        seeds.add(seed)
        if any(k not in v2["strategy_ids"] for k in check["seat_model_names"]):
            raise ValueError("source outside fixed Top Five")
        rec = dict(id=key, seed=seed, sha256=digest(blob), canonical_sha256=digest(canonical(env)),
                   seats=check["seat_model_names"], rewards=check["rewards_by_seat"],
                   actual_last_step=check["last_step"], by_seat=[])
        for seat in (0,1):
            rows, cuts = sample_rows(env["steps"], seat)
            ticks = [t for t in TICKS if t < len(env["steps"])-1]
            if len(ticks) != len(rows):
                raise ValueError("root-tick mismatch")
            for t, r in zip(ticks, rows):
                opp, response = label_contexts(r, env["steps"],seat,t,env["configuration"])
                opportunities.append(opp)
                responses.append(response)
            data.extend(rows)
            groups.extend([key]*len(rows))
            rec["by_seat"].append(dict(seat=seat, **cuts))
        records.append(rec)
    if not data:
        raise ValueError("empty independent set")
    excluded_docs[manifest.name] = digest(manifest.read_bytes())
    return data, np.asarray(groups), np.stack(opportunities), np.stack(responses), records, excluded, excluded_docs


@torch.inference_mode()
def score(path, rows, groups, mask, responses, trainer):
    with np.load(path, allow_pickle=False) as w:
        preset,cfg,_ = resolve_preset_metadata(json.loads(str(w["meta"])))
        if preset != "sapling-1.2m":
            raise ValueError("wrong network")
        net = CNet2(**cfg)
        net.load_state_dict({k:torch.from_numpy(w[k]) for k in w.files if k!="meta"}, strict=True)
    net.eval()
    pred, loss = [], []
    for i in range(0,len(rows),16):
        block=rows[i:i+16]
        batch=trainer.to_dev({k:np.stack([r[k] for r in block]) for k in block[0]},aug=False)
        logits=net(batch)[0]
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite logits")
        pred.extend(logits.argmax(-1).numpy())
        loss.extend(torch.nn.functional.cross_entropy(logits.transpose(1,2), batch["y"],ignore_index=-1,reduction="none").numpy())
    y=np.stack([r["y"] for r in rows])
    free=(y>=0) & (np.stack([r["legal"] for r in rows]).sum(-1)>=2)
    hit=np.asarray(pred)==y
    loss=np.asarray(loss)
    def measure(m):
        n=int(m.sum())
        return dict(n=n, games=len(set(groups[m.any(1)])),
                    correct=int(hit[m].sum()), accuracy=float(hit[m].mean()) if n else None,
                    ce=float(loss[m].mean(dtype=np.float64)) if n else None)
    a,cd,td=np.moveaxis(mask,-1,0)
    masks=dict(all=free,attack=a,defense=(cd|td),cargo_defense=cd,tree_defense=td,
               ordinary=~(a|cd|td))
    per_ctx={k:measure(m & free) for k,m in masks.items()}
    for j,k in enumerate(CONTEXTS):
        m=mask[:,:,j] & free
        roots,slots=np.where(m)
        actual=responses[roots,slots,j,y[m]]
        predicted=responses[roots,slots,j,np.asarray(pred)[m]]
        per_ctx[k]["response"]=dict(eligible=int(m.sum()),expert_yes=int(actual.sum()),
            policy_yes=int(predicted.sum()),both_yes=int((actual & predicted).sum()),
            policy_only=int((~actual & predicted).sum()),expert_only=int((actual & ~predicted).sum()))
    return dict(sha256=digest(path.read_bytes()),contexts=per_ctx,
        forced_slots_excluded=int(((y>=0)&~free).sum()),
        by_game={g:{k:measure(free & m & (groups==g)[:,None]) for k,m in masks.items()} for g in dict.fromkeys(groups)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir",type=Path,required=True)
    p.add_argument("--v1",type=Path,default=WEIGHTS/"elite_bc_v1.npz")
    p.add_argument("--v2",type=Path,default=WEIGHTS/"top5_h2h_v2.npz")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists() or any(k.startswith(("BC_", "CNET2_")) for k in os.environ):
        raise ValueError("use fresh output and unset training environment overrides")
    os.environ["BC_DEVICE"]="cpu"
    import cnet2_train as trainer
    torch.set_num_threads(1)
    rows,groups,masks,response,games,excluded,docs=collect(args.raw_dir)
    result=dict(scope="Later seven archived Top Five head-to-head matches, both seats; each seat observed from first tick with its own carbon and public state. Uniform fixed diagnostic ticks shared with v1 checker. Forced slots excluded. First incompatible prefix slot and suffix excluded. No execution or return inference. Game IDs and map seeds checked against both models' train/validation manifests and the earlier direct-3s2 holdout.",
        revision=REVISION, source_sha256={p.name:digest(p.read_bytes()) for p in (
            Path(__file__),HERE/"elite_3s2_agreement.py",HERE.parents[1]/"code/build_sapling_top5_h2h.py")},
        data_manifests_sha256=docs,raw_ids_expected=list(KEYS),ticks=TICKS,
        definitions=DESCRIPTIONS,precision="FP32 CPU",style_slot=STYLE_SLOT,roots=len(rows),
        games=games,excluded=excluded,
        v1=score(args.v1,rows,groups,masks,response,trainer),
        v2=score(args.v2,rows,groups,masks,response,trainer))
    with args.out.open("x") as f:
        f.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for k in ("v1","v2"):
        print(k,result[k]["contexts"],flush=True)


if __name__=="__main__":
    main()
