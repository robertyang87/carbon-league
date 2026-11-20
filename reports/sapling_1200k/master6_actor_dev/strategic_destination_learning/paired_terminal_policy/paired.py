"""Change the served Sapling policy from signed full-team terminal comparisons."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ACTOR = STUDY.parent
ROOT = ACTOR.parents[2]
JRC = ACTOR / "joint_response_critic_dev"
BOOK = JRC / "signed_roots.json.gz"
GROUPS = JRC / "native_inputs.json.gz"
AUDIT = ACTOR / "cargo_tree_action_learning_dev/paired_response_prefix/signed.json.gz"
CONTRACT = HERE / "protocol.json"
spec = importlib.util.spec_from_file_location("released_training_kernel", STUDY / "learner_seat_corrections/model/fit.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
c = old.c
sys.path.insert(0, str(ROOT / "reports/sapling_1200k/root_recovery_branches"))
from branches import begin, clean, public


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


@lru_cache(maxsize=None)
def file_sha(path):
    return c.sha(path)


def margin_delta(row):
    s = row["seat"]
    base, alt = (row[k + "_scores"] for k in ("parent", "alternative"))
    return round((alt[s] - alt[1-s]) - (base[s] - base[1-s]), 2)


def signed_registry(rule, split):
    for name, path in (("release", old.PARENT), ("weights", old.BASE), ("pairs", BOOK), ("split", GROUPS), ("audit", AUDIT)):
        check(file_sha(path) == rule["artifacts"][name], f"changed {name}")
    book = c.read(BOOK)
    groups = {r["case"]: r for r in c.read(GROUPS)["rows"]}
    audit = {r["case"]: r for r in c.read(AUDIT)["rows"]}
    check(len(book["rows"]) == len(groups) == len(audit) == 108, "unmatched original registry")
    maps = set(rule[split + "_seeds"])
    out = []
    for r in book["rows"]:
        if r["seed"] not in maps or r["family"] not in rule["single_order_families"]:
            continue
        g, a = groups[r["case"]], audit[r["case"]]
        check(g["split"] == ("train" if split == "fit" else "held") and g["terrain"] == r["terrain"], "terrain assignment")
        check(r["examiner_sha256"] == book["original_examiner_sha256"] and r["parent_sha256"] == book["frozen_parent_sha256"], "continuation policies")
        check(r["source_sha256"] == a["source_sha256"] == file_sha(ROOT / r["source"]), "source tape")
        check(r["parent"] != r["alternative"] and r["terrain"] == a["terrain"] and
              a["official_scores_physical"] == [r["parent_scores"], r["alternative_scores"]] and
              margin_delta(r) == a["delta_official_margin"], "paired outcome")
        check(a["prefix_verdict"] == "matched_prefix_and_root_enemy" and a["free_from_preexisting_divergence_and_same_root_enemy"], "unmatched prefix or opponent")
        for arm in r["arms"].values():
            check(file_sha(ROOT / arm["action_file"]) == arm["action_sha256"] and file_sha(ROOT / arm["receipt"]) == arm["receipt_sha256"], "signed terminal evidence")
            check(arm["terminal_step"] - r["t"] == arm["calls"] + 1, "suffix not played to ending")
        out.append(r)
    check(len(out) == len({r["identity"] for r in out}) and {r["seed"] for r in out} == maps, "missing or duplicate roots")
    check(len({r["terrain"] for r in out}) == len(maps), "map terrain collision")
    return out


def materialize(rule, split, path):
    """Reconstruct public history at real roots, force both full prefixes independently."""
    rows = signed_registry(rule, split)
    sources = defaultdict(lambda: defaultdict(list))
    for r in rows:
        sources[r["source"]][r["t"]].append(r)
    model = c.package(path)
    reader = old.prepare.teacher_reader(model)
    free = old.module("deployed_actor_for_public_replay", path)
    features = sys.modules["cnet2_feat"]
    data, cards = [], []
    for source, times in sorted(sources.items()):
        raw = c.read(ROOT / source)
        seat = raw["candidate_seat"]
        env = begin(raw)
        f = features.Feat(seat)
        for t in range(max(times) + 1):
            obs = public(env, seat)
            check(set(obs) == old.PUBLIC and (obs["step"], obs["player"]) == (t, seat), "public history")
            x = f.observe(obs)
            # The release's sole persistent game state is _FEAT. Update it on
            # every public observation, and issue the real public call at each signed root.
            if t in times or t == 0:
                sent = old.normalized(free.agent(obs, raw["game_configuration"]))
            else:
                free._FEAT.observe(obs)
            if t in times:
                enc = []
                for net in model._NETS:
                    tokens, ctx = net.encode(x, 2)
                    enc.append((net, tokens, ctx, net.lin(ctx, "h0")))
                plain = model._return_decode(x, f.aux, enc, False)
                guarded = model._return_decode(x, f.aux, enc, True)
                chosen, rejected = c.policy.guard(model, x, plain, guarded)
                check(sent == old.normalized(f.to_actions(chosen)), "normal vs instrumented free orders")
                for r in sorted(times[t], key=lambda v: v["case"]):
                    digest = old.digest({k:v for k,v in obs.items() if k != "remainingOverageTime"})
                    pair = {k:r[k] for k in ("parent", "alternative")}
                    check((r["seat"],r["seed"],r["terrain"],r["parent_sha256"]) ==
                          (seat,raw["seed"],raw["initial_terrain_u8_sha256"],raw["pack_sha256"]), "archive identity")
                    check(r["public_root_sha256"] == digest and
                          old.digest([r["terrain"],seat,t,digest,pair,r["root_opponent"]]) == r["identity"], "root identity")
                    check(clean(raw["actions"][t][seat]) == r["parent"] and
                          clean(raw["actions"][t][1-seat]) == r["root_opponent"], "physical source actions")
                    available = []
                    for name in ("parent","alternative"):
                        ar = c.actual_features(f, x, r[name], model)
                        cut = int(ar.pop("cut"))
                        fixed = np.zeros_like(ar["residual"])
                        prior = np.zeros_like(fixed)
                        reader(x, f.aux, enc, not rejected, ar["y"], fixed, prior, cut)
                        ar.pop("residual")
                        ar.update(fixed=fixed, parent=prior)
                        available.append(not bool(ar["tok_mask"][cut:features.OWN].any()))
                        data.append(ar)
                    cards.append(dict(**{k:r[k] for k in ("seed","terrain","seat","t","case","identity","parent","alternative","parent_scores","alternative_scores","root_opponent","family","role","source","source_sha256","public_root_sha256","response_rng","arms","parent_sha256","examiner_sha256")},
                                      available=available,delta=margin_delta(r),free=sent,guard_rejected=bool(rejected)))
            if t < max(times):
                env.step(raw["actions"][t])
    return {k:np.stack([a[k] for a in data]) for k in data[0]}, cards


def team_statistics(z, prior, b):
    lp = z.log_softmax(-1)
    lp0 = prior.masked_fill(~b["legal"], -1e9).log_softmax(-1)
    weight = b["eligible"].float()
    index = b["y"].clamp_min(0).unsqueeze(-1)
    joint = (lp.gather(-1,index).squeeze(-1) * weight).sum(1)
    joint0 = (lp0.gather(-1,index).squeeze(-1) * weight).sum(1)
    kl = ((lp0.exp()*(lp0-lp)).sum(-1)*weight).sum(1)
    return joint, joint0, kl


def fit(rule):
    data, cards = materialize(rule,"fit",old.PARENT)
    valid = [i for i,r in enumerate(cards) if all(r["available"])]
    counts = Counter(cards[i]["seed"] for i in valid)
    check(len(counts) == len(rule["fit_seeds"]) and len(valid) >= rule["min_legal_train"], "insufficient complete full orders")
    map_w = np.array([len(valid)/(len(counts)*counts[r["seed"]]) for r in cards],np.float32)
    d = np.array([r["delta"]/rule["scale_official"] for r in cards],np.float32)
    m=c.package(old.PARENT)
    torch.manual_seed(rule["seed"])
    net = c.torch_model(m,1)
    for k,p in net.named_parameters():
        p.requires_grad_(not k.startswith(old.native.VALUE))
    params=[p for p in net.parameters() if p.requires_grad]
    drift=old.parity(data,net)
    optimizer=torch.optim.AdamW(params, lr=rule["lr"], weight_decay=rule["weight_decay"])
    rng=np.random.default_rng(rule["seed"])
    history=[]
    for epoch in range(rule["epochs"]):
        order=rng.permutation(valid)
        for pos in range(0,len(order),rule["batch_roots"]):
            ix=order[pos:pos+rule["batch_roots"]]
            ids=(ix[:,None]*2+np.arange(2)).reshape(-1)
            b=old.native.batch(data,ids)
            new,prior,kl=team_statistics(old.logits(net,b),torch.from_numpy(data["parent"][ids]),b)
            pred=(new-prior).reshape(-1,2).diff(dim=1).squeeze(1)
            w=torch.from_numpy(map_w[ix])
            mse=(w*(pred-torch.from_numpy(d[ix])).square()).mean()
            penalty=(w*kl.reshape(-1,2).mean(1)).mean()
            loss=mse+rule["kl"]*penalty
            check(bool(torch.isfinite(loss)), "nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(params,rule["clip"],error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(epoch=epoch,n=len(ix),mse=float(mse.detach()),kl=float(penalty.detach()),grad=float(grad)))
    with patch.object(old,"HERE",HERE):
        result=old.export(net,m)
    c.save(HERE/"training_pairs.json.gz",cards)
    c.save(HERE/"fit.json",dict(protocol=c.sha(CONTRACT),trainer=c.sha(Path(__file__)),source_cards=c.sha(HERE/"training_pairs.json.gz"),
           allocated=len(cards),usable=len(valid),by_seed=counts,positive=int(sum(d[i]>0 for i in valid)),zero=int(sum(d[i]==0 for i in valid)),
           negative=int(sum(d[i]<0 for i in valid)),params=sum(p.numel() for p in params),dequantized_parity=drift,updates=history,**result))


def team_prefix_lp(data):
    scores=np.where(data["legal"],data["parent"],-1e9).astype(np.float64)
    logp=scores-scores.max(-1,keepdims=True)
    logp-=np.log(np.exp(logp).sum(-1,keepdims=True))
    selected=np.take_along_axis(logp,data["y"].clip(min=0)[...,None],axis=-1)[...,0]
    return (selected*data["eligible"]).sum(1).reshape(-1,2)


def verdict(rows,rule):
    def summarize(group):
        supported=[r for r in group if all(r["available"])]
        changed=[r for r in group if r["free_changed"]]
        priced=[r for r in changed if r["free_margin_change"] is not None]
        return dict(n=len(group),legal=len(supported),mse_reference=float(np.mean([(r["delta"]/rule["scale_official"])**2 for r in supported])) if supported else None,
                    mse_trained=float(np.mean([(r["delta"]/rule["scale_official"]-r["prediction"])**2 for r in supported])) if supported else None,
                    two_choice_improvement=float(np.mean([r["two_choice_margin_delta"] for r in supported])) if supported else None,
                    changed=len(changed),priced_changed=len(priced),unpriced_changed=len(changed)-len(priced),
                    within_pair_parent=sum(r["free_parent_arm"] is not None for r in group),within_pair_trained=sum(r["free_trained_arm"] is not None for r in group),
                    same_free_order=len(group)-len(changed),priced_delta=sum(r["free_margin_change"] for r in priced))
    maps={s:summarize([r for r in rows if r["seed"]==s]) for s in rule["held_seeds"]}
    all_stats=summarize(rows)
    legal_maps=[a for a in maps.values() if a["legal"]]
    risk_reduction=float(np.mean([a["mse_reference"]-a["mse_trained"] for a in legal_maps])) if legal_maps else None
    conditional_gain=float(np.mean([a["two_choice_improvement"] for a in legal_maps])) if legal_maps else None
    required=rule["screen"]
    gates=dict(all_maps_legal=len(legal_maps)==len(maps),
               risk_improved=risk_reduction is not None and risk_reduction>0,
               majority_map_risk=sum(a["mse_trained"]<a["mse_reference"] for a in legal_maps)>len(maps)/2,
               two_choice_positive=conditional_gain is not None and conditional_gain>0,
               enough_free_evidence=all_stats["priced_changed"]>=required["min_priced_changes"] and
                   sum(a["priced_changed"]>0 for a in maps.values())>=required["min_priced_maps"],
               priced_positive=all_stats["priced_delta"]>0,
               no_unknown_changed=(all_stats["unpriced_changed"]==0))
    return dict(by_map=maps,pooled=all_stats,map_equal_mse_reduction=risk_reduction,
                map_equal_scored_two_choice_gain=conditional_gain,gates=gates,qualifies_for_development=all(gates.values()))


def assess(rule):
    f=c.read(HERE/"fit.json")
    trained=HERE/"trained.py"
    check(c.sha(trained)==f["package"] and c.sha(HERE/"weights.npz")==f["weights"] and c.sha(CONTRACT)==f["protocol"],"frozen fit")
    old_data,base=materialize(rule,"held",old.PARENT)
    new_data,child=materialize(rule,"held",trained)
    prior=team_prefix_lp(old_data);new=team_prefix_lp(new_data)
    rows=[]
    for i,(a,b) in enumerate(zip(base,child,strict=True)):
        check(all(a[k]==b[k] for k in a if k not in ("free","guard_rejected")), "held input or labels changed")
        def which(cmd):
            for n,k in enumerate(("parent","alternative")):
                if old.normalized(a[k])==cmd:
                    return n
            return None
        left,right=which(a["free"]),which(b["free"])
        changed=a["free"]!=b["free"]
        delta=a["delta"]
        priced=0. if not changed else ((right-left)*delta if left is not None and right is not None else None)
        complete=all(a["available"])
        rows.append(dict(**a,trained_free=b["free"],free_changed=changed,free_parent_arm=left,free_trained_arm=right,free_margin_change=priced,
                        bc_joint_logp=prior[i].tolist() if complete else None,trained_joint_logp=new[i].tolist() if complete else None,
                        prediction=float(np.diff(new[i]-prior[i])[0]) if complete else None,
                        two_choice_margin_delta=(int(new[i,1]>new[i,0])-int(prior[i,1]>prior[i,0]))*delta if complete else None,
                        trained_guard_rejected=b["guard_rejected"]))
    c.save(HERE/"held_actions.json.gz",rows)
    c.save(HERE/"assessment.json",dict(fit=c.sha(HERE/"fit.json"),rows=c.sha(HERE/"held_actions.json.gz"),new_games=0,
                                       **verdict(rows,rule)))


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("stage",choices=["fit","assess"])
    stage=args.parse_args().stage
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rule=c.read(CONTRACT)
    (fit if stage=="fit" else assess)(rule)
