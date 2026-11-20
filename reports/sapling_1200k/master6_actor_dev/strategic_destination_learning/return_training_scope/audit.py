"""Count conditional score-function evidence and test both owned towers against one real original tape."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn.functional import log_softmax

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]/"cargo_tree_action_learning_dev"
LEAGUE = EXPERIMENT/"native_actor_corpus/league_policy"
INTAKE = LEAGUE/"native_return_intake"
sys.path.insert(0, str(INTAKE))
import learning
o = learning.o


def range_(values):
    return [min(values), max(values)]


def game(seed, seat, terrain, scores, digest, n, picks, advantage=None):
    m,u = o.payoff(scores, seat)
    return dict(seed=seed, seat=seat, terrain=terrain, physical_tape_sha256=digest, official_margin=m,
                u=u, wdl=1 if m>0 else .5 if m==0 else 0, supported=n, alternate=picks, advantage=advantage)


def describe(rows):
    advantages = [a["advantage"] for a in rows if a["advantage"] is not None]
    return dict(episodes=len(rows), physical_maps=len({a["terrain"] for a in rows}),
                map_seats=len({(a["terrain"],a["seat"]) for a in rows}), wins=sum(a["wdl"]==1 for a in rows),
                draws=sum(a["wdl"]==.5 for a in rows), losses=sum(a["wdl"]==0 for a in rows),
                margin_range=range_([a["official_margin"] for a in rows]), utility_range=range_([a["u"] for a in rows]),
                paired_advantage_range=range_(advantages) if advantages else None,
                nonzero_advantage_episodes=sum(abs(a)>1e-14 for a in advantages),
                score_terms=sum(a["supported"] for a in rows), chosen_nondefault=sum(a["alternate"] for a in rows),
                rows=rows)


def policy_census():
    path=INTAKE/"hosted/training_manifest.json"
    plan=o.read(INTAKE/"learning.json")
    manifest,eps=learning.training_manifest(plan,path)
    new,old,controls,perturbed=[],[],[],[]
    priors=[]; count=Counter()
    for e in eps:
        options=0; switched=0
        for d in e["events"]:
            if not d["supported"]:
                o.require(len(d["options"])==1 and d["issued"]==d["options"][0], "invalid singleton")
                continue
            pair=d["options"]; pick=d["choice"]; p=d["probabilities"]
            o.require(len(pair)==len(p)==2 and pair[0]!=pair[1] and d["issued"]==pair[pick] and
                      all(0<x<1 for x in p) and abs(sum(p)-1)<1e-12 and
                      abs(math.log(p[pick])-d["logp"])<1e-12, "invalid physical choice")
            l=np.array(d["native_logits"],float);pred=np.exp(l-l.max());pred/=pred.sum()
            o.require(np.max(np.abs(pred-p))<1e-12, "deployed softmax mismatch")
            priors.append(p[0]);count[d["role"]]+=1;options+=1;switched+=pick
        new.append(game(e["seed"],e["seat"],e["terrain"],e["official_scores_by_seat"],e["tape_sha256"],options,switched,e["advantage"]))
    earlier=o.read(LEAGUE/"native_gradient/alignment.json")["original_inputs"]
    rootdir=EXPERIMENT/"onpolicy_native"
    for g in earlier["games"]:
        s,seat=g["seed"],g["seat"]; traces=[]
        for ev in g["opportunities"]:
            trace=o.read(rootdir/"traces"/f"{s}_{seat}"/f"{ev['t']:03d}.json")
            if ev["supported"]:
                q=trace["probabilities"]; pick=trace["choice"]
                o.require(trace["options"][pick]==trace["issued"] and trace["options"][0]!=trace["options"][1] and
                          abs(math.log(q[pick])-trace["logp"])<1e-12 and 0<q[0]<1 and 0<q[1]<1,
                          "old sampled return has no matching full command and propensity")
                traces.append(pick)
        sampled=g["sampled"]; base=g["baseline"]
        old.append(game(s,seat,g["terrain"],sampled["scores"],sampled["tape_sha256"],len(traces),sum(traces),g["sampled_u"]-g["baseline_u"]))
        controls.append(game(s,seat,g["terrain"],base["scores"],base["tape_sha256"],0,0))
    history=o.read(INTAKE/"win_loss_sources/inventory.json.gz")
    paths=[]
    for r in history["rows"]:
        if r["candidate_return_use"] != "previous_paired_perturbation_terminal_fit":
            continue
        path=o.ROOT/r["source"]; raw=o.read(path)
        o.require((raw["seed"],raw["candidate_seat"],raw["initial_terrain_u8_sha256"],raw["pack_sha256"])==
                  (r["seed"],r["seat"],r["terrain"],r["candidate_sha256"]) and
                  len(raw["actions"])==r["transitions"] and raw["official_score_by_seat"]==r["official_scores_by_seat"] and
                  not raw["errors"],"perturbation source mismatch")
        paths.append(r["registration"])
        row=game(r["seed"],r["seat"],r["terrain"],r["official_scores_by_seat"],r["actions_sha256"],0,0)
        row.update(source=r["source"],registration=r["registration"],actor=r["candidate_sha256"],
                   observed_joint_steps=len(raw["actions"]))
        perturbed.append(row)
    pools=[set(r["terrain"] for r in group) for group in (new,old,perturbed)]
    o.require((len(new),len(old),len(perturbed))==(24,8,20) and len(set.union(*pools))==sum(map(len,pools)),
              "unexpected signed cohorts or overlapping geography")
    summary=dict(intake_851c=describe(new), historical_8da4_sampler=describe(old),
                 historical_greedy_baseline_only=describe(controls), parameter_perturbation=describe(perturbed),
                 maps_across_three_training_sets=len(set.union(*pools)),map_overlap=0,
                 intake_851c_policy=manifest["policy_sha256"], earlier_sampler=earlier["sampled_package"],
                 earlier_greedy_parent=earlier["parent_package"], candidate0_probability_range=range_(priors),
                 active_opportunities_by_role=dict(count), intake_scheduled=sum(len(e["events"]) for e in eps),
                 earlier_scheduled=earlier["scheduled"],perturbation_registrations=sorted(set(paths)))
    return summary,eps,plan


def one_gradient(e,plan):
    """Use actual signed option index and full public observations. No sampling or game against an agent."""
    base=o.package(learning.BEHAVIOR)
    roots,sites=learning.replay_episode(e,base,o.teacher_reader(base),plan)
    nets=[o.torch_model(base,i) for i in (0,1)]
    data,rows=learning.a.network_data(roots)
    b=o.tensorbatch(data,np.arange(len(rows)))
    idx=torch.arange(len(rows))
    slots=torch.tensor([r["slot"] for r in rows])
    acts=torch.tensor([r["moves"] for r in rows])
    choice=torch.tensor([r["pick"] for r in rows])
    mix_weights=[base._HYBRID["expert_weight"],base._HYBRID["student_weight"]]
    proj=[net(b)[0][idx,slots].gather(1,acts).double() for net in nets]
    correction=[]
    for (record,meta,public) in roots:
        slot=meta["slot"];x,aux,_=public
        residual=np.zeros(5)
        if meta["route"] and x["tok_kind"][slot]==0:
            residual=base._return_basis(x,aux,slot,record["mf"][slot])@base._RETURN_THETA
        correction.append(residual[meta["moves"]])
    z=mix_weights[0]*proj[0]+mix_weights[1]*proj[1]+torch.tensor(np.array(correction),dtype=torch.float64)
    lp=log_softmax(z,-1);observed=np.array([r["old_logp"] for r in rows])
    probs=np.array([r["observed_p0"] for r in rows])
    p_error=float(np.max(abs(lp[:,0].exp().detach().numpy()-probs)))
    lp_error=float(np.max(abs(lp[idx,choice].detach().numpy()-observed)))
    o.require(p_error<5e-5 and lp_error<1e-4,"dual tower is not the deployed distribution")
    # Analytic derivative with respect to both real policy readout biases for each selected physical order.
    # Every direct focal point is a worker. Subtracting alternative indices cancels any legal-logit centering.
    tests=[]
    for i,r in enumerate(rows):
        a,bact=r["moves"];pick=r["pick"];observed_p=[probs[i],1-probs[i]]
        chosen=[a,bact][pick];other=[a,bact][1-pick]
        gradients=torch.autograd.grad(lp[i,pick], [net.whead.bias for net in nets],retain_graph=True)
        row=dict(step=r["step"], slot=r["slot"], physical_choice=pick, chosen_action=chosen, old_logp=r["old_logp"], towers=[])
        for w,g in zip(mix_weights,gradients):
            expected=np.zeros(5);expected[chosen]=w*(1-observed_p[pick]);expected[other]=-expected[chosen]
            error=float(np.max(abs(g.detach().numpy()-expected)))
            o.require(error<5e-5, "incorrect analytical physical log-odds gradient")
            row["towers"].append(dict(weight=w, chosen_bias_derivative=float(g[chosen]), alternative_bias_derivative=float(g[other]), max_bias_error=error))
        tests.append(row)
    objective=e["advantage"]*lp[idx,choice].sum()/len(plan["scheduled_turns"])
    objective.backward()
    norms=[]
    for net in nets:
        groups={}
        for name,param in net.named_parameters():
            if param.grad is not None:
                k=name.split(".")[0];groups[k]=groups.get(k,0.)+param.grad.detach().double().square().sum().item()
        norms.append({k:math.sqrt(v) for k,v in groups.items()})
        o.require(all(norms[-1].get(k,0)>0 for k in ("stem","gru","whead")), "trunk/decoder/worker path missing")
    return dict(seed=e["seed"], seat=e["seat"], arm=e["arm"], official_scores=e["official_scores_by_seat"],
                terrain=e["terrain"], input_archive_sha256=e["archive_sha256"],tape_sha256=e["tape_sha256"],
                audited_transitions=e["official_transitions"], scheduled=len(sites), supported=len(rows),
                max_selected_logp_error=lp_error,max_probability_error=p_error,score_bias_checks=tests,
                return_weighted_gradient_norms=dict(it50=norms[0],sapling=norms[1]),
                comparison="Frozen t0 behavior at actual two-order menu; physical menu transitions and all ordinary greedy outputs are piecewise deterministic.",
                optimizer_steps=0,greedy_and_singleton_score=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(621758460)
    counts,eps,plan=policy_census()
    episode=next(e for e in eps if e["seed"]==621758460 and e["seat"]==0 and e["arm"]=="draw_a")
    result=dict(census=counts,dual_tower_replay=one_gradient(episode,plan),
                inputs={str(p.relative_to(o.ROOT)):o.sha(p) for p in (INTAKE/"hosted/training_manifest.json",
                    INTAKE/"hosted/settlement.json",INTAKE/"learning.py",LEAGUE/"native_gradient/alignment.json",
                    INTAKE/"win_loss_sources/inventory.json.gz",LEAGUE/"candidate.py",learning.BEHAVIOR,Path(__file__))})
    o.save(HERE/"audit.json",result)
    print("Wrote audit",o.sha(HERE/"audit.json"),flush=True)
