"""Identify the partial support of a deterministic joint policy in the original return archive."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parents[1]
BOOK = ACTOR / "joint_response_critic_dev/signed_roots.json.gz"


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {"n":len(values),"minimum":min(values),"mean":mean(values),"maximum":max(values)} if values else {"n":0}


def summarize(rows):
    def value(key):
        return [r[key] for r in rows]
    own, rival = [], []
    for r in rows:
        s = r["seat"]
        own.append(r["alternative_scores"][s]-r["parent_scores"][s])
        rival.append(r["alternative_scores"][1-s]-r["parent_scores"][1-s])
    return dict(rows=len(rows),terrain=len({r["terrain"] for r in rows}),seats=Counter(value("seat")),family=Counter(value("family")),
                changed_role=Counter(value("role")),ordering_sign=Counter(("positive" if x>0 else "negative" if x<0 else "zero") for x in value("delta")),
                delta_official_margin=stats(value("delta")),delta_own=stats(own),delta_rival=stats(rival),
                own_up_rival_up_net_down=sum(x>0 and y>x for x,y in zip(own,rival)),
                own_down_but_rival_down_more=sum(x<0 and y<x for x,y in zip(own,rival)))


def observed_changes(rows):
    sys.path.insert(0,str(ACTOR.parent/"root_recovery_branches"))
    from branches import begin, public
    from collections import defaultdict
    sources=defaultdict(list)
    for r in rows:
        if r["free_changed"]:
            sources[r["source"]].append(r)
    result=[]
    for name,subset in sources.items():
        raw=read(ACTOR.parents[2]/name)
        env=begin(raw)
        seat=raw["candidate_seat"]
        needed={r["t"]+i for r in subset for i in (-10,-5,-3,0)}
        history={}
        spans=int(raw["game_configuration"]["size"])
        def dist(a,b):
            dx,dy=abs(a%spans-b%spans),abs(a//spans-b//spans)
            return min(dx,spans-dx)+min(dy,spans-dy)
        for t in range(max(needed)+1):
            if t in needed:
                o=public(env,seat)
                history[t]=(json.loads(json.dumps(o["players"])))
            if t<max(needed):
                env.step(raw["actions"][t])
        for r in subset:
            def differences(first,second):
                return sorted(k for k in first.keys()|second.keys() if first.get(k,"NONE")!=second.get(k,"NONE"))
            pr=r["free"]
            new=r["trained_free"]
            changed=differences(pr,new)
            signed=differences(r["parent"],r["alternative"])
            changed_set=sorted(set(changed+signed))
            context=[]
            for offset in (-10,-5,-3,0):
                ours,rival=(history[r["t"]+offset][s] for s in (seat,1-seat))
                pieces={}
                for uid in changed_set:
                    entry=ours[2].get(uid)
                    if entry is None:
                        pieces[uid]=None
                        continue
                    cell,cargo,role=entry
                    closest=sorted((dist(cell,w[0]),k,w[1],w[2]) for k,w in rival[2].items())
                    pieces[uid]=dict(role=role,cell=cell,cargo=cargo,own_home_distance=min(dist(cell,k) for k in ours[1].values()),
                               nearest_rival=list(closest[0]) if closest else None,
                               own_tree_here=any(v[0]==cell for v in ours[3].values()),rival_tree_here=any(v[0]==cell for v in rival[3].values()))
                context.append(dict(offset=offset,own_cash=ours[0],rival_cash=rival[0],own_workers=len(ours[2]),rival_workers=len(rival[2]),
                                    own_trees=len(ours[3]),rival_trees=len(rival[3]),affected_units=pieces))
            result.append(dict(seed=r["seed"],seat=seat,t=r["t"],source=r["source"],public_sha256=r["public_root_sha256"],family=r["family"],
                 old_order=pr,trained_order=new,signed_alternative=r["alternative"],signed_change=signed,free_change=changed,
                 shared_changed_uid=sorted(set(changed)&set(signed)),registered_pair_official_delta=r["delta"],
                 old_arm=r["free_parent_arm"],trained_arm=r["free_trained_arm"],verified_increment=r["free_margin_change"],
                 source_pre_action_context=context))
    return result


def main():
    contract = read(HERE/"protocol.json")
    historical = read(BOOK)
    train,held = (read(HERE/f) for f in ("training_pairs.json.gz","held_actions.json.gz"))
    cohort = {r["identity"]:r for r in train+held}
    primaries = {r["identity"]:r for r in historical["rows"]}
    repeats=[]
    for r in historical["repeated_streams"]:
        p=primaries[r["identity"]]
        if any(r[k]!=p[k] for k in ("seed","terrain","parent","alternative","root_opponent","examiner_sha256","parent_sha256")):
            raise ValueError("repeat does not match its primary intervention")
        s=r["seat"]
        a,b=(r[n+"_scores"] for n in ("parent","alternative"))
        delta=round((b[s]-b[1-s])-(a[s]-a[1-s]),2)
        a,b=(p[n+"_scores"] for n in ("parent","alternative"))
        primary_delta=round((b[s]-b[1-s])-(a[s]-a[1-s]),2)
        repeats.append(dict(identity=r["identity"],case=r["case"],included_primary=r["identity"] in cohort,seed=r["seed"],
                            primary_delta=primary_delta,repeat_delta=delta,sign_changed=delta*primary_delta<0,
                            rng=r["response_rng"],primary_rng=p["response_rng"],terminal_receipts=r["arms"]))
    out=dict(hashes={f:sha(HERE/f) for f in ("protocol.json","fit.json","assessment.json","training_pairs.json.gz","held_actions.json.gz")},
             official_margin="(alternative own - alternative rival) - (parent own - parent rival); physical acting seat, native official termination",
             counts=dict(signed_primary=len(primaries),selected_fit=len(train),selected_held=len(held),
                         excluded_outside_maps=sum(r["seed"] not in contract["fit_seeds"]+contract["held_seeds"] for r in primaries.values()),
                         excluded_outside_oneshot_family_on_selected_maps=sum(r["seed"] in contract["fit_seeds"]+contract["held_seeds"] and
                             r["family"] not in contract["single_order_families"] for r in primaries.values())),
             repeats_excluded_from_all_gradients_and_means=repeats,allocated_fit=summarize(train),allocated_held=summarize(held),
             changed_free_actions=observed_changes(held))
    (HERE/"economics.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
