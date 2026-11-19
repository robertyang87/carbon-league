"""Fit only original-response labels; evaluate cross-map and next-game intervention without tuning."""
import argparse
from collections import defaultdict
import math
import numpy as np
import actions
import policy
from core import HERE, OLD, HOLD_SEEDS, read, save, sha

NATIVE=HERE/"native"

def labelled(split):
    res=read(NATIVE/f"{split}_labels.json")
    assert res["source_protocol_sha256"]==sha(HERE/"native_input.json") and res["runner_sha256"]==sha(NATIVE/"run.py")
    assert res["verifier_sha256"]==sha(NATIVE/"labels.py") and res["counter_sha256"]==sha(OLD/"evidence/account.py")
    cards=read(HERE/"cards.json")["cases"]
    expected={c["id"]:c for c in cards if c["split"]==split}
    if split=="held":
        sel=read(HERE/"held_selection.json")
        expected={i:expected[i] for i in sel["active_ids"]}
        assert res["held_selection_sha256"]==sha(HERE/"held_selection.json")
    assert len(res["cases"])==len(expected) and {r["id"] for r in res["cases"]}==set(expected)
    for r in res["cases"]:
        c=expected[r["id"]]
        assert all(c[k]==r[k] for k in ("seat","seed","map","t","proposal","state_id","diagnostic","stream"))
        assert all(np.isclose(r["delta"][k],r["arms"]["alt"][k]-r["arms"]["base"][k]) for k in policy.TARGETS)
    return res["cases"]


def fitted(rows):
    assert rows and not any(r["diagnostic"] for r in rows)
    maps=[r["map"] for r in rows]
    weight=np.array([1./(len(set(maps))*maps.count(m)) for m in maps])
    model={"enabled":True,"features":list(actions.FEATURES),"kinds":list(actions.KINDS)}
    x=np.asarray([policy.vector(r["proposal"],model) for r in rows],dtype=np.float64)
    y=np.asarray([[r["delta"][k] for k in policy.TARGETS] for r in rows],dtype=np.float64)
    coefs=np.linalg.solve((x.T*weight)@x+.5*np.eye(x.shape[1]),(x.T*weight)@np.clip(y,-400.,400.))
    assert np.isfinite(coefs).all()
    model["heads"]={k:coefs[:,i].tolist() for i,k in enumerate(policy.TARGETS)}
    return model


def aggregate(rows):
    return {k:math.fsum(r["delta"][k] for r in rows)/len(rows) for k in policy.TARGETS}


def summary(rows):
    bymap=defaultdict(list)
    for r in rows:bymap[r["map"]].append(r)
    map_mean={k:aggregate(v) for k,v in bymap.items()}
    return map_mean,{k:math.fsum(r[k] for r in map_mean.values())/len(map_mean) for k in policy.TARGETS}


def fit():
    allrows=labelled("train")
    primary=[r for r in allrows if not r["diagnostic"]]
    repeat=[r for r in allrows if r["diagnostic"]]
    raw=read(HERE/"support.json")
    assert raw["eligible"] and len(primary)==raw["train_primary"] and len(repeat)==raw["train_diagnostic"]==2
    zero=dict.fromkeys(policy.TARGETS,0.0)
    primary_by_id={r["id"]:r for r in primary}
    repetitions=[{"primary_id":r["primary_id"],"repetition":r["id"],"stream0":primary_by_id[r["primary_id"]]["delta"],
                  "stream1":r["delta"],"sign_flip":{k:primary_by_id[r["primary_id"]]["delta"][k]*r["delta"][k]<0 for k in policy.TARGETS}} for r in repeat]
    loo=[]
    for m in sorted({r["map"] for r in primary}):
        omit=fitted([r for r in primary if r["map"]!=m])
        for r in (r for r in primary if r["map"]==m):
            p=r["proposal"]; hit=policy.choose([p],omit) is not None
            loo.append({"id":r["id"],"map":m,"seed":r["seed"],"seat":r["seat"],"t":r["t"],"kind":p["kind"],
                        "predicted":policy.predictions(p,omit),"hit":hit,"delta":r["delta"] if hit else zero,"actual_delta":r["delta"]})
    bymap,means=summary(loo)
    hits=[r for r in loo if r["hit"]]
    checks={"three_hits":len(hits)>=3,"two_hit_maps":len({r["map"] for r in hits})>=2,
            "margin_up":means["margin"]>0,"principal_preserved":means["principal"]>=0,
            "own_delivery_preserved":means["own_delivery"]>=0,"opponent_delivery_bounded":means["enemy_delivery"]<=5}
    model=fitted(primary)
    model.update(training_labels_sha256=sha(NATIVE/"train_labels.json"),protocol_sha256=sha(HERE/"protocol.json"))
    save(HERE/"weights.json",model)
    save(HERE/"fit_verdict.json",{"decision":"PASS" if all(checks.values()) else "HOLD","checks":checks,
                                 "weights_sha256":sha(HERE/"weights.json"),"train_primary":len(primary),"by_map":bymap,"mean":means,
                                 "leave_one_map_out":loo,"paired_rng_diagnostics":repetitions})
    print("train",checks,means)


def select():
    assert read(HERE/"fit_verdict.json")["decision"]=="PASS"
    model=read(HERE/"weights.json")
    grouped=defaultdict(lambda:defaultdict(list))
    for c in read(HERE/"cards.json")["cases"]:
        if c["split"]=="held":grouped[c["seed"],c["seat"]][c["t"]].append(c)
    registered={(s,p) for s in HOLD_SEEDS for p in (0,1)}
    assert not set(grouped)-registered
    selected=[]; flow=[]
    for seed,seat in sorted(registered):
        hit=None
        for t,cards in sorted(grouped[seed,seat].items()):
            options=sorted(cards,key=lambda c:c["priority_rank"])
            p=policy.choose([c["proposal"] for c in options],model)
            if p is not None:
                hit=next(c for c in options if p==c["proposal"])
                selected.append(hit["id"]);break
        flow.append({"seed":seed,"seat":seat,"first_changed_card":hit and hit["id"],"raw_eligible_turns":len(grouped[seed,seat])})
    save(HERE/"held_selection.json",{"weights_sha256":sha(HERE/"weights.json"),"active_ids":selected,"all_registered":flow,
                                    "cards_sha256":sha(HERE/"cards.json")})


def held():
    rows=labelled("held")
    get={(r["seed"],r["seat"]):r for r in rows}
    panel=[]
    for row in read(HERE/"source/inventory.json")["games"]:
        if row["split"]!="held":continue
        r=get.get((row["seed"],row["seat"]))
        panel.append({k:row[k] for k in ("seed","seat","map")}|{"id":r and r["id"],"delta":r["delta"] if r else dict.fromkeys(policy.TARGETS,0.)})
    assert len(panel)==4 and len(rows)==len(get)
    maps,m=summary(panel)
    seats={str(s):aggregate([r for r in panel if r["seat"]==s]) for s in (0,1)}
    checks={"two_changed_maps":len({r["map"] for r in rows})>=2,"overall_margin_up":m["margin"]>0,
            "each_seat_safe":min(v["margin"] for v in seats.values())>=0,"principal_preserved":m["principal"]>=0,
            "team_collections_up":m["own_delivery"]>0,"enemy_collections_do_not_grow":m["enemy_delivery"]<=0,
            "packaged_actions_verified":read(HERE/"held_deploy_validation.json")["pass"] is True}
    save(HERE/"held_verdict.json",{"decision":"PASS" if all(checks.values()) else "HOLD","checks":checks,"all_seats":panel,
                                  "by_map":maps,"by_seat":seats,"mean":m,"weights_sha256":sha(HERE/"weights.json"),
                                  "package_sha256":sha(HERE/"candidate.py"),"label_sha256":sha(NATIVE/"held_labels.json")})
    print("held",checks,m)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["fit","select","held"]);arg=ap.parse_args()
    {"fit":fit,"select":select,"held":held}[arg.mode]()
