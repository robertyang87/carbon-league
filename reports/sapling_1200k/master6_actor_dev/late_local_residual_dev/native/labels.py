"""Independently recompute every official score, continuation order and attributable cargo flow."""
from collections import Counter
import argparse
import sys
import run

H,DEV=run.HERE,run.DEV
sys.path.insert(0,str(DEV/"evidence"))
import account
sys.path.insert(0,str(run.ROOT/"code"))
from fastsim import FastCarbon


def check_arm(d, row, root, key, snapshot):
    s=row["seat"]; op=1-s; t=root["t"]
    base=d/f"{key}.json.gz"; out=run.read(base); tape=run.source(row); joint=out["orders"][0]
    run.check(joint[s]==root[key] and joint[op]==root["root_enemy"]==tape["actions"][t][op] and
              out["root_digest"]==run.dig(root) and out["rng_seed"]==root["paired_seed"],"different simultaneous root")
    cert=account.account_arm(snapshot,tape["game_configuration"],out["orders"],s,[root["proposal"]["uid"]])
    run.check(cert["official_scores_physical"]==out["official_scores"] and cert["official_margin"]==out["official_margin"]
              and cert["terminal_cash_physical"]==out["final_cash"] and cert["terminal_t"]==out["final_step"],
              "independent source engine/official score mismatch")
    env=run.live.begin(tape);actor,_,_=run.fresh_actor()
    def own():
        return actor.agent(run.live.public(env,s),tape["game_configuration"])
    for i,j in enumerate(tape["actions"][:t]):
        if i==0:run.check(own()==j[s],"parent opening")
        else:actor._FEAT.observe(run.live.public(env,s))
        env.step(j)
    run.check(own()==root["base"],"common full prefix")
    for i,j in enumerate(out["orders"]):
        if i:
            turn=out["search"][i-1]
            run.check(turn["step"]==t+i and turn["own_public_sha256"]==run.dig(run.live.public(env,s)) and
                      turn["enemy_public_sha256"]==run.dig(run.live.public(env,op)) and own()==j[s],"future policy or public state mismatch")
        run.check(not env.done,"post-terminal orders")
        env.step(j)
    run.check(env.done and [z.reward for z in env.states]==out["official_scores"] and out["original_calls"]==len(out["orders"])-1 and
              [z.status for z in env.states]==out["final_status"]==["DONE","DONE"],"unverified terminal")
    p=cert["per_worker"][root["proposal"]["uid"]]; team=cert["team"]; econ=team["treasury_flows_physical"]
    measures={"margin":out["official_margin"],"U":out["U"],"principal":p["strict_principal_delivered"],
              "mined":p["strict_fresh_mined"],"new_direct":p["strict_fresh_delivered"],"own_delivery":team["own_delivery"],
              "focal_delivery":team["focal_delivery"],"others_delivery":team["other_delivery"],
              "team_fresh_direct":team["strict_new_delivered"],"enemy_delivery":econ[op].get("own_delivery",0.),
              "our_terminal_score":out["official_scores"][s],"enemy_terminal_score":out["official_scores"][op]}
    payload={"account":cert,"arm_evidence_sha256":run.sha(base),"measures":measures,"search_calls":out["original_calls"]}
    location=d/f"{key}_account.json.gz"
    run.freeze(location,payload)
    return measures,run.sha(location)


def existing_duplicate(row,root):
    d=(run.ROOT/row["raw"]).parent.parent
    matched=[]
    for folder in d.glob(f'case_{row["seed"]}_{row["seat"]}_*'):
        p=folder/"root.json.gz"
        if not p.exists():continue
        q=run.read(p)
        if q["public_at_root"]["step"]==root["t"] and q["v1"]==root["base"] and q["alternative"]==root["alt"]:
            matched.append(str(p.relative_to(run.ROOT)))
    return matched

def main(split):
    authority=run.read(H/f"{split}_complete.json")
    expected=run.identities()
    run.check(authority["identities"]==expected,"wrong identity")
    weights=run.read(DEV/"weights.json") if split=="held" else None
    output=[]
    for row in run.read(DEV/"protocol.json")["source_groups"]:
        if row["split"]!=split:continue
        key=f'{row["seed"]}_{row["seat"]}'; record=run.read(H/f"selection_{key}.json");root=record["root"]
        line={"seed":row["seed"],"seat":row["seat"],"map":row["map"],"split":split,"t":None,
              "selection_sha256":run.sha(H/f"selection_{key}.json")}
        output.append(line)
        if not root:
            line["status"]="no_eligible_root";continue
        line.update(t=root["t"],proposal=root["proposal"])
        if split=="held":
            p=run.runtime.predict(root["proposal"],weights) if weights.get("enabled") else {}
            if not (p and p["margin"]>10 and p["principal"]>=-5 and p["own_delivery"]>=-5):
                line.update(status="frozen_model_kept_parent",delta={name:0.0 for name in ("margin","principal","own_delivery","new_direct","others_delivery","enemy_delivery")})
                continue
        run.check(not existing_duplicate(row,root),"previous paired work exists; stop and reconcile")
        tape=run.source(row)
        sim=FastCarbon(tape["game_configuration"]).reset(seed=row["seed"])
        for j in tape["actions"][:root["t"]]:sim.step(*j)
        snap=sim.get_state()
        results={}; hashes={}
        for arm in ("base","alt"):
            results[arm],hashes[arm]=check_arm(H/f"{split}_{key}",row,root,arm,snap)
        line.update(status="paired_full_original",arms=results,account_sha256=hashes,
                    delta={k:results["alt"][k]-results["base"][k] for k in results["base"]})
    out={"split":split,"registered_map_seats":len(output),"maps":len({r["map"] for r in output}),
         "source_protocol_sha256":expected["protocol"],"weights":authority["weights"],"runner_sha256":expected["runner"],
         "verifier_sha256":run.sha(H/"labels.py"),"counter_sha256":run.sha(DEV/"evidence/account.py"),"cases":output}
    run.freeze(H/f"{split}_labels.json",out)
    print(split,len(output),Counter(r["status"] for r in output))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("split",choices=("train","held"));main(parser.parse_args().split)
