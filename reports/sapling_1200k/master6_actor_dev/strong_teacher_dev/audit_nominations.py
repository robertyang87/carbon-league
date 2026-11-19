"""Reconcile every actually nominated archived alternative using official rule-only replay."""
from pathlib import Path
import sys
from propose import ACTOR,ROOT,HERE,read,clean,sha,check,dump
sys.path.insert(0,str(ACTOR.parent/"root_recovery_branches"))
from branches import begin


def main():
    packet=read(HERE/"proposals.json.gz")
    rows=[r for r in packet["rows"] if r["nominates_registered_alternative"]]
    out=[]
    for r in rows:
        raw=read(ROOT/r["source"]);wd=ROOT/r["case"]
        check(sha(ROOT/r["source"])==r["source_sha256"],"wrong source")
        data={}
        for arm in ("parent","alternative"):
            seal=read(wd/("live_"+arm+".json"));archive=wd/seal["action_tape"]
            tape=read(archive); check(sha(archive)==r["native"][arm]["tape_sha256"],"changed continuation")
            env=begin(raw)
            for cmd in raw["actions"][:r["t"]]+tape:
                check(not env.done,"truncated or overrun terminal")
                env.step(cmd)
            scores=[s.reward for s in env.states];state=[s.status for s in env.states]
            check(env.done and state==["DONE","DONE"] and scores==seal["score"] and len(tape)==1+seal["search_calls"],"mismatched official outcome")
            data[arm]={"score":scores,"official_steps":int(env.states[0].observation.step),"status":state,
                       "continuation_steps":len(tape),"margin":scores[r["seat"]]-scores[1-r["seat"]]}
        diff=data["alternative"]["margin"]-data["parent"]["margin"]
        check(abs(diff-r["margin_delta"])<.011,"different realized sign")
        out.append({"case":r["case"],"t":r["t"],"apex_focal":r["proposal_action"],"arms":data,"official_margin_delta":diff})
    dump(HERE/"nomination_audit.json",{"nominated_training_roots":len(out),"held_nominations":sum(r["split"]=="hold" for r in rows),
       "verified_new_native_matches":0,"archive":sha(HERE/"proposals.json.gz"),"implementation_sha256":sha(Path(__file__)),"rows":out})


if __name__=="__main__":main()
