"""Probe information isolation and decision influence on the selected arm's new proxy holdout."""
import argparse
import gzip
import json
import sys

import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", action="store_true")
    args=parser.parse_args()
    src=run.HERE/"results.json"
    result=json.loads(src.read_bytes())
    selection=json.loads((run.HERE/"selection.json").read_bytes())
    if args.original:
        folder=run.HERE/"original_candidate"
        game_report=json.loads((folder/"results.json").read_bytes())
        if game_report.get("status") != "complete": raise ValueError("incomplete original games")
        games=game_report["games"]
    else:
        if result.get("status")!="complete": raise ValueError("incomplete holdout")
        folder=run.HERE
        games=[g for g in result["games"] if g["stage"]=="holdout" and g["arm"]==selection["chosen"]]
    name=selection["chosen"]
    buf=run.packed(run.ARMS[name])
    original=run.arena.load_agent(run.arena.PARENT,"original")
    modules=[run.arena.load_agent(run.arena.PARENT,t,buf) for t in ("actual","noise","zero_sapling")]
    modules[-1][0]._student_fraction=lambda *args: 0.
    records=[]
    for g in games:
        path=folder/g["raw_file"]
        if run.arena.prior.digest(path)!=g["raw_sha256"]: raise ValueError("changed tape")
        raw=json.loads(gzip.decompress(path.read_bytes()))
        sim=run.arena.exam.FastCarbon(raw["game_configuration"]).reset(seed=g["seed"])
        seat=g["candidate_seat"]
        roots=[]; changed=[]; sensitive=[]
        for t,joint in enumerate(raw["actions"]):
            obs=sim.obs(seat)
            perturb=dict(obs,seed=-1000-t,randomSeed=t*355,opponent_name="metadata poison",
                         full_carbon=[(i*17+t)%101 for i in range(sim.n_cells)])
            observers=[(modules[0],obs),(modules[1],perturb),(modules[2],obs),(original,obs)]
            if t%12==0 or t+1==len(raw["actions"]):
                actions=[run.arena.act(m,o,sim.config) for m,o in observers]
                if actions[0]!=joint[seat] or actions[0]!=actions[1]:
                    raise ValueError(f"saved action or metadata invariance failed at {g['seed']}/{seat}/{t}")
                roots.append(t)
                if actions[0]!=actions[2]: sensitive.append(t)
                if actions[0]!=actions[3]: changed.append(t)
            else:
                if args.original:
                    if run.arena.act(modules[0],obs,sim.config)!=joint[seat]:
                        raise ValueError("recorded original-examiner game action differs from frozen actor")
                    observers=observers[1:]
                for (m,deps),o in observers:
                    sys.modules.update(deps)
                    if hasattr(m,"_update_economy"): m._update_economy(o)
                    m._FEAT.observe(o)
            sim.step(*joint)
        records.append({"seed":g["seed"],"seat":seat,"audited":roots,"changed":changed,
                        "student_sensitive":sensitive,"recorded_actor_actions_verified":len(raw["actions"]) if args.original else len(roots),
                        "decision_triggers_in_verified_steps":dict(modules[0][0]._ECON_COVER)})
    run.arena.prior.shared.atomic_json(folder/"attribution.json",{
        "results_sha256":run.arena.prior.digest(folder/"results.json"),"chosen":name,"games":records,
        "scope":"Every twelfth actual root plus final action, full public history preserved. Four independent "
                "selectors: reproduced actor, same public state with poisoned seed/name/hidden-carbon metadata, "
                "zero-Sapling policy, and unchanged parent. Counts are effective action differences within each role mixture."})


if __name__=="__main__": main()
