"""Identify all original-examiner roots where the unchanged parent would choose a different move."""
import gzip
import json

import run


def main():
    folder=run.HERE/"original_candidate"
    results=json.loads((folder/"results.json").read_bytes())
    if results.get("status")!="complete": raise ValueError("incomplete original candidate tape")
    parent=run.arena.load_agent(run.arena.PARENT,"baseline_actor")
    rows=[]
    for g in results["games"]:
        tape=folder/g["raw_file"]
        if run.arena.prior.digest(tape)!=g["raw_sha256"]: raise ValueError("changed tape")
        raw=json.loads(gzip.decompress(tape.read_bytes()))
        sim=run.arena.exam.FastCarbon(raw["game_configuration"]).reset(seed=g["seed"])
        seat=g["candidate_seat"]
        changes=[]
        for turn,joint in enumerate(raw["actions"]):
            pred=run.arena.act(parent,sim.obs(seat),sim.config)
            cmd=joint[seat]
            if pred!=cmd:
                units=[]
                for uid in sorted(pred.keys()|cmd.keys()):
                    if pred.get(uid)==cmd.get(uid): continue
                    w=sim.workers.get(uid)
                    units.append({"unit":uid,"role":"base" if w is None else ("collector" if w.is_collector else "planter"),
                                  "position":None if w is None else w.pos,"cargo":None if w is None else w.carbon,
                                  "parent":pred.get(uid),"played":cmd.get(uid)})
                changes.append({"turn":turn,"units":units})
            sim.step(*joint)
        if run.arena.prior.hashed(sim.get_state())!=g["final_state_sha256"]:
            raise ValueError("final state mismatch")
        rows.append({"seed":g["seed"],"seat":seat,"verified_parent_calls":len(raw["actions"]),"changed":changes})
    run.arena.prior.shared.atomic_json(folder/"all_changes.json",{
        "parent_sha256":run.arena.PARENT_SHA,"candidate_sha256":results["header"]["pack_sha256"],
        "scope":"Same candidate trajectory, original persistent parent recomputed at every root. "
                "Downstream units can change through the action prefix. This is not a counterfactual game outcome.",
        "games":rows})


if __name__=="__main__": main()
