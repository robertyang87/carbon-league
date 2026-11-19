"""Compare four archived D1 overrides with their factual joint move and fixed logged continuation."""
import gzip
import json
from pathlib import Path

from d1_sapling_build import HERE
from tree_audited_game import exam
from turning_point_ledger import sha,compact,write


def state(sim, me):
    return dict(cash_margin=round(sim.cash[me]-sim.cash[1-me],2),
                cargo_margin=round(sum(w.carbon*(1 if w.player==me else -1) for w in sim.workers.values()),6),
                worker_margin=len(sim.player_workers[me])-len(sim.player_workers[1-me]),
                collisions=[[wid,role,round(v,6),rec] for wid,role,v,rec,_ in sim.last_collisions if v>0])


def main():
    source=HERE/"delivery_parent_dev"
    ledger=json.loads((source/"results.json").read_bytes())
    by_id={(g["seed"],g["candidate_seat"]):g for g in ledger["games"]}
    findings=[]
    for suffix in ("310","311"):
        report=json.loads((HERE/f"d1_sapling_probe_{suffix}.json").read_bytes())
        for game in report["games"]:
            if len(game["witnesses"])!=game.get("search_changes",0):
                raise ValueError("probe did not retain every override in the block")
            seat,seed=game["seat"],game["seed"]
            row=by_id[(seed,seat)]
            payload=(source/row["raw_file"]).read_bytes()
            if sha(payload)!=row["raw_sha256"]:
                raise ValueError("unaligned source evidence")
            raw=json.loads(gzip.decompress(payload))
            sim=exam.FastCarbon(raw["game_configuration"]).reset(seed=seed)
            events={w["root"]:w for w in game["witnesses"]}
            for t,commands in enumerate(raw["actions"]):
                if t in events:
                    witness=events[t]
                    if commands[seat]!=witness["a0"]:
                        raise ValueError("archived parent disagrees")
                    snap=sim.get_state()
                    variants={}
                    for label in ("a0","selected"):
                        branch=exam.FastCarbon(raw["game_configuration"]).set_state(snap)
                        trace={}
                        expired=0
                        for offset,inputs in enumerate(raw["actions"][t:t+5],1):
                            acts=[dict(a) for a in inputs]
                            if offset==1:
                                acts[seat]=witness[label]
                            # Later logged commands can address an existing or now-absent unit.
                            ids=[set(branch.player_workers[s])|{uid for uid,_ in branch.bases[s]} for s in (0,1)]
                            expired+=sum(len(set(acts[s])-ids[s]) for s in (0,1))
                            branch.step(*acts)
                            if offset in (1,3,5):
                                trace[str(offset)]={**state(branch,seat),"commands_addressing_missing_units":expired}
                            if any(exam._extinct(branch.get_state(),s) for s in (0,1)):
                                break
                        variants[label]=trace
                    findings.append(dict(seed=seed,seat=seat,root=t, selected=witness["selected"],
                                         baseline=witness["a0"], outcomes=variants))
                sim.step(*commands)
            if sha(compact(sim.get_state()))!=row["final_state_sha256"]:
                raise ValueError("factual replay terminal mismatch")
    write(HERE/"d1_sapling_local.json",dict(scope="Post hoc audit of every natural D1 override on the four archived parent games. Joint first step uses actual Master command. Later steps hold both command streams fixed, without re-planning or crediting an eventual win.",rows=findings))


if __name__=="__main__":
    main()
