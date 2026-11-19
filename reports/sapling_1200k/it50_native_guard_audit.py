"""Re-execute the unchanged production it50 on the recorded information sets of both seats."""
import gzip
import json
import os
from pathlib import Path
import tree_audited_game as arena

HERE=Path(__file__).resolve().parent
SHA="57e2557665636a1bbc92ae46a4c33ee9dac609efcffefa1c30df1233746c2211"


def main():
    pack=arena.exam.ROOT/"code/packs/ppou2_m_it50.py"
    if arena.prior.digest(pack)!=SHA:
        raise ValueError("incorrect production artifact")
    directory=HERE/"tree_exam_it50"
    report=json.loads((directory/"results.json").read_text())
    output={"pack_sha256":SHA, "evaluator_sha256":arena.prior.digest(__file__), "games":[]}
    for row in report["games"]:
        raw=json.loads(gzip.decompress((directory/row["raw_file"]).read_bytes()))
        seat=row["candidate_seat"]
        candidate=arena.prior.shared.module(pack, f"original_it50_seat_{seat}")
        sim=arena.exam.FastCarbon(raw["game_configuration"]).reset(seed=row["seed"])
        options=dict(candidate.CONF)
        if not (options["GUARD"] and options["GUARD_NARROW"] and options["SEARCH"] is False):
            raise ValueError("unexpected runtime options")
        differences=[]
        for step, actions in enumerate(raw["actions"]):
            agent_action=candidate.agent(sim.obs(seat),sim.config)
            if agent_action!=actions[seat]:
                differences.append(step)
            sim.step(*actions)
        closure=dict(zip(candidate._AG.__code__.co_freevars, [c.cell_contents for c in candidate._AG.__closure__]))
        actual_cfg=next((v for v in closure.values() if isinstance(v,dict) and "GUARD" in v and "SEARCH" in v),None)
        if not actual_cfg or any(actual_cfg.get(k)!=v for k,v in options.items()):
            raise ValueError("constructed search agent configuration differs from its published options")
        if differences or arena.prior.hashed(sim.get_state())!=row["final_state_sha256"]:
            raise ValueError(f"trajectory mismatch: {differences[:5]}")
        output["games"].append(dict(seed=row["seed"],seat=seat, identical_agent_actions=len(raw["actions"]),
                                    matches_terminal_hash=True,config_passed_to_live_agent={k:actual_cfg[k] for k in options},
                                    stats=candidate.agent.last_stats))
    if len(output["games"])!=2:
        raise ValueError("required both complete seats")
    arena.prior.shared.atomic_json(directory/"native_guard_replay.json",output)


if __name__=="__main__":
    main()
