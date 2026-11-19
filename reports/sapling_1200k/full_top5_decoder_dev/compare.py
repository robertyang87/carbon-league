"""Compare a single full-pass Sapling to its archived V3 control on the same map and seats."""
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from tree_audited_game import exam, prior
from tree_win_gate import validate_game, verify_raw

SEED = 620106811
ACTORS = ROOT / "data/weights/sapling_1200k"
PARENT_SHA = "4cd4c1fcf40f05217ecba2a59095ba40286c5f96017d2b15285c942a3b0f5348"
METRICS = ("cash", "own_positive_delivery_carbon", "cumulative_tree_absorption_income",
           "cargo", "alive_workers", "alive_trees", "new_trees_planted")


def main():
    results_path = HERE / "results.json"
    old_path = HERE.parent / "tree_exam_v3/results.json"
    results, old = [json.loads(p.read_bytes()) for p in (results_path, old_path)]
    identity = json.loads((HERE / "identity.json").read_bytes())
    if (results["status"] != "complete" or old["status"] != "complete"
            or results["header"]["seeds"] != [SEED] or len(results["games"]) != 2
            or {g["candidate_seat"] for g in results["games"]} != {0, 1}
            or results["header"]["pack_sha256"] != identity["pack_sha256"]
            or results["header"]["weight_sha256"] != identity["weight_sha256"]):
        raise ValueError("incorrect candidate run or test block")
    for r in (results, old):
        if r["header"]["examiner_md5"] != prior.EXPECTED_MD5:
            raise ValueError("incorrect examiner")
    if old["header"]["pack_sha256"] != PARENT_SHA:
        raise ValueError("incorrect parent control")
    parent_file = ACTORS / "cash_v3_agent.py"
    candidate_file = ACTORS / "full_top5_decoder_agent.py"
    if prior.digest(parent_file) != PARENT_SHA or prior.digest(candidate_file) != identity["pack_sha256"]:
        raise ValueError("pack bytes differ from recorded actors")
    parent = prior.shared.module(parent_file, "full_pass_parent")
    candidate = prior.shared.module(candidate_file, "full_pass_deployed")
    rows = []
    for g in results["games"]:
        seat = g["candidate_seat"]
        matches = [p for p in old["games"] if p["seed"] == g["seed"] == SEED and p["candidate_seat"] == seat]
        if len(matches) != 1 or any(matches[0]["agent_errors_by_seat"]):
            raise ValueError("missing or ambiguous parent control")
        prior_game = matches[0]
        validate_game(g)
        verify_raw(HERE, g, identity["pack_sha256"], identity["weight_sha256"])
        raw = json.loads(gzip.decompress((HERE / g["raw_file"]).read_bytes()))
        sim = exam.FastCarbon(raw["game_configuration"]).reset(seed=SEED)
        changed = []
        for t, joint in enumerate(raw["actions"]):
            action = candidate.agent(sim.obs(seat), sim.config)
            if action != joint[seat]:
                raise ValueError(f"deployed output differs on seat {seat} at step {t}")
            if parent.agent(sim.obs(seat), sim.config) != action:
                changed.append(t)
            sim.step(*joint)
        if prior.hashed(sim.get_state()) != g["final_state_sha256"]:
            raise ValueError("candidate terminal state differs")
        def economic(row):
            return {p:{k: row["final"][p][k] for k in METRICS} for p in ("candidate","opponent")}
        rows.append({"seat": seat, "candidate_result": g["result"], "parent_result": prior_game["result"],
                     "candidate_margin": g["margin"], "parent_margin": prior_game["margin"],
                     "candidate_minus_parent_margin": g["margin"] - prior_game["margin"],
                     "candidate_steps": len(raw["actions"]), "parent_steps": prior_game["actual_transitions"],
                     "candidate_end": economic(g), "parent_end": economic(prior_game),
                     "candidate_actions_matching_saved_tape": len(raw["actions"]),
                     "parent_differs_on_candidate_roots": changed})
    prior.shared.atomic_json(HERE / "comparison.json", {
        "seed": SEED, "maps": 1, "control_results_sha256": prior.digest(old_path),
        "candidate_results_sha256": prior.digest(results_path), "games": rows,
        "scope": "Descriptive development comparison on one previously used map with both seats. "
                 "Master3s2 uses unseeded temperature one and real-time search. "
                 "The previously archived V3 report has scores and hashes but no action tapes here. "
                 "Same-observation comparisons use persistent original V3 and full-pass actors on candidate tapes."})


if __name__ == "__main__":
    main()
