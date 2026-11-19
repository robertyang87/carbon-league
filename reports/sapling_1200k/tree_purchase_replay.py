"""Rebuild audited action transcripts and value actual purchases through their first ownership period."""
import gzip
import inspect
import json
from pathlib import Path
import sys
import tree_audited_game as arena

ROOT = Path(__file__).resolve().parent
SIM = arena.exam.FastCarbon


def replay(folder, row, role):
    raw = json.loads(gzip.decompress((folder / row["raw_file"]).read_bytes()))
    cfg = raw["game_configuration"]
    sim = SIM(cfg).reset(seed=row["seed"])
    side = row["candidate_seat"] if role == "candidate" else 1-row["candidate_seat"]
    line_src, lineno = inspect.getsourcelines(SIM.step)
    collision = [i + lineno for i,s in enumerate(line_src) if s.strip() == "self.cash[tree_at[pos].player] += w.carbon; gets = tree_at[pos].player"]
    if len(collision) != 1:
        raise ValueError("cannot locate tree collision credit")
    ledger = []
    live = {}
    def trace(frame, event, arg):
        if frame.f_code is not SIM.step.__code__:
            return None
        if event == "line" and frame.f_lineno == collision[0]:
            x = frame.f_locals
            tree = x["tree_at"][x["pos"]]
            if tree.id in live and tree.player == side:
                live[tree.id]["collision_credit"] += float(x["w"].carbon)
        return trace
    before_hook = sys.gettrace()
    sys.settrace(trace)
    try:
        for tick, acts in enumerate(raw["actions"],1):
            pre = {tid:t.player for tid,t in sim.trees.items()}
            active = sum(t.age < cfg["treeLifespan"] for t in sim.trees.values())
            price = cfg["plantCost"]
            if len(sim.player_trees[side]) > cfg["playerTreeProtectiveNumber"]:
                price += cfg["plantCostInflationRatio"] * cfg["plantCostInflationBase"]**active
            sim.step(*acts)
            for tid in list(live):
                if tid not in sim.trees or sim.trees[tid].player != side:
                    live.pop(tid)
            for tid,t in sim.trees.items():
                if t.player != side:
                    continue
                if tid not in pre:
                    entry = dict(tree=tid, step=tick, cost=price, absorption=0., collision_credit=0.)
                    ledger.append(entry)
                    live[tid] = entry
                if tid in live:
                    live[tid]["absorption"] += float(sim.trees_dict[tid][1])
    finally:
        sys.settrace(before_hook)
    if arena.prior.hashed(sim.get_state()) != row["final_state_sha256"]:
        raise ValueError("action replay did not reproduce the complete terminal state")
    if len(ledger) != row["final"][role]["new_trees_planted"]:
        raise ValueError("plant counts differ from external observer")
    def summarize(trees):
        costs = sum(t["cost"] for t in trees)
        earned = sum(t["absorption"]+t["collision_credit"] for t in trees)
        return dict(plants=len(trees), cost=costs, gross_received=earned,
                    absorption=sum(t["absorption"] for t in trees), collision_credit=sum(t["collision_credit"] for t in trees),
                    realized_cash_minus_purchase=earned-costs,
                    repaid=sum(t["absorption"]+t["collision_credit"]>=t["cost"] for t in trees))
    return dict(seed=row["seed"],seat=side,role=role,terminal_hash_reproduced=True,
                all=summarize(ledger),after_200=summarize([t for t in ledger if t["step"]>200]),
                after_260=summarize([t for t in ledger if t["step"]>260]), purchases=ledger)


def main():
    report = {"scope":"Amounts received by first owner of each actually planted tree, up to first loss or terminal; includes absorption and collision credits. No opponent denial, opportunity cost, or counterfactual behavior.", "sim_sha256":arena.prior.digest(arena.exam.ROOT/'code/fastsim.py'),"source_sha256":arena.prior.digest(__file__), "arms":{}, "original_opponent_on_same_trajectories":{}}
    for name in ("v4r","basecal","it50"):
        folder = ROOT / f"tree_exam_{name}"
        j = json.loads((folder/'results.json').read_text())
        report["arms"][name] = [replay(folder,row,"candidate") for row in j["games"]]
        report["original_opponent_on_same_trajectories"][name] = [replay(folder,row,"opponent") for row in j["games"]]
    arena.prior.shared.atomic_json(ROOT / "tree_purchase_replay.json",report)


if __name__ == "__main__":
    main()
