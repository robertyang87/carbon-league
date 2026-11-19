"""Evaluate tree investment in autonomous Sapling play against held-out scripted policies."""
import argparse
import json
import random
import statistics
from pathlib import Path

from cash_practice import audit, economy, np, ROOT
from eval_v53 import _tdist


SEEDS = tuple(range(620134740, 620134744))
POLICIES = ("policy_v0_8", "policy_v0_2")


class PurchaseObserver(audit.Observer):
    """Record actual price and immediately available funds at the simulator's debit."""
    def __init__(self):
        super().__init__()
        self.plants = {}

    def trace(self, frame, event, arg):
        if frame.f_code is self.code and event == "line" and frame.f_lineno == self.plant_line:
            f = frame.f_locals
            sim, worker = f["self"], f["w"]
            price = float(f["player_plant_cost"][worker.player])
            key = (sim.step_count + 1, worker.player, worker.pos)
            self.plants[key] = (price, float(sim.cash[worker.player] - price))
        return super().trace(frame, event, arg)


def play(pack, rival, seed, side):
    random.seed(7919 * seed + 104729 * side + 31)
    np.random.seed(seed)
    sim = audit.FastCarbon(audit.CFG).reset(seed=seed)
    agents = [None, None]
    agents[side], agents[1-side] = pack.agent, rival.make_agent()
    obs = PurchaseObserver()
    income = [0., 0.]
    trees, snapshots = {}, {}
    seizures_won = seizures_lost = 0
    pre = sim.get_state()
    last = int(sim.config["episodeSteps"]) - 1
    for step in range(1, last + 1):
        actions = [agent(sim.obs(p), sim.config) for p, agent in enumerate(agents)]
        for p, commands in enumerate(actions):
            owned = set(sim.player_workers[p]) | {base for base, _ in sim.bases[p]}
            if not isinstance(commands, dict) or set(commands) - owned:
                raise ValueError("invalid public agent command")
        obs.step(sim, actions)
        post = sim.get_state()
        for tid, tree in post["trees"].items():
            p = tree["player"]
            amount = float(post["trees_dict"][tid][1])
            income[p] += amount
            old = pre["trees"].get(tid)
            if old is None and p == side:
                cell = tree["pos"]
                price, reserve = obs.plants[step, p, cell]
                enemies = [w for w in pre["workers"].values() if w["player"] != side and w["type"] == "PLANTER"]
                trees[tid] = dict(plant_step=step, cell=cell, cost=price, cash_after_debit=reserve,
                                  nearest_enemy_planter_pre=min((_tdist(cell, w["pos"]) for w in enemies), default=None),
                                  first_tenure_absorption=0., first_tenure_end=None, times_seized_from_us=0)
            if old and p != old["player"]:
                if p == side:
                    seizures_won += 1
                else:
                    seizures_lost += 1
                    if tid in trees:
                        trees[tid]["times_seized_from_us"] += 1
                        if trees[tid]["first_tenure_end"] is None:
                            trees[tid].update(first_tenure_end=step, first_tenure_reason="seized")
            if tid in trees and trees[tid]["first_tenure_end"] is None and p == side:
                trees[tid]["first_tenure_absorption"] += amount
        for tid in pre["trees"].keys() - post["trees"].keys():
            if tid in trees and trees[tid]["first_tenure_end"] is None:
                trees[tid].update(first_tenure_end=step, first_tenure_reason="expired")
        extinct = [audit._extinct(post, p) for p in (0, 1)]
        if step in (60, 100, 200, last) or any(extinct):
            snapshots[str(step)] = {key:economy(sim, obs, p, income)
                                    for key,p in (("student", side), ("opponent",1-side))}
        if any(extinct) or step == last:
            break
        pre = post
    for record in trees.values():
        if record["first_tenure_end"] is None:
            record.update(first_tenure_steps=step + 1 - record["plant_step"], first_tenure_reason="censored")
        else:
            record["first_tenure_steps"] = record["first_tenure_end"] - record["plant_step"]
    if len(trees) != sim.tree_counters[side]:
        raise ValueError("tree identity count differs from accepted plants")
    official = [step - last - 2 if extinct[p] else float((pre if any(extinct) else post)["cash"][p]) for p in (0, 1)]
    return dict(seed=seed, side=side, opponent=rival.__name__, end_step=step, extinct_by_seat=extinct,
                official_margin=official[side]-official[1-side], official_scores=official, snapshots=snapshots, own_plants=trees,
                seizures_won=seizures_won, seizures_lost=seizures_lost,
                first_plant_step=min((x["plant_step"] for x in trees.values()), default=None),
                final_state_sha256=audit.hashed(post))


def summarize(games):
    trees = [s for g in games for s in g["own_plants"].values()]
    firsts = [g["first_plant_step"] for g in games if g["first_plant_step"] is not None]
    cutoff = [g["snapshots"]["100"] for g in games if "100" in g["snapshots"]]
    return dict(games=len(games), wins=sum(g["official_margin"]>0 for g in games),
                mean_margin=statistics.mean(g["official_margin"] for g in games),
                t100_observations=len(cutoff), t100_cash=statistics.mean(x["student"]["cash"] for x in cutoff) if cutoff else None,
                games_with_plant=len(firsts), games_with_plant_by_t100=sum(s<=100 for s in firsts),
                first_plant_median_when_planted=statistics.median(firsts) if firsts else None,
                plants=len(trees), lifetime_first_tenure_absorption=sum(s["first_tenure_absorption"] for s in trees),
                total_plant_spend=sum(s["cost"] for s in trees),
                purchases_leaving_less_than_recruit_cost=sum(s["cash_after_debit"] < min(audit.CFG["recPlanterCost"],audit.CFG["recCollectorCost"]) for s in trees),
                censored=sum(s["first_tenure_reason"]=="censored" for s in trees),
                seized_before_first_expiry=sum(s["first_tenure_reason"]=="seized" for s in trees),
                seized_within_ten_transitions=sum(s["first_tenure_reason"]=="seized" and s["first_tenure_steps"]<=10 for s in trees),
                total_times_lost_any_tree=sum(g["seizures_lost"] for g in games),
                total_times_seized_enemy_tree=sum(g["seizures_won"] for g in games))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args=p.parse_args()
    if args.out.exists():
        p.error("use a fresh output file")
    pack = args.pack.resolve()
    policies = [ROOT/"code"/(name+".py") for name in POLICIES]
    sources = [pack, Path(__file__).resolve(), Path(__file__).resolve().with_name("cash_practice.py"),
               Path(audit.__file__), Path(audit.shared.__file__), ROOT/"code/fastsim.py", ROOT/"code/eval_v53.py", *policies]
    hashes = {str(f.relative_to(ROOT)):audit.digest(f) for f in sources}
    candidate=audit.shared.module(pack, "candidate")
    games=[]
    for path in policies:
        rival=audit.shared.module(path, path.stem)
        for seed in SEEDS:
            for side in (0,1):
                games.append(play(candidate, rival, seed, side))
    if any(audit.digest(f)!=hashes[str(f.relative_to(ROOT))] for f in sources):
        raise ValueError("source changed during evaluation")
    report=dict(scope="Four fixed maps, both seats and two held-out scripted opponents. "
                "Absorption includes enemy cargo extraction. Initial uninterrupted ownership income excludes collisions, "
                "later recapture income and other trees. Endpoint-owned trees are right-censored for time-to-loss.",
                pack_sha256=audit.digest(pack), source_sha256=hashes, games=games, summary=summarize(games),
                by_opponent={name:summarize([g for g in games if g["opponent"]==name]) for name in POLICIES})
    audit.shared.atomic_json(args.out, report)
    print(json.dumps(report["summary"]), flush=True)


if __name__=="__main__":
    main()
