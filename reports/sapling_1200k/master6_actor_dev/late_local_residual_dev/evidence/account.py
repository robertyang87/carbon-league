"""Reconcile a recorded two-seat suffix without invoking any policy or new game."""
from collections import Counter, defaultdict
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTORS = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


settled = load("residual_original_lot_ledger", ACTORS / "lot_settlement_dev/lots.py")
strict = load("residual_newly_mined_ledger", ACTORS / "late_advantage_actor_dev/lots.py")


def _frame(obs, joint):
    return [{"observation": obs, "action": joint[0]}, {"action": joint[1]}]


def _extinct(obs, cfg):
    floor = min(cfg["recCollectorCost"], cfg["recPlanterCost"])
    return [not (workers or trees) and (not bases or cash < floor)
            for cash, bases, workers, trees in obs["players"]]


def account_arm(root_snapshot, configuration, orders_by_step, seat, focal_uids):
    """A restored FastCarbon.get_state() with counters; complete seat-ordered joint actions; no future policy input."""
    if seat not in (0, 1) or not {"tid_counter", "wid_counter"} <= root_snapshot.keys():
        raise ValueError("seat and exact native ID counters are required")
    focal = list(focal_uids)
    if not focal or len(set(focal)) != len(focal):
        raise ValueError("expected distinct focal collector IDs")
    sim = settled.FastCarbon(configuration)
    sim.set_state(root_snapshot)
    cfg = sim.config
    initial_obs = sim.full_observation()
    if any(initial_obs["players"][seat][2][uid][2] != "COLLECTOR" for uid in focal):
        raise ValueError("focal worker is not an own active collector")
    frames = [_frame(initial_obs, [{}, {}])]
    t0 = sim.step_count
    incomes = [Counter(), Counter()]
    delivered = Counter()
    principal = {}
    transitions = []
    for i, joint in enumerate(orders_by_step):
        pre = frames[-1][0]["observation"]
        if any(_extinct(pre, cfg)) or pre["step"] >= cfg["episodeSteps"] - 1:
            raise ValueError("orders after terminal")
        v, q, after = settled.one_step(sim, joint, seat, verify=True)
        principal.update({(i, u): n for u, n in v.items()})
        transitions.append(q)
        frames.append(_frame(after, joint))
        for kind, s, amount, *detail in sim.cash.events:
            incomes[s][kind] += amount
            if s == seat and kind == "own_delivery":
                delivered[detail[0]] += amount
    T = len(transitions)
    if T == 0:
        raise ValueError("empty suffix")
    pre, post = (z[0]["observation"] for z in frames[-2:])
    extinct = _extinct(post, cfg)
    if not any(extinct) and sim.step_count != cfg["episodeSteps"] - 1:
        raise ValueError("response suffix incomplete")
    official = [(sim.step_count - cfg["episodeSteps"] - 1 if extinct[p]
                 else (pre if any(extinct) else post)["players"][p][0]) for p in (0, 1)]
    for side, tx in enumerate(incomes):
        settled.equal(sum(tx.values()), sim.cash[side] - root_snapshot["cash"][side], "treasury reconciliation")
    principal.update({(T, u): n for u, n in settled.positive_collectors(post, seat).items()})
    lots = settled.Timeline(principal, transitions, T)
    fresh, tally = strict.measure({"steps": frames, "configuration": cfg}, seat, lo=0)
    # Deposit reconstruction based on public transition data is independent of the treasury write tape.
    settled.equal(tally["official_own_worker_deposit"], sum(delivered.values()), "public/official delivery")
    profile = defaultdict(lambda: {"strict_fresh_mined": 0., "strict_fresh_delivered": 0.})
    for (turn, uid), row in fresh.items():
        profile[uid]["strict_fresh_mined"] += row["gross"]
        profile[uid]["strict_fresh_delivered"] += row["direct"]
    workers = {}
    for uid, (pos, cargo, role) in initial_obs["players"][seat][2].items():
        if role != "COLLECTOR":
            continue
        alloc = (lots.summary(0, uid)["amounts"] if cargo > 0 else dict.fromkeys(settled.CATEGORIES, 0.))
        workers[uid] = {"position": pos, "principal": cargo, "principal_fates": alloc,
                        "strict_principal_delivered": fresh[0, uid]["existing_direct"],
                        **profile[uid], "total_deposited_by_worker": delivered[uid]}
        if workers[uid]["strict_principal_delivered"] + profile[uid]["strict_fresh_delivered"] > delivered[uid] + 1e-5:
            raise ValueError("disjoint worker lots exceeded its real cash deposits")
    focal_deposits = sum(delivered[u] for u in focal)
    return {"root_t": t0, "terminal_t": sim.step_count, "seat": seat, "focal_uids": focal,
            "terminal_extinct_physical": extinct, "root_cash_physical": root_snapshot["cash"],
            "terminal_cash_physical": list(sim.cash), "official_scores_physical": official,
            "official_margin": official[seat] - official[1 - seat], "per_worker": workers,
            "team": {"treasury_flows_physical": list(map(dict, incomes)), "strict_new_mined": tally["new_gross"],
                     "strict_new_delivered": tally["new_fates"][0], "own_delivery": sum(delivered.values()),
                     "focal_delivery": focal_deposits,
                     "other_delivery": sum(delivered.values()) - focal_deposits}}
