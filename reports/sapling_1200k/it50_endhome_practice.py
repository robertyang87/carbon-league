"""Four independent practice maps, both seats. Compare the exact two packs."""
import argparse
import json
from pathlib import Path
import random

from cash_practice import audit, np, ROOT
from build_it50_endhome import PARENT, PARENT_SHA, OUTPUT

SEEDS = tuple(range(620188310, 620188314))
POLICIES = ("policy_v0_8", "policy_v0_5")


def one(pack, opp_path, seed, seat):
    random.seed(7919*seed + 104729*seat + 31)
    np.random.seed(seed)
    sim = audit.FastCarbon(audit.CFG).reset(seed=seed)
    module = audit.shared.module(pack, "endhome_practice_actor")
    rival = audit.shared.module(opp_path, "endhome_practice_rival").make_agent()
    log = audit.Observer()
    acts, overrides = [], []
    for t in range(299):
        pre = sim.get_state()
        joint = [{}, {}]
        joint[seat] = module.agent(sim.obs(seat), sim.config)
        joint[1-seat] = rival(sim.obs(1-seat), sim.config)
        acts.append(audit.hashed(joint))
        if getattr(module, "_RETURN_LAST", ()):
            overrides.append({"t": t, "choices": module._RETURN_LAST})
        log.step(sim, joint)
        post = sim.get_state()
        extinct = [bool(audit._extinct(post, s)) for s in (0, 1)]
        if any(extinct):
            break
    cash = (pre if any(extinct) else post)["cash"]
    final = [(sim.step_count - 301) if extinct[s] else float(cash[s]) for s in (0, 1)]
    own = [e for e in log.events if e["event"] == "deposit" and e["player"] == seat and e["amount"] > 0]
    return {"end": sim.step_count, "margin": final[seat]-final[1-seat], "end_cash": final[seat],
            "terminal_load": sum(w.carbon for w in sim.workers.values() if w.player == seat),
            "normal_deposit": sum(e["amount"] for e in own),
            "last15_normal_deposit": sum(e["amount"] for e in own if e["step"]>284),
            "end_state_sha256": audit.hashed(post), "actions_sha256": audit.hashed(acts),
            "counters": dict(getattr(module, "_RETURN_COUNTS", {})), "overrides": overrides}, acts


def main(out):
    if out.exists() or audit.digest(PARENT) != PARENT_SHA:
        raise ValueError("fresh output and released parent required")
    sha = audit.digest(OUTPUT)
    games = []
    for i, seed in enumerate(SEEDS):
        opp = ROOT / "code" / (POLICIES[i % 2]+".py")
        for seat in (0, 1):
            base, commands = one(PARENT, opp, seed, seat)
            candidate, changed = one(OUTPUT, opp, seed, seat)
            first = next((i for i,(a,b) in enumerate(zip(commands, changed)) if a != b), None)
            events = candidate["overrides"]
            if first != (events[0]["t"] if events else None):
                raise ValueError("trajectory diverged before the rule changed an action")
            if first is None and (commands != changed or base["end_state_sha256"] != candidate["end_state_sha256"]):
                raise ValueError("identical trajectory failed end-state check")
            games.append(dict(seed=seed, seat=seat, opponent=opp.name, opponent_sha256=audit.digest(opp),
                              first_different_root=first, parent=base, candidate=candidate))
            print(seed, seat, first, candidate["normal_deposit"]-base["normal_deposit"], flush=True)
    if audit.digest(OUTPUT) != sha:
        raise ValueError("candidate changed during practice")
    summary = {k:sum(g["candidate"][k]-g["parent"][k] for g in games)/len(games)
               for k in ("margin","normal_deposit","last15_normal_deposit","terminal_load")}
    result = dict(games=games, maps=len(SEEDS), parent_sha256=PARENT_SHA, candidate_sha256=sha,
                  paired_mean_change=summary, changed_games=sum(g["first_different_root"] is not None for g in games))
    audit.shared.atomic_json(out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    main(p.parse_args().out)
