"""Break down the original-opponent tree gap, weighting each physical map equally."""
from collections import Counter, defaultdict
from statistics import mean
from tree_attribution import HERE, replay


def run(game):
    tape = replay.load(replay.ROOT / game["path"])
    seat = game["seat"]
    sim = replay.FastCarbon(tape["game_configuration"]).reset(seed=game["seed"])
    sim.cash = replay.CashEvents(sim.cash)
    phases = [[Counter(), Counter()] for _ in range(3)]
    for t, commands in enumerate(tape["actions"]):
        old_ids = set(sim.trees)
        sim.cash.events.clear()
        sim.step(*commands)
        counters = phases[min(t//100, 2)]
        for tid, tree in sim.trees.items():
            if tid in old_ids:
                counters[tree.player ^ seat]["turns"] += 1
        for kind, owner, amount, *detail in sim.cash.events:
            account = counters[owner ^ seat]
            if kind == "tree_absorption":
                account["ground"] += amount - detail[-1]
                account["freight"] += detail[-1]
            elif kind in ("plant", "seize"):
                account[kind] += 1
    assert list(sim.cash) == tape["official_score_by_seat"]
    return dict(seed=game["seed"], seat=seat, source=game["path"], phases=phases)


def gap(phase):
    a,b = phase
    if not a["turns"] or not b["turns"]:
        return dict(ground=a["ground"]-b["ground"], freight=a["freight"]-b["freight"])
    ra,rb = (c["ground"]/c["turns"] for c in phase)
    return dict(ground=a["ground"]-b["ground"], freight=a["freight"]-b["freight"],
                ground_by_time=(a["turns"]-b["turns"])*(ra+rb)/2,
                ground_by_yield=(ra-rb)*(a["turns"]+b["turns"])/2)


if __name__ == "__main__":
    games = [run(g) for g in replay.load(replay.ACTOR/"tactical_exchange_dev/paired.json.gz")]
    stats = []
    for i in range(3):
        groups = defaultdict(list)
        for game in games:
            groups[game["seed"]].append(gap(game["phases"][i]))
        fields = ("ground","freight","ground_by_time","ground_by_yield")
        stats.append({f:mean(mean(r[f] for r in group) for group in groups.values())
                      for f in fields if all(f in r for group in groups.values() for r in group)})
    replay.write(HERE/"historical_tree_income.json", dict(games=games, map_equal=stats))
    print(stats)
