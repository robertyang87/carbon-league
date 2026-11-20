"""Reproduce N33's first real deposit comparison with the official isolated referee."""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parents[2]
ROOT = DERIVATIVE.parents[1]
MATCH = "798a279b8c4cc1de4d5ab84de34d61ce"
sys.path.insert(0, str(ROOT / "reports/sapling_1200k/all_top5"))
import economy_extract as economy

spec = importlib.util.spec_from_file_location("n33_replay_harness", DERIVATIVE / "freight_response/run.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def state(frame):
    shared = frame[0]["observation"]
    return (shared["step"], shared["players"], shared["trees"], list(shared["full_carbon"]),
            [side["observation"]["carbon"] for side in frame])


def simulate(replay):
    raw = replay.read_bytes()
    game = json.loads(gzip.decompress(raw))
    frames = game["steps"]
    cfg = game["configuration"]
    assert len(frames) == 300 and game["rewards"] == [7289.39, 7460.63]
    assert all(side["status"] == "DONE" for side in frames[-1])
    ledger, _, deposits, _ = economy.reconstruct(frames, cfg)
    random.seed(cfg["randomSeed"])
    harness.np.random.seed(cfg["randomSeed"])
    referee = harness.make("carbon", configuration=cfg)
    initial = frames[0][0]["observation"]["full_carbon"]
    referee.states[0].observation.full_carbon = copy.deepcopy(initial)
    fast = economy.FastCarbon(cfg).reset(full_carbon=initial)
    dist = economy.distances(cfg["size"])

    def verify_official(env, simulator):
        actual = state(env.states)
        public = simulator.obs(0)
        assert actual == (simulator.step_count, public["players"], public["trees"], list(simulator.carbon),
                          [simulator.obs(seat)["carbon"] for seat in (0, 1)])
        return actual

    def run_commands(env, sim, uid, t, forced):
        history = []
        for lag, command in enumerate(forced):
            j = t + lag
            orders = [copy.deepcopy(p["action"] or {}) for p in frames[j + 1]]
            source = orders[0].get(uid, "NONE")
            if command != "LOGGED":
                if command == "NONE":
                    orders[0].pop(uid, None)
                else:
                    orders[0][uid] = command
            sim.cash.events.clear()
            sim.step(*orders)
            env.step(orders, [])
            actual = verify_official(env, sim)
            if command == "LOGGED" and all(c == "LOGGED" for c in forced):
                assert actual == state(frames[j + 1])
            our, enemy = actual[1]
            own_wallet, foe_wallet = our[0], enemy[0]
            cash_events = [list(e) for e in sim.cash.events if e[0] == "own_delivery" and e[3] == uid]
            history.append(dict(
                after=j+1, original_order=source, issued=orders[0].get(uid, "NONE"),
                cash=[own_wallet, foe_wallet], margin=round(own_wallet - foe_wallet, 2),
                roster_sizes=[len(our[2]), len(enemy[2])], live_worker=our[2].get(uid),
                all_collisions=sim.last_collisions, deposit_of_actor=cash_events,
                actual_other_orders_sha256=digest(json.dumps(
                    [{u: v for u, v in orders[0].items() if u != uid}, orders[1]],
                    sort_keys=True, separators=(",", ":")).encode()),
            ))
        return history

    experiments = {
        93: ("player-0-worker-6", ("LEFT", "NONE")),
        118: ("player-0-worker-16", ("UP", "DOWN", "NONE", "NONE", "NONE", "NONE")),
    }
    branches = []
    for t in range(119):
        root = verify_official(referee, fast)
        assert root == state(frames[t])
        if t in experiments:
            uid, forced = experiments[t]
            spot, load, _ = root[1][0][2][uid]
            # Retain exact public prefix, hidden board, referee IDs and RNG. Prior
            # duplicated boards are unnecessary once their identity was verified.
            referee.steps[:-1] = [None]*t
            referee.logs = []
            snapshot = harness.official_state.take(referee, [0, 0])
            curves = {}
            for name, commands in (("original", ("LOGGED",)*len(forced)), ("alternative", forced)):
                harness.official_state.restore(referee, snapshot)
                model = copy.deepcopy(fast)
                model.cash = economy.CashEvents(model.cash)
                curves[name] = run_commands(referee, model, uid, t, commands)
            harness.official_state.restore(referee, snapshot)
            assert all(x["actual_other_orders_sha256"] == y["actual_other_orders_sha256"]
                       for x, y in zip(curves["original"], curves["alternative"]))
            official_receipts = [dict(arrival=j+1, amount=amount)
                                 for j in range(t, len(deposits)) for s, u, amount in deposits[j]
                                 if u == uid and s == 0]
            others = root[1][1][2]
            closest = [(dist[spot][v[0]], u, v) for u, v in others.items()
                       if v[2] == "COLLECTOR" and v[1] <= load]
            branches.append(dict(
                t=t, worker=uid, original_cash=[p[0] for p in root[1]], cell=spot, loaded=load,
                nearest_own_base_distance=min(dist[spot][b] for b in root[1][0][1].values()),
                nearest_lighter_enemy=min(closest, default=None), observed_own_orders=frames[t+1][0]["action"],
                first_later_archived_own_deposit=official_receipts[0], branches=curves,
            ))
        orders = [p["action"] or {} for p in frames[t+1]]
        fast.step(*orders)
        referee.step(orders, [])
    check_all = []
    for t in (90, 95, 100, 103, 110, 117, 118, 119, 120, 124, 126, 127, 140, 173):
        p = frames[t][0]["observation"]["players"]
        check_all.append(dict(t=t, cash=[v[0] for v in p], margin=round(p[0][0]-p[1][0],2),
                              primary_worker=p[0][2].get(experiments[118][0])))
    phases = {}
    for x, y in ((100,110),(110,120),(118,124)):
        phases[f"{x}-{y}"] = dict(
            arrivals=[[dict(t=j+1, uid=u, cash=v) for j in range(x, y)
                       for s, u, v in deposits[j] if seat == s] for seat in (0,1)],
            total_own_deposits=[round(sum(row[s][0] for row in ledger[x:y]),2) for s in (0,1)],
        )
    return dict(match=MATCH, seat=0, official_result=game["rewards"], replay_sha256=digest(raw),
                methods="Two deterministic action arms share the same official root, original opponent and teammate commands; alternative suffixes are frozen probes, with no new policy responses or terminal-strength estimate.",
                observed_phases=phases, observed_path=check_all, alternatives=branches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DERIVATIVE / "ladder/replays" / (MATCH+"-1.json.gz"))
    parser.add_argument("--out", type=Path, default=HERE/"alternatives.json")
    args = parser.parse_args()
    args.out.write_text(json.dumps(simulate(args.replay), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
