"""Reconstruct tree ownership, return and feasible planter/courier opportunities on frozen original games."""

from collections import Counter, defaultdict
import gzip, hashlib, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPL = HERE.parent
BASE = REPL.parents[2]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "all_top5"))
from turning_point_ledger import CashEvents, FastCarbon, compact
from economy_extract import distances

PHASES = ("0–99", "100–199", "200–298")


def stage(t):
    return PHASES[min(2, t // 100)]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(f):
    return json.loads(
        gzip.decompress(f.read_bytes()) if f.suffix == ".gz" else f.read_bytes()
    )


def check(a, b, tol=1.0e-5):
    if not math.isclose(a, b, abs_tol=tol, rel_tol=0):
        raise ValueError((a, b))


def safe_rate(n, d):
    return n / d if d else None


def reconstruct(raw, row, income):
    side = row["candidate_seat"]
    config = raw["game_configuration"]
    seed = row["seed"]
    sim = FastCarbon(config).reset(seed=seed)
    if sim.carbon != raw["initial_full_carbon"]:
        raise ValueError("opening mismatch")
    sim.cash = CashEvents(sim.cash)
    distance = distances(config["size"])
    spells = []
    active = {}
    observations = []
    daily = []
    deposits = defaultdict(list)
    phase = defaultdict(lambda: [Counter(), Counter()])
    bases = set(sim.base_at)
    for t, actions in enumerate(raw["actions"]):
        phase_key = stage(t)
        seats = phase[phase_key]
        prev_cash = list(sim.cash)
        own_counts = [len(sim.player_trees[s]) for s in (0, 1)]
        trees = {w.pos: (w.id, w.player, w.age) for w in sim.trees.values()}
        alive = sum(v[2] < config["treeLifespan"] for v in trees.values())
        cost = (
            config["plantCost"]
            + config["plantCostInflationRatio"]
            * config["plantCostInflationBase"] ** alive
        )
        workers = [
            {
                uid: (
                    sim.workers[uid].pos,
                    sim.workers[uid].carbon,
                    sim.workers[uid].is_collector,
                )
                for uid in sim.player_workers[s]
            }
            for s in (0, 1)
        ]
        observations.append(workers)
        possible = {}
        covered = set()
        for s in (0, 1):
            st = seats[s]
            public = sim.obs(s)["carbon"]
            for uid, (cell, load, is_collector) in workers[s].items():
                if is_collector:
                    st["collector_turns"] += 1
                    continue
                st["planter_turns"] += 1
                if cell in bases:
                    continue
                old = trees.get(cell)
                kind = "plant"
                if old and old[2] < config["treeLifespan"]:
                    if old[1] == s:
                        st["standing_on_owned_live_tree"] += 1
                        continue
                    kind = "seize"
                quote = (
                    config["seizeCost"]
                    if kind == "seize"
                    else (
                        config["plantCost"]
                        if own_counts[s] <= config["playerTreeProtectiveNumber"]
                        else cost
                    )
                )
                if prev_cash[s] < quote:
                    st[kind + "_unaffordable"] += 1
                    continue
                if kind == "plant":
                    horizon = min(
                        config["treeLifespan"], config["episodeSteps"] - 2 - t
                    )
                    ground = sum(
                        public[nb] for nb in sim._nbr4[cell] if nb not in trees
                    )
                    nearest_collector = min(
                        (
                            distance[cell][pos]
                            for pos, _, collector in workers[1 - s].values()
                            if collector
                        ),
                        default=99,
                    )
                    nearest_planter = min(
                        (
                            distance[cell][pos]
                            for pos, _, collector in workers[1 - s].values()
                            if not collector
                        ),
                        default=99,
                    )
                    if (
                        ground * config["cellAbsorptionRate"] * horizon >= quote
                        and nearest_collector >= 2
                        and nearest_planter >= 3
                    ):
                        covered.add((s, uid))
                        st["covered_uncontested_plant_opportunities"] += 1
                possible[s, uid] = kind
                st[kind + "_affordable"] += 1
                if actions[s].get(uid) is not None:
                    st[kind + "_affordable_move"] += 1
        sim.cash.events.clear()
        sim.step(*actions)
        after = {w.pos: w.id for w in sim.trees.values()}
        daily_trees = [Counter(), Counter()]
        funds = [Counter(), Counter()]
        for kind, s, value, *d in sim.cash.events:
            funds[s][kind] += value
            if kind == "own_delivery" and value > 0:
                deposits[s, d[0]].append((t + 1, value))
            if kind in ("plant", "seize"):
                uid, cell = d
                tid = after[cell]
                if kind == "seize":
                    prior = spells[active[tid]]
                    if prior["owner"] == s:
                        raise ValueError("self-seizure")
                    prior.update(end="seized", end_before=t + 1)
                before = trees.get(cell)
                spell = dict(
                    tree=tid,
                    cell=cell,
                    owner=s,
                    worker=uid,
                    kind=kind,
                    pay=-value,
                    root=t,
                    start_age=before[2] if kind == "seize" else 0,
                    phase=phase_key,
                    ticks=0,
                    revenue=0.0,
                    tax=0.0,
                    collision=0.0,
                    stage_revenue=Counter(),
                )
                if kind == "seize":
                    prior["next_spell"] = len(spells)

                    def near(which):
                        hist = observations[max(0, t - which)]
                        dists = [
                            (u, round(w[1], 4), distance[cell][w[0]])
                            for u, w in hist[1 - s].items()
                            if w[2]
                        ]
                        return dict(
                            collectors_within_one=sorted(
                                [(u, c) for u, c, d in dists if d <= 1]
                            ),
                            minimum=min((d for u, c, d in dists), default=None),
                            attacker_distance=(
                                distance[cell][hist[s][uid][0]]
                                if uid in hist[s]
                                else None
                            ),
                        )

                    spell["defense_window"] = {
                        str(l): near(l) for l in (0, 1, 3, 5) if t >= l
                    }
                    # A visible affordable stand opportunity was actually taken by the attacking planter.
                    prior["loss_age"] = before[2]
                active[tid] = len(spells)
                spells.append(spell)
                seats[s][kind + "_count"] += 1
                seats[s][kind + "_cost"] -= value
                if (s, uid) in possible:
                    if possible[s, uid] != kind:
                        raise ValueError("purchase type differs")
                    seats[s][kind + "_from_affordable"] += 1
                else:
                    seats[s][kind + "_same_turn_financed"] += 1
                if (s, uid) in covered:
                    seats[s]["covered_uncontested_planted"] += 1
                if kind == "seize":
                    seats[1 - s]["lost_to_seizure"] += 1
            elif kind == "tree_absorption":
                tid = d[0]
                q = spells[active[tid]]
                if q["owner"] != s:
                    raise ValueError("income to wrong owner")
                q["revenue"] += value
                q["tax"] += d[2]
                q["stage_revenue"][phase_key] += value
                daily_trees[s]["absorption"] += value
                daily_trees[s]["worker_tax"] += d[2]
                daily_trees[s]["positive_absorption_tree_turns"] += 1
            elif kind == "tree_collision_cash":
                tid = d[2]
                q = spells[active[tid]]
                if q["owner"] != s:
                    raise ValueError("collision to wrong owner")
                q["collision"] += value
        for tid in list(active):
            q = spells[active[tid]]
            if tid not in sim.trees:
                q.update(end="expired", end_before=t + 1)
                del active[tid]
            else:
                tree = sim.trees[tid]
                if tree.player != q["owner"]:
                    raise ValueError("tree holder mismatch")
                s = tree.player
                q["ticks"] += 1
                daily_trees[s]["held_tree_turns"] += 1
                if q["kind"] != "plant" or q["root"] != t:
                    daily_trees[s]["eligible_tree_turns"] += 1
        for s in (0, 1):
            check(prev_cash[s] + sum(funds[s].values()), sim.cash[s])
            check(
                daily_trees[s]["absorption"],
                sum(sim.trees_dict[tid][1] for tid in sim.player_trees[s]),
                0.00001,
            )
            seats[s].update(daily_trees[s])
            seats[s]["own_delivery"] += funds[s]["own_delivery"]
        daily.append(
            dict(
                t=t,
                cash=list(sim.cash),
                tree=[dict(v) for v in daily_trees],
                cash_entries=[dict(v) for v in funds],
            )
        )
    for k in active.values():
        spells[k].update(end="terminal", end_before=sim.step_count + 1)
    for q in spells:
        if "defense_window" in q:
            near = q["defense_window"]["0"]["collectors_within_one"]
            q["actual_next5_deposits_of_reachable_defenders"] = [
                [uid, j, round(val, 4)]
                for uid, load in near
                for j, val in deposits[1 - q["owner"], uid]
                if q["root"] < j <= q["root"] + 5
            ]
        q["stage_revenue"] = {k: round(v, 6) for k, v in q["stage_revenue"].items()}
    if (
        list(sim.cash) != raw["last_post_cash"]
        or sha(compact(sim.get_state())) != row["final_state_sha256"]
    ):
        raise ValueError("terminal mismatch")
    total = [Counter(), Counter()]
    for m in phase.values():
        for s in (0, 1):
            total[s].update(m[s])
    for s in (0, 1):
        tally = income["own"] if s == side else income["examiner"]
        for a, b in [
            ("absorption", "tree_absorption"),
            ("own_delivery", "own_delivery"),
        ]:
            check(total[s][a], tally.get(b, 0))
        for k in ("plant", "seize"):
            check(-total[s][k + "_cost"], tally.get(k, 0))
            check(
                total[s][k + "_count"],
                sum(q["kind"] == k and q["owner"] == s for q in spells),
            )
        check(
            total[s]["held_tree_turns"],
            sum(q["ticks"] for q in spells if q["owner"] == s),
        )
        check(
            total[s]["absorption"], sum(q["revenue"] for q in spells if q["owner"] == s)
        )
    return phase, total, spells, daily


def decomp(own, enemy):
    a, b = own["held_tree_turns"], enemy["held_tree_turns"]
    ra, rb = own["absorption"] / a, enemy["absorption"] / b
    volume = (a - b) * (ra + rb) / 2
    rate = (ra - rb) * (a + b) / 2
    check(volume + rate, own["absorption"] - enemy["absorption"])
    return dict(volume=volume, rate=rate, gap=volume + rate, own_rate=ra, enemy_rate=rb)


def run():
    finance = read(REPL / "income.json")
    games = []
    allspells = []
    steps = []
    for arm in ("parent", "learned"):
        folder = REPL / arm
        results = read(folder / "results.json")
        proof = read(folder / "official_replay.json")["runs"][arm]
        if results["status"] != "complete" or len(results["games"]) != 4:
            raise ValueError("native four-game arm incomplete")
        for r in results["games"]:
            seed = r["seed"]
            seat = r["candidate_seat"]
            f = folder / r["raw_file"]
            o = next(
                g for g in proof["games"] if (g["seed"], g["seat"]) == (seed, seat)
            )
            if (
                sha(f.read_bytes()) != r["raw_sha256"]
                or o["transcript_sha256"] != r["raw_sha256"]
                or o["official_reward_by_seat"] != r["official_score_by_seat"]
            ):
                raise ValueError("official/actual mismatch")
            raw = read(f)
            receipt = next(
                q for q in finance[arm] if (q["seed"], q["seat"]) == (seed, seat)
            )
            if receipt["raw_sha256"] != r["raw_sha256"]:
                raise ValueError("cash input identity")
            phase, total, spells, daily = reconstruct(raw, r, receipt)
            phase["all"] = total
            info = dict(arm=arm, seed=seed, seat=seat)
            out = dict(
                **info,
                margin=r["margin"],
                official=r["official_score_by_seat"],
                raw=r["raw_sha256"],
                pack=raw["pack_sha256"],
                phase={}
            )
            for key in list(PHASES) + ["all"]:
                own, en = phase[key][seat], phase[key][1 - seat]
                out["phase"][key] = dict(
                    own=dict(own), examiner=dict(en), decomposition=decomp(own, en)
                )
            losses = {}
            for label, s in [("own", seat), ("examiner", 1 - seat)]:
                ended = [q for q in spells if q["owner"] == s]
                lost = [q for q in ended if q["end"] == "seized"]
                winners = [spells[q["next_spell"]] for q in lost]
                end_kinds = Counter(q["end"] for q in ended)
                defender = [
                    q
                    for q in winners
                    if q["defense_window"]["0"]["collectors_within_one"]
                ]
                losses[label] = dict(
                    held_spells=len(ended),
                    end=dict(end_kinds),
                    after_seizure_opponent_spells=len(winners),
                    subsequent_opponent_absorption=sum(q["revenue"] for q in winners),
                    subsequent_opponent_taxes=sum(q["tax"] for q in winners),
                    subsequent_opponent_ticks=sum(q["ticks"] for q in winners),
                    before_loss_ticks=sum(q["ticks"] for q in lost),
                    loss_age_sum=sum(q["loss_age"] for q in lost),
                    loss_future_absorption_by_stage={
                        ph: sum(q["stage_revenue"].get(ph, 0) for q in winners)
                        for ph in PHASES
                    },
                    adjacent_collector_at_loss=len(defender),
                    adjacent_reachable_with_cargo100=sum(
                        any(
                            c >= 100
                            for uid, c in q["defense_window"]["0"][
                                "collectors_within_one"
                            ]
                        )
                        for q in defender
                    ),
                    adjacent_returned_next5=sum(
                        bool(q["actual_next5_deposits_of_reachable_defenders"])
                        for q in defender
                    ),
                    adjacent_actual_next5_cash=sum(
                        sum(
                            d[2]
                            for d in q["actual_next5_deposits_of_reachable_defenders"]
                        )
                        for q in defender
                    ),
                    own_adjacent_at_minus1=sum(
                        bool(
                            q["defense_window"]
                            .get("1", {})
                            .get("collectors_within_one")
                        )
                        for q in winners
                    ),
                    mean_ticks_per_spell=safe_rate(
                        sum(q["ticks"] for q in ended), len(ended)
                    ),
                )
            out["loss"] = losses
            games.append(out)
            allspells.append(dict(**info, spells=spells))
            steps.append(dict(**info, ticks=daily))
    (HERE / "games.json").write_text(
        json.dumps(games, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    for name, record in [("spells", allspells), ("turns", steps)]:
        (HERE / (name + ".json.gz")).write_bytes(
            gzip.compress(
                json.dumps(
                    record, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode(),
                mtime=0,
            )
        )
    print(
        [
            (
                g["arm"],
                g["seed"],
                g["seat"],
                round(g["phase"]["all"]["decomposition"]["gap"], 2),
            )
            for g in games
        ]
    )


if __name__ == "__main__":
    run()
