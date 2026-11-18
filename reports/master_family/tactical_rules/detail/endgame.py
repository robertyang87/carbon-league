"""Late cash conversion, shortest paths, and observed outcomes for the same worker.

Run through mine.py. All conditions use the pre-action public board. Commands
come from frame t+1. Later positions are used only to score a return, never to
classify urgency or mark a path safe. The optional single-step counterfactual
restores the arbiter state AFTER defining the alternative from public inputs.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import statistics
import sys

MOVES = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
ROOT = Path(__file__).resolve().parents[4]


@lru_cache(maxsize=32)
def geometry(n, homes):
    actions = []
    distances = []
    for cell in range(n * n):
        r, c = divmod(cell, n)
        actions.append({a: ((r + dr) % n) * n + ((c + dc) % n)
                        for a, (dr, dc) in MOVES.items()})
        options = []
        for home in homes:
            hr, hc = divmod(home, n)
            dr, dc = abs(hr - r), abs(hc - c)
            options.append(min(dr, n - dr) + min(dc, n - dc))
        distances.append(min(options))
    return actions, distances


def counterfactual(replay, seat, worker, action):
    """Change one loaded worker's final action; all other submitted actions held fixed."""
    if str(ROOT / "code") not in sys.path:
        sys.path.insert(0, str(ROOT / "code"))
    from fastsim import FastCarbon

    root, final = replay["steps"][-2:]
    baseline = FastCarbon(replay["configuration"]).load_observation(root[0]["observation"])
    initial = baseline.get_state()
    actions = [dict(side["action"] or {}) for side in final]
    baseline.step(*actions)
    observed = final[0]["observation"]["players"]
    original = baseline.full_observation()["players"]
    # Dynamic ID counters lost in replay restoration can make exact parity fail.
    # Do not attribute an effect to the move on any unverified root.
    if original != observed or [s[0] for s in observed] != replay["rewards"]:
        return {"baseline_exact": False}
    alt = FastCarbon(replay["configuration"]).set_state(initial)
    actions[seat][worker] = action
    alt.step(*actions)
    diff = alt.cash[seat] - alt.cash[1 - seat]
    baseline_diff = observed[seat][0] - observed[1 - seat][0]
    # The terminal cash-margin comparison applies only when neither side has
    # been eliminated. If changing this action changes extinction, drop it.
    def extinct(sim, side):
        return (not sim.player_workers[side] and not sim.player_trees[side]
                and (not sim.bases[side] or sim.cash[side] < min(sim.config["recCollectorCost"],
                                                                sim.config["recPlanterCost"])))
    if any(extinct(sim, side) for sim in (baseline, alt) for side in (0, 1)):
        return {"baseline_exact": True, "cash_terminal": False}
    return {"baseline_exact": True, "cash_terminal": True, "actual_margin": round(baseline_diff, 2),
            "alternate_margin": round(diff, 2), "delta_margin": round(diff - baseline_diff, 2)}


def final_home_traffic(frames, seat, config, directions):
    """Resolve same-team traffic only on base-steps with a positive inbound collector."""
    library = ROOT / "reports/master_family/deep_review"
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    from carbon_deep_events import analyze_events

    cap = config["episodeSteps"] - 1
    buckets = defaultdict(Counter)
    example = None
    for t in range(cap - 10, min(cap, len(frames) - 1)):
        own = frames[t][0]["observation"]["players"][seat]
        commands = frames[t + 1][seat]["action"] or {}
        own_bases = set(own[1].values())
        inbound = set()
        for worker, (pos, cargo, kind) in own[2].items():
            action = commands.get(worker)
            if kind == "COLLECTOR" and cargo > 0 and pos not in own_bases and action:
                dst = directions[pos][action]
                if dst in own_bases:
                    inbound.add(dst)
        if not inbound:
            continue
        ledger = analyze_events(frames[t:t + 2], seat, configuration=config)
        if ledger["metrics"]["verified_transitions"] != 1:
            raise ValueError(f"Inconsistent final traffic ledger at {t}: {ledger['anomalies']}")
        period = "last_action" if t == cap - 1 else "previous_nine"
        bucket = buckets[period]
        bucket["positive_inbound_base_steps"] += len(inbound)
        for event in ledger["events"]:
            cell = event["position"]
            if (event["kind"] != "collision" or cell not in inbound
                    or event["contested"] or event["own_participants"] < 2):
                continue
            assert event["recipient_seat"] == seat
            gross = sum(worker["known_cargo"] for worker in event["participants"])
            if event["recipient"] == "base_cash":
                received = event["own_cargo_retained"]
            else:
                deliveries = [x["net_cash"] for x in ledger["events"]
                              if x["kind"] == "delivery" and x["position"] == cell]
                if len(deliveries) != 1:
                    raise ValueError(f"Missing pooled deposit at {t}/{cell}")
                received = deliveries[0]
            if abs(received - gross) > 1e-5:
                raise ValueError(f"Shared cash differs from arriving cargo {t}/{cell}")
            bucket["all_friendly_collision_base_steps"] += 1
            bucket["own_workers_removed"] += event["own_deaths"]
            bucket["cargo_credited_once"] += received
            bucket[event["recipient"] + "_route"] += 1
            if t == cap - 1 and example is None:
                example = {"t": t, "base_cell": cell, "starting_cargo_by_worker": [
                    [w["worker_id"], w["known_cargo"]] for w in event["participants"]],
                    "workers_removed": event["deleted_worker_ids"], "winner": event["winner_worker_id"],
                    "recipient": event["recipient"], "team_credit": round(received, 4)}
    return dict(buckets), example


def observe(replay, seat, key):
    frames = replay["steps"]
    cfg = replay["configuration"]
    cap = cfg["episodeSteps"] - 1
    if cfg["moveCost"] != 0 or cfg["smeltCost"] != 0:
        raise ValueError("Cargo and collision comparisons require the zero move/smelt fees of this cohort")
    counts = defaultdict(Counter)
    examples = {}
    trials = []
    if len(frames) <= cap - 32:
        return {"groups": {}, "trials": [], "examples": {}, "traffic": {}}
    homes = tuple(sorted(frames[0][0]["observation"]["players"][seat][1].values()))
    actions, dist = geometry(cfg["size"], homes)
    future_banks = set()
    for t in range(len(frames) - 2, max(-1, cap - 33), -1):
        pre = frames[t][0]["observation"]
        post = frames[t + 1][0]["observation"]
        if (pre["step"], post["step"]) != (t, t + 1):
            raise ValueError(f"Step discontinuity in {key}")
        own = pre["players"][seat]
        enemy = pre["players"][1 - seat]
        own_cells = {w[0] for w in own[2].values()}
        after = post["players"][seat][2]
        commands = frames[t + 1][seat]["action"] or {}
        enemy_lightest = {}
        enemy_reach = set()
        for loc, load, kind in enemy[2].values():
            targets = [loc, *actions[loc].values()]
            enemy_reach.update(targets)
            if kind == "COLLECTOR":
                for dst in targets:
                    enemy_lightest[dst] = min(load, enemy_lightest.get(dst, float("inf")))
        for worker, (cell, cargo, kind) in own[2].items():
            if kind != "COLLECTOR" or cargo <= 0 or cell in homes:
                continue
            command = commands.get(worker) or "NONE"
            target = cell if command == "NONE" else actions[cell][command]
            d = dist[cell]
            slack = cap - t - d
            toward = dist[target] < d
            paid = (target in homes and worker in after and after[worker][0] == target
                    and after[worker][1] == 0)
            if paid:
                future_banks.add(worker)
            eventually = worker in future_banks
            if slack < 0 and eventually:
                raise ValueError(f"Positive return beyond geometric reach in {key}")
            safer = any(dist[dst] < d and dst not in own_cells
                        and dst not in enemy[1].values()
                        and enemy_lightest.get(dst, float("inf")) > cargo
                        for dst in actions[cell].values())
            groups = []
            if d == 3 and cargo >= 100 and safer:
                if slack == 0:
                    groups.append("d3_heavy_safe_exact")
                elif 4 <= slack <= 8:
                    groups.append("d3_heavy_safe_buffer4_8")
            if d == 3 and 0 < cargo < 30 and safer:
                if slack == 0:
                    groups.append("d3_light_safe_exact")
                elif 4 <= slack <= 8:
                    groups.append("d3_light_safe_buffer4_8")
            if 2 <= d <= 5 and cargo >= 100 and slack == 0:
                groups.append("d2_5_heavy_exact_safe" if safer else "d2_5_heavy_exact_exposed")
                groups.append("d2_5_heavy_exact_toward" if toward else "d2_5_heavy_exact_delay")
            if d == 3 and cargo >= 100 and -3 <= slack < 0 and safer:
                groups.append("d3_heavy_safe_too_late")
            if d == 1:
                near = [dst for dst in actions[cell].values() if dst in homes]
                if len(near) == 1 and near[0] not in own_cells:
                    home = near[0]
                    clear = home not in enemy_reach
                    if t == cap - 1:
                        groups.append("final_step_adjacent_clear" if clear else "final_step_adjacent_enemy_reach")
                        if clear:
                            groups.append("final_step_adjacent_clear_light" if cargo < 30 else
                                          "final_step_adjacent_clear_ge30")
                            if not toward:
                                action = next(a for a, dst in actions[cell].items() if dst == home)
                                trial = counterfactual(replay, seat, worker, action)
                                trials.append({"step": t, "worker": worker, "cargo": cargo,
                                               "actual": command, "alternate": action, **trial})
                    if clear and 4 <= slack <= 8 and cargo < 30:
                        groups.append("d1_light_clear_buffer4_8")
            for group in groups:
                bucket = counts[group]
                bucket["decisions"] += 1
                bucket["toward"] += int(toward)
                bucket["stay"] += int(command == "NONE")
                bucket["same_step_bank"] += int(paid)
                bucket["eventual_bank"] += int(eventually)
                if group not in examples and group in ("d2_5_heavy_exact_delay", "d3_heavy_safe_too_late"):
                    examples[group] = {"t": t, "worker": worker, "cargo": cargo, "dist": d,
                                       "slack": slack, "action": command, "closer": toward,
                                       "positive_self_bank_before_end": eventually}
    traffic, collision = final_home_traffic(frames, seat, cfg, actions)
    if collision:
        examples["final_home_pool"] = collision
    return {"groups": dict(counts), "examples": examples, "trials": trials, "traffic": traffic}


def summarize(rows):
    pairs = {"heavy_exact_vs_buffer": ("d3_heavy_safe_exact", "d3_heavy_safe_buffer4_8"),
             "light_exact_vs_buffer": ("d3_light_safe_exact", "d3_light_safe_buffer4_8"),
             "light_last_step_vs_buffer": ("final_step_adjacent_clear_light", "d1_light_clear_buffer4_8")}
    output = {}
    for name in sorted({r["model"] for r in rows}):
        appearances = [r for r in rows if r["model"] == name]
        grouped = defaultdict(Counter)
        diffs = defaultdict(list)
        examples = {}
        counter = Counter()
        traffic = defaultdict(Counter)
        for r in appearances:
            obs = r["observation"]
            for period, values in obs["traffic"].items():
                traffic[period].update(values)
                traffic[period]["appearances_with_inbound"] += 1
                traffic[period]["appearances_with_friendly_collision"] += int(
                    values.get("all_friendly_collision_base_steps", 0) > 0)
            for group, values in obs["groups"].items():
                grouped[group].update(values)
                grouped[group]["appearances"] += 1
            for contrast, (a, b) in pairs.items():
                va, vb = obs["groups"].get(a), obs["groups"].get(b)
                if va and vb:
                    diffs[contrast].append(va["toward"] / va["decisions"] - vb["toward"] / vb["decisions"])
            for k, v in obs["examples"].items():
                examples.setdefault(k, {"key": r["key"], "seat": r["seat"], **v})
            for trial in obs["trials"]:
                counter["candidate_alternatives"] += 1
                if trial["baseline_exact"] and trial.get("cash_terminal"):
                    gain = trial["delta_margin"]
                    counter["verified"] += 1
                    counter["positive"] += int(gain > 0)
                    counter["negative"] += int(gain < 0)
                    counter["zero"] += int(gain == 0)
                    counter["changed_outcome"] += int((trial["actual_margin"] > 0) !=
                                                     (trial["alternate_margin"] > 0) or
                                                     (trial["actual_margin"] == 0) !=
                                                     (trial["alternate_margin"] == 0))
                    key = "verified_positive" if gain > 0 else "verified_other"
                    examples.setdefault(key, {"key": r["key"], "seat": r["seat"], **trial})
        totals = {g: dict(v) for g, v in grouped.items()}
        matched = {k: {"appearances": len(ds), "positive": sum(v > 0 for v in ds),
                       "mean_difference": statistics.mean(ds)} for k, ds in diffs.items()}
        output[name] = {"appearances": len(appearances), "groups": totals, "same_game": matched,
                        "final_step_counterfactuals": dict(counter), "examples": examples,
                        "friendly_base_traffic": dict(traffic)}
    return {"models": output, "definitions": {
        "slack": "episodeSteps-1-t minus toroidal Manhattan steps to nearest own home; t is decision source.",
        "safe": "At least one strictly shorter next cell has no current friendly worker, is not an enemy base and cannot be reached by a lighter or equal enemy collector. This is a one-step screen; team/enemy simultaneous choices still matter.",
        "groups": "Positive cargo on an off-base collector only. Heavy means >=100; light means (0,30). Outcomes concern the same worker, not team-wide recovery of its cargo. Repeated worker-steps are correlated.",
        "eventual_bank": "Same surviving worker entered own base from off-base carrying positive carbon with observed cargo zero on arrival, on this or a later recorded transition. No future outcomes are features.",
        "adjacent_clear": "One own base is adjacent, no current ally on it and no enemy worker can reach it this step. Other allies can still choose to enter simultaneously.",
        "traffic": "Denominator is a home base-step with at least one positive off-base allied collector commanded to enter. Count collisions with exclusively friendly participants including actual newborns/stayers. Credit is either the pooled surviving collector deposit or direct home cash on tie, never summed twice. Commanded collision may be accidental; no deployment override is tested.",
        "counterfactual": "For each final-step missed clear home, change one worker's final action only, hold all other recorded commands constant. Report terminal cash margin only when exact baseline players and official cash agree and both sides remain nonextinct. Full carbon used only in this simulator audit. Counts are one-worker alternatives, not independent games or policy gain."}}
