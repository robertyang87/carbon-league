"""Replay archived physical roots and compare independent two-policy continuations."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from collections import defaultdict
from statistics import mean
from time import perf_counter, process_time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent
CARGO = CORPUS.parent
ACTOR = CARGO.parent
SAPLING = ACTOR.parent
ROOT = SAPLING.parents[1]
BRIDGE = ACTOR / "own_search_bridge_dev"
OLD = BRIDGE / "opponent_response_dev"
PACK = CARGO / "candidate.py"
MODELS = {"published": OLD / "opponent.npz", "calibrated": OLD / "active_calibration/opponent.npz",
          "new": HERE / "model.npz"}
HIDDEN = {"published": 40, "calibrated": 41, "new": 48}
ROOTS = (70, 175)
HORIZONS = (5, 10)
PROTOCOL_SHA = "a68d40240cfde60299079f59fccead4b17e30398f438aae8b50b4c892173ebef"
OTHER_PACK = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
sys.path[:0] = [str(BRIDGE), str(SAPLING), str(ROOT / "code")]
import numpy as np
import bridge
from fastsim import FastCarbon
from turning_point_ledger import CashEvents


def digest(obj):
    b = json.dumps(obj, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b).hexdigest()


def require(ok, context):
    if not ok:
        raise ValueError(context)


def write(path, obj):
    if path.suffix == ".gz":
        b = gzip.compress((json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(), mtime=0)
    else:
        b = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(b)
    tmp.replace(path)


def extinct(sim):
    floor = min(sim.config["recCollectorCost"], sim.config["recPlanterCost"])
    return [not sim.player_workers[s] and not sim.player_trees[s] and (not sim.bases[s] or sim.cash[s] < floor)
            for s in (0, 1)]


def transition(sim, joint):
    """Reconcile every FastCarbon cash write, isolating the normal worker-to-own-base deposit."""
    before = list(sim.cash)
    sim.cash.events.clear()
    sim.step(*joint)
    ledger, delivered = list(before), [0., 0.]
    for kind, seat, signed, *_ in sim.cash.events:
        ledger[seat] += signed
        if kind == "own_delivery":
            delivered[seat] += signed
    require(all(abs(ledger[s] - sim.cash[s]) < 1e-7 for s in (0, 1)), "unreconciled cash write")
    return delivered


def settlement(sim, cash0, deposited):
    positions = [sorted(sim.trees[tid].pos for tid in sim.player_trees[s]) for s in (0, 1)]
    return {"state_t": sim.step_count, "state_sha256": digest(sim.get_state()), "end_cash": list(sim.cash),
            "net_cash": [round(sim.cash[s] - cash0[s], 8) for s in (0, 1)],
            "normal_deposit": [round(v, 8) for v in deposited], "tree_count": list(map(len, positions)),
            "tree_positions": positions, "extinct": extinct(sim)}


def warm_and_replay(card, inv, actor):
    raw = bridge.load(ROOT / card["path"])
    seed, side, expert = card["seed"], card["candidate_seat"], card["expert_seat"]
    require(inv["verified"] and inv["native_action_supervision"] and inv["full_official_terminal"], "unverified archive")
    require(bridge.sha(ROOT / card["path"]) == inv["raw_sha256"] == card["raw_sha256"], "archive file identity")
    require((raw["seed"], raw["candidate_seat"], raw["pack_sha256"]) ==
            (seed, side, card["opponent_pack_sha256"]) and side == 1 - expert, "archive lineup")
    cfg = raw["game_configuration"]
    public_cfg = {k: v for k, v in cfg.items() if k != "randomSeed"}
    sim = FastCarbon(cfg).reset(seed=seed)
    require(len(sim.carbon) == 225 and np.all(np.asarray(sim.carbon) == np.asarray(sim.carbon, dtype="uint8")),
            "noninteger initial terrain")
    require(hashlib.sha256(bytes(map(int, sim.carbon))).hexdigest() == card["physical_sha256"] == inv["physical_sha256"],
            "physical board identity")
    if "initial_full_carbon" in raw:
        require(sim.carbon == raw["initial_full_carbon"], "opening cells differ")
    sim.cash = CashEvents(sim.cash)
    feat = bridge.F.Feat(expert)
    history = [list(sim.cash)]
    counters = {}
    snapshots, truth = {}, {}
    actions = raw["actions"]
    require(digest(actions) == inv["actions_sha256"] and len(actions) == raw["steps"] == inv["transitions"], "historical action tape")
    for t, joint in enumerate(actions):
        if t in ROOTS:
            snapshots[t] = (copy.deepcopy(sim), copy.deepcopy(actor._FEAT), copy.deepcopy(feat))
            counters[t] = [0., 0.]
            truth[t] = {"root": settlement(sim, sim.cash, [0., 0.]), "horizons": {}}
        if t < max(ROOTS):
            obs = sim.obs(side)
            if t == 0:
                actor.agent(obs, public_cfg)
            else:
                actor._FEAT.observe(obs)
            feat.observe(sim.obs(expert))
        require(sim.step_count == t and not any(extinct(sim)), "post-terminal historical decision")
        deposits = transition(sim, joint)
        history.append(list(sim.cash))
        for root, tally in counters.items():
            lag = sim.step_count - root
            if 0 < lag <= max(HORIZONS):
                for s in (0, 1):
                    tally[s] += deposits[s]
                if lag in HORIZONS:
                    truth[root]["horizons"][lag] = settlement(sim, snapshots[root][0].cash, tally)
    gone = extinct(sim)
    n = len(actions)
    require(n == cfg["episodeSteps"] - 1 or any(gone), "incomplete archive")
    rewards = [(n - cfg["episodeSteps"] - 1 if gone[s] else history[-2][s])
               if any(gone) else sim.cash[s] for s in (0, 1)]
    require(gone == inv["extinct_by_seat"] == raw["extinct_by_seat"] and
            rewards == inv["official_score_by_seat"] == raw["official_score_by_seat"], "terminal reward difference")
    require(history[-2:] == [inv["terminal_pre_cash"], inv["terminal_post_cash"]] and
            history[-2:] == [raw["last_pre_cash"], raw["last_post_cash"]], "terminal cash difference")
    require(len(history) == inv["official_cash_observations"] and digest(history) == inv["official_cash_path_sha256"],
            f"complete archived cash path differs for {card['path']} (FastCarbon {digest(history)})")
    audit = {**card, "historical_action_sha256": inv["actions_sha256"], "cash_observations": len(history),
             "official_cash_path_sha256": digest(history), "terminal_reward": rewards, "terminal_extinct": gone,
             "terminal_state_sha256": digest(sim.get_state())}
    return public_cfg, actions, snapshots, truth, audit


def errors(predicted, observed, expert):
    out = {}
    for metric in ("normal_deposit", "net_cash", "end_cash", "tree_count"):
        pred, actual = predicted[metric], observed[metric]
        diff = [pred[s] - actual[s] for s in (0, 1)]
        for label, value in (("expert", diff[expert]), ("sapling", diff[1 - expert]),
                             ("expert_minus_sapling", diff[expert] - diff[1 - expert])):
            out[f"{label}_{metric}"] = round(value, 8)
    for label, side in (("expert", expert), ("sapling", 1 - expert)):
        out[f"{label}_tree_position_symmetric_difference"] = len(
            set(predicted["tree_positions"][side]) ^ set(observed["tree_positions"][side]))
    return out


def branch(snapshot, truth, public_cfg, expert, model, actor, clocks):
    sim, actor._FEAT, feature = copy.deepcopy(snapshot)
    root = sim.step_count
    side = 1 - expert
    require(digest(sim.get_state()) == truth["root"]["state_sha256"], "branch root differs")
    initial = list(sim.cash)
    cumulative = [0., 0.]
    commands, endpoints, event_amounts = [], {}, []
    for k in range(1, max(HORIZONS) + 1):
        if not any(extinct(sim)) and sim.step_count < sim.config["episodeSteps"] - 1:
            started = perf_counter()
            actor_obs = sim.obs(side)
            actor_command = actor.agent(actor_obs, public_cfg)
            middle = perf_counter()
            expert_obs = sim.obs(expert)
            x = feature.observe(expert_obs)
            expert_command, _ = bridge.decode(feature, x, model)
            decoded = perf_counter()
            joint = [None, None]
            joint[side], joint[expert] = actor_command, expert_command
            deposits = transition(sim, joint)
            ended = perf_counter()
            for key, time_s in (("sapling", middle - started), ("opponent", decoded - middle), ("joint_with_engine", ended - started)):
                for h in HORIZONS:
                    if k <= h:
                        clocks[h][key].append(time_s)
            commands.append(joint)
            event_amounts.append(deposits)
            for s in (0, 1):
                cumulative[s] += deposits[s]
        if k in HORIZONS:
            predicted = settlement(sim, initial, cumulative)
            endpoints[k] = {"outcome": predicted, "error": errors(predicted, truth["horizons"][k], expert),
                            "joint_actions_sha256": digest(commands), "effective_transitions": sim.step_count - root}
    return {"commands": commands, "deposit_by_step_by_seat": event_amounts, "horizons": endpoints}


def summarize(rows):
    metrics = {}
    for split in ("exact", "test"):
        for subset in ("both", "0", "1"):
            chosen = [r for r in rows if r["split"] == split and (subset == "both" or int(subset) == r["expert_seat"])]
            result = {}
            for arm in MODELS:
                result[arm] = {}
                for h in HORIZONS:
                    by_map = defaultdict(list)
                    for r in chosen:
                        by_map[r["seed"]].append(r["arms"][arm]["horizons"][h]["error"])
                    keys = next(iter(by_map.values()))[0].keys()
                    maes = {seed: {k: mean(abs(row[k]) for row in samples) for k in keys} for seed, samples in by_map.items()}
                    result[arm][h] = {"map_equal_mae": {k: mean(d[k] for d in maes.values()) for k in keys},
                                      "map_equal_mean_signed_error": {k: mean(mean(x[k] for x in group) for group in by_map.values()) for k in keys},
                                      "mae_by_map": maes}
            metrics.setdefault(split, {})[subset] = {"maps": len({r["seed"] for r in chosen}), "roots": len(chosen), "arm": result}
    return metrics


def latency(samples):
    return {"calls": len(samples), "wall_seconds": sum(samples), "calls_per_second": len(samples)/sum(samples),
            "median_ms": 1000 * float(np.median(samples)), "p95_ms": 1000 * float(np.percentile(samples, 95))}


def main():
    os.sched_setaffinity(0, {3})
    start_cpu, start_wall = process_time(), perf_counter()
    protocol_path = HERE / "protocol.json"
    require(bridge.sha(protocol_path) == PROTOCOL_SHA, "protocol changed")
    protocol = bridge.load(protocol_path)
    for p, expected in protocol["input_sha256"].items():
        require(bridge.sha(ROOT / p) == expected, f"frozen input changed: {p}")
    fit = bridge.load(HERE / "fit.json")
    require(fit["weights_sha256"] == bridge.sha(MODELS["new"]) and fit["input_protocol_sha256"] == PROTOCOL_SHA and
            fit["source_sha256"] == bridge.sha(HERE / "sources.json.gz"), "new weights are not the frozen fit")
    weights = {}
    for arm, path in MODELS.items():
        with np.load(path, allow_pickle=False) as src:
            weights[arm] = {k: src[k] for k in src.files}
        width = HIDDEN[arm]
        require({k: v.shape for k, v in weights[arm].items()} == {"hidden": (83, width), "bias": (width,), "head": (width, 1)},
                f"{arm} shape")
    spec = importlib.util.spec_from_file_location("closed_public_sapling", PACK)
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    sources = bridge.load(HERE / "sources.json.gz")
    with gzip.open(CORPUS / "expansion/inventory.jsonl.gz", "rt") as inp:
        inventory = {row["path"]: row for line in inp if (row := json.loads(line))}
    held_terrains = {c["physical_sha256"] for tag in ("exact", "test") for c in sources[tag]}
    require(len(held_terrains) == 10 and held_terrains.isdisjoint(c["physical_sha256"] for tag in ("train", "val") for c in sources[tag]),
            "fixed model training and closed map overlap")
    rows, tapes, clocks = [], [], defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for tag in ("exact", "test"):
        panel = sources[tag]
        require(sorted((r["seed"], r["expert_seat"]) for r in panel) ==
                sorted((s, seat) for s in protocol["seeds_by_split"][tag] for seat in (0, 1)), "missing or extra paired terrain")
        for card in panel:
            side, expert = card["candidate_seat"], card["expert_seat"]
            require(card["opponent_pack_sha256"] == (bridge.sha(PACK) if tag == "exact" else OTHER_PACK), "wrong lineup")
            cfg, archived, snapshots, truth, audit = warm_and_replay(card, inventory[card["path"]], actor)
            tapes.append({**audit, "split": tag})
            for root in ROOTS:
                logged = archived[root:root + max(HORIZONS)]
                row = {**card, "split": tag, "root_t": root, "truth": truth[root], "archive_commands": logged,
                       "archive_joint_actions_sha256": {h: digest(logged[:h]) for h in HORIZONS}, "arms": {}}
                for arm, model in weights.items():
                    record = branch(snapshots[root], truth[root], cfg, expert, model, actor, clocks[(tag, arm)])
                    record["actor_matches_logged_root"] = record["commands"][0][side] == logged[0][side]
                    row["arms"][arm] = record
                require(len({digest(r["commands"][0][side]) for r in row["arms"].values()}) == 1, "self actor differs at shared root")
                if tag == "exact":
                    require(all(r["actor_matches_logged_root"] for r in row["arms"].values()), "archived 8da4 action differs at shared root")
                rows.append(row)
            print(tag, card["seed"], expert, f"{process_time() - start_cpu:.1f} CPU seconds", flush=True)
            if process_time() - start_cpu > 590:
                break
        if process_time() - start_cpu > 590:
            break
    complete = len(tapes) == 20 and len(rows) == 40
    binding = {"protocol_sha256": PROTOCOL_SHA, "source_sha256": bridge.sha(HERE / "sources.json.gz"),
               "inventory_sha256": bridge.sha(CORPUS / "expansion/inventory.jsonl.gz"),
               "actor_sha256": bridge.sha(PACK), "fit_sha256": bridge.sha(HERE / "fit.json"),
               "model_sha256": {k: bridge.sha(p) for k, p in MODELS.items()}, "script_sha256": bridge.sha(Path(__file__)),
               "cash_instrument_sha256": bridge.sha(SAPLING / "turning_point_ledger.py")}
    write(HERE / "closed.json.gz", {"binding": binding, "complete": complete, "tapes": tapes, "roots": rows})
    summary = {"binding": binding, "closed_sha256": bridge.sha(HERE / "closed.json.gz"), "complete": complete,
               "tapes": len(tapes), "roots": len(rows), "archived_transitions_verified": sum(t["transitions"] for t in tapes),
               "archived_terminal_verified": len(tapes), "branches": len(rows)*len(MODELS),
               "lineups": {"exact": "Historical and simulated Sapling both 8da4, 4 paired terrains",
                           "test": "Historical Sapling 9c54df17, simulated Sapling 8da4, 6 paired terrains; published baseline previously trained on 621495120"},
               "settlement": "Both actor histories warmed through root-1 on their physical seat's public observations. Archive executes recorded joint commands only. Each continuation gets the same root with both authentic ID counters. For each turn 8da4 and its paired opponent each see only their own branch-local public observation. Only an engine extinction or episode end can absorb a branch. Endpoints settle after 5/10 joint actions. Cash means end minus root. Normal deposit sums the net cash credited by own worker returning to own base, excluding captures, tree income, enemy workers and hire costs. Tree control measures live tree counts and physical cell ownership at the endpoint. Errors are predicted minus archived with magnitudes averaged within terrain first.",
               "throughput_method": "Single process pinned to CPU3, numerical thread counts 1. Wall times include constructing the acting seat observation and the normal callable or opponent feature and decode. Joint includes both policies and instrumented engine settlement. History warming, archive validation, endpoint hashing and disk I/O excluded from per-call times. Fully sequential branches.",
               "throughput": {tag: {arm: {h: {k: latency(v) for k, v in steps.items()} for h, steps in parts.items()} for (split, arm), parts in clocks.items() if split == tag} for tag in ("exact", "test")},
               "process_seconds": process_time() - start_cpu, "wall_seconds": perf_counter() - start_wall}
    if complete:
        summary["evaluation"] = summarize(rows)
        exact = summary["evaluation"]["exact"]["both"]["arm"]
        delta = {h: {metric: exact["new"][h]["map_equal_mae"][metric] - exact["published"][h]["map_equal_mae"][metric]
                     for metric in ("expert_normal_deposit", "expert_tree_count")} for h in HORIZONS}
        summary["closed_gate"] = {"mae_new_minus_published": delta,
                                  "passed_closed_conditions": all(v < 0 for v in delta[10].values()) and all(v <= 0 for v in delta[5].values()),
                                  "interpretation": "The separate protocol free-held NLL criterion must also pass."}
        summary["actor_logged_root_matches"] = {tag: sum(r["arms"]["new"]["actor_matches_logged_root"] for r in rows if r["split"] == tag)
                                                 for tag in ("exact", "test")}
    write(HERE / "closed_summary.json", summary)
    require(complete, "closed evaluation stopped at the frozen CPU budget")


if __name__ == "__main__":
    main()
