"""Audit original-engine observations and cash/cargo in the paired freight development runs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from official_game_replay import ENGINE, make, np
from tree_win_gate import EXAMINER, ROOT, validate_game, verify_raw
from turning_point_ledger import CashEvents, FastCarbon, compact, sha, sums_by_seat, write


PHASES = ((1, 100), (101, 200), (201, 299))
EXAMINER_PATH = "sources/master3s2/master3s2.py"


def fleet(sim, seat):
    own = [sim.workers[w] for w in sim.player_workers[seat]]
    return dict(cash=sim.cash[seat], cargo=round(sum(w.carbon for w in own), 6),
                workers=len(own), trees=len(sim.player_trees[seat]))


def contact_branches(sim, commands):
    """One already-present enemy collector could instead meet a carrying collector."""
    def destination(worker, action):
        return sim._move_nbr[action][worker.pos] if action else worker.pos

    targets = [set(), set()]
    for w in sim.workers.values():
        if w.is_collector and w.carbon > 0:
            targets[w.player].add(destination(w, commands[w.player].get(w.id)))
    for w in sim.workers.values():
        if w.is_collector:
            actual = commands[w.player].get(w.id)
            for change in (None, *sim._move_nbr):
                if change != actual and destination(w, change) in targets[1-w.player]:
                    yield w.player, w.id, actual, change


def possible_extra_capture(sandbox, state, branches, commands, factual_collisions):
    def receipt(collisions, robber):
        return sum(amount for _, owner, amount, recipient, _ in collisions
                   if owner != robber and recipient == robber)

    best = [None, None]
    for robber, worker, actual, change in branches:
        alternative = list(commands)
        alternative[robber] = dict(commands[robber], **{worker:change})
        sandbox.set_state(state)
        sandbox.step(*alternative)
        extra = receipt(sandbox.last_collisions, robber) - receipt(factual_collisions, robber)
        if extra > 1e-6 and (best[robber] is None or extra > best[robber]["extra"]):
            best[robber] = dict(extra=extra, worker=worker, actual=actual, alternative=change,
                               victims=[[wid, round(a,6)] for wid, o, a, r, _ in sandbox.last_collisions
                                        if o != robber and r == robber and a > 0])
    return best


def inspect_game(folder, row, header):
    validate_game(row)
    verify_raw(folder, row, header["pack_sha256"], header["weight_sha256"])
    raw = json.loads(gzip.decompress((folder / row["raw_file"]).read_bytes()))
    seat, seed = row["candidate_seat"], row["seed"]
    cfg = raw["configuration"]
    if (cfg["_S_ENGINE"] is not True or cfg["_TEMP"] != 1 or cfg["_SEED"] is not None
            or cfg["_S_ABORT"] != 0.58 or not cfg["_S_ROLL"] or cfg["_S_D2"] != 8):
        raise ValueError("examiner settings differ from the original")
    if raw["game_configuration"] != header["configuration"] or sha(compact(raw["actions"])) != row["actions_sha256"]:
        raise ValueError("rules or joint-action transcript mismatch")
    np.random.seed(seed)
    env = make("carbon", configuration={**raw["game_configuration"], "randomSeed": seed})
    sim = FastCarbon(raw["game_configuration"]).reset(seed=seed)
    sim.cash = CashEvents(sim.cash)
    sandbox = FastCarbon(raw["game_configuration"])
    initial_board = bytes(env.states[0].observation.full_carbon)
    if list(initial_board) != sim.carbon:
        raise ValueError("initial map mismatch")
    observation_hash = [hashlib.sha256(), hashlib.sha256()]
    start_obs = [sha(compact(sim.obs(p))) for p in (0, 1)]
    states = [[fleet(sim, p) for p in (seat, 1-seat)]]
    balances = [[Counter(), Counter()] for _ in PHASES]
    cargo = [[Counter(), Counter()] for _ in PHASES]
    opportunities = [[[] for p in (0, 1)] for _ in PHASES]
    phase_ends = []
    for t, commands in enumerate(raw["actions"], 1):
        if env.done:
            raise ValueError("commands recorded after official terminal")
        for p in (0, 1):
            # __get_shared_state assembles the same filtered observation sent to an agent.
            obs = sim.obs(p)
            if env._Environment__get_shared_state(p).observation != obs:
                raise ValueError(f"visible observation differs: {seed}/{seat} root {t-1}, player {p}")
            observation_hash[p].update(compact(obs) + b"\n")
        phase = (t - 1) // 100
        cash0 = list(sim.cash)
        branches = list(contact_branches(sim, commands))
        pre = sim.get_state() if branches else None
        sim.cash.events.clear()
        sim.step(*commands)
        env.step(commands)
        if branches:
            best = possible_extra_capture(sandbox, pre, branches, commands, sim.last_collisions)
            for p, event in enumerate(best):
                if event:
                    opportunities[phase][p].append(dict(root=t-1, **event))
        components = sums_by_seat(sim.cash.events)
        for p in (0, 1):
            if abs(sum(components[p].values()) - sim.cash[p] + cash0[p]) > 1e-7:
                raise ValueError("unattributed cash change")
            balances[phase][p].update(components[p])
        for _, owner, amount, recipient, _ in sim.last_collisions:
            losses = cargo[phase][owner]
            losses["worker_deaths_in_collisions"] += 1
            losses["cargo_on_dead_workers"] += amount
            if recipient == 1 - owner:
                losses["cargo_transferred_to_enemy"] += amount
        states.append([fleet(sim, p) for p in (seat, 1-seat)])
        if t in (100, 200) or t == len(raw["actions"]):
            snapshot = row["final"] if t == len(raw["actions"]) else row["snapshots"][str(t)]
            for role, p in (("candidate",seat), ("opponent",1-seat)):
                for key, seen in (("own_delivery","own_positive_delivery_carbon"),
                                  ("tree_absorption","cumulative_tree_absorption_income")):
                    if abs(sum(x[p][key] for x in balances) - snapshot[role][seen]) > 1e-6:
                        raise ValueError(f"{role} {key} differs from live observer at t{t}")
            phase_ends.append(t)
    steps = len(raw["actions"])
    rewards = [p.reward for p in env.states]
    if (not env.done or [p.status for p in env.states] != ["DONE", "DONE"] or
            env.states[0].observation.step != steps or rewards != row["official_score_by_seat"] or
            sha(compact(sim.get_state())) != row["final_state_sha256"]):
        raise ValueError("terminal scoring or state disagrees with original interpreter and archive")
    roles = (("candidate", seat), ("opponent", 1-seat))
    periods = []
    for i, end in enumerate(phase_ends):
        start = PHASES[i][0]
        entry = {"from_state":start-1, "to_state":end, "transitions":end-start+1}
        for r, (role, p) in enumerate(roles):
            components = balances[i][p]
            entry[role] = dict(
                cash_from=states[start-1][r]["cash"], end=states[end][r],
                extra_enemy_cargo_capture=dict(roots_with_positive_branch=len(opportunities[i][p]),
                    three_largest=sorted(opportunities[i][p], key=lambda a:-a["extra"])[:3]),
                cash_movements={k: round(v, 6) for k, v in sorted(components.items())},
                **{k:round(cargo[i][p][k],6) for k in ("worker_deaths_in_collisions",
                    "cargo_on_dead_workers", "cargo_transferred_to_enemy")})
        periods.append(entry)
    leads = [s[1]["cash"]-s[0]["cash"] for s in states]
    span = min(10, steps)
    start = max(range(steps-span+1), key=lambda t: round(leads[t+span]-leads[t], 2))
    return dict(seed=seed, seat=seat, result=row["result"], margin=round(row["margin"],2), steps=steps,
                official_reward_by_seat=rewards, termination=row["termination"], map_sha256=sha(initial_board),
                initial_observation_sha256_by_seat=start_obs,
                replayed_observations_sha256_by_seat=[h.hexdigest() for h in observation_hash],
                commands_sha256=row["actions_sha256"], replay_sha256=row["raw_sha256"],
                terminal_state_sha256=row["final_state_sha256"], settings_sha256=sha(compact(cfg)),
                depth=row["search"]["depth"], search_changes=row["search"]["changed"],
                search_returns=row["search_return_types"], call_seconds_by_seat=row["call_seconds_by_seat"],
                cash_phases=periods, largest_opponent_cash_lead_gain=dict(
                    from_state=start, to_state=start+span, gain=round(leads[start+span]-leads[start],2),
                    snapshots=[dict(state=t, candidate=states[t][0], opponent=states[t][1])
                               for t in sorted({max(0,start-5),start,start+span})]))


def inspect_run(folder):
    content = (folder / "results.json").read_bytes()
    record = json.loads(content)
    header = record["header"]
    seeds = header["seeds"]
    if record["status"] != "complete" or header["plan_sha256"] or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected complete development run with distinct assigned maps")
    expected = sorted((seed, p) for seed in seeds for p in (0, 1))
    if sorted((r["seed"], r["candidate_seat"]) for r in record["games"]) != expected:
        raise ValueError("missing or extra game")
    if header["examiner_md5"] != EXAMINER:
        raise ValueError("examiner identity mismatch")
    for name, digest in header["source_sha256"].items():
        if sha((ROOT / name).read_bytes()) != digest:
            raise ValueError(f"protocol drift: {name}")
    games = sorted((inspect_game(folder, g, header) for g in record["games"]), key=lambda g:(g["seed"], g["seat"]))
    for seed in seeds:
        if len({g["map_sha256"] for g in games if g["seed"] == seed}) != 1:
            raise ValueError("seat swap did not use the same map")
    return dict(run=folder.name, pack_sha256=header["pack_sha256"], results_sha256=sha(content), games=games)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    source = subprocess.check_output(["git", "show", "HEAD:"+EXAMINER_PATH], cwd=ROOT)
    if source != (ROOT/"data/master3s2_source.py").read_bytes() or hashlib.md5(source).hexdigest() != EXAMINER:
        raise ValueError("examiner runtime differs from canonical Git source")
    arms = [inspect_run(f) for f in (args.parent, args.candidate)]
    keyed = [[(g["seed"],g["seat"]) for g in a["games"]] for a in arms]
    if keyed[0] != keyed[1] or arms[0]["pack_sha256"] == arms[1]["pack_sha256"]:
        raise ValueError("unpaired runs or identical candidate")
    pairs = []
    for old, new in zip(arms[0]["games"], arms[1]["games"]):
        if any(old[k] != new[k] for k in ("map_sha256", "initial_observation_sha256_by_seat", "settings_sha256")):
            raise ValueError("initial game or examiner configuration differs across arms")
        pairs.append(dict(seed=old["seed"], seat=old["seat"], delta_margin=round(new["margin"]-old["margin"],2),
                          delta_examiner_depth8=new["depth"].get("8",0)-old["depth"].get("8",0)))
    files = ["zerosum_env/core.py","zerosum_env/envs/carbon/carbon.py","zerosum_env/envs/carbon/helpers.py",
             "zerosum_env/envs/carbon/carbon.json","zerosum_env/envs/carbon/idgen.py"]
    write(args.out, dict(
        scope="Development. Candidate minus parent pairs share map and seat. Master samples independently and adapts depth to wall time. A paired score change has no isolated action-level causal meaning.",
        definitions="States 0–100 cover transitions 1–100; next groups are 101–200 and 201–299. Omit unplayed intervals. Cash movements are signed and include rounding. Cargo lost in collisions is an asset movement; it becomes cash only under the applicable recipient rule. Observation hashes use sorted compact FastCarbon JSON plus newline per pre-action state, compared structurally to official agent views for both seats. Counterfactual extra_enemy_cargo_capture tests redirecting one existing collector into the recorded destination of a carrying opponent while holding every other command fixed. Its positive increments are gross cargo credited in collisions relative to the real turn, including tree/base credits. One-step hindsight comparisons allow no opponent adaptation; events on adjacent roots may refer to the same cargo. The largest ten-transition jump is selected after play.",
        canonical_examiner_path=EXAMINER_PATH, examiner_sha256=sha(source),
        official_engine_sha256={f:sha((ENGINE/f).read_bytes()) for f in files},
        parent=arms[0], candidate=arms[1], pairs=pairs))


if __name__ == "__main__":
    main()
