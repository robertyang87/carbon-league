"""One registered TRAIN source game, then paired eight-step policy interventions."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback

# Runs are serial within a worker. The original examiner always retains its own adaptive deadline.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

HERE = Path(__file__).resolve().parent
STRATEGIC = HERE.parents[1]
REPO = next(folder for folder in HERE.parents if (folder / "code/env_src").is_dir())
LEAGUE = STRATEGIC.parent / "cargo_tree_action_learning_dev/native_actor_corpus/league_policy"
sys.path[:0] = [str(LEAGUE), str(LEAGUE / "native_return_intake/live_curriculum/inprocess_snapshot")]
import collect
import official_state
import original_state
from zerosum_env.envs.carbon import idgen

ROOT_STEPS = (256, 280)
WINDOW = 8
BC = STRATEGIC / "bc.py"
ELITE = STRATEGIC / "elite_direct_policy/trained.py"
SIGNED = {"bc": "ede48ad9770d357cbe4a4be6c24dafebc910be1e46a95c508c706e009fd713a0",
          "elite": "5e4f5da3fd63962ed559ba9efdeb788f454e3f6fb9ce617a3b5c877080795cb2",
          "original": original_state.SOURCE}


def need(test, message):
    if not test:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(collect.canonical_bytes(value))


def save(path, value):
    writer = collect.write_gzip if path.suffix == ".gz" else collect.write_json
    writer(path, value)


def platform_state():
    cg = Path("/sys/fs/cgroup")
    return {"host": socket.gethostname(), "affinity": sorted(os.sched_getaffinity(0)),
            "load": list(os.getloadavg()),
            "cgroup": {n: (cg / n).read_text().strip() for n in ("cpu.max", "memory.max", "cpu.stat")
                       if (cg / n).is_file()}}


def actor_state(player):
    return (vars(player.mod._FEAT), player.mod._COUNTS, [net.trunk for net in player.mod._NETS])


def restore_actor(player, state):
    features, counts, trunks = copy.deepcopy(state)
    vars(player.mod._FEAT).clear()
    vars(player.mod._FEAT).update(features)
    player.mod._COUNTS = counts
    for net, trunk in zip(player.mod._NETS, trunks, strict=True):
        net.trunk = trunk


def public(env, seat, bank):
    obs = collect.public(env, seat)
    obs["remainingOverageTime"] = bank[seat]
    return obs


def board_id(env):
    state = env.states[0].observation
    return digest([state.step, state.full_carbon, state.players, state.trees, dict(idgen._id_dict)])


def call(player, obs, cfg, examiner=False):
    if examiner:
        stats = player.mod._S_STATS
        prior = {k: stats[k] for k in ("single", "searched", "changed", "aborted")}
    wall = time.perf_counter()
    cpu = time.process_time()
    action = player.act(obs, cfg)
    receipt = {"cpu": time.process_time() - cpu, "wall": time.perf_counter() - wall}
    if examiner:
        delta = {k: int(stats[k] - previous) for k, previous in prior.items()}
        if delta["single"]:
            candidates, depth, rollouts, internal = 1, 0, 0, None
        else:
            need(delta["aborted"] or delta["searched"], "Original public call did not report search")
            candidates, depth, rollouts, internal, _ = stats["last"]
        receipt.update(K=int(candidates), H=int(depth), rolls=int(rollouts),
                       internal=internal, cost=list(player.mod._S_COST), **delta)
    return action, receipt


def settle(env, bank, seat, view, opposing_view, own, opposing):
    orders = [None, None]
    clocks = [None, None]
    orders[seat], clocks[seat] = own
    orders[1 - seat], clocks[1 - seat] = opposing
    next_bank = [b - max(c["wall"] - 1., 0.) for b, c in zip(bank, clocks)]
    row = {"step": view["step"], "public": copy.deepcopy(view), "enemy_public_sha256": digest(opposing_view),
           "board_before": board_id(env), "orders": orders, "clocks": clocks,
           "bank_before": bank, "bank_after": next_bank}
    env.step(orders)
    row["board_after"] = board_id(env)
    row["cash_after"] = [p[0] for p in env.states[0].observation.players]
    return next_bank, row


def local_time_limits(rows):
    result = []
    for seat in (0, 1):
        elapsed = [r["clocks"][seat]["wall"] for r in rows]
        excess = sum(max(t - 1., 0.) for t in elapsed)
        slow = sum(t > 1. for t in elapsed)
        result.append({"calls": len(elapsed), "slow": slow, "excess": excess, "max": max(elapsed),
                       "crossed_inprocess_limits": max(elapsed) > 3. or slow > 3 or excess > 5.})
    return result


def terminal(env, seat):
    need(env.done, "Official terminal missing")
    scores = [p.reward for p in env.states]
    margin = scores[seat] - scores[1 - seat]
    return {"scores": scores, "margin": margin, "win": int(margin > 0), "draw": int(margin == 0),
            "statuses": [p.status for p in env.states], "step": int(env.states[0].observation.step),
            "cash": [p[0] for p in env.states[0].observation.players]}


def root_snapshot(env, bank, enemy, actors, enemy_reply, view):
    private = original_state.take(enemy.mod)
    students = {name: copy.deepcopy(actor_state(actor)) for name, actor in actors.items()}
    need(original_state.h(students["bc"][0]) == original_state.h(students["elite"][0]),
         "Elite history differs from BC before intervention")
    return {"ref": official_state.take(env, bank), "native": private, "actors": students,
            "board": board_id(env), "own_public": digest(view), "enemy_reply": copy.deepcopy(enemy_reply),
            "actors_sha": {name: original_state.h(state) for name, state in students.items()}}


def observe_only(actor, own_view):
    with collect.attached(actor.modules):
        actor.mod._FEAT.observe(own_view)


def suffix(env, root, start, seat, cfg, enemy, actors, option):
    bank = official_state.restore(env, root["ref"])
    original_state.restore(enemy.mod, root["native"])
    for name, actor in actors.items():
        restore_actor(actor, root["actors"][name])
        need(original_state.h(actor_state(actor)) == root["actors_sha"][name], "Actor restore drift")
    need(board_id(env) == root["board"] and digest(public(env, seat, bank)) == root["own_public"],
         "Paired root drift")
    rows = []
    while not env.done:
        mine, theirs = (public(env, s, bank) for s in (seat, 1-seat))
        i = mine["step"]
        enemy_reply = root["enemy_reply"] if i == start else call(enemy, theirs, cfg, examiner=True)
        bc_reply = call(actors["bc"], mine, cfg)
        within_window = i < start + WINDOW
        elite_reply = call(actors["elite"], mine, cfg) if within_window else None
        used = option if within_window else "bc"
        chosen = bc_reply if used == "bc" else elite_reply
        bank, row = settle(env, bank, seat, mine, theirs, chosen, enemy_reply)
        row.update(policy=used, common_enemy_root=(i == start))
        if within_window:
            row.update(bc_order=bc_reply[0], elite_order=elite_reply[0],
                       option_disagreement=bc_reply[0] != elite_reply[0])
            need(actors["bc"].mod._FEAT.step == i and actors["elite"].mod._FEAT.step == i,
                 "Both actors must observe the actual branch before decoding")
        rows.append(row)
    return {"option": option, "rows": rows, "terminal": terminal(env, seat),
            "window_decisions": sum("option_disagreement" in r for r in rows),
            "within_branch_disagreements": sum(r.get("option_disagreement", False) for r in rows)}


def compare_window(arms, seat):
    """Expose the first causal contrast and rival clock noise on the still-common board."""
    bc, elite = arms["bc"]["rows"], arms["elite"]["rows"]
    shared = []
    for b, e in zip(bc[:WINDOW], elite[:WINDOW]):
        if b["board_before"] != e["board_before"]:
            break
        same_own = b["orders"][seat] == e["orders"][seat]
        same_enemy = b["orders"][1-seat] == e["orders"][1-seat]
        shared.append({"step": b["step"], "own_orders_equal": same_own,
                       "enemy_orders_equal": same_enemy,
                       "own_observations_equal": digest(b["public"]) == digest(e["public"]),
                       "enemy_observations_equal": b["enemy_public_sha256"] == e["enemy_public_sha256"]})
        if not (same_own and same_enemy):
            break
    needs_review = any(not t["enemy_orders_equal"] for t in shared)
    return {"shared_prefix": shared, "rival_order_differs_on_shared_prefix": needs_review,
            "same_physical_orders_through_window": (len(shared) == min(WINDOW, len(bc), len(elite))
                and len(bc[:WINDOW]) == len(elite[:WINDOW])
                and all(t["own_orders_equal"] and t["enemy_orders_equal"] for t in shared)),
            "all_shadow_actions_identical": all(arms[a]["within_branch_disagreements"] == 0
                                                 for a in ("bc", "elite"))}


def registered(args):
    content = args.protocol.read_bytes()
    plan = json.loads(content)
    entries = [r for r in plan["schedule"] if (r["seed"], r["seat"]) == (args.seed, args.seat)]
    need(len(entries) == 1 and entries[0]["role"] in ("fit", "hold"), "Ticket not uniquely registered")
    return sha(content), plan["physical_boards"][str(args.seed)], entries[0]


def run(args, protocol, terrain, ticket, native_file):
    os.environ["CARBON_DEBUG"] = "1"
    for name, path in [("bc", BC), ("elite", ELITE), ("original", native_file)]:
        need(sha(path.read_bytes()) == SIGNED[name], "Package identity drift: " + name)
    cfg = {**collect.CFG, "agentTimeout": 20, "actTimeout": 1, "runTimeout": 1500}
    pubcfg = {k: v for k, v in cfg.items() if k != "randomSeed"}
    env = collect.begin(args.seed, cfg)
    need(sha(bytes(env.states[0].observation.full_carbon)) == terrain, "Terrain drift")
    actors = {"bc": collect.Player(BC), "elite": collect.Player(ELITE)}
    enemy = collect.Player(native_file)
    need(enemy.mod._TEMP == 1 and enemy.mod._SEED is None and enemy.mod._S_ABORT == .58
         and enemy.mod._SEARCH_OK, "Original settings drift")
    header = {"seed": args.seed, "seat": args.seat, "terrain": terrain, "protocol": protocol,
              "runner_sha256": sha(Path(__file__).read_bytes()), "actors": SIGNED, "kind": ticket["role"],
              "configuration": cfg, "host_start": platform_state()}
    snapshots = {}
    chronology = []
    bank = [5., 5.]
    while not env.done:
        mine, theirs = (public(env, s, bank) for s in (args.seat, 1-args.seat))
        i = mine["step"]
        opposing = call(enemy, theirs, pubcfg, examiner=True)
        if i in ROOT_STEPS:
            snapshots[i] = root_snapshot(env, bank, enemy, actors, opposing, mine)
        own = call(actors["bc"], mine, pubcfg)
        if i == 0:
            # Initialize elite weights once. Prefix history thereafter requires no network forwards.
            call(actors["elite"], mine, pubcfg)
        elif i < max(ROOT_STEPS):
            observe_only(actors["elite"], mine)
        bank, row = settle(env, bank, args.seat, mine, theirs, own, opposing)
        chronology.append(row)
    source_result = terminal(env, args.seat)
    source_file = args.out / "source.json.gz"
    save(source_file, {**header, "rows": chronology, "terminal": source_result, "host_end": platform_state(),
                       "inprocess_time_limits": local_time_limits(chronology)})
    source_sha = sha(source_file.read_bytes())
    print(f"Source {args.seed}/{args.seat} terminal {source_result['scores']}", flush=True)
    summary = []
    for start in ROOT_STEPS:
        path = args.out / f"root_{start}.json.gz"
        common = {**header, "root_step": start, "source_sha256": source_sha}
        if start not in snapshots:
            result = {**common, "status": "unreachable_source_terminal", "source_terminal": source_result}
        else:
            state = snapshots.pop(start)
            order = ticket.get("order")
            if order is None:
                order = ["elite", "bc"] if (args.seed - 540 + ROOT_STEPS.index(start)) % 2 else ["bc", "elite"]
            need(sorted(order) == ["bc", "elite"], "Invalid registered pair order")
            arms = {name: suffix(env, state, start, args.seat, pubcfg, enemy, actors, name) for name in order}
            a, b = (arms[n] for n in ("bc", "elite"))
            for arm in arms.values():
                arm["inprocess_time_limits"] = local_time_limits(chronology[:start] + arm["rows"])
            result = {**common, "status": "complete", "root_public_sha256": state["own_public"],
                      "root_board_sha256": state["board"], "root_enemy_after_call_sha256": state["native"]["expected"],
                      "root_actor_sha256": state["actors_sha"], "arms": arms, "order": order,
                      "intervention": compare_window(arms, args.seat),
                      "margin_delta": b["terminal"]["margin"] - a["terminal"]["margin"],
                      "win_delta": b["terminal"]["win"] - a["terminal"]["win"],
                      "source_terminal": source_result, "host_end": platform_state()}
        save(path, result)
        summary.append({"step": start, "status": result["status"], "sha256": sha(path.read_bytes()),
                        "margin_delta": result.get("margin_delta"), "win_delta": result.get("win_delta")})
        print(summary[-1], flush=True)
    save(args.out / "summary.json", {**header, "roots": summary, "source_sha256": source_sha,
                                    "source_terminal": source_result, "host_end": platform_state()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--seat", required=True, choices=(0, 1), type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--original", type=Path)
    args = parser.parse_args()
    protocol, terrain, ticket = registered(args)
    args.out.mkdir(parents=True, exist_ok=True)
    claim = args.out / "claim.json"
    with claim.open("x") as stream:
        json.dump({"seed": args.seed, "seat": args.seat, "protocol": protocol,
                   "host": socket.gethostname(), "start_unix": time.time()}, stream)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = args.original or REPO / "sources/master3s2/master3s2.py"
            if not path.is_file() and args.original is None:
                path = Path(tmp) / "master3s2.py"
                path.write_bytes(subprocess.check_output(["git", "-C", str(REPO), "show",
                                                          "HEAD:sources/master3s2/master3s2.py"]))
            run(args, protocol, terrain, ticket, path)
    except BaseException:
        save(args.out / "failure.json", {"seed": args.seed, "seat": args.seat, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
