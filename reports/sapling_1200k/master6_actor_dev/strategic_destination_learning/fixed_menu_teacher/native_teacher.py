"""Rank frozen joint orders from visible prefixes. Score first, join terminal outcomes separately."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics as stats
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parents[1]
BASE = ACTOR.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE / "root_recovery_branches"))
from branches import begin, public

REGISTRY = ACTOR / "joint_response_critic_dev"
PLAN = json.loads((HERE / "protocol.json").read_bytes())
SOURCE = "sources/master3s2/master3s2.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_bound(path, expected):
    payload = path.read_bytes()
    check(sha(payload) == expected, str(path))
    return json.loads(gzip.decompress(payload))


def write(path, value):
    payload = encode(value) + b"\n"
    if path.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    part = path.with_suffix(".partial")
    part.write_bytes(payload)
    part.replace(path)


def roots():
    inputs = read_bound(REGISTRY / "native_inputs.json.gz", PLAN["root_registry"])["rows"]
    signed = read_bound(REGISTRY / "signed_roots.json.gz", PLAN["terminal_registry"])["rows"]
    by_case = {r["case"]: r for r in signed}
    check(len(inputs) == len(by_case) == len({r["case"] for r in inputs}) == 108, "case identity")
    return [(r, by_case[r["case"]]) for r in inputs]


def rank(native, obs, config, estimate, prev, menu, terminal):
    native._deviations = lambda players, me, lw, first, fixed, cap: [menu[1]]
    native._S_REC = []
    terminal.clear()
    wall, cpu = time.perf_counter(), time.process_time()
    choice = native._search(obs, config, estimate, menu[0], None, None, prev, wall)
    elapsed, used = time.perf_counter() - wall, time.process_time() - cpu
    depth, q = [0, 0], [None, None]
    if native._S_REC:
        step, seat, actions, depth, q, winner = native._S_REC[0]
        check(len(native._S_REC) == 1 and actions == menu and step == obs["step"]
              and seat == obs["player"] and choice == menu[winner], "native menu contract")
        check(all(math.isfinite(v) for v in q), "invalid native scores")
    else:
        check(choice == menu[0] and native._S_STATS["last"][:2] == (2, 0), "unobserved comparison")
    limit = min(8, 299 - obs["step"])
    full = depth == [limit, limit]
    if full and obs["step"] + limit == 299:
        check(len(terminal) == 2 and all(abs(qi - t["q"]) < 1e-7 for qi, t in zip(q, terminal)), "terminal value mismatch")
    return {"depth": depth, "native_q": q, "full": full, "choose": int(full and q[1] > q[0]),
            "wall_s": elapsed, "cpu_s": used, "terminal": list(terminal)}


def score():
    grouped = defaultdict(list)
    for r, s in roots():
        check(r["signed_parent"] == s["parent"] and r["signed_alternative"] == s["alternative"], "menu changed")
        check(all(r[key] == s[key] for key in ("seat", "seed", "t", "terrain")), "root identity changed")
        # Only project identifiers, public orders and time into the ranking loop.
        grouped[s["source"], s["source_sha256"], r["seat"]].append(
            {**{k:r[k] for k in ("case","seat","seed","t","terrain","split")},
             "public_root_sha256":s["public_root_sha256"], "menu":[s["parent"],s["alternative"]]})
    orig_bytes = subprocess.check_output(["git","show","HEAD:"+SOURCE], cwd=ROOT)
    check(sha(orig_bytes) == PLAN["original_sha256"], "original source changed")
    results, swaps = [], {}
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "master3s2.py"
        path.write_bytes(orig_bytes)
        spec = importlib.util.spec_from_file_location("native_menu_teacher", path)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        native._torch.set_num_threads(1)
        check(native._SEARCH_OK and (native._S_D2, native._S_M, native._S_MMAX, native._S_EPS, native._S_ABORT)
              == (8,1,1,0.0,0.58), "original search configuration")
        native._S_GEN = ((-1,1),)
        real_terminal, events = native._terminal, []
        def observe_terminal(margin):
            value = real_terminal(margin)
            independent = max(-2.5, min(2.5, margin/3000.0)) + 0.5 * ((margin > 0) - (margin < 0))
            check(abs(value - independent) < 1e-9, "terminal formula")
            events.append({"simulated_margin": margin, "q": value})
            return value
        native._terminal = observe_terminal
        for (source, digest, seat), cases in grouped.items():
            raw = read_bound(ROOT / source, digest)
            env = begin(raw)
            belief = native.BeliefTracker(native._Bel, seat)
            prev = None
            by_step = defaultdict(list)
            for case in cases:
                by_step[case["t"]].append(case)
            for step in range(max(by_step) + 1):
                obs = public(env, seat)
                check(obs["step"] == step and not env.done, "invalid prefix")
                estimate, _ = belief.step(obs["players"], step, obs["carbon"])
                for case in by_step.get(step, ()):
                    visible = {k:v for k,v in obs.items() if k != "remainingOverageTime"}
                    check(sha(encode(visible)) == case["public_root_sha256"], "public state mismatch")
                    menu = case["menu"]
                    measured = rank(native, obs, raw["game_configuration"], estimate, prev, menu, events)
                    row = {k:v for k,v in case.items() if k != "menu"}
                    row.update(measured, menu_sha256=sha(encode(menu)))
                    results.append(row)
                    split = case["split"]
                    if measured["full"] and split not in swaps:
                        reversed_result = rank(native, obs, raw["game_configuration"], estimate, prev, menu[::-1], events)
                        diff = None
                        if reversed_result["full"]:
                            diff = max(abs(a-b) for a,b in zip(measured["native_q"],reversed_result["native_q"][::-1]))
                        swaps[split] = dict(case=case["case"], reverse=reversed_result, max_realigned_error=diff,
                                            passed=diff is not None and diff <= 2e-4)
                if step < max(by_step):
                    orders = raw["actions"][step]
                    belief.set_actions(obs["players"], orders[seat])
                    prev = obs["players"]
                    env.step(orders)
            print("ranked", len(results), "of 108", flush=True)
    check(len(results) == 108 and set(swaps) == {"train","held"}, "incomplete menu ranking")
    write(HERE / "native_scores.json.gz", dict(schema=1, original=sha(orig_bytes), protocol=sha((HERE/"protocol.json").read_bytes()),
                                              script=sha(Path(__file__).read_bytes()), swaps=swaps, rows=results))


def evaluate():
    path = HERE/"native_scores.json.gz"
    report = json.loads(gzip.decompress(path.read_bytes()))
    lookup = {r["case"]:r for r in report["rows"]}
    obs = []
    for r, s in roots():
        scored = lookup[r["case"]]
        me = r["seat"]
        delta = [(s["alternative_scores"][i] - s["parent_scores"][i]) for i in (me,1-me)]
        check(max(abs(a-b) for a,b in zip(r["delta"], delta)) < 1e-7, "physical seat return")
        actual = delta[0] - delta[1]
        obs.append(dict(**scored, family=r["family"],role=r["role"],alternate_margin_delta=actual,
                        chosen_margin_delta=actual*scored["choose"]))
    def describe(rows):
        maps = defaultdict(list)
        for row in rows:
            maps[row["terrain"]].append(row)
        per = []
        for terrain, game in maps.items():
            per.append(dict(terrain=terrain, seed=game[0]["seed"],n=len(game),
                            chosen=stats.mean(r["chosen_margin_delta"] for r in game),
                            always_alternate=stats.mean(r["alternate_margin_delta"] for r in game)))
        gains = [v["chosen"] for v in per]
        eligible = [r for r in rows if r["full"] and r["native_q"][0] != r["native_q"][1] and abs(r["alternate_margin_delta"])>1e-7]
        info = dict(n=len(rows),maps=len(per),full_depth=sum(r["full"] for r in rows),chosen_alt=sum(r["choose"] for r in rows),
                    mean_per_map=stats.mean(gains),always_alt_per_map=stats.mean(v["always_alternate"] for v in per),
                    correct_nonzero=sum(r["choose"] == int(r["alternate_margin_delta"]>0) for r in eligible),
                    decisions_nonzero=len(eligible),positive_maps=sum(g>0 for g in gains),per_map=per)
        return info
    parts = {s:describe([r for r in obs if r["split"] == s]) for s in PLAN["expected"]}
    h = parts["held"]
    m = h["mean_per_map"]; err = 3.1824463*stats.stdev(v["chosen"] for v in h["per_map"])/math.sqrt(4)
    held = [r for r in obs if r["split"]=="held"]
    seats = {r["seat"] for r in held if r["full"] and r["native_q"][0] != r["native_q"][1]}
    passed = (all(v["passed"] for v in report["swaps"].values()) and h["positive_maps"]>=3
              and m>max(0,h["always_alt_per_map"]) and seats=={0,1})
    write(HERE/"result.json",dict(scores_sha256=sha(path.read_bytes()),promotion=passed,split=parts,held_map_t95=[m-err,m+err],
                                 swaps_pass=all(v["passed"] for v in report["swaps"].values()),terminal_checked=sum(bool(r["terminal"]) for r in obs),
                                 held_by_role={key:describe([r for r in held if r["role"]==key]) for key in sorted({r["role"] for r in held})},
                                 held_by_seat={str(key):describe([r for r in held if r["seat"]==key]) for key in sorted({r["seat"] for r in held})},
                                 total_search_wall_s=sum(r["wall_s"] for r in obs),total_search_cpu_s=sum(r["cpu_s"] for r in obs)))
    print("held",h,"promotion",passed,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score","evaluate"))
    args = parser.parse_args()
    (score if args.phase == "score" else evaluate)()
