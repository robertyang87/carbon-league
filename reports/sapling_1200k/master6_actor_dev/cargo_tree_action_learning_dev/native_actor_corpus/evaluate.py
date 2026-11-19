"""Compare final public entries on fixed original tapes after binding both package hashes."""
import importlib.util
import os
import sys
from collections import Counter

os.sched_setaffinity(0, {0})
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from common import HERE, PACK, iter_public, read, save, sha

ARMS = ("parent", "head", "full")
PAIRS = (("head", "parent"), ("full", "parent"), ("full", "head"))
GROUPS = ("all", "worker", "collector", "planter", "base")
WORKERS = {"collector", "planter"}


def load_packages(contract):
    fit = read(HERE / "fit.json")
    paths = {"parent": PACK, "head": HERE / "head.py", "full": HERE / "full.py"}
    expected = {"parent": contract["parent_sha256"], **{a: fit["arms"][a]["package_sha256"] for a in ("head", "full")}}
    if {arm: sha(path) for arm, path in paths.items()} != expected:
        raise ValueError("Both frozen final packages and the registered parent must be present")
    models = {}
    for arm, path in paths.items():
        spec = importlib.util.spec_from_file_location("native_corpus_free_" + arm, path)
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        models[arm] = model
    return expected, models


def prefix_audit(observer, x, label, roles):
    features = sys.modules["cnet2_feat"]
    prefix = features.Prefix(x, observer.aux)
    y = observer.labels(label)
    if observer.to_actions(y) != {k: v for k, v in label.items() if v not in (None, "NONE")}:
        raise ValueError("Archived examiner action failed physical round trip")
    active = [i for i, uid in enumerate(observer.own_uid) if uid is not None]
    if len(active) != len(roles):
        raise ValueError("Active slot count differs from the public owned entities")
    blocked, forced = [], 0
    for position, slot in enumerate(active):
        _, _, legal = prefix.feats(slot)
        if not legal[y[slot]]:
            blocked = [observer.own_uid[s] for s in active[position:]]
            break
        forced += int(legal.sum() == 1)
        prefix.advance(slot, int(y[slot]))
    return {"first_blocked_uid": blocked[0] if blocked else None, "blocked_suffix_uids": blocked,
            "single_legal_within_prefix": forced}


def run_tape(card, models, k):
    seat, n = card["expert_seat"], card["steps"]
    if seat != 1 - card["seat"]:
        raise ValueError("Wrong original examiner seat")
    observer = sys.modules["cnet2_feat"].Feat(seat)
    result = []
    history = 0
    for t, obs, config, expert, selected in iter_public(card, k):
        if t != history or t != obs["step"] or obs["player"] != seat:
            raise ValueError("Discontinuous public view or wrong physical seat")
        commands = {}
        for arm, model in models.items():
            if t == 0 or selected:
                cfg = {key: value for key, value in config.items() if key != "randomSeed"}
                action = model.agent(obs, cfg)
                if not isinstance(action, dict):
                    raise ValueError("Normal package entry did not return a physical command dictionary")
                commands[arm] = action
            else:
                model._FEAT.observe(obs)
            if model._FEAT.step != t or model._FEAT.seat != seat:
                raise ValueError("Public history mismatch in " + arm)
        x = observer.observe(obs)
        history += 1
        if not selected:
            continue
        own = obs["players"][seat]
        roles = {uid: "base" for uid in own[1]} | {uid: row[2].lower() for uid, row in own[2].items()}
        for action in [*commands.values(), expert]:
            if set(action) - roles.keys():
                raise ValueError("Physical command targets a non-owned entity")
        result.append({"terrain": card["terrain"], "seed": card["seed"], "split": card["split"],
                       "tape_sha256": card["sha256"], "expert_seat": seat, "step": t,
                       "stage": ("opening", "middle", "late")[min(2, 3 * t // n)],
                       "roles_by_uid": roles, "expert": expert, "deployed": commands,
                       "observed_prefix": prefix_audit(observer, x, expert, roles)})
    if len(result) != k or history != n:
        raise ValueError("Incomplete public history or incorrect root membership")
    return result


def command(raw, uid):
    return raw.get(uid) or "NONE"


def counter_view(counts):
    return {**counts, "accuracy": counts["correct"] / counts["entities"] if counts["entities"] else None}


def paired_view(counts):
    return {**counts, "accuracy_delta": (counts["gained"] - counts["lost"]) / counts["entities"] if counts["entities"] else None}


def summary(rows):
    arms = {a: {g: Counter(entities=0, correct=0, sent_non_none=0) for g in GROUPS} for a in ARMS}
    pairs = {f"{a}_minus_{b}": {g: Counter(entities=0, changed=0, gained=0, lost=0, both_wrong_changed=0)
                                for g in GROUPS} for a, b in PAIRS}
    whole = {a: 0 for a in ARMS}
    joint_sent = {a: 0 for a in ARMS}
    blocked = Counter()
    first = Counter()
    prefix_roots = prefix_forced = expert_sent = 0
    for r in rows:
        correct_joint = {a: True for a in ARMS}
        sent_joint = {a: False for a in ARMS}
        roles, audit = r["roles_by_uid"], r["observed_prefix"]
        if audit["first_blocked_uid"] is not None:
            prefix_roots += 1
            first[roles[audit["first_blocked_uid"]]] += 1
            blocked.update(roles[u] for u in audit["blocked_suffix_uids"])
        prefix_forced += audit["single_legal_within_prefix"]
        for uid, role in roles.items():
            groups = ("all", "worker", role) if role in WORKERS else ("all", role)
            truth = command(r["expert"], uid)
            expert_sent += int(truth != "NONE")
            choices = {arm: command(r["deployed"][arm], uid) for arm in ARMS}
            for arm, cmd in choices.items():
                correct_joint[arm] &= cmd == truth
                sent_joint[arm] |= cmd != "NONE"
                for g in groups:
                    count = arms[arm][g]
                    count["entities"] += 1
                    count["correct"] += cmd == truth
                    count["sent_non_none"] += cmd != "NONE"
            for a, b in PAIRS:
                ca, cb = choices[a], choices[b]
                for g in groups:
                    count = pairs[f"{a}_minus_{b}"][g]
                    count["entities"] += 1
                    count["changed"] += ca != cb
                    count["gained"] += ca == truth and cb != truth
                    count["lost"] += ca != truth and cb == truth
                    count["both_wrong_changed"] += ca != cb and truth not in (ca, cb)
        for arm in ARMS:
            whole[arm] += correct_joint[arm]
            joint_sent[arm] += sent_joint[arm]
    return {
        "roots": len(rows), "tapes": len({r["tape_sha256"] for r in rows}), "expert_non_none": expert_sent,
        "arms": {a: {**{g: counter_view(c) for g, c in v.items()},
                     "whole_joint": {"roots": len(rows), "correct": whole[a], "accuracy": whole[a] / len(rows),
                                     "roots_sent_non_none": joint_sent[a]}} for a, v in arms.items()},
        "pairs": {p: {g: paired_view(c) for g, c in counts.items()} for p, counts in pairs.items()},
        "teacher_prefix": {"roots_with_unrepresentable_action": prefix_roots,
                          "first_unrepresentable_by_role": dict(first), "blocked_suffix_active_by_role": dict(blocked),
                          "representable_single_legal": prefix_forced},
    }


def subset_summary(rows, key):
    return {v: summary([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}


def split_summary(rows, seed):
    maps = subset_summary(rows, "terrain")
    counts = summary(rows)
    draw = np.random.default_rng(seed).integers(0, len(maps), (4000, len(maps)))
    estimates, paired = {a: {} for a in ARMS}, {}
    for g in (*GROUPS, "whole_joint"):
        values = {a: np.asarray([m["arms"][a][g]["accuracy"] for m in maps.values()], dtype=float) for a in ARMS}
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError("An entire held terrain has no measured entities for group " + g)
        for a in ARMS:
            estimates[a][g] = float(values[a].mean())
        for a, b in PAIRS:
            delta = values[a] - values[b]
            boot = delta[draw].mean(axis=1)
            item = paired.setdefault(f"{a}_minus_{b}", {})
            item[g] = {"mean_difference": float(delta.mean()), "paired_percentile_95": np.quantile(boot, [.025, .975]).tolist(),
                       "map_gains": int((delta > 0).sum()), "map_losses": int((delta < 0).sum())}
    return {"micro": counts, "by_terrain": maps, "by_expert_seat": subset_summary(rows, "expert_seat"),
            "by_stage": subset_summary(rows, "stage"), "map_equal_accuracy": estimates, "paired_map_bootstrap": paired}


def qualify(test):
    out = {}
    for arm, other in (("full", "head"), ("head", "full")):
        diffs = [m["pairs"][arm + "_minus_parent"]["worker"]["changed"] for m in test["by_terrain"].values()]
        accuracy = test["map_equal_accuracy"]
        role_drops = {r: test["micro"]["arms"][arm][r]["accuracy"] - test["micro"]["arms"]["parent"][r]["accuracy"]
                      for r in ("collector", "planter")}
        tests = {"strictly_better_map_equal_worker_vs_both": accuracy[arm]["worker"] > max(accuracy[a]["worker"] for a in ("parent", other)),
                 "worker_changes_on_at_least_two_maps": sum(n > 0 for n in diffs) >= 2,
                 "each_pooled_role_delta_at_least_minus_002": min(role_drops.values()) >= -0.02}
        out[arm] = {"passed": all(tests.values()), "tests": tests, "worker_changed_maps": sum(n > 0 for n in diffs),
                    "pooled_role_deltas_vs_parent": role_drops,
                    "map_equal_role_deltas_vs_parent": {r: accuracy[arm][r] - accuracy["parent"][r] for r in role_drops}}
    return out


def main():
    contract = read(HERE / "protocol.json")
    package_hashes, models = load_packages(contract)
    sources = read(HERE / "sources.json")
    held_path = HERE / "held_commands.json.gz"
    records = []
    for c in sources["cards"]:
        if c["split"] in ("val", "test"):
            records.extend(run_tape(c, models, contract["roots_per_tape"][c["split"]]))
            print("verified normal-entry commands", c["split"], c["seed"], c["expert_seat"], flush=True)
    paths = ("protocol.json", "sources.json", "fit.json", "common.py", "evaluate.py")
    binding = {p: sha(HERE / p) for p in paths}
    save(held_path, {"package_sha256": package_hashes, "source_sha256": binding, "rows": records})
    results = {}
    for split in ("val", "test"):
        subset = [r for r in records if r["split"] == split]
        expected = contract["expected_maps"][split]
        if len({r["terrain"] for r in subset}) != expected or len(subset) != 2 * expected * contract["roots_per_tape"][split]:
            raise ValueError("Held split differs from registration")
        results[split] = split_summary(subset, contract["seed"])
    out = {"package_sha256": package_hashes, "source_sha256": binding, "held_commands_sha256": sha(held_path),
           "results": results, "qualification": qualify(results["test"]),
           "method": "All normal free commands are physical dictionaries from the unmodified public entry. Step zero initializes each arm, and every skipped observation updates that arm's public feature history exactly once. Every sampled active entity counts, including implicit NONE and teacher commands outside the constrained prefix. Equal-map mean weights each observed terrain once. All 4000 paired bootstrap draws retain both seats and their full within-map samples. Gained and lost count changes in correctness against the same original command.",
           "scope": "These historical native tapes and parent splits have been used in other Sapling studies. Validation and test terrains are withheld from this fit. Each tape uses one fixed historical path. This evaluates agreement with physical examiner commands, with no new opponent games, income effect or independent strength estimate."}
    save(HERE / "held.json", out)
    print(out["qualification"], flush=True)


if __name__ == "__main__":
    main()
