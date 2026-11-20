"""Measure natural command changes on the exact registered historical observer trajectories."""
import importlib.util
from collections import Counter
from pathlib import Path
import sys
import random

import numpy as np

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "cargo_tree_action_learning_dev" / "native_actor_corpus"
sys.path.insert(0, str(NATIVE))
from common import PACK, ROOT, read, save, sha

sys.path.insert(0, str(NATIVE / "expansion"))
from replay import digest, official_rules

ARMS = ("parent", "bc", "joint")
PAIRS = (("bc", "parent"), ("joint", "parent"), ("joint", "bc"))


def load_models(fit, protocol):
    models = {}
    identities = {}
    for arm in ARMS:
        path = PACK if arm == "parent" else HERE / (arm + ".py")
        identities[arm] = sha(path)
        expected = protocol["parent_sha256"] if arm == "parent" else fit["arms"][arm]["package_sha256"]
        if identities[arm] != expected:
            raise ValueError(f"{arm}: frozen source changed")
        spec = importlib.util.spec_from_file_location("destination_" + arm, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        models[arm] = module
    if len(set(identities.values())) != len(ARMS):
        raise ValueError("Duplicate candidate")
    return identities, models


def public_history(card):
    source = ROOT/card["path"]
    if sha(source) != card["sha256"]:
        raise ValueError("Different original game")
    tape = read(source)
    seat, seed = card["expert_seat"], card["seed"]
    cfg = tape["game_configuration"]
    random.seed(seed)
    np.random.seed(seed)
    env = official_rules(ROOT)("carbon", configuration={**cfg, "randomSeed": seed})
    cash = []
    for t, joint in enumerate(tape["actions"]):
        state = env.states[0].observation
        cash.append([p[0] for p in state.players])
        own = env.states[seat].observation
        obs = dict(step=state.step, player=seat, carbon=own.carbon, players=state.players,
                   trees=state.trees, remainingOverageTime=own.get("remainingOverageTime",60))
        yield t, obs, cfg, joint[seat]
        env.step(joint, [])
    cash.append([p[0] for p in env.states[0].observation.players])
    if not env.done or [st.reward for st in env.states] != tape["official_score_by_seat"] or digest(cash) != card["official_cash_path_sha256"]:
        raise ValueError("Historical game did not reproduce")


def evaluate_tape(card, models):
    selected = set(card["registered_times"])
    if len(selected) != 24 or not all(0 <= step < card["transitions"] for step in selected):
        raise ValueError("Invalid registered root list")
    rows = []
    seat = card["expert_seat"]
    steps = 0
    for t, obs, cfg, expert in public_history(card):
        if t != steps or obs["step"] != t or obs["player"] != seat:
            raise ValueError("Observer history mismatch")
        public_cfg = {k: v for k, v in cfg.items() if k != "randomSeed"}
        commands = {arm: model.agent(obs, public_cfg) for arm, model in models.items()}
        steps += 1
        if t not in selected:
            continue
        player = obs["players"][seat]
        roles = {uid: "base" for uid in player[1]}
        roles.update({uid: w[2].lower() for uid, w in player[2].items()})
        if any(set(cmd) - roles.keys() for cmd in (expert, *commands.values())):
            raise ValueError("Action targets another player's unit")
        rows.append(dict(terrain=card["terrain"], split=card["split"], seed=card["seed"],
                         seat=seat, step=t, stage=min(2, 3*t//card["transitions"]),
                         tape=card["sha256"], roles=roles, expert=expert, commands=commands))
    if steps != card["transitions"] or {r["step"] for r in rows} != selected:
        raise ValueError("Partial replay")
    return rows


def rates(rows):
    entity = {arm: {} for arm in ARMS}
    whole = Counter()
    pair = {a+"_vs_"+b: Counter() for a,b in PAIRS}
    for row in rows:
        joint = {a: True for a in ARMS}
        for uid, role in row["roles"].items():
            truth = row["expert"].get(uid) or "NONE"
            choices = {a: row["commands"][a].get(uid) or "NONE" for a in ARMS}
            for a, cmd in choices.items():
                joint[a] &= cmd == truth
                groups = ("all", role) if role == "base" else ("all", "worker", role)
                for group in groups:
                    count = entity[a].setdefault(group, Counter(total=0, correct=0))
                    count["total"] += 1
                    count["correct"] += cmd == truth
            for a, b in PAIRS:
                count = pair[a+"_vs_"+b]
                count["entities"] += 1
                count["changed"] += choices[a] != choices[b]
                count["gained"] += choices[a] == truth and choices[b] != truth
                count["lost"] += choices[a] != truth and choices[b] == truth
        whole.update(joint)
    for groups in entity.values():
        for count in groups.values():
            count["accuracy"] = count["correct"]/count["total"]
    return dict(roots=len(rows), entities=entity, joint_correct=dict(whole), pairs=pair)


def main():
    protocol = read(HERE/"free_entry_protocol.json")
    fit = read(HERE/"fit.json")
    if sha(HERE/"sources.json") != protocol["sources_sha256"] or sha(HERE/"learning.json") != fit["learning_sha256"]:
        raise ValueError("Registration changed")
    identities, models = load_models(fit, protocol)
    cards = [c for c in read(HERE/"sources.json")["cards"] if c["split"] in ("val", "dev")]
    if len(cards) != 16 or len({c["terrain"] for c in cards}) != 8:
        raise ValueError("Incomplete historical evaluation roster")
    rows = []
    for c in cards:
        tape_rows = evaluate_tape(c, models)
        rows.extend(tape_rows)
        print(c["split"], c["seed"], c["expert_seat"], len(rows), rates(tape_rows)["pairs"], flush=True)
    results = {}
    for split in ("val", "dev"):
        selection = [r for r in rows if r["split"] == split]
        by = {key: {v: rates([r for r in selection if r[key] == v]) for v in sorted({r[key] for r in selection})}
              for key in ("terrain", "seat", "stage")}
        if len(selection) != 192 or len(by["terrain"]) != 4:
            raise ValueError("Incomplete historical split")
        macro = {a: {g: sum(m["entities"][a][g]["accuracy"] for m in by["terrain"].values())/4
                     for g in ("all", "worker", "collector", "planter", "base")} for a in ARMS}
        results[split] = dict(pooled=rates(selection), map_equal=macro, groups=by)
    save(HERE/"free_commands.json.gz", rows)
    save(HERE/"free_entry_results.json", dict(packages=identities, results=results,
         registration=sha(HERE/"free_entry_protocol.json"), evaluator=sha(Path(__file__)),
         commands_sha256=sha(HERE/"free_commands.json.gz"), fit_sha256=sha(HERE/"fit.json")))


if __name__ == "__main__":
    main()
