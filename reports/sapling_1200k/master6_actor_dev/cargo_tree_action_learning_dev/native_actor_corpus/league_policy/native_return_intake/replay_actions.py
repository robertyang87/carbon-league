"""Check where training affects the public policy, including ordinary turns with no sampled action."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "natural_original"
sys.path.insert(0, str(PREVIOUS))
import audit_menu
from physical_support import digest, order_key, probabilities


def distribution(actor, observation, cfg):
    rng = audit_menu.MenuDraw()
    if observation["step"]:
        actor.mod._behavior_rng = rng
    default = actor.act(observation, cfg)
    outcome = rng.menu
    if outcome is None:
        return default, probabilities([default], [1.])
    if default != outcome["options"][0] or observation["step"] not in audit_menu.STEPS:
        raise ValueError("Unexpected sampler or default")
    return default, probabilities(outcome["options"], outcome["probabilities"])


def compare_game(game, archive, parent_path, candidate_path):
    parent = audit_menu.collect.Player(parent_path)
    child = audit_menu.collect.Player(candidate_path)
    seat = game["seat"]
    receipt_digest = game["receipt_sha256"]
    turns = []
    with zipfile.ZipFile(archive) as data, tempfile.TemporaryDirectory() as scratch:
        prefix = game["arm"] + "/"
        registration = Path(scratch) / "registration.json"
        registration.write_bytes(data.read(prefix + "registration.json"))
        info = data.read(prefix + "receipt.json")
        tape = data.read(prefix + "turns.jsonl")
        if hashlib.sha256(info).hexdigest() != receipt_digest or hashlib.sha256(tape).hexdigest() != json.loads(info)["turns_sha256"]:
            raise ValueError("Archive disagrees with official receipt")
        live = audit_menu.open_live(registration)
        env, cfg = live.official()
        bank = [5., 5.]
        for t, line in enumerate(tape.splitlines()):
            frame = json.loads(line)
            obs = live.observe(env, seat, bank[seat])
            if env.done or t != obs["step"] or hashlib.sha256(live.encoded(obs)).hexdigest() != frame["public_observation_sha256"][seat]:
                raise ValueError("Invalid pre-action state")
            real = [r["response"]["action"] for r in frame["calls"]]
            old, p = distribution(parent, obs, cfg)
            new, q = distribution(child, obs, cfg)
            if order_key(real[seat]) not in p:
                raise ValueError("Logged parent command missing")
            tv = sum(abs(p.get(k, 0.) - q.get(k, 0.)) for k in p.keys() | q.keys()) / 2
            old_move, new_move = dict(order_key(old)), dict(order_key(new))
            changes = {}
            for unit in sorted(old_move.keys() | new_move.keys()):
                before, after = old_move.get(unit, "NONE"), new_move.get(unit, "NONE")
                if before == after:
                    continue
                public = obs["players"][seat]
                changes[unit] = {"from": before, "to": after,
                                 "role": "BASE" if unit in public[1] else public[2][unit][2]}
            turns.append({"step": t, "public_sha256":frame["public_observation_sha256"][seat],
                          "scheduled": t in audit_menu.STEPS, "total_variation": tv,
                          "observed_order_unreachable": order_key(real[seat]) not in q,
                          "default_changes": changes})
            bank = frame["overage_remaining_seconds"]
            env.step(real)
        if not env.done or [s.reward for s in env.states] != game["official_scores_by_seat"] or len(turns) != game["official_transitions"]:
            raise ValueError("Whole-game replay is incomplete")
    return {"seed": game["seed"], "seat": seat, "receipt_sha256": receipt_digest,
            "n_turns": len(turns), "scheduled": sum(r["scheduled"] for r in turns),
            "changed_default_on_scheduled": sum(bool(r["default_changes"]) and r["scheduled"] for r in turns),
            "changed_default_on_ordinary": sum(bool(r["default_changes"]) and not r["scheduled"] for r in turns),
            "changed_probability_on_scheduled": sum(r["scheduled"] and r["total_variation"] > 1.e-6 for r in turns),
            "ordinary_changed_steps": [r for r in turns if not r["scheduled"] and r["default_changes"]],
            "scheduled_steps": [r for r in turns if r["scheduled"]]}


def main():
    hosted = PREVIOUS / "fresh_repeat/hosted"
    receipt = hosted / "settlement.json"
    if digest(receipt) != "fe727a57eaab4ebd0cf18fdac124dadc10d0358d587008726224386c31adc603":
        raise ValueError("Expected original four paired development blocks")
    record = json.loads(receipt.read_bytes())
    fit = json.loads((HERE / "fit.json").read_bytes())
    parent, expected_parent = audit_menu.PACKS["candidate"]
    child = HERE / "candidate.py"
    if fit["status"] != "training_mechanism_pass" or digest(parent) != expected_parent or digest(child) != fit["selected_package_sha256"]:
        raise ValueError("Missing selected trained package")
    archives = {(r["seed"], r["seat"]): r for r in record["archives"]}
    games = []
    for g in record["games"]:
        if g["arm"] != "sampled_child":
            continue
        ref = archives[g["seed"], g["seat"]]
        path = hosted / (str(ref["id"]) + ".zip")
        if digest(path) != ref["sha256"] or g["action_support"]["deployment_sha256"] != expected_parent:
            raise ValueError("Different acting policy or prior evidence")
        games.append(compare_game(g, path, parent, child))
    if {(g["seed"], g["seat"]) for g in games} != {(s,p) for s in (621758450,621758451) for p in (0,1)} or len(games) != 4:
        raise ValueError("Incomplete development census")
    out = {"source_run": record["workflow_run"], "parent_sha256": expected_parent, "child_sha256": digest(child),
           "scope": "Diagnostic only. Replays the original two public development terrains under the historical parent state distribution. Actual original responses to the new decisions are unknown here.", "games": games}
    dest = HERE / "replay_actions.json"
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(out, sort_keys=True, separators=(",",":"), allow_nan=False) + "\n")
    temp.replace(dest)


if __name__ == "__main__":
    main()
