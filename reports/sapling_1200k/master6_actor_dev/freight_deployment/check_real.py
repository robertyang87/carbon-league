"""Probe the actual standalone package on both complete, previously exposed source trajectories without looking at outcomes."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEV = HERE.parent
sys.path.insert(0, str(ROOT/"code"))
from fastsim import FastCarbon


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commands(actions):
    return {k:v for k,v in (actions or {}).items() if v not in (None, "NONE")}


def run():
    manifest = json.loads((HERE/"manifest.json").read_bytes())
    if digest(HERE/"agent.py") != manifest["agent_sha256"]:
        raise ValueError("frozen agent changed")
    with tempfile.TemporaryDirectory() as isolated:
        pack = Path(isolated)/"submission.py"
        shutil.copyfile(HERE/"agent.py", pack)
        spec = importlib.util.spec_from_file_location("freight_real_hold", pack)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = module._original_agent
        last_original = []
        def capture(*args):
            actions = original(*args)
            last_original[:] = [dict(actions)]
            return actions
        module._original_agent = capture
        result = []
        for seat in (0, 1):
            source = DEV/"hybrid_candidate"/f"raw_621458920_{seat}.json.gz"
            raw = json.loads(gzip.decompress(source.read_bytes()))
            if raw["pack_sha256"] != manifest["inputs_sha256"]["data/weights/sapling_1200k/master6_actor_hybrid.py"]:
                raise ValueError("recorded actions are from another frozen parent")
            config = raw["game_configuration"]
            sim = FastCarbon(config).reset(seed=raw["seed"])
            cuts = []
            reasons = Counter()
            calls = 0
            seen = 0
            for t, joint in enumerate(raw["actions"]):
                obs = sim.obs(seat)
                if obs["step"] != t:
                    raise ValueError("pre-action frame alignment")
                chosen = commands(module.agent(obs, config))
                before = commands(last_original[0])
                reference = commands(joint[seat])
                if before != reference:
                    raise ValueError(f"recorded actor discrepancy at {t}/{seat}: {before} != {reference}")
                memo = module._FREIGHT.last
                reasons[memo["reason"]] += 1
                calls += len(memo["public_predictions"])
                seen += len(obs["players"][1-seat][2]) > 0
                changed = {uid for uid in set(chosen)|set(before) if chosen.get(uid) != before.get(uid)}
                if bool(changed) != memo["changed"]:
                    raise ValueError("self-reported action change disagrees")
                if changed:
                    offer = memo["proposal"]
                    if changed != {offer["worker"]} or chosen.get(offer["worker"],"NONE") != offer["alternative"]:
                        raise ValueError("recorded command disagrees with proposal")
                    if any(before.get(b) != chosen.get(b) for b in obs["players"][seat][1]):
                        raise ValueError("base command changed")
                    row = {**offer,"step":t,"seat":seat,"baseline_command":before.get(offer["worker"],"NONE"),
                           "joint_before":before,"joint_after":chosen,
                           "target_forecast":memo["public_predictions"][offer["target_worker"]]}
                    cuts.append(row)
                # Continue along the unchanged source tape to verify input and timing compatibility only.
                sim.step(*joint)
            if len(cuts)>1:
                raise ValueError("intervention cap")
            result.append({"seat":seat,"turns":len(raw["actions"]), "source_sha256":digest(source),
                           "parent_tape_exact_turns":len(raw["actions"]), "response_counts":dict(reasons),
                           "forecasts_count":calls,"observed_command_changes":cuts,
                           "unchanged_decoder_network_counters":module._COUNTS})
    out={"agent_sha256":manifest["agent_sha256"],"check_source_sha256":digest(Path(__file__)),
         "scope":"Both public histories of a previously exposed parent development map, replaying all archived joint commands. No closed-loop score or future target used.",
         "games":result, "real_changed_roots":sum(len(g["observed_command_changes"]) for g in result)}
    (HERE/"real_roots.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print([(g["seat"], g["response_counts"]) for g in result], flush=True)


if __name__ == "__main__":
    run()
