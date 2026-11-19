"""Frozen public PPO opponents and a reproducible same-observation diversity check.

Run from any directory with one BLAS thread. Replays advance only the recorded
commands; counterfactual commands are compared and never alter the source game.
"""
import ast
import base64
from collections import Counter
import gzip
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import zlib

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "code" / "packs").is_dir())
BASE = "e848e0b5a29e4f81a88587504e28f02ef064db6d"
FROZEN = {
    "it50": ("code/packs/ppou2_m_it50.py", "57e2557665636a1bbc92ae46a4c33ee9dac609efcffefa1c30df1233746c2211"),
    "it40": ("code/packs/ppou2_m_it40.py", "e3cf5ddf98c52d1a1b9fbbd7dd985ee0f33821f6fd2f39a7f607c4f3dfceb1f7"),
    "it20": ("code/packs/echo4_int8.py", "669e99aebaf2e82da3b3159e0ffabd5e9797aaa4c1f2504b9fffe21684d5a5b2"),
}
PROBES = [
    (621495121, 0, "4238a645ad576a94c08a3897eceba847db3e2159fa350facaaaaf0d5ed08ef05"),
    (621461220, 1, "e352ab447ffbfddf825fe70f00ac01a5ada0ce513cdc46f4556ddcfd6b5d0e15"),
]

def digest(blob):
    return hashlib.sha256(blob).hexdigest()

def load(opponent_id):
    """Return a fresh isolated module whose persistent public entry is .agent."""
    relative, expected = FROZEN[opponent_id]
    path = ROOT / relative
    if digest(path.read_bytes()) != expected:
        raise ValueError(f"opponent package digest differs: {opponent_id}")
    spec = spec_from_file_location(f"league_{opponent_id}", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def identities():
    out = []
    for uid, (path, expected) in FROZEN.items():
        blob = (ROOT / path).read_bytes()
        if digest(blob) != expected:
            raise ValueError(f"wrong bytes for {path}")
        tree = ast.parse(blob)
        constants = {n.targets[0].id: ast.literal_eval(n.value)
                     for n in tree.body if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id in {"_B1", "_B2", "_M1", "CONF", "_BUILD_HASHES"}}
        out.append(dict(id=uid, path=path, sha256=expected, size_bytes=len(blob),
                        iteration=json.loads(constants["_M1"])["it"],
                        actor_blob_sha256=digest(zlib.decompress(base64.b64decode(constants["_B1"]))),
                        secondary_network_present=bool(constants["_B2"]),
                        conf=constants["CONF"], embedded_module_md5=constants["_BUILD_HASHES"]))
    if len({o["actor_blob_sha256"] for o in out}) != len(FROZEN):
        raise ValueError("repeated actor payload")
    return out

def audit_tape(cards, seed, seat, tape_sha):
    from fastsim import FastCarbon

    card = next(c for c in cards if c["sha256"] == tape_sha and c["split"] == "train")
    if (card["seed"], card["seat"]) != (seed, seat):
        raise ValueError("training tape identity differs")
    blob = (ROOT / card["path"]).read_bytes()
    if digest(blob) != tape_sha:
        raise ValueError("training tape bytes differ")
    tape = json.loads(gzip.decompress(blob))
    cfg = tape["game_configuration"]
    sim = FastCarbon(cfg).reset(seed=seed, full_carbon=tape["initial_full_carbon"])
    first = {uid: load(uid) for uid in FROZEN}
    repeat = {uid: load(uid) for uid in FROZEN}
    ticks = 64
    counts = {f"{a}/{b}": Counter() for a,b in combinations(FROZEN, 2)}
    changed_on_repeat = Counter({uid: 0 for uid in FROZEN})
    for step, commands in enumerate(tape["actions"][:ticks]):
        obs = sim.obs(seat)
        if (obs["player"],obs["step"]) != (seat,step):
            raise ValueError("bad reconstructed observation")
        physical_ids = set(obs["players"][seat][1]) | set(obs["players"][seat][2])
        choices = {uid: module.agent(obs,cfg) for uid,module in first.items()}
        for uid,module in repeat.items():
            changed_on_repeat[uid] += module.agent(obs,cfg) != choices[uid]
        for a,b in combinations(FROZEN, 2):
            ca, cb = choices[a], choices[b]
            if not (set(ca) | set(cb)) <= physical_ids:
                raise ValueError("public command for absent unit")
            counter = counts[f"{a}/{b}"]
            counter["observation_count"] += 1
            counter["joints_different"] += ca != cb
            counter["worker_and_base_commands"] += len(physical_ids)
            counter["worker_and_base_commands_different"] += sum(ca.get(k) != cb.get(k) for k in physical_ids)
        sim.step(*commands)
    return dict(seed=seed, seat=seat, train_tape=card["path"], train_tape_sha256=tape_sha,
                pair_counts=counts, changed_on_identical_trajectory_repeat=changed_on_repeat)

def main():
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    import numpy
    sys.path.insert(0,str(ROOT / "code"))
    cards = json.loads((HERE.parent/"sources.json").read_bytes())["cards"]
    registry = dict(schema=1, source_main=BASE, opponents=identities(),
                    provenance="Three separate self-owned PPOU2 main-arm iterations of the same model family. Echo4 is the deployed int8 iteration-20 actor. The preceding policy line used behavioral imitation. These league opponents are surrogate training/development opponents.",
                    runtime="Call untouched module.agent(public_observation, configuration) from step 0 onward using one persistent module per side/trajectory. Guard enabled, search and default OPEN_T sampling disabled, root decoder selects argmax. No examiner or candidate hidden state shared. These packages do not sample a new action seed. Repeated maps with these greedy agents replicate the same path unless the opposing learner explores. Keep map, seat and any learner randomness paired in candidate/parent evaluation.",
                    probe_scope="First 64 chronological steps of two existing TRAIN tapes, advancing logged joint actions; both seats represented on different maps. All three opponents see the same public observation and their own persistent state. Joint and per-unit command disagreement includes base and implicit NONE; collections alone do not establish strength or robustness to novel families.",
                    probe_numpy=numpy.__version__,
                    probes=[audit_tape(cards,*row) for row in PROBES])
    for row in registry["probes"]:
        if any(row["changed_on_identical_trajectory_repeat"].values()):
            raise ValueError("public commands changed under replay")
    combined = {key:sum(x["pair_counts"][key]["joints_different"] for x in registry["probes"])
                for key in registry["probes"][0]["pair_counts"]}
    if min(combined.values()) == 0:
        raise ValueError("opponents identical on the two probes")
    path = HERE / "opponents.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry,indent=2,sort_keys=True,allow_nan=False)+"\n")
    tmp.replace(path)
    print(combined,flush=True)

if __name__ == "__main__":
    main()
