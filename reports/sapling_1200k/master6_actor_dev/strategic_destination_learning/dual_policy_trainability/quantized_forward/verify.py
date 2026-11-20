"""Independently execute the two exported policies and measure the legal, frozen-prefix likelihood."""
import ast
import base64
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import types
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import corpus

PARENT = HERE.parents[2] / "cargo_tree_action_learning_dev/candidate.py"
PARENT_SHA = "8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rebuild(patch):
    original = PARENT.read_bytes()
    delta = (HERE/patch["file"]).read_bytes()
    corpus.require(sha(original) == PARENT_SHA and sha(delta) == patch["sha256"], "parent/patch bytes")
    source = original.decode().splitlines(keepends=True)
    nodes = {a.targets[0].id: a for a in ast.parse(original).body
             if isinstance(a, ast.Assign) and isinstance(a.targets[0], ast.Name)}
    used = 0
    for name, segment in zip(("_B1", "_W64"), patch["towers"], strict=True):
        n = nodes[name]
        corpus.require(n.lineno == n.end_lineno and segment["start"] == used, "patch offset")
        old = zlib.decompress(base64.b64decode(ast.literal_eval(n.value)))
        used += segment["length"]
        flip = zlib.decompress(delta[segment["start"]:used])
        corpus.require(sha(old) == segment["original_raw_sha256"] and len(old) == len(flip), "original tower bytes")
        new = np.bitwise_xor(np.frombuffer(old, np.uint8), np.frombuffer(flip, np.uint8)).tobytes()
        corpus.require(sha(new) == segment["raw_sha256"], "new tower byte identity")
        encoded = base64.b64encode(zlib.compress(new, 9)).decode()
        source[n.lineno-1] = f"{name} = {encoded!r}\n"
    corpus.require(used == len(delta), "trailing patch data")
    result = "".join(source).encode()
    corpus.require(len(result) <= 10_000_000, "exceeds package size")
    return result


def teacher_scores(pack, root, target):
    f = corpus.F.Feat(root["seat"])
    f.own_uid = root["uids"]
    parent_y = f.labels(target["parent"])
    x, aux = root["x"], root["aux"]
    physical = [p.act(x, aux, teacher=pack._HYBRID["style"], forced=parent_y)[1] for p in pack._NETS]
    prefix = corpus.F.Prefix(x, aux)
    out = []
    for s in np.flatnonzero(parent_y >= 0):
        _, move, valid = prefix.feats(s)
        if s in target["taught_slots"]:
            true = int(root["actual"][s])
            corpus.require(root["changed"][s] and root["eligible"][s] and valid[true] and
                           all(parent_y[:s] == root["actual"][:s]), "invalid frozen-prefix intervention")
            legal = np.flatnonzero(valid)
            scores = .75*physical[0][s] + .25*physical[1][s]
            if s < 10 and x["tok_kind"][s] == 0 and target["branch"] == "residual":
                scores = scores + pack._return_basis(x, aux, s, move) @ pack._RETURN_THETA
            restricted = scores[legal].astype(np.float64)
            out.append(float(np.logaddexp.reduce(restricted) - scores[true]))
        prefix.advance(s, int(parent_y[s]))
    return out


def main():
    result = json.loads((HERE/"result.json").read_bytes())
    prior_result = (HERE.parent/"result.json").read_bytes()
    corpus.require(result["parent_result_sha256"] == sha(prior_result) and
                   json.loads(prior_result)["parent_sha256"] == PARENT_SHA, "incorrect parent")
    raw = (HERE/"commands.json.gz").read_bytes()
    corpus.require(sha(raw) == result["orders_sha256"], "wrong expected commands")
    commands = json.loads(gzip.decompress(raw))
    old_commands = json.loads(gzip.decompress((HERE.parent/"orders.json.gz").read_bytes()))
    roots = corpus.roots()
    by_key = {(r["gid"], r["step"]): r for r in roots}
    corpus.require(len(commands) == len(old_commands) == len(by_key) == 64, "complete panel not retained")
    for row, old in zip(commands, old_commands, strict=True):
        for k in ("gid", "seed", "seat", "step", "split", "parent", "teacher", "taught_slots", "branch"):
            corpus.require(row[k] == old[k], "training/probe target changed")
    answer = {}
    for name in ("fp", "qat"):
        policy = rebuild(result["arms"][name]["patch"])
        rows = []
        counts = {split:Counter() for split in ("fit", "probe")}
        scores = {split:[] for split in counts}
        for card in corpus.sources()["cards"]:
            pack = types.ModuleType("dual_quantized_validation_" + name)
            pack.__file__ = str(HERE/(name+"_reconstructed.py"))
            exec(compile(policy, pack.__file__, "exec"), vars(pack))
            raw = corpus.json_file(corpus.ROOT/card["source"], card["source_sha256"])
            response = corpus.json_file(corpus.ROOT/card["response"], card["response_sha256"])
            t0, seat = card["first_step"], card["seat"]
            sim = corpus.FastCarbon(raw["game_configuration"]).reset(seed=card["seed"])
            tape = raw["actions"][:t0-1] + response["tape"]
            expected = {r["step"]:r for r in commands if r["gid"]==card["gid"]}
            for t, both in enumerate(tape[:t0+32]):
                issued = pack.agent(sim.obs(seat), raw["game_configuration"])
                if t in expected:
                    row = expected[t]
                    root = by_key[row["gid"], t]
                    corpus.require(issued == row[name], "real public entry changed signed command")
                    st = counts[row["split"]]
                    st["checked_joint"] += 1
                    st["changed_joint"] += issued != row["parent"]
                    st["changed_units"] += sum(issued.get(u) != row["parent"].get(u) for u in root["uids"] if u)
                    for slot in row["taught_slots"]:
                        uid = root["uids"][slot]
                        st["targeted_units"] += 1
                        st["targeted_physical_changed"] += issued.get(uid) != row["parent"].get(uid)
                        st["targeted_teacher_hit"] += issued.get(uid) == row["teacher"].get(uid)
                    vals = teacher_scores(pack, root, row)
                    scores[row["split"]].extend(vals)
                    rows.append(dict(gid=row["gid"], seat=seat, step=t, split=row["split"], nll=vals))
                sim.step(*both)
        summary = {}
        for split, values in scores.items():
            expected_nll = result["arms"][name]["nll"][split]
            if isinstance(expected_nll, dict):
                expected_nll = expected_nll.get("mean_NLL", expected_nll.get("mean"))
            mean = float(np.mean(values))
            corpus.require(len(values) == (11 if split == "fit" else 12) and abs(mean-float(expected_nll)) < 5e-6,
                           "real package pinned teacher likelihood")
            summary[split] = dict(n=len(values), mean_nll=mean, **counts[split])
        answer[name] = dict(bytes=len(policy), package_sha256=sha(policy), splits=summary, per_root=rows)
    out = dict(result_sha256=sha((HERE/"result.json").read_bytes()), commands_sha256=sha((HERE/"commands.json.gz").read_bytes()),
               source_sha256=sha(Path(__file__).read_bytes()), arms=answer, new_games=0)
    tmp = HERE/"verification.tmp"
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False)+"\n")
    tmp.replace(HERE/"verification.json")
    print({k:a["splits"] for k,a in answer.items()}, flush=True)


if __name__ == "__main__":
    main()
