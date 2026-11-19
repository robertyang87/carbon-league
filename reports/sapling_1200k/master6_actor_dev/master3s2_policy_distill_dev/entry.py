"""Frozen public-entry readout at 12 recorded steps on two held physical maps.

Choose two held original-side cards by terrain, id, seat order, preferring opposite
physical seats for the two primaries. On the same source and map, include other
held seats where provided, identified by their actual source domain. All packs
see every seat-public observation from round zero. Their commands never advance
the environment. There is no teacher invocation, candidate rollout, or full match.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FILES = {
    "parent": ROOT / "data/weights/sapling_1200k/master6_actor_v1.py",
    "control": HERE / "control.py",
    "student": HERE / "student.py",
}
OUT = HERE / "entry.json.gz"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def choose(cards):
    held = [r for r in cards if r["split"] == "held"]
    fit = {r["terrain"] for r in cards if r["split"] == "fit"}
    assert fit.isdisjoint(r["terrain"] for r in held)
    pool = sorted((r for r in held if r["domain"] == "original"),
                  key=lambda r: (r["terrain"], r["id"], r["seat"]))
    a = pool[0]
    remaining = [r for r in pool if r["terrain"] != a["terrain"]]
    b = next((r for r in remaining if r["seat"] != a["seat"]), remaining[0])
    selected = [a, b]
    for primary in (a, b):
        other = sorted((r for r in held if r["terrain"] == primary["terrain"]
                        and r["source_sha256"] == primary["source_sha256"]
                        and r["seat"] != primary["seat"]), key=lambda r: (r["domain"], r["id"]))
        if other:
            selected.append(other[0])
    return selected


def identity(card):
    return str(card["id"]), card["terrain"], int(card["seat"]), card["domain"]


def canonical(cmd, entities):
    if not isinstance(cmd, dict) or not set(cmd) <= entities.keys():
        raise ValueError("Command refers to an unregistered entity")
    out = {}
    for uid, action in cmd.items():
        if action in (None, "NONE", ""):
            continue
        allowed = (("RECCOLLECTOR", "RECPLANTER") if entities[uid] == "BASE"
                   else ("UP", "RIGHT", "DOWN", "LEFT"))
        if action not in allowed:
            raise ValueError((uid, action))
        out[uid] = action
    return out


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def import_pack(path):
    spec = importlib.util.spec_from_file_location("distill_checked_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trace(card, name):
    import builder
    sim, pairs = builder.replay(card)
    pack = import_pack(FILES[name])
    registered = sorted(map(int, card["steps"]))
    if len(registered) != 12 or len(set(registered)) != 12:
        raise ValueError("Expected twelve distinct fixed roots")
    rows = []
    s = int(card["seat"])
    for t, pair in enumerate(pairs):
        obs = sim.obs(s)
        if obs["step"] != t or obs["player"] != s:
            raise ValueError("Replay did not supply the full public prefix")
        actual = pack.agent(obs, sim.config)
        if t in registered:
            own = obs["players"][s]
            entities = {k: "BASE" for k in own[1]}
            entities.update({k: w[2] for k, w in own[2].items()})
            rows.append(dict(t=t, entities=entities, input_sha256=digest(obs),
                             tape=canonical(pair[s], entities), output=canonical(actual, entities)))
        if t == registered[-1]:
            break
        sim.step(*pair)
    if [r["t"] for r in rows] != registered:
        raise ValueError("Incomplete registered public trajectory")
    return dict(card=identity(card), source=card["source"], source_sha256=card["source_sha256"],
                pack_sha256=sha(FILES[name]), full_prefix_calls=registered[-1] + 1, rows=rows)


def diff(a, b, entities):
    changed = [uid for uid in entities if a.get(uid) != b.get(uid)]
    return dict(joint_changed=int(bool(changed)), changed_by_role=dict(Counter(entities[k] for k in changed)),
                changed_ids=changed, all_by_role=dict(Counter(entities.values())))


def collect(manifest):
    selected = choose(manifest["records"])
    primary = [list(identity(r)) for r in selected[:2]]
    out = dict(primary=primary, frozen_selection=__doc__, entry_source_sha256=sha(Path(__file__)),
               manifest_sha256=sha(HERE/"manifest.json"), packs={k:sha(v) for k,v in FILES.items()}, samples=[])
    with tempfile.TemporaryDirectory(prefix="sapling-entry-") as tmp:
        paths = {}
        for name in FILES:
            path = Path(tmp) / (name + ".json")
            subprocess.run([sys.executable, str(Path(__file__)), "--worker", name, "--temp", str(path)], check=True)
            paths[name] = json.loads(path.read_bytes())
        for i, card in enumerate(selected):
            samples = {k: paths[k][i] for k in FILES}
            for name, val in samples.items():
                assert val["card"] == list(identity(card)) and val["pack_sha256"] == out["packs"][name]
            directions = [("control", "parent"), ("student", "parent"), ("student", "control")]
            for label in FILES:
                directions.append((label, "tape"))
            game = dict(card=list(identity(card)), primary=list(identity(card)) in primary,
                        source=card["source"], source_sha256=card["source_sha256"],
                        tape_actor="original_master3s2" if card["domain"]=="original" else card["domain"],
                        prefix_calls_per_pack=samples["parent"]["full_prefix_calls"], rows=[], totals={})
            for j, root in enumerate(samples["parent"]["rows"]):
                outputs = {"tape": root["tape"]}
                for name, val in samples.items():
                    r = val["rows"][j]
                    assert (r["input_sha256"],r["t"],r["tape"],r["entities"]) == (root["input_sha256"],root["t"],root["tape"],root["entities"])
                    outputs[name] = r["output"]
                changes = {a+"_vs_"+b: diff(outputs[a], outputs[b], root["entities"]) for a,b in directions}
                game["rows"].append(dict(t=root["t"], entities=root["entities"], input_sha256=root["input_sha256"],
                                         commands=outputs, differences=changes))
            for a,b in directions:
                label=a+"_vs_"+b
                rows=[r["differences"][label] for r in game["rows"]]
                counts=Counter();total=Counter()
                for r in rows:
                    counts.update(r["changed_by_role"]);total.update(r["all_by_role"])
                game["totals"][label]=dict(joint_changed=sum(r["joint_changed"] for r in rows),
                                          joint_total=len(rows), changed_by_role=dict(counts), total_by_role=dict(total))
            out["samples"].append(game)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker", choices=list(FILES))
    p.add_argument("--temp", type=Path)
    args=p.parse_args()
    manifest=json.loads((HERE/"manifest.json").read_bytes())
    if args.worker is not None:
        cards=choose(manifest["records"])
        # Import each agent in a separate process from other packs to isolate dynamically embedded Python modules.
        args.temp.write_text(json.dumps([trace(r,args.worker) for r in cards],allow_nan=False))
        return
    result=collect(manifest)
    raw=gzip.compress(json.dumps(result,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode(),mtime=0)
    if OUT.exists() and OUT.read_bytes()!=raw:
        raise ValueError("Existing frozen public-entry readout differs")
    temp=OUT.with_suffix(".partial")
    temp.write_bytes(raw);os.replace(temp,OUT)
    print([(r["card"],r["totals"]["student_vs_tape"]["joint_changed"],r["totals"]["student_vs_parent"]["joint_changed"]) for r in result["samples"]])

if __name__=="__main__":
    main()
