"""Reproduce old branch outcomes and test the *frozen* residual proposal on the same public roots."""
import gzip
import hashlib
import json
import os
from pathlib import Path
from functools import lru_cache
from account import account_arm, load, settled

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
ACTORS = DEV.parent
BASE = ACTORS.parent
ROOT = BASE.parents[1]
EXAM = ROOT / "sources/master3s2/master3s2.py"
# Sparse research checkouts keep a byte-identical local source mirror.
if not EXAM.exists():
    EXAM = ROOT / "data/master3s2_source.py"
EXAM_SHA = hashlib.sha256(EXAM.read_bytes()).hexdigest()
if EXAM_SHA != "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5":
    raise ValueError("original examiner digest mismatch")
runtime = load("late_residual_public_proposals", DEV / "runtime.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def pathref(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


@lru_cache(maxsize=48)
def restore(raw_path, t):
    raw = read(raw_path)
    sim = settled.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    terrain = hashlib.sha256(bytes(int(v) for v in sim.carbon)).hexdigest()
    if raw.get("initial_terrain_u8_sha256", terrain) != terrain:
        raise ValueError("start terrain hash differs")
    for j in raw["actions"][:t]:
        sim.step(*j)
    if sim.step_count != t:
        raise ValueError("root frame not reached")
    return sim, raw, terrain


def pair(source, rawfile, t, physical, arms, scope, split, stream, native_id):
    sim, raw, terrain = restore(rawfile, t)
    current_maps = read(DEV / "protocol.json")["source_groups"]
    overlaps = sorted({r["split"] for r in current_maps if r["map"] == terrain})
    cfg = raw["game_configuration"]
    if not raw.get("examiner_md5") == "57f9a594cce4f2ec9e3573d8d9d26ca1" or native_id != EXAM_SHA:
        raise ValueError("reference opponent is not the full archived original")
    view = sim.obs(physical)
    j0, j1 = [a["acts"][0] for a in arms]
    enemy = 1 - physical
    if j0[enemy] != j1[enemy]:
        raise ValueError("simultaneous root response differs")
    changed = {uid: [j0[physical].get(uid) or "NONE", j1[physical].get(uid) or "NONE"]
               for uid in j0[physical].keys() | j1[physical].keys()
               if (j0[physical].get(uid) or "NONE") != (j1[physical].get(uid) or "NONE")}
    if not changed or any(view["players"][physical][2][uid][2] != "COLLECTOR" for uid in changed):
        raise ValueError("unqualified collector change")
    proposals = runtime.variants(view, cfg, j0[physical])
    exact = [p for p in proposals if changed == {p["uid"]: [p["original"], p["action"]]}]
    results = []
    for a in arms:
        ledger = account_arm(sim.get_state(), cfg, a["acts"], physical, sorted(changed))
        if a["score"] != ledger["official_scores_physical"]:
            raise ValueError(("official source and local replay disagree", a["score"], ledger["official_scores_physical"]))
        results.append({**ledger, "tape": a["tape"], "continuation": a["actor"], "native_calls": a["calls"]})
    deltas = {"official_margin": results[1]["official_margin"] - results[0]["official_margin"]}
    for k in ("own_delivery", "focal_delivery", "other_delivery", "strict_new_delivered"):
        deltas[k] = results[1]["team"][k] - results[0]["team"][k]
    for k in ("strict_principal_delivered", "strict_fresh_delivered"):
        deltas[k] = sum(results[1]["per_worker"][u][k] - results[0]["per_worker"][u][k] for u in changed)
    return {"source": str(source.relative_to(ROOT)), "root_tape": pathref(rawfile), "seed": raw["seed"], "terrain_sha256_u8": terrain,
            "current_v1_protocol_map_overlap": overlaps,
            "seat": physical, "t": t, "split": split, "stream": stream, "scope": scope, "changed": changed,
            "root_joint_by_arm": [j0, j1], "actual_actor_root_view": {k: view[k] for k in ("step", "players", "player", "trees", "carbon")},
            "frozen_runtime_proposal": proposals, "exact_single_frozen_proposal_match": bool(exact),
            "scoped_native_examiner_sha256": native_id, "arms": results, "delta": deltas}


def standard(folder, split, stream):
    protocol = read(folder / "live_protocol.json")
    rec = protocol["record"]
    rawfile = ROOT / rec["source"]
    if sha(rawfile) != rec["source_sha256"]:
        raise ValueError("invalid original trajectory")
    arms = []
    for name in ["parent", "alternative"]:
        src = folder / f"live_{name}.json"
        report = read(src)
        tape = folder / report["action_tape"]
        if sha(tape) != report["action_sha256"] or len(read(tape)) - 1 != report["search_calls"]:
            raise ValueError("native controls and complete tape mismatch")
        arms.append(dict(acts=read(tape), tape=[pathref(src), pathref(tape)], actor=report["pack_sha256"],
                         calls=report["search_calls"], score=report["score"]))
    return pair(folder / "live_protocol.json", rawfile, protocol["root_t"], protocol["seat"], arms,
                "different continuation pack; historical external transfer only", split, stream, report["source_sha256"])


def queue():
    q = ACTORS / "endgame_queue_dev"
    spec = read(q / "protocol.json")
    sources = {(r["seed"], r["seat"]): r for b in spec["source_batches"] for r in read(ROOT / b["receipt"])["games"]}
    for c in read(q / "native_protocol.json")["cases"]:
        s, p = c["seed"], c["seat"]
        f = q / f"native_{s}_{p}.json.gz"
        doc = read(f)
        rawfile = ROOT / sources[s, p]["raw_path"]
        if sha(rawfile) != doc["source_sha256"]:
            raise ValueError("queue trajectory identity")
        arms = []
        for k in ["v1", "candidate"]:
            a = doc["arms"][k]
            arms.append(dict(acts=[a["first_joint"], a["last_joint"]], tape=[pathref(f)], actor=spec["source_parent"],
                             calls=a["search_call_count"], score=a["end"]["score"]))
        yield pair(f, rawfile, c["t"], p, arms, "outside frozen window t297; own reply recomputed by v1 on the arm successor; enemy responds with original search",
                   c["receipt_role"] + "/historical", 0, spec["source_examiner"])


def mixed():
    p = ACTORS / "economy_aux_dev/native_dev/paired_edits/native_t288"
    spec, verified = read(p / "protocol.json"), read(p / "verified.json.gz")
    rawfile = p.parents[1] / "candidate" / f"raw_{spec['seed']}_{spec['seat']}.json.gz"
    if sha(rawfile) != spec["input_sha256"]["raw"]:
        raise ValueError("mixed joint trajectory identity")
    arms = []
    for name in ["control_root", "auxiliary_root"]:
        d = p / name
        files = [d / "root.json", *sorted(d.glob("turn_*.json.gz"))]
        v = verified["results"][name]
        arms.append(dict(acts=[read(f)["joint"] for f in files], tape=[pathref(f) for f in files],
                         actor=spec["input_sha256"]["candidate"], calls=v["search_calls"], score=v["official_scores"]))
    return pair(p / "protocol.json", rawfile, spec["turn"], spec["seat"], arms,
                "two simultaneous collector edits; both continue with auxiliary pack; joint label only",
                "historical development", 0, spec["input_sha256"]["original_full_source"])


def main():
    if sha(DEV / "runtime.py") != read(DEV / "protocol.json")["proposal_sha256"]:
        raise ValueError("public proposal differs from preregistration")
    rows = []
    for d in sorted((BASE / "late_response_training/arms").glob("*/*/*")):
        rows.append(standard(d, read(d / "live_protocol.json")["split"] + "/published", int(d.name)))
    for d in sorted((BASE / "planter_routes_dev/endgame_responses/arms").glob("*")):
        rows.append(standard(d, "historical development; mother chosen after near loss", int(d.name.split("_")[-1])))
    rows.extend(queue())
    rows.append(mixed())
    raw = json.dumps(rows, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    output = HERE / "historical_pairs.json.gz"
    tmp = output.with_name(output.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(gzip.compress(raw, mtime=0))
    os.replace(tmp, output)
    print("pairs", len(rows), "distinct roots", len({(r["seed"], r["seat"], r["t"]) for r in rows}))
    for r in rows:
        print(r["seed"], r["seat"], r["t"], r["stream"], len(r["changed"]), r["exact_single_frozen_proposal_match"], round(r["delta"]["official_margin"], 2))


if __name__ == "__main__":
    main()
