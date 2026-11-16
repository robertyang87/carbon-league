"""Fixture build generation for the k0_join acceptance chain.

The annotator fixture games (selfplay-994000 / selfplay-994001, from
gen1r_smoke) have no local cnet2_build generation, and cnet2_build.main's
corpus scan would filter them out entirely (both sides are named
"m3s2r1", which is not a F.TEACHERS side -- BC corpora only build
teacher sides BY DESIGN). The join seam needs BOTH seats' feature rows,
so this script drives cnet2_build's own machinery by import only:

  * per-game feature rows: cnet2_build.build_game with sides=[0, 1]
    (the exact function main() dispatches, same task tuple shape);
  * identity: scan-shaped game dicts, _producer_provenance,
    _game_fingerprint, _cfg_fingerprint -- all imported, unmodified;
  * publish: generation assembled with main()'s own gen_key / gen_id /
    manifest / meta conventions, then verified through
    cnet2_build.load_generation(verify=True).

Nothing in cnet2_build is edited. The one deliberate deviation from a
main() product is declared here and in the receipt: both seats are
built (a main() corpus build of this index would build none), because
the fixture exists to feed the join seat-join, not to train BC.
Provenance (names, scores, weights identity) is recorded verbatim from
the gen1r_smoke index -- nothing is renamed to a teacher.
"""
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "run"))  # repo bytes: cnet2_build, cnet2_feat

import cnet2_build as B  # noqa: E402
import cnet2_feat as F  # noqa: E402

CORPUS = "/home/hatch/workspace/gen1r_smoke"
GAME_IDS = [994000, 994001]
OUT = os.path.join(HERE, "fixture", "build_out_min207")


def scan_shaped_game(entry, rd):
    """The game dict cnet2_build._scan would build for this entry,
    except sides = [0, 1] (see module docstring)."""
    key = os.path.basename(entry["file"])
    order = entry["order"]
    names = [entry["strategies"][order[s]]["name"] for s in (0, 1)]
    scores = [entry["scores"][order[s]] for s in (0, 1)]
    raw_mg = raw_u = None
    if isinstance(entry.get("margin"), dict) and isinstance(entry.get("u"), dict):
        raw_mg = [float(entry["margin"][order[s]]) for s in (0, 1)]
        raw_u = [float(entry["u"][order[s]]) for s in (0, 1)]
    fp = os.path.join(rd, entry["file"])
    run_identity = None
    idp = os.path.join(rd, "identity.json")
    if os.path.exists(idp):
        decl = json.load(open(idp))
        run_identity = decl.get("identity") if isinstance(decl, dict) else None
    return {"file": fp, "key": key, "names": names, "scores": scores,
            "sides": [0, 1], "val": False, "src": os.path.basename(rd.rstrip("/")),
            "terminal": entry.get("terminal") is True,
            "end_reason": entry.get("end_reason"),
            "raw_margin": raw_mg, "raw_u": raw_u,
            "outcome": B._outcome_class(entry),
            "producer_provenance": B._producer_provenance(entry, fp, run_identity),
            "label_inputs": B._label_inputs(entry),
            "steps": entry.get("steps", 0),
            "short_terminal": bool(entry.get("terminal") is True
                                   and entry.get("steps", 0) < 200)}


def main():
    idx = json.load(open(os.path.join(CORPUS, "index.json")))
    entries = {e["game"]: e for e in idx if e.get("game") in GAME_IDS}
    assert sorted(entries) == GAME_IDS, entries.keys()
    os.makedirs(os.path.join(OUT, "shards"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "generations"), exist_ok=True)

    games, tasks = [], []
    for gi, gid_num in enumerate(GAME_IDS):
        g = scan_shaped_game(entries[gid_num], CORPUS)
        g["gi"] = gi
        tids = [B.teacher_id(g["names"][s]) for s in g["sides"]]
        mg = [g["scores"][s] - g["scores"][1 - s] for s in g["sides"]]
        g["fingerprint"] = B._game_fingerprint(g, tids, mg)
        g["shard"] = os.path.join(
            OUT, "shards", g["key"].replace(".json.gz", "") + ".npz")
        if g.get("raw_margin") is not None:
            g["raw_payload"] = {"mg": g["raw_margin"], "u": g["raw_u"],
                                "outcome": B.OUTCOME_CODES[g.get("outcome", "normal")]}
        else:
            g["raw_payload"] = None
        games.append(g)
        tasks.append((g["file"], g["sides"], gi, tids, mg, g["shard"],
                      g["raw_payload"]))

    ns = {}
    for t in tasks:
        gi, n, err = B.build_game(t)
        assert err is None, (t[5], err)
        assert n > 0, t[5]
        ns[gi] = n
        games[gi]["n"] = n
        print(json.dumps({"built": games[gi]["key"], "gi": gi, "rows": n}))

    total = sum(ns.values())
    valid = len(games)
    fields = dict(B.FIELDS)
    raw_games = sum(1 for g in games if g.get("raw_payload") is not None)
    if raw_games:
        fields.update(B.RAW_FIELDS)
    exact_rows = sum(ns[g["gi"]] for g in games
                     if g.get("raw_payload") is not None
                     and g["raw_payload"]["outcome"] != B.OUTCOME_CODES["unusable"])
    label_cov = {"raw_games": raw_games, "games": len(games),
                 "exact_rows": exact_rows, "approx_rows": total - exact_rows,
                 "unusable_rows": 0, "rows": total}
    mirror = B.mirror_signature(games)
    mirror["allowed"] = False
    cfg_fp = B._cfg_fingerprint()
    terminal_games = sum(1 for g in games if g.get("terminal"))
    meta = {"games": games, "total": total, "teachers": F.TEACHERS,
            "cfg_fp": cfg_fp, "failed_games": [], "partial": False,
            "valid_games": valid, "filtered_games": {},
            "labels": label_cov, "terminal_games": terminal_games,
            "terminal_short_games": 0, "mirror_gate": mirror,
            "fixture_note": "join acceptance fixture: built by import of "
                            "cnet2_build.build_game with sides=[0,1]; a "
                            "main() corpus scan of this index builds no "
                            "teacher sides by design (see make_build_fixture.py)"}
    gen_key = {"cfg_fp": cfg_fp, "weighted": False, "total": total,
               "valid": valid, "partial": False, "mirror_gate": mirror,
               "failed": [], "filtered": {}, "labels": label_cov,
               "games": [[g["key"], int(ns.get(g["gi"], 0)), g["fingerprint"]]
                         for g in games]}
    gen_id = "gen-" + hashlib.sha256(
        json.dumps(gen_key, sort_keys=True).encode()).hexdigest()[:16]
    gen_dir = os.path.join(OUT, "generations", gen_id)
    staging = os.path.join(OUT, "generations", f".staging-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    try:
        for k, (dt, sh) in fields.items():
            mm = np.lib.format.open_memmap(
                os.path.join(staging, k + ".npy"), mode="w+", dtype=dt,
                shape=(total,) + sh)
            o = 0
            for gi in range(len(games)):
                D = np.load(games[gi]["shard"])
                n = len(D["y"])
                if k in B.RAW_FIELDS and k not in D:
                    mm[o:o + n] = B._RAW_FILL[k]
                else:
                    mm[o:o + n] = D[k] if k != "gid" else gi
                o += n
            mm.flush()
            del mm
        with open(os.path.join(staging, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        man = {"generation": gen_id, "created": time.time(),
               "cfg_fp": cfg_fp, "weighted": False, "total": total,
               "valid_games": valid, "partial": False,
               "failed_games": [], "filtered_games": {},
               "terminal_games": terminal_games, "terminal_short_games": 0,
               "labels": label_cov,
               "meta_sha256": B._sha256_file(os.path.join(staging, "meta.json")),
               "fields": {k: {"file": k + ".npy", "rows": total,
                              "sha256": B._sha256_file(
                                  os.path.join(staging, k + ".npy"))}
                          for k in fields}}
        with open(os.path.join(staging, "manifest.json"), "w") as fh:
            json.dump(man, fh)
        if os.path.isdir(gen_dir):
            shutil.rmtree(gen_dir)
        os.replace(staging, gen_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ptr = {"generation": gen_id, "dir": f"generations/{gen_id}",
           "cfg_fp": cfg_fp, "total": total, "created": time.time()}
    tmp = os.path.join(OUT, f".current-{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        json.dump(ptr, fh)
    os.replace(tmp, os.path.join(OUT, "current.json"))
    got_dir, got_man = B.load_generation(OUT, verify=True)
    print(json.dumps({"fixture_build_generation": gen_id, "total": total,
                      "verified_via": "cnet2_build.load_generation",
                      "gen_dir": got_dir,
                      "fields": sorted(got_man["fields"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
