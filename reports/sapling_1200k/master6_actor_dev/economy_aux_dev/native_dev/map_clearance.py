"""Check the two registered native development maps against replay terrain and prior reservations."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parents[1]
TREE = ACTOR / "tree_response_training"
BASE = ACTOR.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(TREE))
import batch_clearance as prior
from tree_win_gate import RESERVED_RANGE

SOURCE = TREE / "planter_survival/native_dev/clearance.py"
EXPOSURE = BASE / "all_top5/appendix_97de1ebc/exposure.json"
EXPOSURE_SHA = "8e8c1e0de0f8e18f5daee82e7517c1181bde6d5e1bdd5475b5be541d3ce9d6c4"
SCANNER = ACTOR / "freight_deployment/map_preflight.py"
CATALOG = SCANNER.parent / "first_response/map_preflight.json"


def disjoint(boards, seeds, terrains, source):
    if {int(s) for s in boards} & set(seeds) or set(boards.values()) & set(terrains):
        raise ValueError(f"seed or terrain overlaps {source}")


def public_boards(boards, revision):
    evidence = prior.load(EXPOSURE)
    prior.check(prior.digest(EXPOSURE) == EXPOSURE_SHA and evidence["revision"] == revision,
                "the frozen public exposure declaration changed")
    seeds, terrain, ids = set(), set(), set()
    for name, size, content_sha, id_sha in evidence["identity_sources"]:
        raw = gzip.decompress((BASE / name).read_bytes())
        rows = [json.loads(line) for line in raw.splitlines()]
        group_ids = [r["id"] for r in rows]
        prior.check(len(rows) == size and prior.sha(raw) == content_sha and
                    prior.sha("\n".join(sorted(group_ids)).encode()) == id_sha,
                    "public opening ledger does not match the declaration")
        for r in rows:
            seed, board, game = r["seed"], r["initial_terrain_u8_sha256"], r["id"]
            prior.check(type(seed) is int and len(bytes.fromhex(board)) == 32 and
                        seed not in seeds and board not in terrain and game not in ids,
                        "duplicate or malformed public game identity")
            seeds.add(seed)
            terrain.add(board)
            ids.add(game)
    prior.check(len(ids) == evidence["distinct_terrain_for_exclusion"] == 1488 and
                evidence["verified_scored"] == 1487 and evidence["fault_prefix"] == 1,
                "incomplete scored or fault-game exposures")
    disjoint(boards, seeds, terrain, "all 1,488 published openings")
    return {"source":prior.relative(EXPOSURE), "sha256":EXPOSURE_SHA,
            "revision":revision,"verified_scored":1487,"fault_prefix":1,"unique_terrain":len(terrain)}


def registered(boards):
    spec = importlib.util.spec_from_file_location("previous_native_sources", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    older, files, tapes = mod.exclusion_ledger()
    for directory in (ACTOR / "symmetry_dev", TREE / "planter_survival/native_dev"):
        clearance = directory / "clearance.json"
        plan = directory / "protocol.json"
        doc = prior.load(clearance)
        prior.check(doc["declaration_sha256"] == prior.digest(plan) and
                    doc["verdict"] == "eligible_in_checked_scope", "invalid earlier development clearance")
        files[prior.relative(clearance)] = prior.digest(clearance)
        for seed, terrain in doc["boards"].items():
            prior.check(older.get(seed, terrain) == terrain, "conflicting earlier registries")
            older[seed] = terrain
        for arm in ("parent","candidate"):
            official = directory / arm / "official_replay.json"
            proof = prior.load(official)["runs"][arm]["games"]
            files[prior.relative(official)] = prior.digest(official)
            expected = {(int(seed),seat) for seed in doc["boards"] for seat in (0,1)}
            prior.check({(r["seed"],r["seat"]) for r in proof}==expected and len(proof)==len(expected),
                        "incomplete archived original comparison")
            for r in proof:
                seed=r["seed"]
                tape = directory / arm / f"raw_{seed}_{r['seat']}.json.gz"
                board = prior.mi.terrain_sha256(prior.load(tape)["initial_full_carbon"])
                prior.check(board == r["map_sha256"] == older[str(seed)] and
                            prior.digest(tape)==r["transcript_sha256"],"signed earlier opening differs")
                tapes += 1
    disjoint(boards, (int(s) for s in older), older.values(), "native source games or reserved intakes")
    return {"registries_and_official_sha256":files, "unique_predeclared_or_played_maps":len(older),
            "actual_signed_native_source_and_development_tapes":tapes}


def actor_training(boards):
    manifest = ACTOR / "data_split.json"
    training = HERE.parent / "result.json"
    receipt = prior.load(training)
    all_rows = {g["id"]:g for g in prior.load(manifest)["games"]}
    checked = {}
    for part, src in receipt["sources"].items():
        ids = set(src["shards"])
        prior.check(set(src["replays"])==ids, "training shard/replay set mismatch")
        rows = [all_rows[gid] for gid in sorted(ids)]
        prior.check(all(g["split"]==part and g["sha256"]==src["replays"][g["id"]] for g in rows),
                    "economic actor input split differs")
        disjoint(boards, [g["seed"] for g in rows], [g["terrain_sha256"] for g in rows], part)
        checked[part]=len(rows)
    # Parent model also used two maps excluded from the new fit. Check all ancestor split rows.
    disjoint(boards, [r["seed"] for r in all_rows.values()], [r["terrain_sha256"] for r in all_rows.values()],
             "all parent actor split entries")
    return {"parent_manifest_sha256":prior.digest(manifest),"child_training_receipt_sha256":prior.digest(training),
            "parent_manifest_entries":len(all_rows),"child_maps_by_split":checked}


def full_scan(seed, terrain, plan, work):
    protocol = HERE / "protocol.json"
    scratch = work / f"{seed}.json"
    if not scratch.exists():
        subprocess.run([sys.executable,str(SCANNER),"--seed",str(seed),"--revision",plan["source_revision"],
                        "--declaration",prior.relative(protocol),"--registry",str(CATALOG),"--out",str(scratch)],
                       cwd=ROOT,check=True,env={**os.environ,"GIT_CONFIG_COUNT":"1","GIT_CONFIG_KEY_0":"grep.threads",
                                               "GIT_CONFIG_VALUE_0":"1"})
    raw = scratch.read_bytes()
    row = prior.validate_scan(raw,seed,terrain,protocol,plan,{prior.relative(protocol):prior.digest(protocol)})
    prior.check(not row["conflicts"] and not row["same_terrain_matches"] and
                row["verdict"]=="eligible_in_checked_scope" and row["historical_catalog_loaded_sha256"]==prior.digest(CATALOG) and
                row["sha256_of_this_script"]==prior.digest(SCANNER) and row["formal_reserved_half_open"]==list(RESERVED_RANGE),
                "full historical or formal scan failed")
    name = f"clearance_audit_{seed}.zip"
    member=zipfile.ZipInfo(scratch.name)
    member.compress_type=zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(work/name,"w") as z:
        z.writestr(member,raw)
    rec={"archive":name,"report_sha256":prior.sha(raw),"sha256":prior.digest(work/name),
         "coverage":row["coverage"],"frozen_ladder":row["frozen_ladder"],"reserved_formal_half_open":list(RESERVED_RANGE)}
    return rec


def issue(work):
    protocol=HERE/"protocol.json"
    plan=prior.load(protocol)
    seeds=plan["maps"]
    prior.check(plan["formal"] is False and plan["physical_seats"]==[0,1] and len(set(seeds))==len(seeds)==2,
                "unexpected matched development protocol")
    prior.check(all(type(s) is int and not (RESERVED_RANGE[0]<=s<RESERVED_RANGE[1]) for s in seeds),
                "overlap with formal seed reservation")
    boards={str(s):prior.official_board(s) for s in seeds}
    prior.check(len(set(boards.values()))==len(seeds),"registered seeds have duplicate terrain")
    final=HERE/"clearance.json"
    if final.exists():
        obj=prior.load(final)
        prior.check(obj["verdict"]=="eligible_in_checked_scope" and obj["declaration_sha256"]==prior.digest(protocol) and
                    obj["boards"]==boards and obj["audit_issuer_sha256"]==prior.digest(Path(__file__)) and
                    obj["exposure"]["sha256"]==prior.digest(EXPOSURE) and all(prior.digest(HERE/r["archive"])==r["sha256"]
                    for r in obj["full_audits"].values()),"sealed clearance differs")
        return
    exposure=public_boards(boards,plan["source_revision"])
    predeclared=registered(boards)
    actual=actor_training(boards)
    cached_seeds, cached_terrains, cached=prior.ladder()
    disjoint(boards,cached_seeds,cached_terrains,"additional workspace ladder replays")
    historical=prior.historical_catalogs(prior.load(TREE/"clearance.json"),boards)
    scans={str(s):full_scan(s,boards[str(s)],plan,work) for s in seeds}
    for rec in scans.values():
        dest=HERE/rec["archive"]
        draft=dest.with_suffix(".writing")
        draft.write_bytes((work/rec["archive"]).read_bytes())
        draft.replace(dest)
    prior.save(final,{"verdict":"eligible_in_checked_scope","declaration_sha256":prior.digest(protocol),
                      "audit_issuer_sha256":prior.digest(Path(__file__)),"source_revision":plan["source_revision"],
                      "boards":boards,"exposure":exposure,"training_and_native_games":predeclared,"actor_data":actual,
                      "formal_reserved_half_open":list(RESERVED_RANGE),"original_catalogs":historical,
                      "ladder":cached,"full_audits":scans,
                      "scope":"Two newly declared development maps. The frozen Git tree, known workspaces, earlier original games and queued intakes, public terrain and prospective formal seed interval are checked. Historical sources lacking true openings are limited to their recorded seed or inferred default terrain."})
    print(prior.digest(final), boards,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work",type=Path,help="reuse original scanner outputs for this frozen declaration")
    args=ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="economic-native-map-audit-") as tmp:
        issue(args.work or Path(tmp))
