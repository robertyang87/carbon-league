"""Verify and archive the next numeric top-five replay batch from frozen score indexes."""
import argparse
import gzip
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import fetch
from appendix_97de1ebc.collect import inventories
from appendix_batch import prepare
from appendix_f7109df6.report import build
from validation_delta import run as validate

TOP = Path(__file__).resolve().parent
BASE = TOP.parent


def previous_manifests(last):
    manifest = fetch.load(last / "manifest.json")
    revisions = [*manifest.get("baseline_revisions", []), manifest["revision"]]
    if not manifest.get("baseline_revisions") and manifest.get("baseline_revision"):
        revisions.insert(0, manifest["baseline_revision"])
    paths = {}
    for path in [TOP / "manifest.json", *TOP.glob("appendix_*/manifest.json")]:
        revision = fetch.load(path)["revision"]
        if revision in paths:
            raise ValueError(f"multiple published manifests for {revision}")
        paths[revision] = path
    return [paths[revision] for revision in revisions]


def check_archive(manifest):
    for entry in manifest["new_archives"]:
        path = fetch.ROOT / entry["archive"]
        if fetch.sha(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"raw replay archive differs: {path}")
        games = [game for game in manifest["games"] if game["archive_location"]["archive"] == entry["archive"]]
        with tarfile.open(path) as archive:
            if len(archive.getmembers()) != len(games):
                raise ValueError(f"unexpected archive membership: {path}")
            for game in games:
                if fetch.sha(archive.extractfile(game["path"]).read()) != game["sha256"]:
                    raise ValueError(f"raw replay differs: {game['id']}")


def compare_boards(prior, current, manifest, faults):
    """Bind the independent exclusion table to each historical ledger, including nonnumeric games."""
    ids, seeds, terrain = set(), set(), set()
    sources = []
    for path in [*prior, current / "manifest.json"]:
        directory = path.parent
        ledgers = [directory / "validation_maps.jsonl.gz"]
        if directory / "unscored/manifest.json" in faults:
            ledgers.append(directory / "unscored/validation_maps.jsonl.gz")
        for ledger in ledgers:
            raw = gzip.decompress(ledger.read_bytes())
            rows = [json.loads(line) for line in raw.splitlines()]
            row_ids = [row["id"] for row in rows]
            for row in rows:
                if row["id"] in ids or row["seed"] in seeds or row["initial_terrain_u8_sha256"] in terrain:
                    raise ValueError(f"repeated game or map identity at {row['id']}")
                ids.add(row["id"])
                seeds.add(row["seed"])
                terrain.add(row["initial_terrain_u8_sha256"])
            sources.append([str(ledger.relative_to(BASE)), len(rows), fetch.sha(raw),
                            fetch.sha("\n".join(sorted(row_ids)).encode())])
    def digest(items):
        return fetch.sha(json.dumps(items, separators=(",", ":")).encode())

    previous_file = prior[-1].parent / "exposure.json"
    previous = fetch.load(previous_file)
    if digest(sources[:-1]) != previous["all_source_sha256"]:
        raise ValueError("historical game/map exclusion source digest changed")
    fault_count = sum(len(fetch.load(path)["games"]) for path in faults)
    if len(ids) != manifest["accumulated_games"] + fault_count:
        raise ValueError("incomplete union of numeric and nonnumeric games")
    cross = {}
    actor = BASE / "master6_actor_dev"
    for relative in ("symmetry_dev/clearance.json", "economy_aux_dev/native_dev/clearance.json",
                     "tree_response_training/clearance.json", "tree_response_training/next_intake.json",
                     "tree_response_training/supplement_01/registration.json"):
        path = actor / relative
        data = fetch.load(path)
        sets = [batch["terrain_at_queue_time"] for batch in data["batches"]] if "batches" in data else [data["boards"]]
        for boards in sets:
            if set(map(int, boards)) & seeds or set(boards.values()) & terrain:
                raise ValueError(f"development or training map overlaps: {relative}")
        cross[relative] = {"maps_checked": sum(map(len, sets)), "registration_sha256": fetch.sha(path.read_bytes())}
    receipt = {"revision": manifest["revision"], "complete_two_scores": manifest["accumulated_games"],
               "official_fault_prefix": fault_count, "distinct_ids_seeds_and_initial_terrains": len(ids),
               "new_source": sources[-1], "all_source_sha256": digest(sources),
               "previous_receipt_sha256": fetch.sha(previous_file.read_bytes()),
               "checked_development_and_training_zero_overlap": cross}
    fetch.dump(current / "exposure.json", receipt)
    return receipt


def verify(prior_dir, revision):
    rev = fetch.git("rev-parse", "--verify", revision + "^{commit}").decode().strip()
    prior = previous_manifests(prior_dir)
    relative = [str(p.parent.relative_to(TOP)) for p in prior]
    old = fetch.load(prior[-1])["revision"]
    fetch.git("merge-base", "--is-ancestor", old, rev)
    current = TOP / ("appendix_" + rev[:8])
    current.mkdir(exist_ok=True)
    fault_ledgers = [p.parent / "unscored/manifest.json" for p in prior if (p.parent / "unscored/manifest.json").exists()]
    fault_ids = {g["id"] for p in fault_ledgers for g in fetch.load(p)["games"]}
    delta = inventories(old, rev, relative)
    for gid in delta["unscored_new_ids"]:
        if gid not in fault_ids:
            raise ValueError(f"new null score requires a separate status and prefix audit: {gid}")
        entry = delta["all_added_references_by_id"][gid]
        entry["class"] = "old_fault"
        delta["existing_references"] += len(entry["paths"])
    delta["previous_fault_references"] = delta["unscored_new_ids"]
    delta["unscored_new_ids"] = []
    new = {gid:r for gid,r in delta["all_added_references_by_id"].items() if r["class"] == "top"}
    delta["new_mutual"] = sum(row["mutual"] for row in new.values())
    fetch.dump(current / "score_delta.json", delta)
    manifest_path = current / "manifest.json"
    if not manifest_path.exists():
        partial = {}
        prepare(rev, prior, current, (delta["new_count"], delta["new_mutual"]), unscored=partial)
        if set(partial) != fault_ids:
            raise ValueError("historical incomplete score cohort changed")
    manifest = fetch.load(manifest_path)
    if (manifest["revision"] != rev or manifest["accumulated_games"] != delta["union_games"]
            or {g["id"] for g in manifest["games"]} != set(new)):
        raise ValueError("independent full score catalog and score diff disagree")
    check_archive(manifest)
    registries = []
    for directory in relative:
        p = TOP / directory / "validation_maps.jsonl.gz"
        if p.read_bytes() != fetch.git("show", rev + ":" + p.relative_to(fetch.ROOT).as_posix()):
            raise ValueError(f"historical map registry differs: {p}")
        registries.append((directory, fetch.sha(p.read_bytes())))
    validate(fetch.CACHE, current, rev, old, registries, previous_fault_ledgers=fault_ledgers)
    exposure = compare_boards(prior, current, manifest, fault_ledgers)

    def run(script, *args):
        subprocess.run([sys.executable, str(TOP/script), *map(str, args)], check=True)
    run("appendix_499886f1/cases.py", "--manifest", manifest_path, "--cache", fetch.CACHE)
    run("appendix_0137949d/cash.py", "--manifest", manifest_path, "--cache", fetch.CACHE)
    run("turn_action_windows.py", current, "--cache", fetch.CACHE)
    index = TOP / f"context_{current.name}_index.tsv.gz"
    receipt = TOP / f"context_{current.name}_receipt.json"
    run("context_catalog.py", "--manifest", manifest_path, "--output", index)
    run("context_audit.py", "--index", index, "--validation", current / "validation_games.jsonl.gz",
        "--out", receipt, "--jobs", "1")
    mutual = sum(row["cohort"] == "mutual" for p in [*prior, manifest_path] for row in fetch.load(p)["games"])
    build(current, total_mutual=mutual)
    report = current / "README.md"
    desc = (f"含另列故障局，共 {exposure['distinct_ids_seeds_and_initial_terrains']:,} 张已见初盘。"
            f"{manifest['accumulated_games']:,} 场有双席完整官方分数。")
    with report.open("a") as out:
        out.write("\n" + desc + "[初盘排除与实验隔离核对](exposure.json)\n")
        if (current / "matched.md").is_file():
            out.write("\n同一报名对手、同一物理席的实际观察可逐工复核。[同对手的防守与经营分解](matched.md) · [逐步命令与结算](turn_actions.json.gz)\n")
        else:
            out.write("\n[双方转折前后的动作与结算](turn_actions.json.gz)\n")
    run("future_economy_check.py", "--replay-cache", fetch.CACHE / "replays", "--expect-games",
        manifest["accumulated_games"], "--out", current / "future_economy_check.json")
    run("verify_appendix_turns.py", current)
    return current


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--prior", required=True, help="previous appendix directory inside all_top5")
    args.add_argument("--revision", required=True, help="score index commit to publish")
    options = args.parse_args()
    print(verify(TOP / options.prior, options.revision), flush=True)
