"""Reserve the first two eligible development seeds after 621758355 at a frozen public main."""
import ast
import collections
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent
CARGO = HERE.parent.parent
SAPLING = CARGO.parents[1]
ROOT = SAPLING.parents[1]
SNAPSHOT = "02ea663a3986fb791cedb92b552416c93f392efb"
SUPPLEMENT = "56e1e6137fcff237aa212ea9423a29e8d486501d"
SUPPLEMENT_LEDGER = "4e5ff61d38cb0b1d1b2602fe2146921e03f4c67f21dc163897e7b378df8b6c5d"
FIRST = 621758356
LEGACY = CARGO / "territory_sequence/fresh_original/check_maps.py"
REPLAY_RECEIPT = SAPLING / "all_top5/appendix_b243ff82/exposure.json"
RECENT = CARGO / "territory_sequence/windowed/fresh_original/clearance.json"
CORPUS = HERE.parent
MAX_SEED = 2**32
MARKERS = ("seed", "plan", "map", "panel", "protocol", "register", "registr", "clearance",
           "exposure", "declar", "split", "sources", "reserved", "roster", "terrain")
NUMBER = re.compile(rb"(?<![.a-zA-Z0-9])\d{8,10}(?![.a-zA-Z0-9])")
HASH = re.compile(rb"(?<![a-zA-Z0-9])[0-9a-f]{64}(?![a-zA-Z0-9])")
OPENING = re.compile(rb'"(?:initial_full_carbon|full_carbon)"\s*:\s*(\[[\s0-9.,]+\])')
CANDIDATE = re.compile(rb"(?<![.a-zA-Z0-9])621758[3-9][0-9]{2}(?![.a-zA-Z0-9])")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def frozen(path):
    return git("show", SNAPSHOT + ":" + relative(path))


def validate_endpoint_evidence():
    source = json.loads(frozen(CORPUS / "sources.json"))
    protocol = json.loads(frozen(CORPUS / "protocol.json"))
    code = ast.parse(frozen(CORPUS / "audits.py"))
    declaration = [node for node in code.body if isinstance(node, ast.Assign) and
                   any(isinstance(target, ast.Name) and target.id == "EXTERNAL" for target in node.targets)]
    interval = [621758300, 621758400]
    if not (source["external_development_seed_half_open"] == protocol["all_new_dev_seed_interval"] == interval
            and len(declaration) == 1 and list(ast.literal_eval(declaration[0].value)) == interval):
        raise ValueError("Unverified previous lookup range")
    enumerated = {int(seed): board for seed, board in source["registered_and_external_physical_terrains"].items()}
    if not set(range(*interval)) <= enumerated.keys() or interval[1] in enumerated:
        raise ValueError("Previous overlap lookup differs")
    # Test any other occurrences in these files normally. The end operand is exclusive.
    del source["external_development_seed_half_open"]
    del source["registered_and_external_physical_terrains"]
    del protocol["all_new_dev_seed_interval"]
    checker_lines = frozen(CORPUS / "audits.py").splitlines(keepends=True)
    del checker_lines[declaration[0].lineno - 1:declaration[0].end_lineno]
    sanitized = {relative(CORPUS / "sources.json"): json.dumps(source).encode(),
                 relative(CORPUS / "protocol.json"): json.dumps(protocol).encode(),
                 relative(CORPUS / "audits.py"): b"".join(checker_lines)}
    return enumerated, sanitized


def main():
    if LEGACY.read_bytes() != frozen(LEGACY):
        raise ValueError("The physical-map checker differs from the snapshot")
    spec = importlib.util.spec_from_file_location("frozen_development_maps", LEGACY)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    enumerated, sanitized = validate_endpoint_evidence()
    prior = json.loads(frozen(REPLAY_RECEIPT))
    recent = json.loads(frozen(RECENT))
    window = recent["explicit_earlier_development_reservations"] | recent["physical_boards"]
    if sorted(map(int, window)) != list(range(621758350, FIRST)):
        raise ValueError("Six prior window identities are incomplete")
    tree = git("ls-tree", "-rz", SNAPSHOT)
    files = {}
    for line in tree.split(b"\0"):
        if line:
            entry, path = line.split(b"\t", 1)
            files[path.decode()] = entry.split()[2].decode()
    nominal, declared, published = set(), set(), set()
    ladder, faults, blocks = {}, set(), []
    references = collections.defaultdict(set)
    played = collections.defaultdict(set)
    counts = collections.Counter()
    watched = collections.defaultdict(list)
    scan_digest = hashlib.sha256()
    for path, oid in sorted(files.items()):
        p = Path(path)
        # Full repository paths include seed-bearing directories and filenames.
        nominal.update(map(int, NUMBER.findall(path.encode())))
        for s in map(int, CANDIDATE.findall(path.encode())):
            references[s].add(path)
            if p.name.startswith(("raw_", "receipt_")):
                played[s].add(path)
        if p.suffix in (".json", ".jsonl", ".gz", ".tgz", ".tar", ".zip"):
            watched[oid].append(path)
            scan_digest.update((path + "\0" + oid + "\n").encode())

    def inspect(path, data):
        p = Path(path)
        is_metadata = any(s in p.name.lower() for s in MARKERS)
        if path in sanitized:
            data = sanitized[path]
        if p.name == "validation_maps.jsonl.gz" and path.startswith(relative(SAPLING) + "/all_top5/"):
            counts["ladder_registries"] += 1
            for line in data.splitlines():
                row = json.loads(line)
                if row["id"] in ladder:
                    raise ValueError(f"Duplicate historical replay {row['id']}")
                ladder[row["id"]] = (row["seed"], row["initial_terrain_u8_sha256"])
                if "/unscored/" in path:
                    faults.add(row["id"])
        if path.startswith("data/k0_calib/") and "seed_blocks_" in p.name:
            for b in json.loads(data)["blocks"]:
                low, high = map(int, b["seeds"].split("-"))
                blocks.append((low, high, path))
        if is_metadata:
            counts["metadata_files"] += 1
            nominal.update(map(int, NUMBER.findall(data)))
            declared.update(match.decode() for match in HASH.findall(data))
        if b"621758" in data:
            for s in set(map(int, CANDIDATE.findall(data))):
                references[s].add(path)
                if (p.name.startswith("raw_") or p.name == "official_replay.json") and str(s).encode() in data:
                    played[s].add(path)
        # Big archived native and ladder replays repeat full_carbon on later frames.
        # Only the actual opening belongs to this map. Multi-game registers bind each record by its terrain hash.
        opening = OPENING.search(data, 0, min(len(data), 2**20))
        if opening:
            cells = json.loads(opening[1])
            if len(cells) == 225:
                published.add(ref.map_identity.terrain_sha256(cells))
                counts["recorded_openings_with_225_cells"] += 1

    def archive(path, raw):
        if path.endswith((".tar", ".tar.gz", ".tgz")):
            # The published top-five tar volumes and plant timing data are covered by all per-game physical ledgers.
            if path.startswith("data/sapling_all_top5_20261119/") or "/all_top5/plant_timing_data/" in path:
                counts["volumes_covered_by_ladder_ledgers"] += 1
                return
            with tarfile.open(fileobj=io.BytesIO(raw)) as container:
                for member in container:
                    if member.isfile() and member.name.endswith((".json", ".jsonl", ".gz")):
                        archive(path + "/" + member.name, container.extractfile(member).read())
        elif path.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as container:
                for name in container.namelist():
                    if name.endswith((".json", ".jsonl", ".gz")):
                        archive(path + "/" + name, container.read(name))
        else:
            inspect(path, gzip.decompress(raw) if path.endswith(".gz") else raw)

    with subprocess.Popen(["git", "-C", str(ROOT), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as batch:
        for oid, paths in watched.items():
            batch.stdin.write((oid + "\n").encode())
            batch.stdin.flush()
            header = batch.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError(f"Frozen object is missing {oid}")
            raw = batch.stdout.read(int(header[2]))
            if batch.stdout.read(1) != b"\n":
                raise ValueError("Malformed git object")
            for path in paths:
                archive(path, raw)
        batch.stdin.close()
        if batch.wait():
            raise ValueError("Cannot read complete frozen input set")
    seeds, terrains = {x[0] for x in ladder.values()}, {x[1] for x in ladder.values()}
    if (len(seeds) != len(terrains) or len(ladder) != len(seeds) or len(ladder) - len(faults) != 1629
            or sorted(faults) != prior["fault_ids"] or len(ladder) != prior["distinct_archived_ids_including_fault"]):
        raise ValueError("The 1,629 scored and one exceptional public map are incomplete")
    # The frozen main includes this later scorecard while the reproducible per-game map ledger
    # was published after that checkout. Bind both the frozen card and supplemental replay evidence.
    folder = SAPLING / "all_top5/appendix_56e1e613"
    extra = (folder / "validation_maps.jsonl.gz").read_bytes()
    validation = json.loads((folder / "validation.json").read_bytes())
    card = json.loads((folder / "manifest.json").read_bytes())["games"][0]
    official_cards = json.loads(git("show", SUPPLEMENT + ":" + card["score_sources"][0]))
    if (sha(extra) != SUPPLEMENT_LEDGER or validation["revision"] != SUPPLEMENT
            or validation["map_registry_sha256"] != SUPPLEMENT_LEDGER or card["card"] not in
            [{k: r[k] for k in card["card"] if k in r} for r in official_cards]):
        raise ValueError("Supplemental official scorecard or observed ladder terrain differs")
    additional = json.loads(gzip.decompress(extra))
    identity = (additional["seed"], additional["initial_terrain_u8_sha256"])
    if (additional["id"] != card["id"] or additional["id"] in ladder or identity[0] in seeds or identity[1] in terrains
            or card["sha256"] != "23a8a417ff5e4aa0c07b92b327f99e599f2ca49a8484ad549a526c893e3fbd5d"
            or validation["original_board_transitions"] != card["transitions"]):
        raise ValueError("Supplemental opening or original engine validation failed")
    ladder[additional["id"]] = identity
    seeds.add(identity[0])
    terrains.add(identity[1])
    _, known_seed, known_terrain = ref.exam_exposures(sorted(n for n in nominal if n < MAX_SEED))
    known_terrain.update(window.values())
    known_seed.update(map(int, window))
    recorded = known_terrain | declared | published | terrains
    rows, boards = [], {}
    seed = FIRST
    while len(boards) < 2:
        actual = ref.map_identity.local_board(seed)
        ref.np.random.seed(seed)
        env = ref.make("carbon", configuration={**ref.exam.CFG, "randomSeed": seed})
        official = ref.map_identity.terrain_sha256(env.states[0].observation.full_carbon)
        if (actual != official or env.states[0].observation.full_carbon !=
                ref.exam.FastCarbon(ref.exam.CFG).reset(seed=seed).carbon):
            raise ValueError(f"Physical engines disagree on opening {seed}")
        evidence = []
        if seed in played:
            evidence.append("documented historical tape or official replay")
        if seed in enumerated or actual in enumerated.values():
            evidence.append("identity previously enumerated solely for overlap check" +
                            ("; independent game evidence also present" if seed in played else "; no actual game found"))
        if seed in seeds or actual in terrains:
            evidence.append("published ladder opening")
        if ref.RESERVED_RANGE[0] <= seed < ref.RESERVED_RANGE[1]:
            evidence.append("formal seed reservation")
        overlaps = [p for low, high, p in blocks if low <= seed <= high]
        if overlaps:
            evidence.append("registered K0 seed block")
        if seed in known_seed or actual in recorded:
            evidence.append("previous local exposure or declared physical identity")
        paths = sorted(references.get(seed, set()))
        if paths:
            evidence.append("seed in repository path or structured evidence")
        if actual in boards.values():
            evidence.append("same physical terrain as earlier accepted map")
        direct = []
        if not evidence:
            # Search source, scripts, docs and other tracked plaintext for either candidate identity.
            q = subprocess.run(["git", "-C", str(ROOT), "grep", "-Il", "-F", "-e", str(seed), "-e", actual, SNAPSHOT],
                               stdout=subprocess.PIPE, check=False)
            if q.returncode not in (0, 1):
                raise ValueError("Whole-repository text check failed")
            for line in q.stdout.decode().splitlines():
                path = line.split(":", 1)[1]
                if path in sanitized and str(seed).encode() not in sanitized[path] and actual.encode() not in sanitized[path]:
                    continue
                direct.append(path)
            if direct:
                evidence.append("additional exact identity in repository text")
        if not evidence:
            boards[str(seed)] = actual
        rows.append({"seed": seed, "physical_terrain_sha256": actual, "official_and_fast_225_cells_match": True,
                     "eligible": not evidence, "reasons": evidence, "actual_game_evidence": sorted(played.get(seed, set())),
                     "structured_matches": paths, "additional_text_matches": direct, "k0_seed_blocks": overlaps})
        seed += 1
    receipt = {"snapshot": SNAPSHOT, "scanned_git_tree": git("rev-parse", SNAPSHOT + "^{tree}").decode().strip(),
               "physical_boards": boards, "seed_collisions": 0, "terrain_collisions": 0,
               "supplemental_ladder_source": {"score_revision": SUPPLEMENT, "terrain_ledger": relative(folder / "validation_maps.jsonl.gz"),
                                              "terrain_ledger_sha256": SUPPLEMENT_LEDGER, "replay_sha256": card["sha256"], "row": additional},
               "selection_rule": "First two distinct eligible seeds in increasing numerical order starting at 621758356. Only recorded identities, declarations and opening-cell evidence determine selection.",
               "official_and_fast_opening_sources": ["zerosum_env.make(carbon)", "FastCarbon.reset"],
               "scope": "Frozen Git repository paths, JSON and JSONL including compressed evidence and archived members, ledger-backed top-five volumes, conservative seed references in metadata, observed opening digests, and exact seed/hash matches in tracked plaintext. Previously unplayed identity-only lookup entries are separately identified. The right endpoint of the earlier half-open development lookup is not an occupied seed.",
               "checks": dict(counts), "whole_repository_paths": len(files), "structured_blob_ids": len(watched),
               "structured_input_manifest_sha256": scan_digest.hexdigest(), "known_local_or_conservatively_inferred_seed_ids": len(known_seed),
               "known_local_or_conservatively_inferred_terrain_hashes": len(known_terrain), "recorded_opening_hashes_outside_or_overlapping_ledgers": len(published),
               "declared_metadata_sha256_values": len(declared), "score_complete_ladder_games": len(ladder)-len(faults),
               "exceptional_ladder_ids": sorted(faults), "distinct_ladder_terrain_hashes": len(terrains),
               "k0_seed_block_files": len({p for _, _, p in blocks}), "formal_seed_half_open": ref.RESERVED_RANGE,
               "previous_window_development_boards": window, "prior_enumeration_file": relative(CORPUS / "sources.json"),
               "scope_inputs_sha256": {relative(p): sha(frozen(p)) for p in (LEGACY, REPLAY_RECEIPT, RECENT, CORPUS / "sources.json", CORPUS / "protocol.json", CORPUS / "audits.py")},
               "attempts": rows, "games_launched": 0}
    destination = HERE / "clearance.json"
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"snapshot": SNAPSHOT, "physical_boards": boards, "attempts": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
