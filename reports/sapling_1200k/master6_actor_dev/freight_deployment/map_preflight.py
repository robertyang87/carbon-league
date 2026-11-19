"""Identity-only audit of the named development map against frozen records and locally available game histories."""

from collections import Counter, defaultdict
import argparse, gzip, hashlib, json, os, re, subprocess, sys, tarfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = HERE.parents[3]
WORK = ROOT.parent
sys.path.insert(0, str(BASE))
import map_identity as mi
from tree_win_gate import RESERVED_RANGE

REV = "8ec53bc5f774ddcb2e4f0b1eb0be7bbb85770efa"
SEED = 621460120
DECL = (
    "reports/sapling_1200k/master6_actor_dev/freight_deployment/development_gate.json"
)
H = lambda b: hashlib.sha256(b).hexdigest()
INT = r"([0-9][0-9_]{5,13})"
RANGE = re.compile(
    r"(?<![\w.])" + INT + r"\s*(?:\.\.|[~:–—-]|\bto\b)\s*" + INT + r"(?![\w.])"
)
PYRANGE = re.compile(
    r"\brange\(\s*" + INT + r"\s*,\s*" + INT + r"(?:\s*,\s*([1-9][0-9_]*))?\s*\)"
)
TEXT = {
    ".json",
    ".jsonl",
    ".md",
    ".sh",
    ".log",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".txt",
    ".py",
    ".out",
    ".stdout",
}


def rel(p):
    return str(p.relative_to(WORK))


def git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, check=False
    )


def intish(x):
    return type(x) is int and 0 <= x < 2**32


def objhash(x):
    return H(json.dumps(x, sort_keys=True, separators=(",", ":")).encode())


def addw(index, terrain, seed, how, origin):
    index[terrain, seed, how].add(origin)


def build(SEED, REV, DECL, dest, catalog=None):
    if not intish(SEED):
        raise ValueError("seed outside uint32")
    if Path(DECL).is_absolute() or ".." in Path(DECL).parts:
        raise ValueError("declaration must be a repository-relative path")
    witness = git("show", REV + ":" + DECL)
    frozen_declaration = witness.returncode == 0
    declaration_bytes = (ROOT / DECL).read_bytes()
    if (
        not str(SEED).encode() in declaration_bytes
        or frozen_declaration
        and declaration_bytes != witness.stdout
    ):
        raise ValueError(
            "wrong trial declaration or worktree has edited frozen declaration"
        )
    if git("rev-parse", "--verify", REV + "^{commit}").returncode:
        raise ValueError("unavailable revision")
    seedset = mi.ladder_boards()
    # Separate what existed at the pinned decision from later published exposures.
    frozen_rules = {}
    for ledger, *_ in mi.SOURCES:
        obj = git("show", REV + ":" + str((BASE / ledger).relative_to(ROOT)))
        if obj.returncode:
            continue
        if H(obj.stdout) != H((BASE / ledger).read_bytes()):
            raise ValueError("a pre-existing historical ladder ledger was replaced")
        frozen_rules[ledger] = obj.stdout
    if not frozen_rules:
        raise ValueError("missing frozen ladder evidence")
    frozen_ladder_fingerprint = H(
        b"".join(bytes.fromhex(H(b)) for b in frozen_rules.values())
    )
    frozen_ladder_count = sum(
        len(gzip.decompress(b).splitlines()) for b in frozen_rules.values()
    )
    terrain = mi.local_board(SEED)
    from tree_audited_game import exam
    from official_game_replay import make, ENGINE

    np.random.seed(SEED)
    official = make("carbon", configuration={**exam.CFG, "randomSeed": SEED})
    official_terrain = mi.terrain_sha256(official.states[0].observation.full_carbon)
    if terrain != official_terrain:
        raise ValueError("independent game interpreter and native generator disagree")
    reserved = RESERVED_RANGE[0] <= SEED < RESERVED_RANGE[1]
    globally_seen = SEED in seedset[1] or terrain in seedset[2]
    # Exact whole-repository scan is pinned to the old Git tree, independent of incidental worktree edits.
    r = git("grep", "-l", "-I", "-F", "-e", str(SEED), "-e", terrain, REV, "--", ".")
    if r.returncode not in (0, 1):
        raise ValueError("frozen git search unavailable")
    frozen_hits = [s.split(":", 1)[1] for s in r.stdout.decode().splitlines()]
    hits = []
    allowed = []
    index = defaultdict(set)
    files = {}
    skipped = Counter()
    sizes = Counter()
    counts = Counter()
    rules = {}
    raw_ladder = set()
    for n, *_ in mi.SOURCES:
        q = BASE / n
        records = [
            json.loads(row) for row in gzip.decompress(q.read_bytes()).splitlines()
        ]
        man = json.loads((q.parent / "manifest.json").read_bytes())
        for c in man["games"]:
            raw_ladder.add(c["sha256"])
        rules[str(q.relative_to(ROOT))] = {
            "rows": len(records),
            "compressed_sha256": H(q.read_bytes()),
        }
    all_dirs = [ROOT / "code", ROOT / "reports", ROOT / "data", ROOT / "automation"]
    for d in sorted(WORK.iterdir()):
        if not d.is_dir() or d == ROOT:
            continue
        if (d / ".git").exists():
            if (d / "reports/sapling_1200k").is_dir():
                all_dirs.append(d / "reports/sapling_1200k")
        elif d.name.startswith(
            (
                "sapling",
                "cash-v3",
                "tree-v",
                "restore-top5",
                "replay-return",
                "elite-rules",
                "top5-108",
            )
        ):
            all_dirs.append(d)
    cash_shards = []
    content_aliases = defaultdict(list)
    localrefs = defaultdict(set)
    skipped_files = []
    if catalog:
        previous = json.loads(catalog.read_bytes())
        if (
            not previous["ladder"]["inputs"]
            or any(
                rules.get(path) != rec
                for path, rec in previous["ladder"]["inputs"].items()
            )
            or objhash(previous["local_source_sha256"])
            != previous["local_sources_digest"]
            or previous.get("schema_version") != 2
        ):
            raise ValueError(
                "incompatible saved evidence: require schema 2, original component ledger hashes contained in current archive, and intact local source digest; omit --registry to rebuild"
            )
        for value in previous["local_native_or_training_maps"]:
            for name in value["source"]:
                addw(index, value["terrain"], value["seed"], value["method"], name)
                files[name] = previous["local_source_sha256"][name]

    def find_terrains(obj, src, seed=None):
        if not isinstance(obj, (dict, list)):
            return
        if isinstance(obj, list):
            for el in obj:
                if isinstance(el, (dict, list)):
                    find_terrains(el, src, seed)
            return
        seed = next(
            (obj[k] for k in ("seed", "randomSeed", "map_seed") if intish(obj.get(k))),
            seed,
        )
        if intish(seed) and (
            600000000 <= seed < 700000000
            or "actions" in obj
            or "official_score_by_seat" in obj
            or ("shard" in obj and "seat" in obj)
        ):
            localrefs[seed].add(src)
        for k in ("seeds", "map_seeds", "train_seeds", "dev_seeds", "holdout_seeds"):
            vals = obj.get(k)
            if isinstance(vals, list):
                for sv in vals:
                    if intish(sv) and 600000000 <= sv < 700000000:
                        localrefs[sv].add(src)
        for a, b in [
            ("seed_start", "seed_end"),
            ("start_seed", "end_seed"),
            ("seed_min", "seed_max"),
        ]:
            lo, hi = obj.get(a), obj.get(b)
            if intish(lo) and intish(hi) and lo <= SEED <= hi:
                hits.append([src, "declared_seed_bounds", lo, hi])
        for k in ("seed_range", "seed_interval", "map_seed_range"):
            bounds = obj.get(k)
            if (
                isinstance(bounds, list)
                and len(bounds) == 2
                and all(intish(v) for v in bounds)
                and bounds[0] <= SEED <= bounds[1]
            ):
                hits.append([src, "declared_seed_interval", *bounds])
        for k in (
            "initial_terrain_u8_sha256",
            "initial_terrain_sha256",
            "map_sha256",
            "terrain_sha256",
        ):
            h = obj.get(k)
            if isinstance(h, str) and re.fullmatch("[0-9a-f]{64}", h):
                addw(index, h, seed, "recorded_digest", src)
        for k in ("initial_full_carbon", "initial_terrain", "initial_carbon"):
            a = obj.get(k)
            if isinstance(a, list) and len(a) == 225:
                try:
                    addw(index, mi.terrain_sha256(a), seed, "observed_225_cells", src)
                except ValueError:
                    skipped["nonintegral_or_not_full_initial_array"] += 1
        for k, item in obj.items():
            if (
                isinstance(item, dict)
                or isinstance(item, list)
                and item
                and isinstance(item[0], (dict, list))
            ):
                find_terrains(item, src, seed)

    def scantext(src, raw, gz=False):
        # A source is identified by the bytes on disk, and nested members by the byte stream payload.
        sig = H(raw)
        content_aliases[sig].append(src)
        if sig in raw_ladder:
            counts["public_replay_by_exact_compressed_hash"] += 1
            return
        if len(content_aliases[sig]) > 1:
            return
        data = gzip.decompress(raw) if gz else raw
        if b"\0" in data[:8000]:
            return
        counts["unique_text_payloads"] += 1
        if src == rel(ROOT / DECL) and sig == H(declaration_bytes):
            allowed.append(src)
            return
        if str(SEED).encode() in data or terrain.encode() in data:
            if src == rel(ROOT / DECL):
                allowed.append(src)
            else:
                hits.append([src, "exact_number_or_terrain"])
        if len(data) < 6_000_000 and any(
            x in data
            for x in (
                b"..",
                b"range(",
                b"to ",
                b"-",
                "–".encode(),
                "—".encode(),
                b"~",
                b":",
            )
        ):
            for line in data.splitlines():
                if len(line) > 10000 or not re.search(rb"[0-9][0-9_]{5}", line):
                    continue
                txt = line.decode("utf-8", "replace")
                for rex, py in ((RANGE, False), (PYRANGE, True)):
                    for mat in rex.finditer(txt):
                        a, b = map(lambda v: int(v.replace("_", "")), mat.group(1, 2))
                        if not a <= SEED <= b or (
                            py
                            and (
                                SEED >= b
                                or (SEED - a)
                                % int((mat.group(3) or "1").replace("_", ""))
                            )
                        ):
                            continue
                        hits.append([src, "numeric_interval", mat.group(0)[:120]])
        low = src.lower()
        if not (low.endswith((".json", ".json.gz", ".jsonl", ".jsonl.gz"))):
            return
        # Full raw replay arrays are examined separately. Avoid Python objects for unrelated plots/receipts.
        if not any(
            s in data
            for s in (
                b'"initial_full_carbon"',
                b'"terrain_sha256"',
                b'"initial_terrain',
                b'"map_sha256"',
                b'"initial_map_sha256"',
                b'"full_carbon"',
                b'"seed"',
                b'"randomSeed"',
                b'"seed_start"',
                b'"seed_range"',
            )
        ):
            return
        if len(data) > 35_000_000 and not any(
            s in data for s in (b'"initial_full_carbon"', b'"full_carbon"')
        ):
            skipped_files.append(
                [
                    src,
                    len(data),
                    "large summary; exact integer and terrain lookup already applied",
                ]
            )
            return
        try:
            if low.endswith((".jsonl", ".jsonl.gz")):
                values = [json.loads(l) for l in data.splitlines() if l.strip()]
            else:
                values = [json.loads(data)]
        except (ValueError, UnicodeDecodeError):
            skipped["unparsed_json_after_literal_scan"] += 1
            return
        for obj in values:
            if (
                isinstance(obj, dict)
                and isinstance(obj.get("steps"), list)
                and obj["steps"]
            ):
                try:
                    ob = obj["steps"][0][0]["observation"]
                    a = ob["full_carbon"]
                    seed = obj.get("configuration", {}).get("randomSeed")
                    if ob["step"] == 0:
                        addw(
                            index,
                            mi.terrain_sha256(a),
                            seed,
                            "observed_replay_opening",
                            src,
                        )
                        if intish(seed) and 600000000 <= seed < 700000000:
                            localrefs[seed].add(src)
                except (KeyError, ValueError, TypeError, IndexError):
                    skipped["replays_without_step0_full_carbon"] += 1
                continue
            find_terrains(obj, src)
        files[src] = sig

    for d in all_dirs:
        if not d.exists():
            continue
        for dp, dirs, names in os.walk(d):
            dirs[:] = [
                n
                for n in dirs
                if n
                not in (".git", "__pycache__", ".venv", "node_modules", ".pytest_cache")
            ]
            for n in names:
                p = Path(dp) / n
                if not p.is_file() or p.absolute() in (
                    Path(__file__).absolute(),
                    (HERE / "map_preflight.json").absolute(),
                    dest.absolute(),
                ):
                    continue
                src = rel(p)
                ext = p.suffix
                gz = ext == ".gz"
                inner = Path(p.stem).suffix if gz else ext
                if inner not in TEXT and ext not in (".tar", ".tgz", ".npz"):
                    continue
                if ext == ".npz":
                    # These two trace families genuinely store full simulator carbon; the soft and tree
                    # DAgger auxiliary carbon arrays are explicit zero placeholders. Other NPZ files often
                    # contain only weights/features. Their provenance manifests are instead inspected.
                    if p.parent.name.startswith(
                        ("sapling-cash-data", "sapling-cash-independent-dataset")
                    ):
                        cash_shards.append(p)
                    skipped["NPZ_by_manifest_unless_verified_full_carbon"] += 1
                    continue
                # Validated public replay/decision/label caches are already covered by the stricter
                # seed + original terrain attestations for all 1,382 boards. Hash matching raw replays
                # still checks every local payload against that catalog; unpublished files are parsed.
                if "sapling-all-top5-20261119" in p.parts and "replays" not in p.parts:
                    skipped["derivatives_of_public_archive"] += 1
                    continue
                if "all_top5" in p.parts and (
                    "economy_games" in p.parts
                    or "economy_data" in p.parts
                    or n in ("layout.json.gz", "defense_traces.jsonl.gz")
                    or n.endswith("-1.md")
                ):
                    skipped["derivatives_of_public_archive"] += 1
                    continue
                if p.stat().st_size > 70_000_000:
                    skipped_files.append(
                        [
                            src,
                            p.stat().st_size,
                            "above scan ceiling; Git tracked exact text checked independently",
                        ]
                    )
                    continue
                if ext in (".tar", ".tgz") or n.endswith(".tar.gz"):
                    if "sapling_all_top5" in src:
                        skipped["archive_covered_by_frozen_ladder_registry"] += 1
                        continue
                    try:
                        with tarfile.open(p, mode="r:*") as ar:
                            for m in ar:
                                if not m.isfile() or not m.name.endswith(
                                    (".json", ".jsonl", ".json.gz", ".jsonl.gz")
                                ):
                                    continue
                                if m.size > 35_000_000:
                                    skipped_files.append(
                                        [
                                            src + "!" + m.name,
                                            m.size,
                                            "large nested member",
                                        ]
                                    )
                                    continue
                                scantext(
                                    src + "!" + m.name,
                                    ar.extractfile(m).read(),
                                    m.name.endswith(".gz"),
                                )
                    except tarfile.TarError:
                        skipped_files.append([src, p.stat().st_size, "unreadable tar"])
                        continue
                else:
                    scantext(src, p.read_bytes(), gz)
    # Reconstruct true full-carbon values in DAgger NPZs only for an actual step-zero row with
    # known seat and exact invertibility under the generator's recorded q8 encoding.
    fm = json.loads((WORK / "sapling-cash-data/manifest.json").read_bytes())
    cg = {r["shard"]: r for r in fm["games"]}
    for f in sorted(cash_shards):
        n = f.name
        g = cg.get(n)
        if not g:
            continue
        if H(f.read_bytes()) != g["sha256"]:
            raise ValueError("cash shard identity mismatch " + str(f))
        with np.load(f, allow_pickle=False) as z:
            ix = np.flatnonzero(z["step"] == 0)
            if len(ix) != 1:
                skipped["cash_shard_without_unique_opening"] += 1
                continue
            encoded = z["carbon"][ix[0]]
            cells = np.rint(encoded.astype("float64") * 100 / 255).astype("uint8")
            if cells.size != 225 or not np.array_equal(
                np.rint(cells.astype(float) * 255 / 100).astype("uint8"), encoded
            ):
                raise ValueError("noninvertible opening " + n)
            physical = cells.reshape(15, 15)[:, ::-1].ravel() if g["seat"] else cells
            addw(
                index,
                mi.terrain_sha256(physical.tolist()),
                g["seed"],
                "observed_teacher_shard_opening_q8",
                rel(f),
            )
            files[rel(f)] = H(f.read_bytes())
    # Generator-derived hashes supplement, never replace, observed terrain. Keep the two types distinct.
    for seed, refs in sorted(localrefs.items()):
        if seed in seedset[1]:
            continue
        if seed == SEED:
            hits.extend([src, "recorded_seed"] for src in refs if src not in allowed)
        default_terrain = mi.local_board(seed)
        for src in refs:
            addw(index, default_terrain, seed, "inferred_default_native_generator", src)
    boardhits = [x for x in index if x[0] == terrain]
    expected = [DECL] if frozen_declaration else []
    if sorted(frozen_hits) != expected:
        hits.append(["frozen_git", "unexpected_exact_references", frozen_hits])
    # Collapse repeated evidence paths into a digest-bound source table; preserve a concrete
    # path and source SHA for every distinct terrain class and alternative seed.
    registry = []
    sources = {}
    for (h, s, method), w in sorted(
        index.items(), key=lambda q: (q[0][0], str(q[0][1]), q[0][2])
    ):
        if h in seedset[2]:
            continue
        for name in w:
            if name not in files:
                raise ValueError("missing provenance " + name)
            sources[name] = files[name]
        registry.append(
            {"terrain": h, "seed": s, "method": method, "source": sorted(w)}
        )
    uniq = lambda method: {
        h for h, s, m in index if m == method and h not in seedset[2]
    }
    recorded = {
        q["terrain"] for q in registry if not q["method"].startswith("inferred")
    }
    ret = {
        "verdict": (
            "eligible_in_checked_scope"
            if not hits and not boardhits and not globally_seen and not reserved
            else "blocked"
        ),
        "frozen_git_revision": REV,
        "candidate_seed": SEED,
        "candidate_initial_terrain_u8_sha256": terrain,
        "formal_reserved_half_open": list(RESERVED_RANGE),
        "reserved": reserved,
        "ladder_seed_or_terrain_match": globally_seen,
        "official_opening_sha256": official_terrain,
        "official_configuration": exam.CFG,
        "formal_and_generator_source_sha256": {
            str(q.relative_to(ROOT)): H(q.read_bytes())
            for q in (
                BASE / "map_identity.py",
                BASE / "tree_win_gate.py",
                Path(sys.modules[exam.FastCarbon.__module__].__file__),
                ENGINE / "zerosum_env/envs/carbon/carbon.py",
            )
        },
        "historical_catalog_loaded_sha256": (
            H(catalog.read_bytes()) if catalog else None
        ),
        "schema_version": 2,
        "frozen_ladder": {
            "games": frozen_ladder_count,
            "fingerprint": frozen_ladder_fingerprint,
            "added_local_ledger_games": sum(v["rows"] for v in rules.values())
            - frozen_ladder_count,
            "added_local_ledgers": sorted(
                set(rules) - {str((BASE / q).relative_to(ROOT)) for q in frozen_rules}
            ),
        },
        "ladder": {
            "games": len(seedset[1]),
            "distinct_terrains": len(seedset[2]),
            "ledger_fingerprint": seedset[0],
            "inputs": rules,
        },
        "current_declaration": {
            "path": DECL,
            "sha256": H(declaration_bytes),
            "present_in_revision": frozen_declaration,
            "frozen_exact_references": frozen_hits,
            "local_exact_references": allowed,
        },
        "conflicts": hits,
        "same_terrain_matches": [[h, s, m] for h, s, m in boardhits],
        "local_native_or_training_maps": registry,
        "local_source_sha256": sources,
        "local_sources_digest": objhash(sources),
        "coverage": {
            "roots": [rel(d) for d in all_dirs if d.exists()],
            "unique_file_or_member_payloads": len(content_aliases),
            "file_or_member_paths": sum(map(len, content_aliases.values())),
            "distinct_nonladder_recorded_terrains": len(recorded),
            "distinct_nonladder_inferred_terrains": len(
                uniq("inferred_default_native_generator")
            ),
            "recorded_observation_or_hash_methods": dict(
                Counter(q["method"] for q in registry)
            ),
            "expectations": dict(counts),
            "distinct_nominal_native_game_or_training_seeds": len(localrefs),
            "skips": dict(skipped),
            "exceptions": skipped_files,
        },
        "limitations": [
            "Scope is the frozen repository and listed local directories. No access claim for unregistered remote jobs or undocumented private maps.",
            "Only terrain digests attested under the row-major 225 unsigned-byte encoding (or native auditor output) enter the terrain set. A public initial_map_sha256 encoded from JSON has different bytes and is not compared.",
            "Inferred local native boards assume the recorded FastCarbon generator with default map rules. Historical runs without an opening, certified digest or invertible full-carbon row only support nominal-seed / generator-based checks.",
            "Weight tensors, compressed feature arrays without verified full initial carbon, and duplicate public-derived summaries cannot independently establish historical map identity. Frozen public archives override locally regenerated terrain. Literal seed and standard closed/half-open range checks cover enumerated metadata and scripts; arbitrary calculations or private schedules are not exhaustively recovered.",
        ],
        "sha256_of_this_script": H(Path(__file__).read_bytes()),
    }
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(ret, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, dest)
    print(ret["verdict"], terrain, len(hits), len(boardhits), len(registry), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--revision", default=REV)
    ap.add_argument(
        "--declaration",
        default=DECL,
        help="single immutable repository-relative declaration, tracked at --revision or freshly written in the working tree",
    )
    ap.add_argument("--out", type=Path, default=HERE / "map_preflight.json")
    ap.add_argument(
        "--registry",
        type=Path,
        help="optional previous receipt to include retained local evidence even if an old workspace was removed; current files are also audited",
    )
    opt = ap.parse_args()
    if opt.seed != SEED and (
        opt.out.resolve() == HERE / "map_preflight.json" or opt.declaration == DECL
    ):
        ap.error("for another map provide its own --declaration and --out")
    build(opt.seed, opt.revision, opt.declaration, opt.out.resolve(), opt.registry)
