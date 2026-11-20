"""Re-run the archive and public-score exposure checks before collecting policy-window returns."""
import argparse
import gzip
import io
import importlib.util
import json
from pathlib import Path
import re
import tarfile
import zipfile
import types

HERE = Path(__file__).resolve().parent
STRATEGIC = HERE.parent
TEMPLATE = STRATEGIC.parent / "cargo_tree_action_learning_dev/native_actor_corpus/league_policy/clearance.py"
SCORE_TEMPLATE = STRATEGIC / "elite_direct_policy/development/clearance.py"
SOURCE = "8072d127b1120adf9b90492e121eaebc3f23594e"
SEEDS = list(range(621758540, 621758548))


def save(name, value):
    target = HERE / name
    raw = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if target.exists() and target.read_text() != raw:
        raise ValueError(f"Existing inconsistent {target}")
    target.write_text(raw)


def module(path):
    spec = importlib.util.spec_from_file_location("elite_census", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def capture():
    earlier = module(SCORE_TEMPLATE)
    earlier.SOURCE = SOURCE
    earlier.SEEDS = SEEDS
    files = earlier.git("ls-tree", "-r", "--name-only", SOURCE).decode().splitlines()
    canonical, proofs, faults = earlier.top_ledgers(files)
    roots = earlier.boards()
    original = earlier.frozen(TEMPLATE)
    code = original.decode()
    for old, new in (("len(top5)==1637", f"len(top5)=={len(canonical)}"),
                     ("if b'6217584' in b and CANDIDATE.search(b)", "if b'62175854' in b and CANDIDATE.search(b)")):
        if code.count(old) != 1:
            raise RuntimeError("Changed archive census")
        code = code.replace(old, new)
    archive = types.ModuleType("policy_window_archive_census")
    archive.__file__ = str(TEMPLATE)
    exec(compile(code, str(TEMPLATE), "exec"), archive.__dict__)
    groups = {"fit": SEEDS[:5], "hold": SEEDS[5:]}
    archive.SNAPSHOT, archive.SEEDS, archive.GROUPS = SOURCE, SEEDS, groups
    archive.CANDIDATE = re.compile(rb"(?<![.\d])62175854[0-7](?![.\d])")
    result = archive.scan()
    if result["by_seed"] != roots:
        raise ValueError("Opening verification mismatch")
    result.update(schema="sapling-eight-policy-window-maps/1",
                  purpose="Previously unseen development-derived maps for five fitting and three held policy-window response collections. All eight excluded from independent formal evaluation.",
                  canonical_ledgers=proofs, archive_census_sha256=earlier.sha(original))
    prior = STRATEGIC / "elite_direct_policy/development/score_delta.json"
    result["previous_score_clearance_sha256"] = earlier.sha(earlier.frozen(prior))
    previous = json.loads(earlier.frozen(prior))
    earlier.BASE = previous["through"]
    extra = earlier.score_delta()
    for row in previous["additional_public_replays"]:
        if row["seed"] in SEEDS or row["terrain"] in roots.values():
            raise ValueError("Prior public score map collision")
    result["score_increment"] = extra
    save("clearance.json", result)
    print(f"{len(roots)} layouts cleared against {len(canonical)} top-five openings, {len(faults)} retained fault(s)", flush=True)


def increment(through):
    earlier = module(SCORE_TEMPLATE)
    earlier.SOURCE, earlier.SEEDS = through, SEEDS
    roots = earlier.boards()
    prior = json.loads((HERE / "clearance.json").read_bytes())
    if prior["by_seed"] != roots:
        raise ValueError("Changed candidate layouts")
    candidate = re.compile(rb"(?<![.\d])62175854[0-7](?![.\d])")
    hashes = [h.encode() for h in roots.values()]
    paths = earlier.git("ls-tree", "-r", "--name-only", through).decode().splitlines()
    current, proof, faults = earlier.top_ledgers(paths)
    if any(seed in SEEDS or h in roots.values() for seed, h in current.values()):
        raise ValueError("New canonical map collision")
    earlier.BASE = SOURCE
    score = earlier.score_delta()
    changed = earlier.git("diff", "--name-only", "--diff-filter=ACMRT", SOURCE, through).decode().splitlines()
    def inspect(path, data):
        if candidate.search(path.encode()) or candidate.search(data) or any(h in data for h in hashes):
            raise ValueError("Registered identity appeared in " + path)
        if path.endswith((".tgz", ".tar", ".tar.gz")):
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                for member in archive:
                    if member.isfile():
                        inspect(member.name, archive.extractfile(member).read())
        elif path.endswith((".zip", ".npz")):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.namelist():
                    inspect(member, archive.read(member))
        elif path.endswith(".gz"):
            inspect(path[:-3], gzip.decompress(data))
    digests = {}
    for path in changed:
        raw = earlier.git("show", through + ":" + path)
        inspect(path, raw)
        digests[path] = earlier.sha(raw)
    save("clearance_increment.json", dict(after=SOURCE, through=through, paths=digests,
         physical_boards=roots, top5_gate=len(current), top5_faults=faults,
         canonical_ledgers=proof, score_increment=score, seed_intersections=[], physical_intersections=[]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--increment", help="Current full commit after the complete baseline census")
    args = parser.parse_args()
    if args.increment:
        increment(args.increment)
    else:
        capture()
