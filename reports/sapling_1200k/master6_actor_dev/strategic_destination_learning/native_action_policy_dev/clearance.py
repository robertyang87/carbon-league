"""Check fresh development boards against the pre-registration tree and its latest ladder scores."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import types

HERE = Path(__file__).resolve().parent
STRATEGY = HERE.parent
PREVIOUS = STRATEGY / "midgame_economic_policy/development"
SEEDS = (621758560, 621758561)
BASE = "d17f45ddf983006b713191154947344c48d733ae"
TOKEN = re.compile(rb"(?<![.\da-zA-Z])(?:621758560|621758561)(?![.\da-zA-Z])")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load("registered_economic_census", PREVIOUS / "clearance.py")
git = prior.git
sha = prior.sha


def generate():
    prior.old.SEEDS = list(SEEDS)
    prior.old.SOURCE = BASE
    names = git("ls-tree", "-r", "--name-only", BASE).decode().splitlines()
    top5, ledger_hashes, faults = prior.old.top_ledgers(names)
    source = prior.frozen(prior.TEMPLATE, BASE)
    code = source.decode()
    for key, value in (
        ("len(top5)==1637", f"len(top5)=={len(top5)}"),
        ("if b'6217584' in b and CANDIDATE.search(b)", "if b'62175856' in b and CANDIDATE.search(b)"),
    ):
        if code.count(key) != 1:
            raise ValueError("Unexpected historical archive scanner version")
        code = code.replace(key, value)
    scanner = types.ModuleType("native_actions_archive_scan")
    scanner.__file__ = str(prior.TEMPLATE)
    exec(compile(code, str(prior.TEMPLATE), "exec"), scanner.__dict__)
    scanner.SNAPSHOT = BASE
    scanner.SEEDS = list(SEEDS)
    scanner.GROUPS = {"paired_development_only": list(SEEDS)}
    scanner.CANDIDATE = TOKEN
    census = scanner.scan()
    boards = prior.old.boards()
    if boards != census["by_seed"] or len(top5) != census["top5_unique_openings"]:
        raise ValueError("Conflicting registered physical roots or ladder denominator")
    last = json.loads(prior.frozen(PREVIOUS / "clearance_increment.json", BASE))
    if set(boards.values()) & set(last["physical_boards"].values()):
        raise ValueError("Unplayed but reserved two-map experiment intersects")
    prior.old.BASE = last["through"]
    delta = prior.old.score_delta()
    if delta["unresolved_incremental_replays"]:
        raise ValueError("New score source missing its replay")
    census.update(
        purpose="Two development maps with both seats, never used for training or final exam.",
        top5_canonical_ledgers=ledger_hashes,
        official_score_delta=delta,
        reviewed_ladder_faults=faults,
        verifier_sha256=sha(Path(__file__).read_bytes()),
        archive_scanner_sha256=sha(source),
        preceding_map_registration=last["physical_boards"],
    )
    return census


def save(path, result):
    raw = (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Existing receipt differs: {path}")
    if not path.exists():
        temp = path.with_suffix(".partial")
        temp.write_bytes(raw)
        temp.replace(path)
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--through", help="Extend the fixed census to this full, pre-publication commit")
    end = args.parse_args().through
    if end is None:
        result = generate()
        print(save(HERE / "clearance.json", result), result["by_seed"])
    else:
        baseline = (HERE / "clearance.json").read_bytes()
        receipt = json.loads(baseline)
        if receipt["source_snapshot"] != BASE or sha(Path(__file__).read_bytes()) != receipt["verifier_sha256"]:
            raise ValueError("Census source or code drift")
        prior.SEEDS = list(SEEDS)
        prior.old.SEEDS = list(SEEDS)
        prior.IDS = TOKEN
        state = {"by_seed": receipt["by_seed"], "remote_extension": []}
        prior.extend(state, BASE, end)
        layouts = prior.verify_added_boards(BASE, end, receipt["by_seed"])
        state.update(after=BASE, through=end, through_tree=git("rev-parse", end+"^{tree}").decode().strip(),
                     source_census_sha256=sha(baseline), added_replay_openings=layouts)
        print(save(HERE / "clearance_at_freeze.json", state))
