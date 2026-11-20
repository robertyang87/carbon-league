"""Reproduce the four development openings and exclude previously exposed maps."""
import importlib.util
import json
import re
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path / ".git").exists())
STRATEGY = ROOT / "reports/sapling_1200k/master6_actor_dev/strategic_destination_learning"
BASE = "fb0140af2474a7049e86d05e3a1ff56154729fd2"
SEEDS = list(range(621758580, 621758584))


def generate():
    previous_path = STRATEGY / "native_action_policy_dev/clearance.py"
    spec = importlib.util.spec_from_file_location("preceding_development_clearance", previous_path)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    prior = previous.prior
    prior.old.SOURCE = BASE
    prior.old.SEEDS = SEEDS
    paths = prior.git("ls-tree", "-r", "--name-only", BASE).decode().splitlines()
    top5, receipts, faults = prior.old.top_ledgers(paths)

    # The archive crawler's historical denominator and cheap seed prefix were
    # specialized to the first panel. Bind them to this panel before scanning.
    source = prior.frozen(prior.TEMPLATE, BASE)
    code = source.decode()
    for before, after in [
        ("len(top5)==1637", f"len(top5)=={len(top5)}"),
        ("if b'6217584' in b and CANDIDATE.search(b)",
         "if b'62175858' in b and CANDIDATE.search(b)"),
    ]:
        if code.count(before) != 1:
            raise ValueError("Frozen archive crawler changed")
        code = code.replace(before, after)
    registration = "reports/master3s2_derivative/root_policy/register.py"
    reservation = b'    need(not {g["seed"] for g in selected}&set(range(621758580,621758584)))'
    registration_hash = "2b32f59cc339e523f88fb7d8561a633ad7fa90ed2423cc0e63375c30410064c2"
    registration_bytes = prior.git("show", BASE + ":" + registration)
    if prior.sha(registration_bytes) != registration_hash or registration_bytes.count(reservation) != 1:
        raise ValueError("The previous reservation no longer only excludes these maps from fitting")
    hook = "data=git('show',blob)"
    if code.count(hook) != 1:
        raise ValueError("Unexpected plain-text registry scanner")
    code = code.replace(hook, hook + "\n        if blob == SNAPSHOT + ':' + REGISTRY: data=data.replace(RESERVATION, b'')")
    scanner = types.ModuleType("development_archive_scan")
    scanner.__file__ = str(prior.TEMPLATE)
    exec(compile(code, str(prior.TEMPLATE), "exec"), scanner.__dict__)
    scanner.REGISTRY, scanner.RESERVATION = registration, reservation
    scanner.SNAPSHOT = BASE
    scanner.SEEDS = SEEDS
    scanner.GROUPS = {"paired_development_only": SEEDS}
    scanner.CANDIDATE = re.compile(rb"(?<![.\da-zA-Z])62175858[0-3](?![.\da-zA-Z])")
    census = scanner.scan()
    layouts = prior.old.boards()
    if layouts != census["by_seed"] or len(top5) != census["top5_unique_openings"]:
        raise ValueError("Conflicting opening fingerprints or denominator")

    fit_contract = json.loads(prior.git("show", BASE + ":reports/master3s2_derivative/root_policy/protocol.json"))
    expert_games = fit_contract["games"]
    if len(expert_games) != 80 or any(row["seed"] in SEEDS or row["terrain_sha256"] in layouts.values()
                                     for row in expert_games):
        raise ValueError("Derivative expert training or holdout panel overlaps")
    # Include public score-only updates, whose raw opening can live outside Git.
    previous_receipt = prior.frozen(previous_path.with_name("clearance.json"), BASE)
    preceding = json.loads(previous_receipt)
    known = {(int(seed), layout) for seed, layout in preceding["by_seed"].items()}
    known.update((row["seed"], row["terrain"]) for row in
                 preceding["official_score_delta"]["additional_public_replays"])
    prior.old.BASE = preceding["official_score_delta"]["through"]
    delta = prior.old.score_delta()
    for row in delta["additional_public_replays"]:
        known.add((row["seed"], row["terrain"]))
    if delta["unresolved_incremental_replays"] or any(
        seed in SEEDS or terrain in layouts.values() for seed, terrain in known
    ):
        raise ValueError("Missing or previously used opening")

    census.update(schema="master3s2-derivative/paired-development-clearance/1",
                  purpose="Four two-seat development maps, excluded from training and final evaluation.",
                  prior_development_reservation_registration_sha256=registration_hash,
                  derivative_expert_training_and_holdout_maps_checked=len(expert_games),
                  official_score_delta=delta, top5_ledger_sha256=receipts,
                  reviewed_ladder_faults=faults, previous_census_sha256=prior.sha(previous_receipt),
                  archive_scanner_sha256=prior.sha(source), verifier_sha256=prior.sha(Path(__file__).read_bytes()))
    return census


if __name__ == "__main__":
    result = generate()
    out = HERE / "clearance.json"
    raw = (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if out.exists() and out.read_bytes() != raw:
        raise ValueError(f"Existing receipt differs: {out}")
    if not out.exists():
        temporary = out.with_suffix(".partial")
        temporary.write_bytes(raw)
        temporary.replace(out)
    print(result["by_seed"], flush=True)
