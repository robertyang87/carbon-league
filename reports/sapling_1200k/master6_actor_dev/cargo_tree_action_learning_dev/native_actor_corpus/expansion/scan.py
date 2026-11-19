"""Inventory a frozen Git tree and replay the commands already played. No policy invocation."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import platform

from lineups import lineups

HERE = Path(__file__).resolve().parent
ROOT = Path(subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True).strip())
REV = "5567aaddaabd3cc86eb7863fdc4c38a640878e48"
RAW = re.compile(r"(?:^|/)raw_(\d+)_(0|1)\.json\.gz$")
ORIGINAL = "data/master3s2_source.py"


def digest(b):
    return hashlib.sha256(b).hexdigest()


def encode(v):
    return json.dumps(v, sort_keys=True, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()


def save(name, data):
    path = HERE / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    if name.endswith(".jsonl.gz"):
        with tmp.open("wb") as f, gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as z:
            for row in data:
                z.write(encode(row) + b"\n")
    else:
        tmp.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    os.replace(tmp, path)


class Snapshot:
    def __init__(self, repo):
        self.repo = repo
        output = subprocess.check_output(["git", "-C", repo, "ls-tree", "-rlz", REV])
        self.tree = {}
        for ln in output.split(b"\0")[:-1]:
            label, path = ln.split(b"\t", 1)
            mode, kind, oid, size = label.split()
            if kind == b"blob":
                self.tree[os.fsdecode(path)] = (oid.decode(), int(size))

    @lru_cache(maxsize=None)
    def local(self, path):
        f = self.repo / path
        data = f.read_bytes()
        oid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if self.tree[path] != (oid, len(data)):
            raise ValueError(f"Working-tree file differs from {REV}: {path}")
        return f

    def bytes(self, path):
        if (self.repo / path).exists():
            return self.local(path).read_bytes()
        return subprocess.check_output(["git", "-C", self.repo, "cat-file", "blob", self.tree[path][0]])

    @lru_cache(maxsize=None)
    def prove(self, path):
        return digest(self.bytes(path))


def count(rows):
    outcomes = Counter(r["expert_outcome"] for r in rows if r["expert_outcome"] is not None)
    return {"records": len(rows), "terrains": len({r["physical_sha256"] for r in rows}),
            "map_expert_seats": len({(r["physical_sha256"], r["expert_seat"]) for r in rows}),
            "opponent_packages": len({r["opponent_pack_sha256"] for r in rows}),
            "opponent_weight_identities": len({r["opponent_weight_identity"] for r in rows}),
            "action_turns": sum(r["transitions"] for r in rows if r["native_action_supervision"]),
            "official_results_for_native": sum(bool(r["native_outcome_supervision"]) for r in rows),
            "official_wdl_from_expert_seat": dict(outcomes),
            "official_wdl_from_opponent_seat": dict(Counter(r["result"] for r in rows if r["expert_outcome"])),
            "search_return_types": dict(sum((Counter(r["clock"]["search"]["return_counts"]) for r in rows), Counter())),
            "resource_tiers": dict(Counter(r["clock"]["resources"]["tier"] for r in rows)),
            "complete_action_labels_by_expert_seat": dict(Counter(r["expert_seat"] for r in rows if r["native_action_supervision"]))}


def admit_source(row):
    native = row["clock"]
    original = (native["configuration"]["natural_t1_058"]
                and native["identity"]["physical_seat"] == row["expert_seat"]
                and row["examiner_md5"] == "57f9a594cce4f2ec9e3573d8d9d26ca1"
                and native["identity"]["level"] == "observed_raw_identity"
                and native["identity"]["receipt_source_sha256"] == native["identity"]["canonical_source_sha256"] == native["identity"]["expected_sha256"]
                and row["has_bound_native_source"] and not row["raw_errors"])
    row["native_outcome_supervision"] = bool(original and row["verified"] and row["full_official_terminal"])
    row["native_action_supervision"] = bool(row["native_outcome_supervision"] and native["search"]["every_action_logged"]
                                            and row["transitions"] > 0 and not any(row["unrecognized_or_unowned_command_ids_by_seat"]))
    row["expert_outcome"] = None
    if row["full_official_terminal"]:
        a, b = row["official_score_by_seat"][row["expert_seat"]], row["official_score_by_seat"][row["candidate_seat"]]
        row["expert_outcome"] = "win" if a > b else "loss" if a < b else "draw"


def assemble(snapshot, replay_cache):
    from clock_audit import clock
    from partition import classify
    from replay import replay
    registry = json.loads((HERE / "exclusions.json").read_bytes())
    if registry["revision"] != REV:
        raise ValueError("terrain exclusion registry cutoff differs")
    raw_paths = sorted(p for p in snapshot.tree if RAW.search(p))
    if len(raw_paths) != 314:
        raise ValueError("frozen raw inventory differs")
    if replay_cache is not None:
        cache = [json.loads(l) for l in Path(replay_cache).read_text().splitlines()]
        results = {r["path"]: r for r in cache}
        if len(cache) != len(raw_paths) or set(results) != set(raw_paths):
            raise ValueError("incomplete or duplicate official replay cache")
    rows = []
    for path in raw_paths:
        data = snapshot.bytes(path)
        raw = json.loads(gzip.decompress(data))
        row = results[path].copy() if replay_cache is not None else {"path": path, **replay(raw, snapshot.repo / path, snapshot.repo)}
        expected = RAW.search(path)
        if (row["path"], row["raw_sha256"], row["seed"], row["candidate_seat"]) != (
                path, digest(data), int(expected[1]), int(expected[2])):
            raise ValueError("raw filename, bytes or physical seat differs from official replay")
        timing = clock(raw, snapshot.repo / path, snapshot.repo)
        refs = list(timing["evidence"].values()) + row["historical_native_receipts"] + row["historical_official_receipts"]
        for ref in refs:
            if snapshot.prove(ref["path"]) != ref["sha256"]:
                raise ValueError("receipt differs from frozen Git tree: " + ref["path"])
        row.update({"git_blob": snapshot.tree[path][0], "bytes": len(data), "clock": timing,
                    "opponent_weight_identity": raw["weight_identity"], "opponent_pack_sha256": raw["pack_sha256"],
                    "expert_command_tape_sha256": row["physical_seat_action_sha256"][row["expert_seat"]],
                    "examiner_md5": raw["examiner_md5"], "native_config_sha256": digest(encode(raw["configuration"])),
                    "raw_errors": raw["errors"],
                    "raw_audit_messages": raw["audit_messages"], "has_bound_native_source": bool(row["historical_native_receipts"])})
        row["geography"] = classify(row, registry)
        admit_source(row)
        rows.append(row)
    return rows, registry


def main():
    params = argparse.ArgumentParser(description=__doc__)
    params.add_argument("--replay-cache", help="JSONL from replay.py for all 314 raw records, verified against Git at assembly")
    args = params.parse_args()
    snapshot = Snapshot(ROOT)
    rows, registry = assemble(snapshot, args.replay_cache)
    opponents = lineups(snapshot, rows)
    settle(rows, registry, snapshot, opponents)



def settle(rows, registry, snapshot, opponents):
    from replay import ENGINE, ENGINE_INPUTS
    import numpy as np
    by_board = defaultdict(list)
    for row in rows:
        by_board[row["physical_sha256"]].append(row)
    by_trajectory = defaultdict(list)
    for r in rows:
        if r["native_action_supervision"] or r["native_outcome_supervision"]:
            by_trajectory[r["trajectory_sha256"]].append(r)
    for group in by_trajectory.values():
        # Physical copies can differ in filenames or clocks. Prefer a registered recording, then sorted path.
        primary = min(group, key=lambda r: (not r["geography"]["previously_enrolled_sha"], r["path"]))
        for r in group:
            r["duplicate_trajectory_of"] = primary["path"] if r is not primary else None
    pools = defaultdict(list)
    cards = []
    for board, runs in sorted(by_board.items(), key=lambda kv: (min(r["seed"] for r in kv[1]), kv[0])):
        splits = {r["geography"]["split"] for r in runs}
        if len(splits) != 1:
            raise ValueError(("mixed partition on one physical terrain", board, splits))
        split = splits.pop()
        hard = sorted({reason for r in runs for reason in r["geography"]["hard_protection"]})
        strict_barriers = sorted({reason for r in runs for reason in r["geography"]["strict_history_exclusions"]})
        if any(r["geography"]["identity_conflict"] for r in runs):
            hard.append("physical_identity_conflict")
        if hard:
            map_pool = "protected"
        elif split == "train":
            map_pool = "current_13"
        elif strict_barriers:
            map_pool = "retirement_outside_29" if split == "outside" else "retirement_quarantine_7"
        else:
            map_pool = "strict_new"
        roles = defaultdict(set)
        for r in runs:
            for role, evidence in r["geography"]["evidence_by_prior_role"].items():
                roles[role].update(evidence)
            r.setdefault("duplicate_trajectory_of", None)
            if r["duplicate_trajectory_of"]:
                role = "duplicate_trajectory"
            elif not r["native_action_supervision"] or not r["native_outcome_supervision"]:
                role = "unverified_or_failed"
            elif map_pool == "current_13":
                role = "current_13_enrolled" if r["geography"]["previously_enrolled_sha"] else "current_13_extra_trajectory"
            else:
                role = map_pool
            r["sampling_pool"] = role
            r["use_on_new_terrain_without_retirement"] = role == "strict_new"
            r["use_after_versioned_retirement_on_new_terrain_vs_13"] = role in (
                "strict_new", "retirement_quarantine_7", "retirement_outside_29")
            r["additive_trajectory_same_current_training_terrain"] = role == "current_13_extra_trajectory"
            r["additive_without_retiring_any_prior_holdout"] = (
                role == "strict_new" or (role == "current_13_extra_trajectory" and not strict_barriers))
            pools[role].append(r)
        cards.append({"seeds": sorted({r["seed"] for r in runs}), "physical_sha256": board, "current_split": split,
                      "pool": map_pool, "prior_roles": sorted({role for r in runs for role in r["geography"]["prior_roles"]}),
                      "historical_role_evidence": {k: sorted(v) for k, v in roles.items()},
                      "match_directories": sorted({str(Path(r["path"]).parent) for r in runs}),
                      "historical_opponents": sorted({r["opponent_pack_sha256"] for r in runs}),
                      "hard_protection": hard, "strict_history_exclusions": strict_barriers,
                      "count": count(runs), "record_pools": dict(Counter(r["sampling_pool"] for r in runs)),
                      "tapes_by_physical_expert_seat": {s: [r["raw_sha256"] for r in runs if r["expert_seat"] == s]
                                                      for s in (0, 1)}})
    retired = pools["retirement_outside_29"] + pools["retirement_quarantine_7"]
    new_geography = pools["strict_new"] + retired
    valid = [r for r in rows if r["native_action_supervision"] and r["native_outcome_supervision"]]
    strict_same = [r for r in pools["current_13_extra_trajectory"] if r["additive_without_retiring_any_prior_holdout"]]
    require_retire_same = [r for r in pools["current_13_extra_trajectory"] if not r["additive_without_retiring_any_prior_holdout"]]
    newly_available_files = [r for r in (new_geography + pools["current_13_extra_trajectory"]) if not r["geography"]["previously_enrolled_sha"]]
    original = snapshot.bytes(ORIGINAL)
    summary = {"frozen_commit": REV, "selection": RAW.pattern, "exclusions_sha256": digest((HERE / "exclusions.json").read_bytes()),
               "original": {"path": ORIGINAL, "md5": hashlib.md5(original).hexdigest(), "sha256": digest(original)},
               "engine": {str(ENGINE / f): snapshot.prove(str(ENGINE / f)) for f in ENGINE_INPUTS},
               "clock_reference": {p: snapshot.prove(p) for p in ("reports/agent-guide.md", "reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/ladder_live/protocol.json")
                                   if p in snapshot.tree},
               "software": {"python": platform.python_version(), "numpy": np.__version__, "interpreter": sys.implementation.name,
                            "source_sha256": {p.name: digest(p.read_bytes()) for p in sorted(HERE.glob("*.py"))}},
               "pool_accounting": "Subsets are fixed by physical map and native identity. Complete games with slow, zero-depth or single-candidate returns are retained. Resource tiers describe provenance, not official container equivalence.",
               "sampling_unit": "Sample terrain first, then expert physical seat, then played opponent pack and whole trajectory. Replay observation history from t0, using only the acting seat view and the actual current-turn expert-unit prefix. Choose turns within each trajectory independently of outcomes and depth. The joint tape, opposite commands and terminal cash are for reconstruction or the separate terminal outcome, not features.",
               "previous_raw_tapes": len(registry["previous_tapes"]),
               "complete_duplicate_paths": sum(max(0, len(group)-1) for group in by_trajectory.values()),
               "all_314": count(rows), "full_native_action_and_outcome_evidence": count(valid),
               "retire_only_outside_current_13": count(retired),
               "same_current_geography_incremental_without_any_prior_holdout_retirement": count(strict_same),
               "same_current_geography_incremental_requiring_prior_holdout_retirement": count(require_retire_same),
               "additional_trajectories_without_retiring_any_prior_holdout": count(strict_same + pools["strict_new"]),
               "retired_plus_strict_additional_files_vs_original_58": count(newly_available_files),
               "new_terrain_union_after_retirement_outside_13": count(new_geography),
               "new_terrain_union_after_retirement_outside_original_29": count(pools["strict_new"] + pools["retirement_outside_29"]),
               "new_trajectories_vs_old_training_after_retirement": count(new_geography + pools["current_13_extra_trajectory"]),
               "all_cumulative_if_retired": count(new_geography + pools["current_13_extra_trajectory"] + pools["current_13_enrolled"]),
               "pool_counts": {p: count(g) for p, g in sorted(pools.items())},
               "horizons": dict(Counter(r["transitions"] for r in rows)),
               "total_official_transitions": sum(r["transitions"] for r in rows),
               "eligible_dual_seat_cash_observations": sum(r["official_cash_observations"] for r in valid),
               "eligible_archived_cash_snapshot_ticks_checked": sum(len(r["historical_cash_snapshot_ticks_checked"]) for r in valid),
               "source_site_game_horizon": dict(Counter(r["clock"]["search"]["official_transitions_reported"] for r in rows)),
               "step_zero_failed_archive_unexecuted_joint_actions": sum(r["clock"]["search"]["action_pairs"]-r["transitions"] for r in rows if not r["full_official_terminal"]),
               "game_level_data_clock": {**{key: dict(Counter(r["clock"]["search"]["tier"] for r in group))
                                                 for key, group in [("all", rows), ("official",valid)]},
                                        "native_searched_calls_above_058_seconds": sum((r["clock"]["search"].get("searched_native_seconds") or {}).get("over_native_budget",0) for r in rows),
                                        "per_call_public_wall_above_058_seconds": sum((r["clock"]["search"].get("all_public_wall_seconds") or {}).get("over_native_budget",0) for r in rows),
                                        "per_call_public_wall_above_1s": sum((r["clock"]["search"].get("all_public_wall_seconds") or {}).get("over_one_second",0) for r in rows)},
               "preserved_test_or_val_map_hashes": [c["physical_sha256"] for c in cards if c["current_split"] in ("val","test")],
               "registered_current_game_ids": registry["active_full_game_seed_ids"],
               "formal_seed_half_open": registry["formal_seed_half_open"],
               "eligible_full_games_with_independent_official_archive": sum(bool(r["historical_official_receipts"]) for r in valid),
               "counterfactual_suffixes_counted_as_raw": 0,
               "strictly_platform_resource_proven_full_games": sum(r["clock"]["resources"]["official_same_host_quota"] is True for r in valid),
               "resource_issues_by_code": dict(sum((Counter(r["clock"]["issues"]) for r in rows), Counter())),
               "search_clock_tiers": dict(Counter(r["clock"]["search"]["tier"] for r in valid)),
               "original_reference_sha_receipt": dict(Counter(str(r["clock"]["identity"]["receipt_source_sha256"]) for r in rows)),
               "natural_search_config_sha256_by_raw": dict(Counter(r["native_config_sha256"] for r in rows)),
               "agent_or_audit_errors": [r["path"] for r in rows if r["raw_errors"] or r["raw_audit_messages"]],
               "full_games_with_a_zero_depth_return": sum(bool(r["clock"]["search"]["return_counts"].get("depth_zero_return_base")) for r in valid),
               "full_games_with_a_single_candidate_return": sum(bool(r["clock"]["search"]["return_counts"].get("single")) for r in valid),
               "source_opening_evidence": dict(Counter(r["opening_evidence"] for r in rows)),
               "complete_positive_depth_histogram_games": sum(bool(r["clock"]["search"]["searched_completed_depth_histogram"]) for r in valid)}
    save("lineups.json", opponents)
    save("terrains.jsonl.gz", cards)
    save("inventory.jsonl.gz", rows)
    for name in ("lineups.json", "terrains.jsonl.gz", "inventory.jsonl.gz"):
        summary.setdefault("output_sha256", {})[name] = digest((HERE / name).read_bytes())
    save("summary.json", summary)
    print(json.dumps({name: (v["terrains"], v["records"], v["action_turns"]) for name,v in summary["pool_counts"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
