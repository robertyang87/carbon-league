"""Build the frozen terrain registry and classify original-agent candidates without opening games."""
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent
CARGO = CORPUS.parent
ACTOR = CARGO.parent
SAPLING = ACTOR.parent
ROOT = SAPLING.parents[1]
REVISION = "5567aaddaabd3cc86eb7863fdc4c38a640878e48"
RAW = re.compile(r"raw_(\d+)_([01])\.json\.gz$")
TERRAIN = re.compile(r"[0-9a-f]{64}$")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def name(path):
    return str(path.relative_to(ROOT))


def build():
    identities, panels, raw_members = {}, {}, {}
    provenance = {}
    records = defaultdict(lambda: defaultdict(set))
    labels = defaultdict(set)
    tree = [ROOT / p for p in git("ls-tree", "-r", "--name-only", REVISION).decode().splitlines()]

    def read(path, json_data=True):
        raw = git("show", f"{REVISION}:{name(path)}")
        provenance[name(path)] = hashlib.sha256(raw).hexdigest()
        if path.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        return json.loads(raw) if json_data else raw

    def put(seed, role, path, physical=None):
        seed = int(seed)
        if physical is not None:
            if not TERRAIN.fullmatch(physical):
                raise ValueError(f"Bad physical map at {path}")
            identities.setdefault(seed, set()).add(physical)
        labels[seed].add(role)
        records[seed][role].add(name(path))

    source = CORPUS / "sources.json"
    audit = read(source)
    panel_path = CARGO / "panel.json"
    panel = read(panel_path)
    if hashlib.sha256(git("show", f"{REVISION}:{name(panel_path)}")).hexdigest() != audit["source_sha256"][name(panel_path)]:
        raise ValueError("Reference panel digest changed")
    for c in audit["cards"]:
        seed, board = c["seed"], c["terrain"]
        if panel["split"][board] != c["prior_split"]:
            raise ValueError("Parent and frozen audit disagree")
        panels[seed] = {"terrain": board, "parent_split": c["prior_split"], "corpus_split": c["split"]}
        put(seed, "parent_" + c["prior_split"], panel_path, board)
        put(seed, "corpus_" + c["split"], source, board)
        raw_members[c["sha256"]] = {"seed": seed, "expert_seat": c["expert_seat"], "path": c["path"], "split": c["split"]}
    if Counter(c["parent_split"] for c in panels.values()) != {"train": 19, "val": 4, "test": 6} or Counter(c["corpus_split"] for c in panels.values()) != {"train": 13, "val": 3, "test": 6, "excluded": 7}:
        raise ValueError("Reference splits changed")

    for seed, physical in audit["registered_and_external_physical_terrains"].items():
        put(seed, "overlap_lookup_only", source, physical)
    carbon = CARGO / "learned_carbon_observation/protocol.json"
    for c in read(carbon)["cards"]:
        put(c["seed"], "carbon_" + c["split"], carbon, c["terrain"])
    release = CORPUS / "original_response/release.json"
    for c in read(release)["cases"]:
        put(c["seed"], "original_response_qualification", release, c["terrain"])
    fresh = CORPUS / "fresh_original/protocol.json"
    fresh_register = read(fresh)
    for seed, physical in fresh_register["physical_boards"].items():
        put(seed, "current_registered_game", fresh, physical)

    local = SAPLING / "known_local_maps/exposure.json"
    history = read(local)
    for row in history["maps"]:
        put(row["seed"], "older_local_exposure", local, row["initial_terrain_u8_sha256"])
    for src in history["sources"]:
        path = ROOT / src["path"]
        for seed in src["seeds"]:
            if path.name in ("sapling_exam_results.tgz", "sapling_exam_r2_results.tgz", "sapling_exam_v2_200_results.tgz"):
                put(seed, "historical_completed_exam", path)
            elif any(part.startswith("tree_exam") for part in path.parts):
                if seed not in (620106811, 620106812, 620098101):
                    raise ValueError("The early native development set changed")
                put(seed, "historical_development", path)
            elif ("terminal_return_fit/train_" in src["path"] or
                  "tree_response_training/" in src["path"] and "/source/" in src["path"]):
                put(seed, "historical_training", path)
            elif path.name == "results.json" and ("dev" in src["path"] or "original_" in src["path"]):
                put(seed, "historical_development", path)

    fixed_exam = ROOT / "reports/REGISTRATION_MASTER3S2_EXAMINER_20261116.md"
    if "20291101–20292100" not in read(fixed_exam, False).decode():
        raise ValueError("Original fixed examiner set differs")
    fixed_seeds = set(range(20291101, 20292101))
    if {seed for seed, roles in labels.items() if "historical_completed_exam" in roles} != fixed_seeds:
        raise ValueError("Legacy fixed examiner blocks differ from the three frozen source archives")
    for seed in fixed_seeds:
        put(seed, "historical_completed_exam", fixed_exam)
    for seed, file in [(620106811, "TREE_90_EXAM.md"), (620106812, "TREE_90_EXAM.md"), (620098101, "IT50_SAPLING_HYBRID.md")]:
        doc = SAPLING / file
        text = read(doc, False).decode()
        if str(seed) not in text or not any(label in text for label in ("开发场", "研究图")):
            raise ValueError("Historical diagnostic registration changed")
        put(seed, "historical_development", doc)

    # The official 100-map draw contract includes a plan nonce; discover registered published plans
    # by their data contract, without generating a draw or running an examination.
    evidence = SAPLING / "known_local_maps/README.md"
    if "正式考卷未抽取" not in read(evidence, False).decode():
        raise ValueError("Registered draw status changed")
    found = subprocess.run(["git", "-C", str(ROOT), "grep", "-l", "-F", '"nonce"', REVISION, "--", "*.json"],
                           capture_output=True, check=False)
    if found.returncode not in (0, 1):
        raise RuntimeError("Frozen JSON plan search failed")
    draw_plans = []
    for line in found.stdout.decode().splitlines():
        path = ROOT / line.split(":", 1)[1]
        doc = read(path)
        if isinstance(doc, dict) and str(doc.get("schema", "")).startswith(("sapling-one-shot-90/", "sapling-90-and-income/")):
            if len(doc["maps"]) != 100:
                raise ValueError("Incomplete published formal map registry")
            for row in doc["maps"]:
                put(row["seed"], "formal_published_draw", path, row["initial_terrain_u8_sha256"])
            draw_plans.append(name(path))

    distill = ACTOR / "master3s2_policy_distill_dev/manifest.json"
    for c in read(distill)["records"]:
        put(c["seed"], "distill_" + c["split"], distill, c["terrain"])
    bridge = ACTOR / "own_search_bridge_dev/fit.json"
    for c in read(bridge)["trajectories"]:
        put(c["seed"], "own_search_" + c["split"], bridge, c["terrain"])
    for file, stem in [("joint_response_critic_dev/source_registry.json", "joint_critic_"),
                       ("late_harvest_corridor/source_registry.json", "late_harvest_")]:
        path = ACTOR / file
        for c in read(path)["games"]:
            put(c["seed"], stem + c["split"], path, c.get("terrain", c.get("terrain_sha256")))
    cargo_defense = ACTOR / "loaded_cargo_defense_dev/protocol.json"
    for split, triples in read(cargo_defense)["sources"].items():
        for seed, _, physical in triples:
            put(seed, "loaded_cargo_" + split, cargo_defense, physical)

    h24 = CARGO / "territory_sequence/sources.json"
    for c in read(h24):
        put(c["seed"], "h24_" + c["split"], h24, c["terrain"])
    terminal = ACTOR / "terminal_return_fit"
    p = terminal / "protocol.json"
    rule = read(p)
    put(rule["training_seed"], "historical_training", p)
    put(rule["blind_development_seed"], "historical_development", p)
    p = terminal / "multimap/protocol.json"
    rule = read(p)
    terrain = read(terminal / "multimap/clearance.json")["boards"]
    for role, key in [("historical_training", "train_maps"), ("historical_development", "development_maps")]:
        for seed in rule[key]:
            put(seed, role, p, terrain[str(seed)])
    prior_check = ACTOR / "own_search_bridge_dev/protocol.json"
    rule = read(prior_check)
    if "Last two" not in rule["rollout_holdout"]:
        raise ValueError("Original behavior holdout contract differs")
    for folder in rule["training_tapes"][-2:]:
        if "multimap/train/" not in folder:
            raise ValueError("Ambiguous cross-project holdout")
        put(int(Path(folder).parent.name), "prior_behavior_holdout", prior_check)
    late = SAPLING / "late_response_training/protocol.json"
    for c in read(late)["maps"]:
        put(c["seed"], "historical_development", late)
        put(c["seed"], "historical_training" if c["split"] == "train" else "prior_behavior_holdout", late)

    tree_response = ACTOR / "tree_response_training"
    for p in (tree_response / "next_intake.json", tree_response / "supplement_01/registration.json"):
        rule = read(p)
        for seed in rule["protected_unopened_development_maps"]:
            put(seed, "protected_development", p)
        for batch in rule["batches"]:
            for seed, board in batch["terrain_at_queue_time"].items():
                put(seed, "historical_training", p, board)

    old_controls = CARGO / "learned_carbon_observation/source_clearance.json"
    for seed in read(old_controls)["excluded_first_win_controls_and_343_355"]:
        put(seed, "historical_development", old_controls)
    for p in (ACTOR / "stratified_action_value_dev/source/pilot_plan.json",
              ACTOR / "blended_actor_arena/protocol.json", ACTOR / "blended_policy_return_dev/map_declaration.json"):
        rule = read(p)
        values = rule.get("maps") or rule.get("new_proxy_development") or rule["new_dev"]["seeds"]
        for seed in values:
            put(seed, "protected_development", p)

    preflight = ACTOR / "blended_policy_return_dev/map_clearance.json"
    for seed, rec in read(preflight)["development"].items():
        put(seed, "protected_development", preflight, rec["terrain"])
    for i in (0, 1):
        archive = ACTOR / f"stratified_action_value_dev/source/pilot_preflight_{i}.zip"
        with zipfile.ZipFile(io.BytesIO(read(archive, False))) as bundle:
            identity = json.loads(bundle.read("identity.json"))
        if identity["official_opening_sha256"] != identity["candidate_initial_terrain_u8_sha256"]:
            raise ValueError("Pilot preflight physical receipt differs")
        put(identity["candidate_seed"], "protected_development", archive, identity["official_opening_sha256"])

    for p in tree:
        if not p.is_relative_to(ACTOR) or p.name != "clearance.json" or p.is_relative_to(HERE):
            continue
        rule = read(p)
        registered = {k: v for field in ("boards", "physical_boards") if isinstance(rule.get(field), dict) for k, v in rule[field].items()}
        for seed, board in registered.items():
            if not str(seed).isdecimal() or not isinstance(board, str) or not TERRAIN.fullmatch(board):
                continue
            put(seed, "physical_registration", p, board)
            if 621758320 <= int(seed) <= 621758355:
                put(seed, "historical_development", p)
    # These reserved/hosted maps were assigned without a local raw_<seed>_<seat> archive.
    special = {
        621758329: ACTOR / "tree_screen_response_dev/protocol.json",
        621758331: ACTOR / "tree_screen_payoff_dev/protocol.json",
        621758332: ACTOR / "harvest_release_dev/protocol.json",
        621758336: ACTOR / "elite_role_mix_dev/collector_only_dev/protocol.json",
        621758350: CARGO / "territory_sequence/windowed/fresh_original/clearance.json",
        621758351: CARGO / "territory_sequence/windowed/fresh_original/clearance.json",
        621758333: ACTOR / "harvest_tax_exit_dev/protocol.md",
        621758334: ACTOR / "elite_role_mix_dev/conditional/fresh_paired_dev/protocol.json",
        621758337: ACTOR / "midgame_plant_site_v2_dev/site_protocol.json",
    }
    for seed, p in special.items():
        if str(seed).encode() not in read(p, False):
            raise ValueError(f"Missing development declaration for {seed}")
        put(seed, "historical_development", p)
    for p in tree:
        m = RAW.fullmatch(p.name)
        if p.is_relative_to(ACTOR) and m and 621758300 <= int(m[1]) < 621758400:
            put(int(m[1]), "archived_development_game", p)
    for part in ("fresh_original", "windowed/fresh_original"):
        p = CARGO / "territory_sequence" / part / "clearance.json"
        for seed, board in read(p)["physical_boards"].items():
            put(seed, "completed_hosted_development", p, board)
    for seed in range(621758320, 621758356):
        if "historical_development" not in labels[seed]:
            raise ValueError(f"Missing explicit development registration for {seed}")

    # The public ledgers provide both actual terrain IDs and nominal seed membership. Never draw a map here.
    public = {}
    for p in tree:
        if p.name == "validation_maps.jsonl.gz" and p.is_relative_to(SAPLING / "all_top5"):
            for line in read(p, False).splitlines():
                d = json.loads(line)
                if d["id"] in public:
                    raise ValueError("Duplicate public ladder game")
                public[d["id"]] = d["initial_terrain_u8_sha256"]
                put(d["seed"], "published_ladder", p, d["initial_terrain_u8_sha256"])
    if len(public) != 1631 or len(set(public.values())) != len(public):
        raise ValueError("Frozen ladder roster differs from 1,630 scored and one exceptional map")
    by_seed = {str(s): {"terrain_hashes": sorted(identities.get(s, set())),
                        "roles": {k: sorted(v) for k, v in sorted(groups.items())}}
               for s, groups in sorted(records.items())}
    by_physical = defaultdict(list)
    for seed, row in by_seed.items():
        for terrain in row["terrain_hashes"]:
            by_physical[terrain].append(int(seed))
    return {"revision": REVISION, "schema": 1, "physical_encoding": "SHA-256 of exactly 225 row-major unsigned carbon bytes",
            "parent_split_maps": {"train": 19, "val": 4, "test": 6}, "corpus_split_maps": {"train": 13, "val": 3, "test": 6, "excluded": 7},
            "formal_seed_half_open": audit["formal_seed_half_open"], "lookup_only_interval": audit["external_development_seed_half_open"],
            "active_full_game_seed_ids": fresh_register["seeds"], "public_ladder_maps": len(public),
            "registered_formal_history": {"fixed_benchmark_terrains":len(fixed_seeds), "seed_interval_inclusive":[20291101,20292100],
                                          "published_one_shot_100_map_plans":draw_plans, "reference":name(evidence),
                                          "draw_contract":"Frozen tracked JSON queried for the nonce required by tree_win_gate. No new seeds or unseen physical hashes are generated."},
            "cross_study_exposure_on_parent_panel": {str(seed): {role: sorted(paths) for role, paths in records[seed].items()
                  if role.startswith(("distill_","own_search_","carbon_","h24_","late_harvest_","loaded_cargo_","joint_critic_"))}
                  for seed in sorted(panels)},
            "previous_tapes": raw_members, "previous_paths": {c["path"]: h for h, c in raw_members.items()},
            "by_seed": by_seed, "by_physical_sha256": dict(sorted(by_physical.items())),
            "source_sha256": dict(sorted(provenance.items())),
            "scope": "Fixed registrations and physical map evidence only. Candidate tape and receipt verification is a separate scanner step. Strict additions are new to this corpus and preserve active and earlier evaluation splits. A new examiner trajectory on an admitted training terrain supplies labels but no new geography. Previously held, development or completed exam data can be proposed only as a separate retrospective cohort after explicit retirement. Current games and formal reservations stay isolated."}


def classify(candidate, registry):
    """Return terrain/role eligibility; the caller independently certifies the original agent and directions."""
    seed, board = candidate["seed"], candidate["physical_sha256"]
    seat, owner = candidate["expert_seat"], candidate["candidate_seat"]
    path = candidate["path"]
    if type(seed) is not int or not TERRAIN.fullmatch(board) or seat not in (0, 1) or owner != 1-seat:
        raise ValueError("Expected an audited game seed, canonical physical map and complementary seats")
    by_seed = registry["by_seed"]
    physical_seeds = registry["by_physical_sha256"].get(board, [])
    matched = sorted({seed, *physical_seeds})
    nominal = by_seed.get(str(seed), {})
    mismatch = bool(nominal.get("terrain_hashes")) and board not in nominal["terrain_hashes"]
    roles = set()
    refs = defaultdict(set)
    for s in matched:
        for role, paths in by_seed.get(str(s), {}).get("roles", {}).items():
            roles.add(role)
            refs[role].update(paths)
    parent_split = next((s for s in ("test", "val", "train") if "parent_" + s in roles), None)
    split = next((s for s in ("test", "val", "excluded", "train") if "corpus_" + s in roles), None)
    lo, hi = registry["formal_seed_half_open"]
    if any(lo <= s < hi for s in matched):
        roles.add("formal_reserved_interval")
    # Identify exact physical source bytes rather than treating an opposite seat as a second expert.
    raw_hash = candidate.get("raw_sha256") or candidate.get("tape_sha256") or candidate.get("sha256")
    known_path_hash = registry["previous_paths"].get(path)
    if known_path_hash and raw_hash and known_path_hash != raw_hash:
        raise ValueError("Physical path reused with a different tape digest")
    enrolled = registry["previous_tapes"].get(raw_hash or known_path_hash)
    if enrolled and (seat != enrolled["expert_seat"] or enrolled["seed"] != seed or
                     board not in by_seed[str(seed)]["terrain_hashes"]):
        raise ValueError("Enrolled tape and proposed seed/seat/terrain disagree")
    prior_tape = enrolled is not None
    hard = roles & {"formal_reserved_interval", "formal_published_draw", "historical_completed_exam", "current_registered_game",
                    "corpus_val", "corpus_test"}
    strict_barriers = roles & {"corpus_excluded", "parent_val", "parent_test", "carbon_val", "h24_val", "h24_test",
                               "prior_behavior_holdout", "original_response_qualification", "protected_development",
                               "historical_development", "archived_development_game", "completed_hosted_development",
                               "published_ladder", "distill_held", "own_search_rollout_hold", "late_harvest_eval", "loaded_cargo_held", "loaded_cargo_reserve", "joint_critic_held", "joint_critic_holdout", "joint_critic_test"}
    if split == "train":
        # This is already-admitted geography, including any explicitly disclosed cross-generation roles.
        # If a separately protected test or qualification also used this terrain, keep that role protected.
        strict_barriers.difference_update({"historical_development"})
    if not split and not strict_barriers and not hard and "historical_training" not in roles:
        if "older_local_exposure" in roles or "physical_registration" in roles:
            strict_barriers.add("historical_purpose_unresolved")
        # A map newly found in an archived held/evaluation directory still has an evidence-backed role.
        directory = path.split("/")[:-1]
        if any(s in ("holdout", "held", "heldout", "test") for s in directory):
            strict_barriers.add("archived_holdout_directory")
        elif any(s in ("native_dev", "original_dev", "fresh_original", "replication") or s.endswith("_dev")
                 for s in directory if s not in ("master6_actor_dev", "cargo_tree_action_learning_dev")):
            strict_barriers.add("archived_development_directory")
    strict_terrain = not mismatch and not hard and not strict_barriers
    strict_label = strict_terrain and not prior_tape
    optional_terrain = not mismatch and not hard
    optional_label = optional_terrain and not prior_tape
    optional_vs_train26 = optional_terrain and not (enrolled and enrolled["split"] == "train")
    prior_training = bool(roles & {"historical_training", "carbon_train", "h24_train", "distill_fit", "own_search_fit", "joint_critic_train", "loaded_cargo_train"})
    if mismatch:
        layer = "mismatched_nominal_and_physical_identity"
    elif hard:
        layer = "preserved_evaluation_or_formal"
    elif strict_barriers:
        layer = "retired_history_only"
    elif prior_tape:
        layer = "previously_enrolled_tape"
    elif split == "train":
        layer = "distinct_trajectory_on_existing_corpus_train_terrain"
    elif prior_training:
        layer = "previous_training_terrain_new_to_this_corpus"
    else:
        layer = "additional_terrain_without_registered_evaluation"
    return {"seed": seed, "physical_sha256": board, "expert_seat": seat, "candidate_seat": owner,
            "parent_split": parent_split, "split": split or "outside", "layer": layer,
            "role": "original_examiner", "training_role_source": "scanner verifies original package identity and physical role",
            "prior_roles": sorted(roles), "physical_alias_seed_ids": physical_seeds,
            "identity_conflict": mismatch, "previously_enrolled_sha": prior_tape,
            "in_original_29": parent_split is not None, "in_current_training_13": split == "train",
            "previously_trained_in_another_study": prior_training,
            "hard_protection": sorted(hard), "strict_history_exclusions": sorted(strict_barriers),
            "strict_additional_tape_if_verified": strict_label,
            "strict_new_training_map_for_this_corpus_if_verified": strict_label and split != "train",
            "same_training_terrain_additional_tape_if_verified": strict_label and split == "train",
            "old_train_tape_can_supply_unsampled_roots": prior_tape and split == "train" and strict_terrain,
            "retired_optional_if_new_version": optional_terrain,
            "retired_additional_tape_if_verified": optional_label,
            "retired_new_training_map_vs_13_if_verified": optional_vs_train26 and split != "train",
            "retired_additional_training_tape_vs_original26_if_verified": optional_vs_train26,
            "retired_distinct_source_file_vs_original58_if_verified": optional_label,
            "history_retirement_required": optional_terrain and bool(strict_barriers),
            "previous_identity_lookup_alone": roles == {"overlap_lookup_only"},
            "evidence_by_prior_role": {k: sorted(v) for k, v in sorted(refs.items())}}


def retired_optional_if_new_version(candidate, registry):
    """Is this geography available for an expressly retrospective new version, after provenance checks?"""
    return classify(candidate, registry)["retired_optional_if_new_version"]


if __name__ == "__main__":
    registry = build()
    # Compact maps keep the frozen public ladder ledger practical to load in repeated inventory scans.
    (HERE / "exclusions.json").write_text(json.dumps(registry, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"Registered {len(registry['by_seed'])} seeds and {len(registry['by_physical_sha256'])} physical hashes")
