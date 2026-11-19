"""Read genuine v1 tree-response training pairs. Never launch an agent or open development tapes."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from statistics import mean
import sys

CODE = Path(__file__).resolve().parent.parent
PARENT = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
WEIGHTS = "8a5544ce41cf7fef04424eb735d43c7a659411f872bbd8d90f0046843bad33a2"
STUDENT = "f82bfd47a904aa1fbfa4e95fbfcdd13e623c1f1eb216a8e736694a1d282cc713"
MASTER = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
EXAM_SETTINGS = "9d37b28c3bfc3cb28411c437d24d49c3dc8af7b813d9d7a16b0b0896aa9a37a6"
FIRST_PROTOCOL = "00ff49427af57f07905bea5c655cf49078713fda4d73cdc136d680d4beec4023"


def read(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf8") as stream:
        return json.load(stream)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def equal(actual, expected, what):
    require(canonical(actual) == canonical(expected), what)


def index(rows, keys):
    indexed = {tuple(row[k] for k in keys): row for row in rows}
    require(len(indexed) == len(rows), f"duplicate {keys}")
    return indexed


def batches(root):
    root = Path(root)
    require(sha(root / "protocol.json") == FIRST_PROTOCOL, "unexpected initial tree-response protocol")
    yield "intake_01", root, None
    queue_file = root / "next_intake.json"
    if not queue_file.exists():
        require(not (root / "batches").exists() or not any((root / "batches").glob("intake_*")),
                "continuation exists without a registered queue")
        return
    queue = read(queue_file)
    equal(queue["source_data_sha256"], sha(root / "training_labels.json"), "continuation upstream label hash")
    equal(queue["policy_sha256"], PARENT, "continuation parent")
    names = [row["name"] for row in queue["batches"]]
    require(names == sorted(set(names)) and all(re.fullmatch(r"intake_[0-9]{2}", name) and name > "intake_01" for name in names),
            "invalid continuation names")
    require(not {p.name for p in (root / "batches").glob("intake_*")} - set(names), "unregistered continuation")
    for entry in queue["batches"]:
        yield entry["name"], root / "batches" / entry["name"], entry


def progress(intake_root):
    """Counts published source, arm and case receipts. An individual file is not a verified training row."""
    out = {}
    for name, area, _ in batches(Path(intake_root)):
        source = area / "source"
        out[name] = dict(started=(area / "protocol.json").is_file(),
                         raw_source_games=sum(1 for _ in source.glob("raw_*_*.json.gz")),
                         arm_receipts=sum(1 for a in ("parent", "alternative") for _ in area.glob(f"case_*/live_{a}.json")),
                         case_labels=sum(1 for _ in area.glob("case_*/training_label.json")),
                         labels_published=(area / "training_labels.json").is_file(),
                         aggregate_present=(area / ("training_labels.json" if name == "intake_01" else "aggregate.json")).is_file())
    return out


def load_authority(area, plan):
    # label.py's replay layer only executes saved commands. Give each intake its own
    # module namespace, as batch_cli.modules does, without altering other imports.
    sys.path.insert(0, str(CODE))
    try:
        spec = importlib.util.spec_from_file_location("tree_response_evidence_authority", CODE / "label.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CODE))
    module.HERE, module.PLAN = area, plan
    return module


def verify_sources(area, plan, receipt, clearance):
    source = area / "source"
    cert_file = source / "official_replay.json"
    equal(receipt["official_sha256"], sha(cert_file), "source official replay hash")
    official = read(cert_file)["runs"]["source"]
    result = read(source / "results.json")
    header = result["header"]
    equal([official["pack_sha256"], header["pack_sha256"], header["weight_sha256"], header["seeds"], result["status"]],
          [PARENT, PARENT, WEIGHTS, plan["train_maps"], "complete"], "source original pack identity")
    expected = {(s, seat) for s in plan["train_maps"] for seat in (0, 1)}
    rows = index(receipt["games"], ("seed", "seat"))
    certified = index(official["games"], ("seed", "seat"))
    played = index(result["games"], ("seed", "candidate_seat"))
    require(expected == rows.keys() == certified.keys() == played.keys(), "source map/seat coverage")
    outputs = {}
    for (seed, seat), row in rows.items():
        cert, game = certified[seed, seat], played[seed, seat]
        fname = f"raw_{seed}_{seat}.json.gz"
        source_file = source / fname
        raw = read(source_file)
        h = sha(source_file)
        require(Path(row["raw_path"]).name == fname and game["raw_file"] == fname, "source path")
        equal([row["raw_sha256"], cert["transcript_sha256"], game["raw_sha256"]], [h] * 3, "source tape hash")
        grid = raw["initial_full_carbon"]
        require(len(grid) == 225 and all(isinstance(x, (int, float)) and 0 <= x <= 255 and int(x) == x for x in grid),
                "initial carbon terrain")
        board = hashlib.sha256(bytes(map(int, grid))).hexdigest()
        equal([raw["initial_terrain_u8_sha256"], row["map_sha256"], cert["map_sha256"],
               game["initial_terrain_u8_sha256"], clearance["boards"][str(seed)]], [board] * 5, "terrain identity")
        n = len(raw["actions"])
        equal([raw["seed"], raw["candidate_seat"], raw["pack_sha256"], raw["weight_identity"],
               raw["examiner_md5"], raw["errors"], raw["audit_messages"]], [seed, seat, PARENT, WEIGHTS, plan["examiner_md5"], [], []],
              "source identity/errors")
        require(hashlib.sha256(canonical(raw["configuration"])).hexdigest() == EXAM_SETTINGS, "original examiner settings")
        equal([raw["steps"], row["steps"], cert["steps"], game["actual_transitions"]], [n] * 4, "source length")
        require(game["valid"] and game["search_invocations"] == n and cert["statuses"] == ["DONE", "DONE"], "source termination/search")
        scores = raw["official_score_by_seat"]
        equal([row["scores"], cert["official_reward_by_seat"], game["official_score_by_seat"]], [scores] * 3, "source official scores")
        equal(row["official_margin"], scores[seat] - scores[1-seat], "source margin")
        outputs[seed,seat] = (row, raw, source_file)
    return outputs


def verify_case(area, reg, source, selected_candidate, label, authority, protocol_hash):
    seed, seat, t, uid, window = (reg[k] for k in ("seed", "seat", "step", "uid", "window"))
    row, raw, source_file = source
    require(reg["source"] == row["raw_path"] and reg["source_sha256"] == row["raw_sha256"], "registered source mismatch")
    case = area / f"case_{seed}_{seat}_{window}"
    require(case.name == reg["case_directory"] and label["features"] == f"{case.name}/root.json.gz", "case directory identity")
    root_file = case / "root.json.gz"
    root_hash = sha(root_file)
    equal([reg["root_sha256"], label["root_sha256"]], [root_hash] * 2, "root hash")
    root = read(root_file)
    c = root["candidate"]
    equal(c, selected_candidate, "selected public opportunity and root disagree")
    equal([c["seed"], c["seat"], c["t"], c["uid"], c["position"], c["tape_sha256"], c["stratum_bits"]],
          [seed, seat, t, uid, reg["cell"], reg["source_sha256"], reg["source_stratum"]], "opportunity identity")
    require(c["actions"]["NONE"]["can_pay_now"] and c["actions"]["NONE"]["payable_site"] and reg["price"] > 0,
            "registered action not feasible")
    equal(c["actions"]["NONE"]["price"], reg["price"], "pay price")
    equal(root["v1"], authority.clean(raw["actions"][t][seat]), "parent differs from chosen real source")
    a, b = root["v1"], root["alternative"]
    require({who for who in a.keys() | b.keys() if a.get(who) != b.get(who)} == {uid}, "nonisolated intervention")
    old, new = a.get(uid, "NONE"), b.get(uid, "NONE")
    require((old == "NONE") != (new == "NONE") and {old, new} <= {"NONE", "UP", "RIGHT", "DOWN", "LEFT"}, "pay/move orientation")
    equal([reg["old"], reg["new"], root["old"], root["new"], reg["paid_arm"], root["paid_arm"]],
          [old, new, old, new] + ["parent" if old == "NONE" else "alternative"] * 2, "wrong intervention orientation")
    qf, sf = case / "live_protocol.json", case / "live_shared_init.json"
    q, shared = read(qf), read(sf)
    equal([q["family"], q["seed"], q["seat"], q["root_t"], q["training_protocol_sha256"], q["training_root_sha256"]],
          ["tree_terminal_training", seed, seat, t, protocol_hash, root_hash], "different registered family/pair/root")
    equal(q["record"], {k:reg[k] for k in ("seed", "seat", "source", "source_sha256")}, "live source registration")
    equal(q["paired_responder_source_sha256"], sha(CODE.parents[1] / "root_recovery_branches/live.py"), "live responder implementation")
    equal(q["root"], {"t":t, "worker":uid, "old":old, "alternative":new, "tree":""}, "live root registration")
    from numpy.random import RandomState
    mt, keys, idx, normal, saved = RandomState(reg["rng_seed"]).get_state()
    equal([shared[k] for k in ("source_sha256", "mt_algorithm", "mt_keys", "mt_index", "has_normal", "stored_normal")],
          [MASTER, mt, keys.tolist(), idx, normal, saved], "shared registered random stream")
    observed = []
    for name in ("parent", "alternative"):
        arm = read(case / f"live_{name}.json")
        equal([arm[k] for k in ("arm", "seed", "seat", "source_sha256", "pack_sha256", "original_trajectory_sha256",
                                "protocol_sha256", "shared_init_sha256", "root", "action_tape")],
              [name, seed, seat, MASTER, PARENT, row["raw_sha256"], sha(qf), sha(sf), q["root"],
               f"live_{name}_actions.json.gz"], "branch identity/stream/tape")
        tape = read(case / arm["action_tape"])
        equal(tape[0], arm["root_actions"], "receipt/root tape mismatch")
        observed.append(arm["root_opponent_observation_sha256"])
    require(observed[0] == observed[1] and re.fullmatch("[0-9a-f]{64}", observed[0]), "opponent root observation mismatch")
    equal(read(case / "training_label.json"), label, "published local and aggregate labels disagree")
    # Both interpreters replay both full branches. This verifies exact debit by
    # uid/cell (including a genuine miss), acquired tree, future cash, and true
    # official rewards even for extinction. No forecast or native search runs.
    rebuilt = authority.label({**reg, "source": str(source_file.resolve())})
    equal(rebuilt, label, "independently reconstructed terminal response or cash debit differs")
    return dict(board=row["map_sha256"], seed=seed, seat=seat, t=t, uid=uid, root=root, reg=reg, label=label, root_hash=root_hash,
                cfg=raw["game_configuration"])


def read_batch(area, name, entry, reserved, prior_hashes):
    plan = read(area / "protocol.json")
    protocol_hash = sha(area / "protocol.json")
    seeds = plan["train_maps"]
    equal([plan["parent_pack_sha256"], plan["parent_weight_manifest_sha256"], plan["student_npz_sha256"], plan["physical_seats"]],
          [PARENT, WEIGHTS, STUDENT, [0, 1]], "wrong family of training actors/seats")
    require(len(seeds) == len(set(seeds)) == 2 and not (reserved & set(seeds)), "unregistered or development map")
    if entry is not None:
        require(plan["schema"] == "sapling-tree-response-training-continuation-v1", "unexpected continuation schema")
        equal([plan["intake_name"], seeds, plan["training_job_registry"], plan["prior_receipts"], plan["queue_sha256"]],
              [name, entry["maps"], entry["jobs"], prior_hashes, sha(area.parents[1] / "next_intake.json")], "continuation chain")
        equal(sorted(plan["development_maps"]), sorted(reserved), "development allocation change")
    receipt, clearance = read(area / "source_receipt.json"), read(area / "clearance.json")
    selection, agg = read(area / "selection.json"), read(area / "training_labels.json")
    equal([receipt["protocol_sha256"], selection["protocol_sha256"], agg["protocol_sha256"], clearance["declaration_sha256"]],
          [protocol_hash] * 4, "intake protocol hash chain")
    equal([receipt["clearance_sha256"], selection["source_receipt_sha256"], selection["all_opportunities_sha256"], agg["selection_sha256"]],
          [sha(area / f) for f in ("clearance.json", "source_receipt.json", "opportunities.jsonl.gz", "selection.json")], "intake evidence hash chain")
    require(clearance["verdict"] == "eligible_in_checked_scope" and agg["schema"] == "tree-response-training-terminal-v1"
            and agg["reserved_development_games"] == 0 and selection["opponent_future_or_terminal_used_in_selection"] is False,
            "unsupported batch provenance")
    equal(selection["covered_maps"], seeds, "selected source maps")
    selector = CODE / ("batch_cli.py" if entry else "select_roots.py")
    equal(selection["selector_sha256"], sha(selector), "unknown selector implementation")
    if entry:
        equal(selection["shared_decode_sha256"], sha(CODE / "select_roots.py"), "unknown shared legal decoder")
    sources = verify_sources(area, plan, receipt, clearance)
    boards = {sources[s,0][0]["map_sha256"] for s in seeds}
    reserved_boards = {clearance["boards"][str(s)] for s in reserved}
    require(len(boards) == 2 and not (boards & reserved_boards), "training/development terrain overlap")
    if entry:
        equal({str(s): sources[s,0][0]["map_sha256"] for s in seeds}, entry["terrain_at_queue_time"], "queue terrain identity")
    cases = index(selection["cases"], ("seed", "seat", "window"))
    labels = index(agg["pairs"], ("seed", "seat", "window"))
    offered = {(s,p,w) for s in seeds for p in (0,1) for w in range(len(plan["windows"]))}
    empty = {tuple(x) for x in selection["empty_windows"]}
    require(cases.keys() == labels.keys() and not (empty & cases.keys()) and cases.keys() | empty == offered, "lost or duplicate case window")
    keys = [(r["seed"], r["seat"], r["step"], r["uid"]) for r in cases.values()]
    require(len(set(keys)) == len(keys), "duplicate root in named intake")
    chosen = dict.fromkeys(keys)
    with gzip.open(area / "opportunities.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["seed"], row["seat"], row["t"], row["uid"]
            if key in chosen:
                require(chosen[key] is None, "repeated eligible selected root in public index")
                chosen[key] = row
    jobs = index(entry["jobs"], ("seed", "seat", "window")) if entry else {}
    authority = load_authority(area, plan)
    records = []
    for (seed, seat, window), reg in cases.items():
        t = reg["step"]
        lo, hi = plan["windows"][window]
        require(lo <= t <= hi, "root outside frozen window")
        if entry:
            job = jobs[seed,seat,window]
            equal([reg["job_key"], reg["rng_seed"], reg["expected_stratum"]], [job["case_id"], job["rng_seed"], job["desired_exposure_bits"]], "job registration")
        else:
            equal(reg["rng_seed"], 813700 + 4*seeds.index(seed) + 2*seat + window, "registered original stream")
        rec = verify_case(area, reg, sources[seed,seat], chosen[seed,seat,t,reg["uid"]], labels[seed,seat,window], authority, protocol_hash)
        records.append({**rec, "batch":name})
    paid = sum(r["label"]["pay_intent_filled"] for r in records)
    margins = [r["label"]["target_delta_official_margin"] for r in records]
    included = {r["seed"] for r in records}
    equal([agg["unique_source_maps"], agg["paid"], agg["positive_labels"], agg["negative_labels"]],
          [len(included), paid, sum(x>0 for x in margins), sum(x<0 for x in margins)], "aggregate class counts")
    equal(agg["context_counts"], Counter("".join(map(str,r["label"]["observed_stratum"])) for r in records), "aggregate strata")
    for s in included:
        effects = [r["label"]["target_delta_official_margin"] for r in records if r["seed"] == s]
        equal(agg["map_summary"][str(s)], {"pairs":len(effects), "mean_local_effect":mean(effects)}, "map aggregate")
    return records, boards


def check_aggregate(root, area, records, prior, count_boards):
    """Cross-check the continuation publisher's cumulative identities, labels and fit support."""
    saved = read(area / "aggregate.json")
    provenance = [{"dir": k, "labels_sha256": v} for k, v in prior.items()]
    examples = []
    for r in records:
        folder = root if r["batch"] == "intake_01" else root / "batches" / r["batch"]
        features = folder / r["label"]["features"]
        label = features.with_name("training_label.json")
        item = r["label"]
        examples.append(dict(key=[PARENT, r["board"], r["seat"], r["t"], r["uid"]],
                             source_intake=str(folder.relative_to(root)), features=str(features.relative_to(root)),
                             label=str(label.relative_to(root)), label_sha256=sha(label), filled=item["pay_intent_filled"],
                             target=item["target_delta_official_margin"], seed=r["seed"], seat=r["seat"],
                             observed_stratum=item["observed_stratum"]))
    paid = [e for e in examples if e["filled"]]
    pos = sorted({e["seed"] for e in paid if e["target"] > 0})
    neg = sorted({e["seed"] for e in paid if e["target"] < 0})
    equal(saved, dict(policy_sha256=PARENT, source_intakes=provenance, distinct_maps=count_boards,
                      complete_pairs=len(examples), filled_pairs=len(paid), positive_maps=pos, negative_maps=neg,
                      fit_support_gate=(count_boards >= 8 and len(paid) >= 32 and len(pos) >= 2 and len(neg) >= 2),
                      held_out_games=0, fit_performed=False, examples=examples), "cumulative intake publisher disagrees")


def collect(intake_root):
    """Return (verified intent records, registered unfinished batches) in frozen intake/window order."""
    root = Path(intake_root)
    registry = list(batches(root))
    repo = CODE.parents[3]
    equal([sha(repo / f) for f in ("data/master3s2_source.py", "data/weights/sapling_1200k/master6_actor_v1.npz",
                                   "data/weights/sapling_1200k/terminal_return_fit_v1.py")],
          [MASTER, STUDENT, PARENT], "wrong original opponent or training actor files")
    reserved = set(read(root / "protocol.json")["development_maps"])
    records, pending, prior = [], [], {}
    covered_seeds, covered_boards, roots = set(), set(), set()
    for name, area, entry in registry:
        if not (area / "training_labels.json").is_file() or (entry and not (area / "aggregate.json").is_file()):
            pending.append(name)
            continue
        require(not pending, "later intake claims completion before a registered predecessor")
        batch, boards = read_batch(area, name, entry, reserved, prior)
        seeds = set(read(area / "protocol.json")["train_maps"])
        require(not covered_seeds & seeds and not covered_boards & boards, "reused training terrain or seed")
        covered_boards.update(boards)
        covered_seeds.update(seeds)
        for r in batch:
            key = r["board"], r["seat"], r["t"], r["uid"]
            require(key not in roots, "duplicate paired training unit")
            roots.add(key)
        records.extend(batch)
        prior[str(area.relative_to(root))] = sha(area / "training_labels.json")
        if entry:
            check_aggregate(root, area, records, prior, len(covered_boards))
    return records, pending
