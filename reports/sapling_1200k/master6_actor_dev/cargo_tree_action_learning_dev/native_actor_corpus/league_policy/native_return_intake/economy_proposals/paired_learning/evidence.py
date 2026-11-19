"""Rebuild the acting student's public input before separately reading an audited terminal label."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ECON = HERE.parent
INTAKE = ECON / "terminal_intake"
LEAGUE = ECON.parents[1]
sys.path[:0] = [str(HERE), str(INTAKE)]
import model
import select_root

play = select_root.play
PLAN_SHA = "8486ff0b4c087bb168a5523db3828159046b32d5bb4befd65a1e2b1a1fefbb8e"
FIT = (621758462, 0)
TRANSFER = (621758463, 1)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def card(path):
    return json.loads(path.read_bytes())


def require(test, reason):
    if not test:
        raise ValueError(reason)


def ticket(seed, seat):
    require((seed, seat) in (FIT, TRANSFER), "Unregistered map or physical seat")
    registration = (HERE / "protocol.json").read_bytes()
    require(sha(registration) == PLAN_SHA, "Changed learning protocol")
    for name, expected in json.loads(registration)["input_anchors"].items():
        require(sha((LEAGUE / name).read_bytes()) == expected, "Changed source " + name)
    plan = card(INTAKE / "protocol.json")
    fixture = next(f for f in plan["fixtures"] if (f["seed"], f["seat"]) == (seed, seat))
    return plan, fixture, INTAKE / "outputs" / f"{seed}_{seat}"


def serve(actor, obs, cfg, issued):
    rng = actor.mod._behavior_rng
    try:
        return play.audit_menu.per_order(actor, "candidate", obs, cfg, issued)
    finally:
        if obs["step"] != 0:
            actor.mod._behavior_rng = rng


def probe(actor, obs, cfg, issued, category, with_features):
    """Replay the historical natural output. All proposal and readout inputs originate with this seat."""
    require("full_carbon" not in obs, "Private carbon given to acting student")
    mod = actor.mod
    capture = []
    original = mod._mixed_actions

    def keep(x, aux):
        capture.append((x, aux))
        return original(x, aux)

    mod._mixed_actions = keep
    try:
        event = serve(actor, obs, cfg, issued)
    finally:
        mod._mixed_actions = original

    require(len(capture) == 1, "Missing action-time network features")
    native = None
    if event and event["supported"]:
        require(event["options"][event["choice"]] == issued, "Natural menu choice and issued order differ")
        native = {"p": event["probabilities"], "pick": event["choice"]}
    trunks = [net.trunk for net in mod._NETS]
    phi = None
    try:
        with play.audit_menu.collect.attached(actor.modules):
            menu = select_root.proposals.build(mod, *capture[0], obs)
            expected_default = event["options"][0] if native else issued
            require(menu["baseline"] == expected_default, "Catalog baseline diverged from the parent")
            selected = select_root.pick(menu, obs, cfg, issued, category, native)
            if selected and native is None:
                selected["native_A_probability"] = 1.
            if selected and with_features:
                commands = [row["command"] for row in menu["orders"]]
                phi = model.features(mod, *capture[0], commands)
    finally:
        for net, trunk in zip(mod._NETS, trunks):
            net.trunk = trunk
    return selected, menu, phi


def public_case(seed, seat):
    """Open only the signed prefix for one ticket. Expose no root enemy command or suffix data."""
    plan, fixture, folder = ticket(seed, seat)
    result = {"seed": seed, "seat": seat, "category": fixture["category"], "terrain": fixture["board_sha256"]}
    prefix_path = folder / "prefix.json"
    if not prefix_path.is_file():
        if (folder / "failure.json").is_file():
            return dict(result, status="no_root", reason="The registered execution failed before publishing a prefix")
        gate = INTAKE / "preflight.json"
        if gate.is_file() and not card(gate)["permitted"][fixture["category"]]:
            return dict(result, status="no_root", reason="The preregistered category failed its historical preflight")
        return dict(result, status="pending", reason="Registered pre-action prefix has not been published")
    meta = card(prefix_path)
    claim_path = folder / "claim.json"
    claim = card(claim_path)
    for item in (meta, claim):
        require((item["seed"], item["seat"], item["terrain"]) == (seed, seat, result["terrain"]), "Wrong ticket")
    require(claim["protocol_sha256"] == sha((INTAKE / "protocol.json").read_bytes()) and
            claim["arm_order"] == fixture["arm_order"] and claim["category"] == fixture["category"] and
            claim["source_sha256"] == meta["sources"] and claim["release"] == meta["release"],
            "Prefix differs from registered source, role or arm order")
    if meta["root_step"] is None:
        require(meta["status"] == "no_registered_supported_root", "Missing root identity")
        return dict(result, status="no_root", reason="No supported root at the preregistered turns")
    step = meta["root_step"]
    require(meta["status"] == "root" and step in plan["probe_turns"] and
            "official_scores_by_seat" not in meta, "Unexpected root input")
    tape_path = folder / "prefix.jsonl.gz"
    compressed = tape_path.read_bytes()
    raw = gzip.decompress(compressed)
    require(sha(compressed) == meta["turns_sha256"] and sha(raw) == meta["raw_turns_sha256"], "Prefix bytes changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == step == meta["pre_end_step"], "Prefix truncation")
    env, cfg, live = play.official(seed, result["terrain"], seat)
    actor = play.audit_menu.collect.Player(play.student_history.PACK)
    bank = [5., 5.]
    for t, row in enumerate(rows):
        require(not env.done and row["step"] == t == env.states[0].observation.step, "Broken action sequence")
        obs = live.observe(env, seat, bank[seat])
        require(sha(live.encoded(obs)) == row["public_before_by_seat"][seat] and
                row["student"]["order"] == row["orders"][seat] and row["original"]["order"] == row["orders"][1-seat],
                "Public history or physical role mismatch")
        if t in plan["probe_turns"]:
            selected, _, _ = probe(actor, obs, cfg, row["orders"][seat], fixture["category"], False)
            require(selected is None, "A valid earlier root was skipped")
        else:
            serve(actor, obs, cfg, row["orders"][seat])
        for side, side_name in ((seat,"student"), (1-seat,"original")):
            wall = row[side_name]["wall"]
            require(math.isfinite(wall) and wall >= 0, "Invalid published duration")
            bank[side] -= max(0., wall - 1.)
        require(bank == row["bank_after"], "Clock does not close")
        env.step(row["orders"])
        require(play.board_hash(env, result["terrain"]) == row["board_after_sha256"], "Official state or entity IDs differ")
    require(not env.done and env.states[0].observation.step == step and
            play.board_hash(env, result["terrain"]) == meta["root_board_sha256"], "Wrong official pre-action root")
    obs = live.observe(env, seat, bank[seat])
    observed_sha = sha(live.encoded(obs))
    registered = meta["root_menu"]
    require(observed_sha == meta["root_public_by_seat"][seat] == registered["public_observation_sha256"],
            "Root was built for a different public observation")
    selected, menu, phi = probe(actor, obs, cfg, registered["native_A"], fixture["category"], True)
    require(selected is not None and all(selected[k] == registered[k] for k in selected if k != "native_choice"),
            "Root physical orders, support or witness do not reproduce")
    original_draw = registered["native_choice"]
    reconstructed = selected["native_choice"]
    require((reconstructed is None and original_draw is None) or
            (reconstructed is not None and original_draw is not None and reconstructed["p"] == original_draw["p"]
             and reconstructed["pick"] == original_draw["pick"] == original_draw["unforced_choice"]
             and not original_draw["forced"]), "Natural distribution or realized A choice differs")
    require(registered["options"] == meta["root_orders"] and menu["baseline"] == registered["catalog_baseline"],
            "Recorded physical menu changed")
    try:
        index_a, index_b = model.require_pair(menu, *selected["options"])
    except ValueError as exc:
        return dict(result, status="unsupported_pair", reason=str(exc))
    require(phi.dtype == np.float64 and phi.shape == (len(menu["orders"]), 8, 96) and np.isfinite(phi).all(),
            "Invalid neural readout features")
    files = {path.name: sha(path.read_bytes()) for path in (claim_path, prefix_path, tape_path)}
    return dict(result, status="ready", step=step, catalog=menu, options=selected["options"],
                index_A=index_a, index_B=index_b, phi=phi, observation=obs,
                public_observation_sha256=observed_sha, files=files)


def require_frozen_transfer(seal):
    """Check the complete published prediction against the real public root before opening its label."""
    path = Path(seal)
    require(path.name == "holdout_prediction.json", "Expected the published blind prediction")
    pred = card(path)
    weights_path, fit_path = path.parent / "readout.npy", path.parent / "fit.json"
    require((pred["seed"], pred["seat"]) == TRANSFER and pred["protocol_sha256"] == PLAN_SHA and
            pred["weights_sha256"] == sha(weights_path.read_bytes()) and pred["fit_sha256"] == sha(fit_path.read_bytes()),
            "Sealed prediction does not bind its fit, weights or held-out ticket")
    fit = card(fit_path)
    require((fit["seed"], fit["seat"]) == FIT and fit["weights_sha256"] == pred["weights_sha256"] and
            fit["protocol_sha256"] == PLAN_SHA, "Foreign fit weights or unregistered learner")
    case = public_case(*TRANSFER)
    require(case["status"] == "ready" and pred["case_files"] == case["files"], "Different blind input prefix")
    orders = case["catalog"]["orders"]
    p = [x["probability"] for x in orders]
    with weights_path.open("rb") as stream:
        w = np.load(stream, allow_pickle=False)
    q = model.probabilities(p, case["phi"], w)
    require(pred["physical_orders"] == [x["command"] for x in orders] and pred["old_probabilities"] == p and
            np.array_equal(pred["new_probabilities"], q) and (pred["A_index"], pred["B_index"]) == (case["index_A"], case["index_B"]) and
            pred["public_features_sha256"] == sha(np.asarray(case["phi"],dtype="<f8").tobytes()),
            "The sealed full-support prediction cannot be reproduced")


def label_case(seed, seat, *, sealed_transfer=None):
    plan, fixture, folder = ticket(seed, seat)
    if (seed, seat) == TRANSFER:
        if sealed_transfer is None:
            return dict(eligible=False, status="sealed", reason="Publish frozen weights and every transfer probability first")
        require_frozen_transfer(sealed_transfer)
    verified_path = folder / "verified.json"
    if not verified_path.is_file():
        return dict(eligible=False, status="pending", reason="Independent per-ticket rule and timing audit not published")
    actual = card(verified_path)
    require((actual["seed"],actual["seat"]) == (seed,seat) and actual["category"] == fixture["category"] and
            actual["terrain"] == fixture["board_sha256"] and actual["arm_order"] == fixture["arm_order"] and
            actual["protocol_sha256"] == sha((INTAKE/"protocol.json").read_bytes()) and
            actual["verification_sha256"] == sha((INTAKE/"verify.py").read_bytes()), "Signature or label assignment mismatch")
    result = dict(seed=seed, seat=seat, eligible=False, status=actual["status"], files={"verified.json":sha(verified_path.read_bytes())})
    if not actual["eligible"]:
        return dict(result, reason=actual.get("eligibility",actual["status"]))
    require(actual["status"] == "verified_pair" and actual["single_shared_native_root"] and
            actual["actual_root_requests"] == 1 and actual["eligibility"]["cold"] and actual["eligibility"]["warm_clock"] and
            actual["eligibility"]["realized_semantic_event"] and actual["eligibility"]["depth_prefix_A_B"] == [True]*3,
            "Missing independent official, economic or clock evidence")
    require(actual["root_menu"] == card(folder / "prefix.json")["root_menu"], "Label points to different public menu")
    for name, digest in actual["receipt_hashes"].items():
        require(Path(name).name == name and sha((folder / name).read_bytes()) == digest, "Official receipt changed")
        result["files"][name] = digest
    require(sha((folder / "shared.json").read_bytes()) == actual["shared_sha256"], "Shared original root changed")
    result["files"]["shared.json"] = actual["shared_sha256"]
    scores, margins, utilities = [], [], []
    for name in ("A","B"):
        arm = actual["branches"][name]
        for fn, digest in ((name+".json",arm["sha256"]),(name+".jsonl.gz",arm["tape_sha256"])):
            require(sha((folder / fn).read_bytes()) == digest, "Conditional trajectory changed")
            result["files"][fn] = digest
        pair = arm["official_scores_by_seat"]
        require(len(pair) == 2 and all(math.isfinite(v) for v in pair), "Invalid official terminal")
        margin = pair[seat] - pair[1-seat]
        wdl = float(margin > 0) + .5 * (margin == 0)
        utility = .5 * wdl + .25 * (1 + math.tanh(margin / 2000.))
        require(math.isclose(arm["margin"], margin, rel_tol=0., abs_tol=1e-9) and arm["wdl"] == wdl and
                abs(arm["u"]-utility) < 1e-12, "Wrong seat's terminal reward")
        scores.append(pair); margins.append(margin); utilities.append(utility)
    require(abs(margins[1]-margins[0]-actual["official_margin_B_minus_A"]) < 1e-8 and
            abs(utilities[1]-utilities[0]-actual["delta_u_B_minus_A"]) < 1e-12, "B-minus-A sign or utility mismatch")
    return dict(result, eligible=True, reason=None, official_scores_by_arm=scores, margins=margins,
                utilities=utilities, delta_u_B_minus_A=utilities[1]-utilities[0], gates=actual["eligibility"])
