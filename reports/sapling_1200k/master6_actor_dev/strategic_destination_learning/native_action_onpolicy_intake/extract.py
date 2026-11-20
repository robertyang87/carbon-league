"""Reconstruct the probability of each command actually sent by the signed training sampler."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SAMPLER = STUDY / "native_action_onpolicy/sampler.py"
WEIGHTS = SAMPLER.with_name("behavior.npz")
SAMPLER_SHA = "87655af5309b6c5468f76441da46ce9f4bd7ff923315b2196b7d636994dd0a9b"
WEIGHT_SHA = "6851b91122d2e0b699ae05fe4513b7eaa6b6c2098e1ace14dc0e699470368bf2"
ORIGINAL_SHA = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
LEAGUE = STUDY.parent / "cargo_tree_action_learning_dev/native_actor_corpus/league_policy"
sys.path.insert(0, str(LEAGUE / "natural_original"))
import audit_menu

collect = audit_menu.collect


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_sampler():
    require(collect.sha_file(SAMPLER) == SAMPLER_SHA and collect.sha_file(WEIGHTS) == WEIGHT_SHA,
            "Changed acting package or weights")
    with np.load(WEIGHTS, allow_pickle=False) as weights:
        require(np.array_equal(weights["W1"], np.zeros((24, 1), np.float32)), "Expected the zero residual")


def physical_mass(features, n_orders):
    if features is None:
        require(n_orders == 1, "Multiple orders without a pivot")
        return np.ones(1)
    z = np.asarray(features["logits"], np.float64)
    hidden = np.asarray(features["hidden"], np.float32)
    actions = features["actions"]
    routes = features["route_to_order"]
    base = features["base_index"]
    require(z.ndim == 1 and len(z) == len(actions) == len(routes) and
            1 < len(z) <= 5 and hidden.shape == (len(z), 24), "Malformed pivot features")
    require(np.isfinite(z).all() and np.isfinite(hidden).all(), "Non-finite policy features")
    require(all(type(a) is int and 0 <= a <= 4 for a in actions) and len(set(actions)) == len(actions),
            "Invalid or duplicate pivot moves")
    require(all(type(i) is int and 0 <= i < n_orders for i in [base, *routes]) and
            set([base, *routes]) == set(range(n_orders)), "Malformed grouped physical support")
    mass = np.zeros(n_orders)
    mass[base] = .5
    exp = np.exp(z - np.max(z))
    np.add.at(mass, routes, (.5 / exp.sum()) * exp)
    return mass


def action_step(actor, public, cfg, issued, public_sha):
    # The fresh replay draw is irrelevant. Look up the wire command in the full distribution.
    actor.act(public, cfg)
    trace = actor.mod._LAST_POLICY
    step, seat = public["step"], public["player"]
    require((trace["seat"], trace["step"]) == (seat, step), "Policy history is out of sync")
    features, options = trace["policy_features"], trace["orders"]
    keys = [tuple(sorted(r["action"].items())) for r in options]
    require(len(set(keys)) == len(keys), "Duplicate physical command")
    key = tuple(sorted(issued.items()))
    require(key in keys, f"Logged command missing from physical support at {step}")
    idx = keys.index(key)
    mass = physical_mass(features, len(options))
    observed = np.array([[r["probability"], r["prior_probability"]] for r in options], np.float64)
    require(np.isfinite(observed).all() and (mass > 0).all() and
            np.allclose(observed, mass[:, None], rtol=0, atol=1e-10), "Incorrect physical masses")
    require(features is None or
            (0 < step <= 294 and step % 6 == 0 and options[features["base_index"]]["action"] == trace["parent"]),
            "Pivot off schedule or parent misbound")
    cash = [float(p[0]) for p in public["players"]]
    return dict(step=step, public_sha256=public_sha, cash_margin_before=cash[seat]-cash[1-seat],
                orders=[dict(action=o["action"], probability=float(p)) for o,p in zip(options,mass)],
                issued_order_index=idx, behavior_probability=float(mass[idx]), logp=math.log(mass[idx]),
                policy_features=features)


def complete_game(game, archive, archive_sha, terrain):
    seat, seed = game["seat"], game["seed"]
    require(collect.sha_file(archive) == archive_sha, "Archive SHA differs from settlement")
    with zipfile.ZipFile(archive) as z, TemporaryDirectory() as temp:
        require(len(z.namelist()) == len(set(z.namelist())), "Duplicate ZIP member")
        reg = Path(temp) / "registration.json"
        reg.write_bytes(z.read("collector/registration.json"))
        plan = json.loads(reg.read_bytes())
        receipt_bytes = z.read("collector/receipt.json")
        receipt = json.loads(receipt_bytes)
        require((plan["seed"], plan["candidate_seat"], plan["opponent_seat"],
                 plan["physical_map_sha256"]) == (seed, seat, 1-seat, terrain), "Wrong registered map or seat")
        require(plan["files"]["candidate"]["sha256"] == SAMPLER_SHA and
                plan["files"]["opponent"]["sha256"] == ORIGINAL_SHA, "Wrong policy identity")
        require(sha(receipt_bytes) == game["receipt_sha256"] and
                receipt["protocol_sha256"] == collect.sha_file(reg), "Wrong match or receipt")
        for k in ("status","counted","winner_seat","official_scores_by_seat","official_transitions"):
            require(game[k] == receipt[k], "Divergent archived and settled " + k)
        require(game["status"] == "complete" and game["counted"], "Expected a completed training game")
        tape = z.read("collector/turns.jsonl")
        require(sha(tape) == receipt["turns_sha256"], "Wrong action ledger")
        raw_obs = z.read("collector/observations.jsonl.gz")
        require(sha(raw_obs) == game["public_observation_archive_sha256"], "Wrong public observation archive")
        live = audit_menu.open_live(reg)
        env, cfg = live.official()
    actor = collect.Player(SAMPLER)
    bank = [5., 5.]
    steps = []
    for t,(turn,sidecar) in enumerate(zip(tape.splitlines(), gzip.decompress(raw_obs).splitlines(), strict=True)):
        turn, sidecar = json.loads(turn), json.loads(sidecar)
        require(not env.done and t == turn["step"] == sidecar["step"], "Invalid transition")
        observations = [live.observe(env, side, bank[side]) for side in (0,1)]
        hashes = [sha(live.encoded(o)) for o in observations]
        require(hashes == turn["public_observation_sha256"] == sidecar["public_observation_sha256"] and
                observations == sidecar["observations"], "Pre-action public observation mismatch")
        calls = turn["calls"]
        require(len(calls) == 2 and all(c["status"] == "ok" and c["response"]["step"] == t for c in calls),
                "Missing action in completed trajectory")
        commands = [c["response"]["action"] for c in calls]
        steps.append(action_step(actor, observations[seat], cfg, commands[seat], hashes[seat]))
        env.step(commands)
        bank = turn["overage_remaining_seconds"]
    require(env.done and all(s.status == "DONE" for s in env.states) and
            [s.reward for s in env.states] == game["official_scores_by_seat"] and
            len(steps) == game["official_transitions"], "Numeric terminal does not reproduce")
    actor.diagnostics()
    return steps


def labels(game):
    wdl = None
    if game["counted"]:
        wdl = .5 if game["winner_seat"] is None else float(game["winner_seat"] == game["seat"])
    if game["status"] != "complete":
        return wdl, None, None
    s = game["official_scores_by_seat"]
    require(game["counted"] and len(s) == 2 and all(math.isfinite(k) for k in s), "Invalid numeric terminal")
    margin = float(s[game["seat"]] - s[1-game["seat"]])
    require(wdl == (1. if margin > 0 else 0. if margin < 0 else .5), "Official WDL and score disagree")
    return wdl, margin, .5 * wdl + .5 * (1. + math.tanh(margin/2000.)) / 2.


def extract(hosted):
    settled_path = hosted / "settlement.json"
    settlement = json.loads(settled_path.read_bytes())
    require(settlement["protocol_sha256"] == collect.sha_file(HERE / "protocol.json"), "Wrong training protocol")
    require(settlement["schema"] == "sapling-physical-policy-training-intake/1" and
            settlement["use"] == "on-policy training only", "Wrong sampling cohort")
    design = json.loads((HERE / "protocol.json").read_bytes())
    registered = {(card["seed"],card["seat"]) for card in design["schedule"]}
    games = settlement["games"]
    require(len(games) == len(registered) == 8 and registered == {(g["seed"],g["seat"]) for g in games}
            and all(g["arm"] == "collector" for g in games), "Incomplete registered denominator")
    archives = {(a["seed"],a["seat"]):a for a in settlement["archives"]}
    result = []
    for game in sorted(games, key=lambda g: g["seed"]):
        key = (game["seed"],game["seat"])
        terrain = design["physical_boards"][str(key[0])]
        wdl, margin, u = labels(game)
        summary = {k:game.get(k) for k in ("seed","seat","status","counted","winner_seat","official_scores_by_seat",
                                           "official_transitions","reason","receipt_sha256")}
        summary.update(terrain=terrain, wdl=wdl, official_margin=margin, u=u, steps=[],
                       gradient_eligible=False, archive_sha256=archives.get(key,{}).get("sha256"))
        if game["status"] == "complete":
            archive = archives[key]
            summary["steps"] = complete_game(game, hosted / (archive["blob_filename"]), archive["sha256"], terrain)
            summary["gradient_eligible"] = True
        result.append(summary)
    return dict(schema="physical_order_onpolicy_v1", behavior_sha256=SAMPLER_SHA, weight_sha256=WEIGHT_SHA,
                settlement_sha256=collect.sha_file(settled_path), source_commit=settlement["commit"],
                source_run=settlement["run_id"], games=result)


def fixture():
    sys.path.insert(0, str(STUDY / "native_action_return_intake"))
    from select_orders import source_for
    source, identity = source_for(621758540)
    cfg = {k:v for k,v in source["configuration"].items() if k != "randomSeed"}
    actor = collect.Player(SAMPLER)
    actor.mod._NATIVE_RNG = np.random.default_rng(10244)
    rows = []
    redraw_disagreed = 0
    for turn in source["rows"]:
        public = turn["public"]
        wire = turn["orders"][source["seat"]]
        row = action_step(actor, public, cfg, wire, sha(collect.canonical_bytes(public)))
        rows.append(row)
        redraw_disagreed += int(wire != actor.mod._LAST_POLICY["issued"])
    require(len(rows) == source["terminal"]["step"] and redraw_disagreed > 0, "Uninformative fixture")
    expect = labels(dict(seat=0, counted=True, status="forfeit", winner_seat=1))
    require(expect == (0.,None,None) and
            labels(dict(seat=0,counted=False,status="missing_or_unverified",winner_seat=None)) == (None,None,None),
            "Invented a score or policy-gradient label")
    return dict(scope="Off-policy integration fixture on one archived BC training game. Excluded from fitting.",
                sampler_sha256=SAMPLER_SHA, **identity, transitions=len(rows),
                scheduled_with_pivot=sum(r["policy_features"] is not None for r in rows),
                distinct_physical_choices=sum(len(r["orders"])>1 for r in rows),
                wire_command_differs_from_offline_draw=redraw_disagreed,
                all_logged_commands_found=True, official_training_episodes=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    check_sampler()
    if args.fixture:
        receipt = fixture()
        collect.write_json(HERE / "extract_fixture.json", receipt)
        print(receipt)
    else:
        dataset = extract(HERE / "hosted")
        collect.write_gzip(HERE / "training_rows.json.gz", dataset)
        print("registered", len(dataset["games"]), "usable", sum(g["gradient_eligible"] for g in dataset["games"]))
