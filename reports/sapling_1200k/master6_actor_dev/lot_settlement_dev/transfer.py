"""Freeze existing paired native branches by pre-action physical eligibility; score only after freezing."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
TREE = HERE.parent / "tree_response_training"
sys.path.insert(0, str(ROOT / "code"))
from fastsim import FastCarbon


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes())


def require(v, m):
    if not v:
        raise ValueError(m)


def put(p, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    if p.exists():
        require(p.read_bytes() == payload, "immutable output differs: " + str(p))
    else:
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(p)


@lru_cache(maxsize=30)
def original(src):
    raw = load(src)
    official = src.parent / "official_replay.json"
    cert = load(official)["runs"][src.parent.name]
    (game,) = [g for g in cert["games"] if (g["seed"], g["seat"]) == (raw["seed"], raw["candidate_seat"])]
    sim = FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    board = hashlib.sha256(bytes(int(x) for x in sim.carbon)).hexdigest()
    require(board == game["map_sha256"], "source terrain differs")
    return raw, board


def prior():
    receipts = [TREE / "source_receipt.json"] + [
        TREE / f"batches/intake_{b:02d}/source_receipt.json" for b in (2, 3, 4, 5)
    ]
    maps = {g["seed"]: g["map_sha256"] for p in receipts for g in load(p)["games"]}
    require(len(maps) == len(set(maps.values())) == 10, "wrong model source terrain split")
    return maps, receipts


def rootstate(raw, t, history=False):
    cfg = raw["game_configuration"]
    s = raw["candidate_seat"]
    sim = FastCarbon(cfg).reset(seed=raw["seed"])
    past = []
    for i, cmd in enumerate(raw["actions"][:t]):
        if history and i >= t - 10:
            past.append(sim.obs(s))
        sim.step(*cmd)
    return sim, past


def freeze():
    dest = HERE / "branch_manifest.json"
    if dest.exists():
        return load(dest)
    maps, receipts = prior()
    accepted = []
    excluded = []
    paths = sorted(BASE.rglob("live_protocol.json"))
    covers = {
        k: []
        for k in (
            "terminal_economy",
            "late_response_training",
            "master6_actor_dev/freight_response",
            "master6_actor_dev/freight_deployment",
            "root_recovery_branches",
            "planter_routes_dev",
            "master6_actor_dev/tree_response_training",
        )
    }
    for p in paths:
        d = p.parent
        cfg = load(p)
        rel = str(d.relative_to(ROOT))
        counts = dict(directory=rel, protocol_sha256=sha(p))
        for k in covers:
            if str(d.relative_to(BASE)).startswith(k):
                covers[k].append(rel)
        if "return_value_dev" in rel or "early_lane_dev" in rel:
            excluded.append(dict(**counts, reason="new_or_concurrent_experiment_outside_archived_roster"))
            continue
        if "root" not in cfg or "record" not in cfg:
            # Earliest manually registered branch selected capture_now; its missing record is obtained from
            # the stage's recorded source identity and root (same as original responder's default lookup).
            root = load(d / "live_parent.json")["root"]
            matches = [
                (g, v["root"])
                for g in load(d / "results.json")["games"]
                for family, v in g["roots"].items()
                if family == cfg["family"] and (g["seed"], g["seat"]) == (cfg["seed"], cfg["seat"])
            ]
            ((reg, frozenroot),) = matches
            require(
                root["worker"] == frozenroot["worker"] and root["t"] == frozenroot["t"],
                "legacy root mismatch",
            )
        else:
            root, reg = cfg["root"], cfg["record"]
        seat, t = cfg["seat"], cfg["root_t"]
        src = ROOT / reg["source"]
        uid = root["worker"]
        require(sha(src) == reg["source_sha256"], "source tape changed")
        raw, board = original(src)
        state, _ = rootstate(raw, t)
        worker = state.workers.get(uid)
        require(
            worker
            and worker.player == seat
            and raw["seed"] == cfg["seed"]
            and raw["candidate_seat"] == seat
            and root["old"] == (raw["actions"][t][seat].get(uid) or "NONE"),
            "wrong physical source worker",
        )
        row = dict(
            **counts,
            seed=cfg["seed"],
            seat=seat,
            t=t,
            uid=uid,
            cargo=worker.carbon,
            role="collector" if worker.is_collector else "planter",
            terrain_sha256=board,
            source=reg["source"],
            source_sha256=reg["source_sha256"],
            source_pack=raw["pack_sha256"],
            old=root["old"],
            alternative=root.get("alternative", root.get("replacement")),
            family=cfg["family"],
        )
        reasons = []
        if cfg["seed"] in maps or board in maps.values():
            reasons.append("same_seed_or_board_as_model_source")
        if not worker.is_collector:
            reasons.append("focal_worker_is_not_collector")
        if worker.carbon <= 0:
            reasons.append("no_initial_cargo")
        if row["old"] == row["alternative"]:
            reasons.append("same_action")
        alternatives = [
            arm
            for arm in ("alternative", "capture")
            if (d / f"live_{arm}_actions.json.gz").is_file() and (d / f"live_{arm}.json").is_file()
        ]
        if (
            not (d / "live_parent.json").is_file()
            or not (d / "live_parent_actions.json.gz").is_file()
            or len(alternatives) != 1
        ):
            reasons.append("incomplete_paired_archive")
        if reasons:
            excluded.append(dict(**row, reason="+".join(reasons)))
            continue
        row["treatment_name"] = alternatives[0]
        row["arms"] = {
            arm: {
                "tape_sha256": sha(d / f"live_{arm}_actions.json.gz"),
                "receipt_sha256": sha(d / f"live_{arm}.json"),
            }
            for arm in ("parent", alternatives[0])
        }
        row["shared_rng_sha256"] = sha(d / "live_shared_init.json")
        accepted.append(row)
    groups = {(r["terrain_sha256"], r["seat"], r["t"], r["uid"]) for r in accepted}
    out = dict(
        source_maps=maps,
        model_source_receipts={str(p.relative_to(ROOT)): sha(p) for p in receipts},
        query="All existing native live_protocol.json under reports/sapling_1200k at discovery, excluding current return_value/early_lane. Positive own focal collector, different registered action. Both complete original Master3s2 arm and paired action files required. No selection on return, state after root, or sign of difference. Same seed OR terrain as a model source is excluded. Earlier return/denial/freight studies intentionally selected particular root types; distinct old policy packs may differ from current v1.",
        matched_registries=len(paths),
        included_pairs=len(accepted),
        distinct_source_roots=len(groups),
        distinct_source_maps=len(set(r["terrain_sha256"] for r in accepted)),
        covered_prefix_registries={k: len(v) for k, v in covers.items()},
        included=accepted,
        excluded=excluded,
    )
    put(dest, out)
    return out


def scenario(r):
    raw, terrain = original(ROOT / r["source"])
    require(
        terrain == r["terrain_sha256"] and raw["pack_sha256"] == r["source_pack"],
        "physical board/model mismatch",
    )
    s, t = r["seat"], r["t"]
    sim, history = rootstate(raw, t, True)
    obs = sim.obs(s)
    orders = raw["actions"][t][s]
    alter = dict(orders)
    if r["alternative"] == "NONE":
        alter.pop(r["uid"], None)
    else:
        alter[r["uid"]] = r["alternative"]
    return raw, sim, obs, history, [orders, alter]


def predictions(roots):
    import head

    model_path = HERE / "weights.json.gz"
    control_path = HERE / "distance_control.json.gz"
    weights = load(model_path)
    control = load(control_path)
    rows = []
    for r in roots:
        _, _, obs, h, a = scenario(r)
        s, u = r["seat"], r["uid"]
        estimate = {}
        for name, orders in zip(("parent", "alternative"), a):
            chosen = orders.get(u) or "NONE"
            estimate[name] = head.predict(weights, obs, h, s, u, action=chosen, own_orders=orders)
        ctrl = head.predict(control, obs, h, s, u)
        rows.append(
            dict(
                directory=r["directory"],
                parent=estimate["parent"],
                alternative=estimate["alternative"],
                distance_control=ctrl,
                chose=(
                    "alternative"
                    if estimate["alternative"]["expected_cash"] > estimate["parent"]["expected_cash"] + 1e-6
                    else "parent"
                ),
            )
        )
    doc = dict(
        weights_sha256=sha(model_path),
        distance_control_sha256=sha(control_path),
        head_sha256=sha(HERE / "head.py"),
        manifest_sha256=sha(HERE / "branch_manifest.json"),
        rows=rows,
        control="Order-blind conditional estimator. An expected-cash difference within 0.000001 retains original order.",
    )
    put(HERE / "branch_predictions.json", doc)
    return rows


def observed(r, branch, raw):
    import lots
    import math

    sys.path.insert(0, str(BASE / "root_recovery_branches"))
    from branches import begin

    directory = ROOT / r["directory"]
    key = r["treatment_name"] if branch == "alternative" else "parent"
    receipt_path = directory / f"live_{key}.json"
    tape = directory / f"live_{key}_actions.json.gz"
    binding = r["arms"][key]
    require(
        sha(tape) == binding["tape_sha256"]
        and sha(receipt_path) == binding["receipt_sha256"]
        and sha(directory / "live_protocol.json") == r["protocol_sha256"]
        and sha(directory / "live_shared_init.json") == r["shared_rng_sha256"],
        "archive changed",
    )
    receipt = load(receipt_path)
    tail = load(tape)
    require(
        receipt["action_sha256"] == sha(tape)
        and receipt["protocol_sha256"] == r["protocol_sha256"]
        and receipt["shared_init_sha256"] == r["shared_rng_sha256"]
        and receipt["original_trajectory_sha256"] == r["source_sha256"]
        and receipt["pack_sha256"] == r["source_pack"]
        and receipt["statuses"] == ["DONE", "DONE"]
        and receipt["source_sha256"] == "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
        and receipt["search_calls"] == len(tail) - 1,
        "archive omitted original search or changed paired identity",
    )
    seat, t, u = r["seat"], r["t"], r["uid"]
    clean = lambda a: {k: v for k, v in a.items() if v and v != "NONE"}
    recorded = raw["actions"][t]
    base = clean(recorded[seat])
    issued = clean(tail[0][seat])
    require(clean(recorded[1 - seat]) == clean(tail[0][1 - seat]), "root enemy action differs")
    expected = dict(base)
    if branch == "alternative":
        if r["alternative"] == "NONE":
            expected.pop(u, None)
        else:
            expected[u] = r["alternative"]
    require(issued == expected, "unregistered extra modifications")
    sim, _ = rootstate(raw, t)
    raw = dict(raw, initial_terrain_u8_sha256=r["terrain_sha256"])
    env = begin(raw)
    for actions in raw["actions"][:t]:
        env.step(actions)
    transitions = []
    values = {}
    for j, actions in enumerate(tail):
        require(not env.done, "extra branch action after terminal")
        # The official Board and env share the id generator. Step the persistent
        # environment before the independent checker restores this pre-step ID state.
        env.step(actions)
        cargo, forward, _ = lots.one_step(sim, actions, seat, verify=True)
        require(list(sim.cash) == [p[0] for p in env.states[0].observation.players], "branch cash mismatch")
        values.update({(j, k): v for k, v in cargo.items()})
        transitions.append(forward)
    steps = len(transitions)
    require(
        env.done
        and [g.status for g in env.states] == ["DONE", "DONE"]
        and [g.reward for g in env.states] == receipt["score"]
        and sim.step_count == receipt["terminal_t"] == steps + t,
        "official native terminal does not match",
    )
    values.update({(steps, k): v for k, v in lots.positive_collectors(sim.full_observation(), seat).items()})
    lot = lots.Timeline(values, transitions, steps)
    summary = lot.summary(0, u)
    events = lot.events(0, u)
    require(abs(summary["root_cargo"] - r["cargo"]) < 1e-6, "wrong original lot")
    payouts = [
        {"after_root": e["step"], "at_step": t + e["step"], "kind": e["kind"], "amount": e["amount"]}
        for e in events
        if e["kind"] in ("own_direct", "own_relay")
    ]
    cash = float(sum(e["amount"] for e in payouts))
    deadlines = {
        str(h): sum(e["amount"] for e in payouts if e["after_root"] <= h)
        for h in (1, 3, 6, 10, 16, 25, 40, 64, 100)
    }
    deadlines["terminal"] = cash
    return dict(
        current_lot=summary["root_cargo"],
        actual_own_cash=cash,
        payouts=payouts,
        first_own_cash_after=payouts[0]["after_root"] if payouts else None,
        actual_conditional_cash_time=summary["mean_steps_to_own_cash_conditional"],
        actual_cause_amounts=summary["amounts"],
        cash_by_deadline=deadlines,
        statuses=receipt["statuses"],
        native_search_calls=receipt["search_calls"],
        physical_terminal_scores=receipt["score"],
        terminal_u=receipt["u"],
        terminal_margin=receipt["margin"],
        terminal_t=receipt["terminal_t"],
    )


def mean(seq):
    return sum(seq) / len(seq) if seq else None


def sign(x):
    return 1 if x > 1e-6 else -1 if x < -1e-6 else 0


def evaluate():
    roots = load(HERE / "branch_manifest.json")["included"]
    predicted = predictions(roots)
    pairs = []
    groups = defaultdict(list)
    maps = defaultdict(list)
    for r, q in zip(roots, predicted):
        require(r["directory"] == q["directory"], "prediction identity")
        raw, _ = original(ROOT / r["source"])
        arms = {arm: observed(r, arm, raw) for arm in ("parent", "alternative")}
        a, b = arms["parent"], arms["alternative"]
        pd = q["alternative"]["expected_cash"] - q["parent"]["expected_cash"]
        observed_delta = b["actual_own_cash"] - a["actual_own_cash"]
        chosen = arms[q["chose"]]
        row = dict(
            seed=r["seed"],
            seat=r["seat"],
            at=r["t"],
            uid=r["uid"],
            family=r["family"],
            source_pack=r["source_pack"],
            directory=r["directory"],
            arms=arms,
            predicted_terminal_delta=pd,
            actual_terminal_delta=observed_delta,
            predicted10_delta=q["alternative"]["cash_by_deadline"]["10"]
            - q["parent"]["cash_by_deadline"]["10"],
            actual10_delta=b["cash_by_deadline"]["10"] - a["cash_by_deadline"]["10"],
            outcome_margin_delta=b["terminal_margin"] - a["terminal_margin"],
            outcome_u_delta=b["terminal_u"] - a["terminal_u"],
            selected=q["chose"],
            selected_real_lot_cash=chosen["actual_own_cash"],
            change_against_parent=chosen["actual_own_cash"] - a["actual_own_cash"],
            change_against_uniform_choice=chosen["actual_own_cash"]
            - (a["actual_own_cash"] + b["actual_own_cash"]) / 2,
            realized_lot_abs_error_model=mean(
                [abs(q[arm]["expected_cash"] - arms[arm]["actual_own_cash"]) for arm in arms]
            ),
            realized_lot_abs_error_distance=mean(
                [abs(q["distance_control"]["expected_cash"] - arms[arm]["actual_own_cash"]) for arm in arms]
            ),
        )
        pairs.append(row)
        groups[(r["seed"], r["seat"], r["t"], r["uid"])].append(row)
    cases = []
    fields = (
        "predicted_terminal_delta",
        "actual_terminal_delta",
        "predicted10_delta",
        "actual10_delta",
        "outcome_margin_delta",
        "change_against_parent",
        "change_against_uniform_choice",
        "realized_lot_abs_error_model",
        "realized_lot_abs_error_distance",
    )
    for (seed, s, t, uid), rows in groups.items():
        summary = {k: mean([r[k] for r in rows]) for k in fields}
        cases.append(dict(seed=seed, seat=s, at=t, uid=uid, streams=len(rows), **summary))
        maps[seed].append(summary)
    map_summary = {str(seed): {k: mean([r[k] for r in rows]) for k in fields} for seed, rows in maps.items()}
    expected_signs = Counter(
        (sign(r["predicted_terminal_delta"]), sign(r["actual_terminal_delta"])) for r in cases
    )
    out = dict(
        manifest_sha256=sha(HERE / "branch_manifest.json"),
        predictions_sha256=sha(HERE / "branch_predictions.json"),
        lots_sha256=sha(HERE / "lots.py"),
        included_pairs=len(pairs),
        unique_roots=len(cases),
        maps=len(maps),
        all_pairs=pairs,
        by_root=cases,
        by_map=map_summary,
        map_equal_average={k: mean([r[k] for r in map_summary.values()]) for k in fields},
        unique_root_direction_counts=[
            {"predicted": k[0], "realized": k[1], "count": n} for k, n in sorted(expected_signs.items())
        ],
        scope="Historical complete original-examiner continuations for old Sapling-family policies. At one root only the issued own collector order differs. Each root receives one vote after averaging its shared-randomness paired replicates; map summaries then weight distinct roots equally. No proposed action is deployed here. Correctly predicting a delay does not establish added receipts.",
    )
    put(HERE / "branch_transfer.json", out)
    print(
        {
            k: v
            for k, v in out.items()
            if k
            in ("included_pairs", "unique_roots", "maps", "unique_root_direction_counts", "map_equal_average")
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("freeze", "evaluate"))
    a = ap.parse_args()
    if a.mode == "freeze":
        print(
            {
                k: v
                for k, v in freeze().items()
                if k
                in ("matched_registries", "included_pairs", "distinct_source_roots", "distinct_source_maps")
            }
        )
    else:
        evaluate()
