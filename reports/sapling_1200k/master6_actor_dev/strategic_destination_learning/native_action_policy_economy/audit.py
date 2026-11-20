"""Measure a candidate's exact one-step economic outcomes on held Master3s2 action tapes."""

import argparse
import copy
import gzip
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
STRATEGIC = HERE.parent
WINDOWS = STRATEGIC / "policy_window_returns"
SAPLING = STRATEGIC.parents[1]
INTAKE = (
    STRATEGIC.parent
    / "cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake"
)
sys.path.insert(0, str(WINDOWS / "runner"))
import run as source_run
import verify as source_verify

sys.path.insert(0, str(INTAKE))
from physical_support import digest, order_key, probabilities

spec = importlib.util.spec_from_file_location(
    "economic_coverage", INTAKE / "menu_coverage/coverage.py"
)
economic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(economic)
need = source_run.need
EPS = 1e-7


def load_tapes():
    protocol = json.loads((HERE / "protocol.json").read_bytes())
    manifest = WINDOWS / "hosted/manifest.json"
    sources = STRATEGIC / "sources.json"
    need(
        digest(manifest) == protocol["source_manifest_sha256"]
        and digest(WINDOWS / "protocol.json") == protocol["source_protocol_sha256"]
        and digest(sources) == protocol["strategic_sources_sha256"],
        "Source list changed",
    )
    census = json.loads(sources.read_bytes())["maps"]
    train_terrains = {row["terrain"] for row in census}
    expected = [
        row for row in json.loads(manifest.read_bytes())["tickets"] if row["role"] == "hold"
    ]
    need(len(expected) == len(protocol["games"]) == 3, "Wrong map count")
    for frozen, ticket in zip(protocol["games"], expected, strict=True):
        path = manifest.parent / (str(ticket["artifact"]) + ".zip")
        need(
            frozen["archive_id"] == ticket["artifact"]
            and frozen["archive_sha256"] == digest(path) == ticket["sha256"],
            "Archive mismatch",
        )
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("source.json.gz")
            verification = archive.read("verified.json")
            need(
                source_run.sha(verification) == ticket["verification_receipt_sha256"],
                "Receipt mismatch",
            )
            source = json.loads(gzip.decompress(raw))
            need(
                json.loads(archive.read("summary.json"))["source_sha256"] == source_run.sha(raw),
                "Unbound archive member",
            )
        need(
            source["actors"]["original"] == protocol["examiner_sha256"]
            and source["actors"]["bc"] == protocol["parent_sha256"]
            and all(frozen[k] == source[k] for k in ("seed", "seat", "terrain"))
            and frozen["terrain"] not in train_terrains
            and source["kind"] == "hold",
            "Source identity mismatch",
        )
        yield source


def support(actor, obs, cfg):
    sent = actor.act(obs, cfg)
    info = actor.mod._LAST_POLICY
    options = info["orders"]
    weights = probabilities(
        [r["action"] for r in options], [float(r["probability"]) for r in options]
    )
    prior = probabilities(
        [r["action"] for r in options], [float(r["prior_probability"]) for r in options]
    )
    sampled = order_key(sent)
    need(
        sampled == order_key(info["issued"])
        and sampled in weights
        and abs(math.log(weights[sampled]) - info["logp"]) < 1e-6,
        "Issued action and logp differ",
    )
    _, bases, workers, _ = obs["players"][obs["player"]]
    for order in weights:
        for uid, move in order:
            legal = (
                ("RECCOLLECTOR", "RECPLANTER")
                if uid in bases
                else (("UP", "DOWN", "LEFT", "RIGHT") if uid in workers else ())
            )
            need(move in legal, f"Invalid physical order {uid} {move}")
    return weights, prior, order_key(info["parent"])


def transition(env, state, pair, seat):
    counters = source_run.idgen._id_dict
    original_counts = dict(counters)
    try:
        fork = copy.copy(env)
        fork.states = copy.deepcopy(env.states)
        fork.steps = env.steps[:-1] + [fork.states]
        fork.logs = []
        fork.error = env.error.copy()
        fork.step(pair, [])
        sim = economic.FastCarbon(env.configuration).set_state(state)
        outcome = economic.account(sim, pair, state)
        economic.check_board(sim, fork)
        own, enemy = [outcome["by_seat"][p] for p in (seat, 1 - seat)]
        net = {k: own[k] - enemy[k] for k in economic.KEYS}
        outcome["domain"] = {
            "bank": net["warehouse_deposit"],
            "cargo": net["own_root_load_still_alive"],
            "intercept": net["opponent_cargo_looted"],
            "trees": net["plants"] + 2 * net["seizures"] - net["tree_expiries"],
        }
        outcome["board_after"] = source_run.board_id(fork)
        outcome["terminal_scores"] = [s.reward for s in fork.states] if fork.done else None
        return outcome
    finally:
        counters.clear()
        counters.update(original_counts)


def recent_actions(tape, t):
    return {
        lag: dict(
            step=t - lag,
            team_and_enemy=tape[t - lag]["orders"],
            visible_players=tape[t - lag]["public"]["players"],
        )
        for lag in (10, 5, 3, 0)
        if t >= lag
    }


def game(source, candidate):
    cfg = source["configuration"]
    seat = source["seat"]
    env = source_run.collect.begin(source["seed"], cfg)
    sim = economic.FastCarbon(env.configuration).reset(seed=source["seed"])
    actor = source_run.collect.Player(candidate)
    public_cfg = {k: v for k, v in cfg.items() if k != "randomSeed"}
    tape = source["rows"]
    rows = []
    for t, recorded in enumerate(tape):
        view = source_run.public(env, seat, recorded["bank_before"])
        need(
            recorded["board_before"] == source_run.board_id(env)
            and source_run.digest(view) == source_run.digest(recorded["public"]),
            "Source prefix mismatch",
        )
        pair = recorded["orders"]
        old = order_key(pair[seat])
        weights, prior, reported_parent = support(actor, view, public_cfg)
        need(
            reported_parent == old and weights.get(old, 0.0) >= 0.5 - EPS,
            "Archived parent command or probability floor violated",
        )
        delta = {k: 0.0 for k in ("bank", "cargo", "intercept", "trees")}
        evidence = dict(
            step=t,
            board_before=recorded["board_before"],
            enemy_order=pair[1 - seat],
            parent_order=pair[seat],
            options=[
                dict(action=dict(k), probability=v, prior_probability=prior[k])
                for k, v in weights.items()
            ],
            changed_mass=sum(p for k, p in weights.items() if k != old),
            expected_delta=delta,
            learned_minus_prior=dict.fromkeys(delta, 0.0),
        )
        if set(weights) != {old}:
            economic.check_board(sim, env)
            snapshot = sim.get_state()
            outcomes = {}
            for key in dict.fromkeys([old, *weights]):
                changed = pair.copy()
                changed[seat] = dict(key)
                outcomes[key] = transition(env, snapshot, changed, seat)
            need(
                outcomes[old]["board_after"] == recorded["board_after"],
                "Replayed parent transition mismatch",
            )
            evidence["parent_outcome"] = outcomes[old]
            for option in evidence["options"]:
                option["outcome"] = outcomes[order_key(option["action"])]
            for field in delta:
                delta[field] = sum(
                    p * (outcomes[k]["domain"][field] - outcomes[old]["domain"][field])
                    for k, p in weights.items()
                )
                evidence["learned_minus_prior"][field] = sum(
                    (p - prior[k]) * (outcomes[k]["domain"][field] - outcomes[old]["domain"][field])
                    for k, p in weights.items()
                )
            evidence["expected_by_seat_delta"] = [
                {
                    key: sum(
                        p
                        * (outcomes[k]["by_seat"][side][key] - outcomes[old]["by_seat"][side][key])
                        for k, p in weights.items()
                    )
                    for key in economic.KEYS
                }
                for side in (0, 1)
            ]
        env.step(pair)
        sim.step(*pair)
        need(source_run.board_id(env) == recorded["board_after"], "Wrong next source state")
        rows.append(evidence)
    economic.check_board(sim, env)
    need(source_run.terminal(env, seat) == source["terminal"], "Source end mismatch")
    actor.diagnostics()
    details = []
    for field in ("bank", "cargo", "intercept", "trees"):
        for sign in (-1, 1):
            row = max(rows, key=lambda r: sign * r["expected_delta"][field])
            if sign * row["expected_delta"][field] > EPS:
                details.append(
                    dict(
                        field=field,
                        direction=sign,
                        step=row["step"],
                        delta=row["expected_delta"][field],
                        history=recent_actions(tape, row["step"]),
                    )
                )
    return dict(
        seed=source["seed"], terrain=source["terrain"], seat=seat, rows=rows, examples=details
    )


def describe(game):
    rows = game["rows"]
    changed = [r for r in rows if r["changed_mass"] > 0]
    domains = ("bank", "cargo", "intercept", "trees")
    useful = [
        r
        for r in changed
        if any(
            abs(option["outcome"]["domain"][field] - r["parent_outcome"]["domain"][field]) > EPS
            for option in r["options"]
            for field in domains
        )
    ]
    domain_mass = {}
    for field in domains:
        positive = negative = 0.0
        roots_with_gain = roots_with_loss = 0
        for row in changed:
            base = row["parent_outcome"]["domain"][field]
            gains = sum(
                o["probability"]
                for o in row["options"]
                if o["outcome"]["domain"][field] > base + EPS
            )
            losses = sum(
                o["probability"]
                for o in row["options"]
                if o["outcome"]["domain"][field] < base - EPS
            )
            roots_with_gain += int(gains > 0)
            roots_with_loss += int(losses > 0)
            positive += gains
            negative += losses
        domain_mass[field] = dict(
            roots_with_gain=roots_with_gain,
            roots_with_loss=roots_with_loss,
            expected_favorable_turns=positive,
            expected_adverse_turns=negative,
        )
    return dict(
        seed=game["seed"],
        seat=game["seat"],
        terrain=game["terrain"],
        turns=len(rows),
        changed_support=len(changed),
        economically_distinguishable=len(useful),
        economic_opportunities=domain_mass,
        mean_changed_mass=statistics.mean(r["changed_mass"] for r in rows),
        probabilities_evaluated=sum(len(r["options"]) for r in rows),
        simulated_branches_including_baseline=sum(
            len({order_key(r["parent_order"]), *(order_key(o["action"]) for o in r["options"])})
            for r in changed
        ),
        mean_delta={k: statistics.mean(r["expected_delta"][k] for r in rows) for k in domains},
        mean_learned_minus_prior={
            k: statistics.mean(r["learned_minus_prior"][k] for r in rows) for k in domains
        },
        expected_components_by_physical_seat=[
            {
                k: sum(r["expected_by_seat_delta"][seat][k] for r in changed) / len(rows)
                for k in economic.KEYS
            }
            for seat in (0, 1)
        ],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path)
    p.add_argument("--sha")
    args = p.parse_args()
    if args.candidate is None:
        for source in load_tapes():
            source_verify.replay(source, source["rows"], source["terminal"])
            print(
                source["seed"],
                source["seat"],
                len(source["rows"]),
                source["terminal"]["scores"],
                flush=True,
            )
        return
    package = args.candidate.read_bytes()
    need(args.sha and source_run.sha(package) == args.sha, "Candidate SHA does not match")
    with tempfile.TemporaryDirectory() as scratch:
        frozen = Path(scratch) / "candidate.py"
        frozen.write_bytes(package)
        games = [game(source, frozen) for source in load_tapes()]
    summaries = [describe(g) for g in games]
    passing_domains = [
        k
        for k in summaries[0]["mean_delta"]
        if sum(m["mean_delta"][k] > EPS for m in summaries) >= 2
        and statistics.mean(m["mean_delta"][k] for m in summaries) > EPS
    ]
    ok = sum(g["economically_distinguishable"] > 0 for g in summaries) >= 2 and bool(
        passing_domains
    )
    source_run.save(HERE / "steps.json.gz", games)
    result = dict(
        candidate_sha256=args.sha,
        protocol_sha256=digest(HERE / "protocol.json"),
        program_sha256=digest(Path(__file__)),
        steps_sha256=digest(HERE / "steps.json.gz"),
        games=summaries,
        passing_domains=passing_domains,
        warrant_development=ok,
        new_original_games=0,
    )
    source_run.save(HERE / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
