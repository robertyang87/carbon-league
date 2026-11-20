"""Account for all registered elite/control tickets using only signed, executed orders."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
import zipfile

HERE = Path(__file__).resolve().parent
STRATEGY = HERE.parents[1]
PRIOR_PATH = STRATEGY / "development/economy.py"
spec = importlib.util.spec_from_file_location("registered_strategic_cash_ledger", PRIOR_PATH)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
ledger = prior.ledger
check = ledger.require

SEEDS = tuple(range(621758520, 621758524))
ARMS = ("control", "elite")
PACKAGES = {
    "elite": "5e4f5da3fd63962ed559ba9efdeb788f454e3f6fb9ce617a3b5c877080795cb2",
    "control": "ede48ad9770d357cbe4a4be6c24dafebc910be1e46a95c508c706e009fd713a0",
    "opponent": "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5",
}
TREE = ("ground_rent", "cargo_tax", "plant", "seize")
QUESTION_SHA = "31f144b12eedd3c94e3e7c573db69f382aeff3da2230100a6840369950e439c9"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(g):
    if not g["counted"]:
        return "unresolved"
    if g["winner_seat"] is None:
        return "draw"
    return "win" if g["winner_seat"] == g["seat"] else "loss"


def finances(game):
    """Learner and examiner have separate own delivery and tree cash accounts."""
    if game["official_margin"] is None:
        return None
    s = game["seat"]
    metrics = {}
    for name, sides in [
        ("official_score", game["official_scores_by_seat"]),
        ("tree_operating_cash", [sum(a.get(k, 0) for k in TREE) for a in game["signed_account_by_seat"]]),
        *((k, [a.get(k, 0) for a in game["signed_account_by_seat"]]) for k in ledger.KIND)
    ]:
        own, examiner = sides[s], sides[1-s]
        metrics[name] = dict(own=own, examiner=examiner, net=own-examiner)
    for role in ("own", "examiner", "net"):
        ledger.check_equal(sum(metrics[name][role] for name in ledger.KIND), metrics["official_score"][role],
                           "Cash sources and official scoring adjustment")
    return metrics


def minus(upper, lower):
    return {key: {role: v-lower[key][role] for role, v in values.items()} for key, values in upper.items()}


def aggregate(rows):
    valid = [r["delta"] for r in rows if r["delta"] is not None]
    means = None
    if len(valid) == len(rows) and valid:
        means = {key: {role: mean(v[key][role] for v in valid) for role in roles} for key, roles in valid[0].items()}
    return dict(planned_map_seats=len(rows), numeric_map_seats=len(valid), elite_minus_control=means)


def pivot(game):
    if not game["transitions"]:
        return None
    sign = "gain" if outcome(game) == "win" else "loss"
    start, end = ledger.swing_windows(game["cash_margin_by_state"])[sign + "_1"]
    before = {n: ledger.profile(game["seat"], game["transitions"], max(0, start-n), start) for n in (3, 5, 10)}
    return dict(turn=start, margin_t=game["cash_margin_by_state"][start], margin_after=game["cash_margin_by_state"][end],
                prior_windows=before, observed_turns=game["transitions"][max(0, start-10):end])


def case_pair(games, pairs, arm_metrics, desired):
    eligible = [g for (_,_,arm), g in games.items() if arm == "elite" and
                g["official_margin"] is not None and outcome(g) == desired]
    if not eligible:
        return None
    # The largest negative margin is the closest loss. For wins take the largest positive margin.
    g = min(eligible, key=lambda r: (-r["official_margin"], r["seed"], r["seat"]))
    seed, seat = g["seed"], g["seat"]
    return dict(seed=seed, seat=seat, candidate_outcome=desired, candidate_margin=g["official_margin"],
                control_margin=games[seed, seat, "control"]["official_margin"],
                comparison=pairs[seed, seat], levels={a: arm_metrics[seed, seat, a] for a in ARMS},
                candidate_observed_pivot=pivot(g))


def read_matches(doc, plan):
    cards = {(g["seed"], g["seat"], g["arm"]): g for g in doc["games"]}
    order = {(s, seat, a) for s in SEEDS for seat in (0, 1) for a in ARMS}
    check(set(cards) == order and len(cards) == len(doc["games"]) == 16, "Registered game denominator changed")
    cfg = ledger.config()
    games = {}
    blocks = set()
    for b in doc["archives"]:
        seed, seat = b["seed"], b["seat"]
        check(seed in SEEDS and seat in (0, 1) and (seed, seat) not in blocks, "Unexpected or duplicate host ticket")
        blocks.add((seed, seat))
        path = HERE / "hosted" / (str(b["id"]) + ".zip")
        check(digest(path) == b["sha256"], "Archive digest changed")
        with zipfile.ZipFile(path) as z:
            for arm in ARMS:
                key = seed, seat, arm
                card = cards[key]
                if not card.get("receipt_sha256"):
                    continue
                reg = json.loads(z.read(arm + "/registration.json"))
                terrain = plan["physical_boards"][str(seed)]
                check((reg["seed"], reg["candidate_seat"], reg["arm"], reg["opponent_seat"]) == (*key, 1-seat)
                      and reg["matrix_sha256"] == doc["protocol_sha256"] and int(reg["workflow_run"]) == doc["workflow_run"]
                      and reg["physical_map_sha256"] == terrain == card["terrain"], "Wrong signed game or physical board")
                check(reg["files"]["candidate"]["sha256"] == PACKAGES[arm] and
                      reg["files"]["opponent"]["sha256"] == PACKAGES["opponent"], "Archive package identity differs")
                games[key] = ledger.replay(card, z, cfg)
    for key in order - games.keys():
        card = cards[key]
        check(not doc["records_reconciled"] and card.get("official_scores_by_seat") is None and
              not card.get("receipt_sha256"), "Dropped a signed or scored game")
        games[key] = dict(seed=key[0], seat=key[1], arm=key[2], **{k:card.get(k) for k in
                          ("status","counted","winner_seat","reason","official_scores_by_seat")},
                          terrain=plan["physical_boards"][str(key[0])], official_margin=None,
                          cash_margin_by_state=[], transitions=[], executed_transitions=None)
    return cards, games


def main():
    path = HERE / "hosted/settlement.json"
    doc = json.loads(path.read_bytes())
    plan_path = HERE / "protocol.json"
    plan = json.loads(plan_path.read_bytes())
    check(doc["protocol_sha256"] == digest(plan_path) and digest(HERE / "economy_question.md") == QUESTION_SHA,
          "Current protocol or economics definitions differ")
    check(tuple(plan["arms"]) == ARMS and tuple(plan["seeds"]) == SEEDS and
          doc["registered_games"] == plan["expected_games"] == 16 and len(set(plan["physical_boards"].values())) == 4,
          "Incorrect map, arm or denominator")
    for key, expected in PACKAGES.items():
        f = plan["files"][key]
        check(expected == doc["packages"][key] == f["sha256"] == digest(ledger.ROOT / f["path"]),
              "Model or original examiner bytes differ")
    cards, games = read_matches(doc, plan)
    metrics = {k: finances(g) for k, g in games.items()}
    signed = {(p["seed"], p["seat"]): p for p in doc["results"]["paired_map_seats"]}
    pairs = {}
    for seed in SEEDS:
        for seat in (0, 1):
            e, c = (games[seed, seat, a] for a in ("elite", "control"))
            calc = ledger.comparison(e,c)
            delta = minus(metrics[seed, seat, "elite"], metrics[seed, seat, "control"]) if calc["numeric"] else None
            official = signed[seed, seat]["elite_minus_control_margin"]
            if delta is not None:
                ledger.check_equal(delta["official_score"]["net"], official, "Official ticket contrast")
                ledger.check_equal(calc["official_margin_delta"], official, "Both sides' official contrast")
            else:
                check(official is None, "Numeric official score missing from economic ledger")
            pairs[seed, seat] = dict(seed=seed, seat=seat, terrain=e["terrain"], statuses=calc["statuses"],
                                     results={g["arm"]:outcome(g) for g in (e,c)}, delta=delta,
                                     first_order_disagreement_physical_seats=calc.get("first_different_order_by_physical_seat"))
    maps = {s: aggregate([pairs[s, 0], pairs[s, 1]]) for s in SEEDS}
    byseat = {p: aggregate([pairs[s, p] for s in SEEDS]) for p in (0, 1)}
    negative = [s for s,v in maps.items() if v["elite_minus_control"] is not None and
                v["elite_minus_control"]["official_score"]["net"] < 0]
    worst = min(negative, key=lambda s: (maps[s]["elite_minus_control"]["official_score"]["net"], s)) if negative else None
    slim = []
    for k in sorted(games):
        g = games[k]
        row = {f:g.get(f) for f in ("seed","seat","arm","terrain","status","counted","winner_seat","official_margin",
                    "official_scores_by_seat","executed_transitions","action_tape_sha256","physical_cash_by_seat")}
        row.update(outcome=outcome(g), reason=cards[k].get("reason"), receipt_sha256=cards[k].get("receipt_sha256"),
                   amounts=metrics[k], observed_signed_flows=(
                       g.get("signed_account_by_seat") if g.get("executed_transitions") else None))
        slim.append(row)
    report = dict(registered=16, observed_statuses=len(slim), workflow_run=doc["workflow_run"], commit=doc["commit"],
                  hashes=dict(protocol=doc["protocol_sha256"],question=QUESTION_SHA,packages=PACKAGES,
                              settlement=digest(path),analyzer=digest(Path(__file__)),previous_economy=digest(PRIOR_PATH),
                              original_ledger=digest(prior.BASE/"economy.py"),environment=ledger.CFG_HASH),
                  records_reconciled=doc["records_reconciled"], missing_tickets=doc["missing_tickets"],
                  verification_errors=doc["verification_errors"], games=slim, per_ticket=list(pairs.values()),
                  per_map=maps, per_seat=byseat, all=aggregate(list(pairs.values())),
                  strongest_candidate_win=case_pair(games, pairs, metrics, "win"),
                  closest_candidate_loss=case_pair(games, pairs, metrics, "loss"),
                  most_negative_map=(dict(seed=worst, map_average=maps[worst], paired_by_seat=[pairs[worst,p] for p in (0,1)])
                                     if worst is not None else None),
                  semantics=dict(own="competitor in that arm",examiner="original Master3s2 on the same trajectory",
                                 net="own minus examiner", contrast="elite trajectory minus control trajectory",
                                 tree_cash="ground_rent + cargo_tax + plant + seize, signed",
                                 workers="position, cargo, is collector, physical seat, distance to own base",
                                 selected_cases="Only executed joint orders. This attribution is descriptive."))
    out = HERE / "hosted/economy_analysis.json.gz"
    raw = json.dumps(report, allow_nan=False, separators=(",",":"), sort_keys=True).encode()
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(raw, mtime=0)); tmp.replace(out)
    print(out, digest(out))
    print("Whole group", report["all"])


if __name__ == "__main__":
    main()
