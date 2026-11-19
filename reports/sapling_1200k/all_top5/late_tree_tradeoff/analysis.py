"""Audit realized late planting and courier opportunities without sampling new actions."""

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ALL = HERE.parent
BASE = ALL.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ALL))
sys.path.insert(0, str(BASE))
import plant_timing_extract as plant
from economy_extract import distances, reconstruct
from income_target import reconstruct as native_income

REV = "41e4177e"
BATCH = ALL / ("appendix_" + REV)
MASTER = "ecde65e596acbeb9ba5feca22e05f100"
CACHE = Path("/workspace/sapling-all-top5-20261119")


def read(p):
    b = p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix == ".gz" else b)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def result(game, seat):
    x = game["score_by_seat"][seat] - game["score_by_seat"][1 - seat]
    return "win" if x > 0 else "loss" if x < 0 else "draw"


def sample():
    batch = read(BATCH / "manifest.json")
    archives = {rev[:8] for rev in batch["baseline_revisions"][1:]}
    paths = [
        ALL / "manifest.json",
        *[ALL / ("appendix_" + rev) / "manifest.json" for rev in sorted(archives)],
    ]
    old = {}
    for path in paths:
        for g in read(path)["games"]:
            if g["id"] in old:
                raise ValueError("duplicate historical map")
            old[g["id"]] = g
    if len(old) != batch["baseline_games"]:
        raise ValueError("missing predecessor archive")
    historical = [g for g in old.values() if MASTER in g["seat_strategy_ids"]]
    latest = [g for g in batch["games"] if MASTER in g["seat_strategy_ids"]]

    def pair(g):
        s = g["seat_strategy_ids"].index(MASTER)
        return g["seat_names"][1 - s], s

    groups = defaultdict(list)
    for g in historical:
        groups[pair(g)].append(g)
    hist_loss = [g for g in historical if result(g, pair(g)[1]) == "loss"]
    controls = {}
    for key in {pair(g) for g in hist_loss}:
        wins = sorted(
            (g for g in groups[key] if result(g, key[1]) == "win"),
            key=lambda g: g["id"],
        )
        if wins:
            controls[key] = wins[0]
    source = (
        [(g, "new") for g in latest]
        + [(g, "earlier_loss") for g in hist_loss]
        + [(g, "earlier_win_control") for _, g in sorted(controls.items())]
    )
    receipts = {}
    for g, cohort in source:
        name, seat = pair(g)
        prior = groups[name, seat]
        if g["id"] in receipts:
            raise ValueError("reused map")
        receipts[g["id"]] = dict(
            cohort=cohort,
            opponent=name,
            seat=seat,
            official=result(g, seat),
            previous_same_opponent_any_seat=sum(len(groups[name, k]) for k in (0, 1)),
            previous_same_opponent_seat=len(prior),
            previous_same_opponent_seat_losses=sum(
                result(r, seat) == "loss" for r in prior
            ),
        )
    return (
        source,
        receipts,
        dict(
            predecessor_all_maps=len(old),
            earlier_master6=len(historical),
            new_master6=len(latest),
            history_loss_cells=len({pair(g) for g in hist_loss}),
            history_matched_win_cells=len(controls),
            previous_master6_wdl=dict(
                Counter(result(g, pair(g)[1]) for g in historical)
            ),
        ),
        paths,
    )


def game_rows(game, info, new_cash, *, replay_bytes=None):
    blob = (CACHE / game["path"]).read_bytes() if replay_bytes is None else replay_bytes
    if sha(blob) != game["sha256"]:
        raise ValueError("original replay identity changed")
    env = json.loads(gzip.decompress(blob))
    frames = env["steps"]
    cfg = env["configuration"]
    seat = info["seat"]
    sid = game["id"]
    if env["rewards"] != game["score_by_seat"] or env["rewards"] != [
        s["reward"] for s in frames[-1]
    ]:
        raise ValueError("terminal reward mismatch")
    cache = CACHE / "plant_work" / (sid + ".gz")
    if cache.exists():
        d = read(cache)
        if (
            d["replay_sha256"] != game["sha256"]
            or d["extractor_sha256"] != plant.version()
        ):
            raise ValueError("stale purchase ledger")
    else:
        d = plant.extract(blob)
    if d["turns"] != len(frames) - 1 or d["rewards"] != env["rewards"]:
        raise ValueError("history identity mismatch")
    if sid in new_cash:
        cash = new_cash[sid]
        if cash["replay_sha256"] != game["sha256"]:
            raise ValueError("cash receipt mismatch")
        ledger = [c[2] for c in cash["turns"]]
        bank = [c[3] for c in cash["turns"]]
    else:
        ledger, _, bank, _ = reconstruct(frames, cfg)
    if len(bank) != len(frames) - 1:
        raise ValueError("cash ledger length")
    rows = []
    dist = distances(cfg["size"])
    for lc in d["choices"]:
        c = dict(zip(plant.CHOICE, lc))
        t = c["root"]
        if (
            c["seat"] != seat
            or c["other_cash_financed"]
            or not (1 <= c["remaining_turns"] <= 50)
        ):
            continue
        period = c["remaining_turns"]
        h = (
            min(period, max(0, cfg["treeLifespan"] - c["old_tree_age"]))
            if c["kind"] == "seize"
            else min(period - 1, cfg["treeLifespan"])
        )
        estimate = c["neighbor_carbon"] * cfg["cellAbsorptionRate"] * h
        own = frames[t][0]["observation"]["players"][seat]
        cell = c["cell"]
        routes = []
        for uid, (pos, load, role) in own[2].items():
            home = min(dist[pos][v] for v in own[1].values())
            if (
                role == "COLLECTOR"
                and load >= 100
                and home <= 4
                and dist[pos][cell] == 1
                and min(dist[cell][v] for v in own[1].values()) < home
            ):
                routes.append((-load, uid))
        paid = c["paid"] is not None
        if paid:
            p = dict(zip(plant.PURCHASE, d["purchases"][c["paid"]]))
            if (p["seat"], p["worker"], p["root"], p["cell"], p["kind"]) != (
                seat,
                c["worker"],
                t,
                cell,
                c["kind"],
            ):
                raise ValueError("purchase keyed to another root")
            net = p["absorption"] + p["collision_cash"] - p["price"]
            economics = dict(
                price=p["price"],
                gross=round(p["absorption"] + p["collision_cash"], 5),
                enemy_cargo_absorption=p["enemy_worker_absorption"],
                collision_cash=p["collision_cash"],
                net=round(net, 5),
                paid_back=p["payback_transitions"] is not None,
                end=p["end_reason"],
            )
        else:
            economics = None
        corridor = None
        if routes:
            neg, uid = min(routes)
            post = frames[t + 1][0]["observation"]["players"][seat]
            following = [
                (i + t + 1, round(amt, 4))
                for i, step in enumerate(bank[t : t + 5])
                for s, worker, amt in step
                if s == seat and worker == uid
            ]
            corridor = dict(
                uid=uid,
                cargo=-neg,
                entered_source_cell=uid in post[2] and post[2][uid][0] == cell,
                paid_in_next_5=following,
            )
        rows.append(
            dict(
                id=sid,
                seat=seat,
                t=t,
                worker=c["worker"],
                cell=cell,
                kind=c["kind"],
                remaining=period,
                quoted_price=c["price"],
                passive=round(estimate, 5),
                below=estimate < c["price"],
                decision="paid" if paid else "moved" if c["command"] else "held_unpaid",
                purchase=economics,
                corridor=corridor,
            )
        )
    phase = {}
    for side in (seat, 1 - seat):
        v = [sum(step[side][k] for step in ledger[-30:]) for k in range(8)]
        phase["master" if side == seat else "opponent"] = dict(
            bank=round(v[0], 2),
            absorb=round(v[2], 2),
            plant_seize=round(v[5] + v[6], 2),
        )
    return rows, dict(
        seed=game["seed"],
        sha256=game["sha256"],
        official_margin=round(env["rewards"][seat] - env["rewards"][1 - seat], 2),
        terminal_t=len(frames) - 1,
        end30=phase,
        last30_cash_delta=[
            (
                round(
                    frames[-1][0]["observation"]["players"][s][0]
                    - frames[-31][0]["observation"]["players"][s][0],
                    2,
                )
                if len(frames) > 30
                else None
            )
            for s in (0, 1)
        ],
        **info
    )


def summary(rows):
    p = [x["purchase"] for x in rows if x["purchase"] is not None]
    return dict(
        opportunities=len(rows),
        opportunity_maps=len({x["id"] for x in rows}),
        moves=sum(x["decision"] == "moved" for x in rows),
        paid=len(p),
        paid_maps=len({r["id"] for r in rows if r["purchase"]}),
        repaid=sum(v["paid_back"] for v in p),
        net=round(sum(v["net"] for v in p), 3),
        avg_net=round(sum(v["net"] for v in p) / len(p), 3) if p else None,
    )


def main():
    chosen, info, design, paths = sample()
    new_cash = {g["id"]: g for g in read(BATCH / "turn_cash.json.gz")["games"]}
    rows = []
    games = {}
    for g, _ in chosen:
        a, b = game_rows(g, info[g["id"]], new_cash)
        rows.extend(a)
        games[g["id"]] = b
    rows.sort(key=lambda x: (x["id"], x["t"], x["worker"]))

    def collect(r):
        i = games[r["id"]]
        return [i["cohort"]] + (
            [
                (
                    "new_unseen_opponent"
                    if i["previous_same_opponent_any_seat"] == 0
                    else "new_seen_opponent"
                )
            ]
            if i["cohort"] == "new"
            else []
        )

    categories = defaultdict(list)
    corridors = defaultdict(list)
    for r in rows:
        for k in collect(r):
            for stage in [
                "1–50",
                (
                    "1–10"
                    if r["remaining"] <= 10
                    else "11–30" if r["remaining"] <= 30 else "31–50"
                ),
            ]:
                categories[k, r["kind"], stage, r["below"]].append(r)
            if r["corridor"]:
                corridors[k, r["decision"]].append(r)
    cube = [
        dict(cohort=k, kind=a, period=b, below=c, **summary(rr))
        for (k, a, b, c), rr in sorted(categories.items())
    ]
    route = []
    for (cohort, decision), items in sorted(corridors.items()):
        entered = [r for r in items if r["corridor"]["entered_source_cell"]]
        paid = sum(bool(r["corridor"]["paid_in_next_5"]) for r in items)
        route.append(
            dict(
                cohort=cohort,
                decision=decision,
                roots=len(items),
                maps=len({r["id"] for r in items}),
                distinct_courier_map_pairs=len(
                    {(r["id"], r["corridor"]["uid"]) for r in items}
                ),
                arrived_on_vacated_cell=len(entered),
                own_courier_paid_in_5=paid,
                entered_and_paid_in_5=sum(
                    bool(r["corridor"]["paid_in_next_5"]) for r in entered
                ),
            )
        )
    native = {}
    parent = BASE / "master6_actor_dev/terminal_return_fit"
    dev = read(parent / "development.json")
    for family, dirname in [("parent", "dev_parent"), ("candidate", "dev_learned")]:
        g = read(parent / dirname / "results.json")["games"]
        native[family] = []
        for r in sorted(g, key=lambda rr: rr["candidate_seat"]):
            seat = r["candidate_seat"]
            rawpath = parent / dirname / r["raw_file"]
            ledger = native_income(r, read(rawpath))
            if (
                sha(rawpath.read_bytes()) != dev[family][seat]["raw_sha256"]
                or abs(r["margin"] - dev[family][seat]["margin"]) > 1e-6
            ):
                raise ValueError("native diagnostic differs")
            cash = ledger["cash_entries_by_seat"]
            native[family].append(
                dict(
                    seat=seat,
                    official_margin=round(r["margin"], 2),
                    raw=r["raw_sha256"],
                    cash_entries_by_seat=cash,
                )
            )
    out = dict(
        design=design,
        source_sha256={
            str(p.relative_to(ROOT)): sha(p.read_bytes())
            for p in [
                *paths,
                BATCH / "manifest.json",
                BATCH / "turn_cash.json.gz",
                BATCH / "master6_pressure.json.gz",
                Path(plant.__file__),
                HERE / "SCOPE.md",
            ]
        },
        games=games,
        cube=cube,
        corridors=route,
        native=native,
    )
    (HERE / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    data = json.dumps(
        rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()
    (HERE / "opportunities.json.gz").write_bytes(gzip.compress(data, mtime=0))
    print(design, len(rows), len(games), flush=True)


if __name__ == "__main__":
    main()
