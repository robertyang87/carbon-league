"""Apply the unchanged study condition to six recorded Master3s2 games and count available decisions."""

import gzip
import json
from analysis import BASE, HERE, ROOT, game_rows, read, sha, distances
from economy_extract import FastCarbon

RUNS = (
    "terminal_return_fit/dev_parent",
    "terminal_return_fit/dev_learned",
    "freight_deployment/first_response/transfer_parent",
)


def normalize(raw):
    cfg = raw["game_configuration"]
    sim = FastCarbon(cfg).reset(seed=raw["seed"])
    if list(sim.carbon) != raw["initial_full_carbon"]:
        raise ValueError("opening mismatch")

    def frame(actions):
        item = [
            dict(observation=sim.obs(s), action=actions[s], reward=None) for s in (0, 1)
        ]
        if not sim.step_count:
            item[0]["observation"]["full_carbon"] = list(sim.carbon)
        return item

    frames = [frame([{}, {}])]
    for acts in raw["actions"]:
        sim.step(*acts)
        frames.append(frame(acts))
    for s in (0, 1):
        frames[-1][s]["reward"] = raw["official_score_by_seat"][s]
    if list(sim.cash) != raw["last_post_cash"]:
        raise ValueError("terminal mismatch")
    return (
        dict(configuration=cfg, steps=frames, rewards=raw["official_score_by_seat"]),
        sim,
    )


def exits(row, frames, sim):
    t, seat, uid = row["t"], row["seat"], row["worker"]
    obs = frames[t][0]["observation"]
    own, enemy = obs["players"][seat], obs["players"][1 - seat]
    joint = frames[t + 1][seat]["action"]
    neighbors = sim._move_nbr

    def target(pos, move):
        return neighbors[move][pos] if move in neighbors else pos

    # Conservative same-side feasibility under untouched downstream unit orders.
    blocked = {target(w[0], joint.get(k)) for k, w in own[2].items() if k != uid}
    blocked.update(own[1].values())
    blocked.update(enemy[1].values())
    distance = distances(sim.size)
    cell = own[2][uid][0]
    candidate = []
    for move, nbs in neighbors.items():
        dest = nbs[cell]
        if dest in blocked:
            continue
        threat = min(
            (distance[dest][w[0]] for w in enemy[2].values() if w[2] == "COLLECTOR"),
            default=99,
        )
        candidate.append(dict(move=move, landing=dest, min_enemy_collector_dist=threat))
    return candidate


def main():
    allrows = []
    games = []
    for name in RUNS:
        folder = BASE / "master6_actor_dev" / name
        report = read(folder / "results.json")
        proofpath = folder / (
            "official_check.json"
            if name.startswith("freight")
            else "official_replay.json"
        )
        proof = read(proofpath)
        truth = (
            proof["games"] if "games" in proof else proof["runs"][folder.name]["games"]
        )
        if report["status"] != "complete" or sorted(r["seat"] for r in truth) != [0, 1]:
            raise ValueError("uncertified arm")
        for r in sorted(report["games"], key=lambda v: v["candidate_seat"]):
            seat = r["candidate_seat"]
            path = folder / r["raw_file"]
            raw = read(path)
            official = next(x for x in truth if x["seat"] == seat)
            if (
                sha(path.read_bytes()) != r["raw_sha256"]
                or r["raw_sha256"] != official["transcript_sha256"]
                or raw["official_score_by_seat"] != official["official_reward_by_seat"]
            ):
                raise ValueError("original transcript/score failed verification")
            env, sim = normalize(raw)
            memory = gzip.compress(
                json.dumps(env, separators=(",", ":")).encode(), mtime=0
            )
            key = name.replace("/", "_") + "_" + str(seat)
            rows, _ = game_rows(
                dict(
                    id=key,
                    sha256=sha(memory),
                    seed=r["seed"],
                    score_by_seat=raw["official_score_by_seat"],
                ),
                dict(seat=seat),
                {},
                replay_bytes=memory,
            )
            for row in rows:
                if row["below"] or row["corridor"]:
                    row["own_slot_distinct_exits"] = exits(row, env["steps"], sim)
            low = [x for x in rows if x["below"]]
            paid = [x for x in low if x["purchase"]]
            net = [x["purchase"]["net"] for x in paid]
            games.append(
                dict(
                    name=name,
                    seed=r["seed"],
                    seat=seat,
                    raw=str(path.relative_to(ROOT)),
                    sha256=r["raw_sha256"],
                    official_margin=round(r["margin"], 2),
                    actor_sha256=report["header"]["pack_sha256"],
                    examiner_md5=report["header"]["examiner_md5"],
                    actual_trees_purchased_all_late=sum(
                        bool(x["purchase"]) for x in rows
                    ),
                    total_opportunities=len(rows),
                    below_affordable_roots=len(low),
                    below_paid=len(paid),
                    below_paid_seize=sum(x["kind"] == "seize" for x in paid),
                    below_paid_repaid=sum(x["purchase"]["paid_back"] for x in paid),
                    below_paid_gross_cost=round(
                        sum(x["purchase"]["price"] for x in paid), 3
                    ),
                    below_paid_actual_gross=round(
                        sum(x["purchase"]["gross"] for x in paid), 3
                    ),
                    below_paid_cash_net=round(sum(net), 3),
                    perfect_hindsight_first_spell_saving=round(
                        -sum(min(x, 0) for x in net), 3
                    ),
                    below_paid_with_distinct_exit=sum(
                        bool(x["own_slot_distinct_exits"]) for x in paid
                    ),
                    below_paid_with_distinct_unthreatened_exit=sum(
                        any(
                            v["min_enemy_collector_dist"] > 1
                            for v in x["own_slot_distinct_exits"]
                        )
                        for x in paid
                    ),
                    corridor_roots=sum(bool(x["corridor"]) for x in rows),
                    below_paid_and_corridor=sum(bool(x["corridor"]) for x in paid),
                    corridor_with_unthreatened_distinct_exit=sum(
                        bool(x["corridor"])
                        and any(
                            v["min_enemy_collector_dist"] > 1
                            for v in x["own_slot_distinct_exits"]
                        )
                        for x in rows
                    ),
                )
            )
            allrows.extend(rows)
    (HERE / "native_opportunities.json.gz").write_bytes(
        gzip.compress(
            json.dumps(allrows, separators=(",", ":"), allow_nan=False).encode(),
            mtime=0,
        )
    )
    (HERE / "native.json").write_text(
        json.dumps(games, indent=2, allow_nan=False) + "\n"
    )
    print(
        [
            (
                r["name"],
                r["seat"],
                r["below_paid"],
                r["below_paid_cash_net"],
                r["corridor_roots"],
            )
            for r in games
        ]
    )


if __name__ == "__main__":
    main()
