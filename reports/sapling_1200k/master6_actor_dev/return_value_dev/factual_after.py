"""Follow each held-out real v1 courier after a prospective route edit. No simulated alternative tail."""

from collections import Counter
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cashflow_pairs", HERE / "branches.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def track(game, pack, reg):
    seed, seat = game["seed"], game["seat"]
    cache = HERE / f"validation_{seed}_{seat}.json.gz"
    shard = api.read(cache)
    src = api.ROOT / game["raw_path"]
    api.require(
        shard["identity"]["candidate"] == pack
        and shard["identity"]["evaluator"] == api.sha(HERE / "validate.py")
        and shard["identity"]["source"] == game["raw_sha256"] == api.sha(src),
        "validation identity changed",
    )
    raw = api.read(src)
    raw["initial_terrain_u8_sha256"] = game["map_sha256"]
    actual = next(g for g in reg["games"] if (g["seat"], g["seed"]) == (seat, seed))
    api.require(
        actual["map_sha256"] == game["map_sha256"] and raw["pack_sha256"] == api.V1, "unmatched original game"
    )
    env = api.begin(raw)
    cfg = raw["game_configuration"]
    api.check_cfg(cfg)
    sim = api.FastCarbon(cfg).reset(seed=seed)
    sim.cash = api.FreightCash(sim.cash)
    roots = shard["rows"]
    keys = {(x["t"], x["uid"]) for x in roots}
    api.require(len(roots) == len(keys) == shard["count"].get("edits", 0), "duplicate or missing edits")
    active = []
    for i, joint in enumerate(raw["actions"]):
        pre = {u: (w.pos, w.carbon, w.player) for u, w in sim.workers.items()}
        for row in roots:
            if i != row["t"]:
                continue
            uid = row["uid"]
            api.equal(pre[uid][1], row["cargo"], "initial principal")
            api.require(joint[seat].get(uid, "NONE") == row["old"], "factual action mismatch")
            active.append(
                dict(
                    root_t=i,
                    uid=uid,
                    old=row["old"],
                    alternative=row["alternative"],
                    lot=pre[uid][1],
                    holder=uid,
                    remaining=pre[uid][1],
                    allocation=Counter(),
                    parent_immediate_retained=row["parent"]["retained_root"],
                    alternate_immediate_retained=row["candidate"]["retained_root"],
                    fresh_harvest=0.0,
                    same_worker_gross_receipts=[],
                    root_receipts=[],
                    first_10=[],
                )
            )
        env.step(joint)
        ev = api.add_step(sim, joint, env.states[0].observation.players)
        hits = {u: (s, load, beneficiary, winner) for u, s, load, beneficiary, winner in sim.last_collisions}
        winners = {w.pos: u for u, w in sim.workers.items() if w.is_collector}
        incoming = Counter()

        def landed(uid, s):
            o = joint[s].get(uid)
            return sim._move_nbr[o][pre[uid][0]] if o in sim._move_nbr else pre[uid][0]

        for dead, (side, amt, _, winning) in hits.items():
            if amt and winning is not None:
                incoming[winners[landed(dead, side)]] += amt
        for x in active:
            uid = x["uid"]
            h = x["holder"]
            q = x["remaining"]
            flow = x["allocation"]
            first10 = i < x["root_t"] + 10
            oldq = q
            gross = 0.0
            fresh = 0.0
            tax = 0.0
            if uid in sim.workers:
                gross = sim.cash.deposited.get(uid, 0.0)
                fresh = (
                    sim.workers[uid].carbon
                    + sim.cash.reductions.get(uid, 0.0)
                    + gross
                    - pre[uid][1]
                    - incoming[uid]
                )
                api.require(fresh > -1e-6, "negative fresh harvest")
                if joint[seat].get(uid) in sim._move_nbr:
                    api.equal(fresh, 0.0, "moved and harvested")
                fresh = max(0.0, fresh)
                x["fresh_harvest"] += fresh
                if gross and sim.cash.owner[uid] == seat:
                    x["same_worker_gross_receipts"].append(dict(t=i + 1, amount=gross))
            if q and h in hits:
                side, amt, beneficiary, winner = hits[h]
                api.require(side == seat and q <= amt + 1e-6, "tag on wrong worker")
                if winner == seat:
                    h = winners[landed(h, seat)]
                else:
                    cause = (
                        "our_tree_or_base_collision_cash"
                        if beneficiary == seat
                        else "enemy_collision_transfer" if beneficiary is not None else "ground_or_overflow"
                    )
                    flow[cause] += q
                    if beneficiary == seat:
                        x["root_receipts"].append(dict(t=i + 1, uid=h, kind=cause, amount=q))
                    h, q = None, 0.0
            if q and h in ev["payments"]:
                side, amt = ev["payments"][h]
                api.require(q <= amt + 1e-6, "root tag exceeds settlement")
                cause = "own_base" if side == seat else "enemy_base"
                flow[cause] += q
                x["root_receipts"].append(dict(t=i + 1, uid=h, kind=cause, amount=q, gross=amt))
                h, q = None, 0.0
            if q:
                api.require(h in sim.workers, "unclassified loaded worker death")
                tax = q * (1.0 - ev["ret"].get(h, 1.0))
                flow["enemy_tree_tax"] += tax
                q -= tax
            api.equal(q + sum(flow.values()), x["lot"], "full attributed lot")
            if i == x["root_t"]:
                api.equal(q, x["parent_immediate_retained"], "registered first transition")
            if first10:
                x["first_10"].append(
                    dict(
                        after_step=i + 1,
                        old_order=joint[seat].get(uid, "NONE") if uid in pre else None,
                        worker_position=sim.workers[uid].pos if uid in sim.workers else None,
                        worker_load=sim.workers[uid].carbon if uid in sim.workers else None,
                        root_carrier=h,
                        root_remaining=q,
                        root_tax=tax,
                        worker_new_harvest=fresh,
                        worker_own_base_receipt=gross if sim.cash.owner.get(uid) == seat else 0.0,
                        cash_by_seat=list(sim.cash),
                    )
                )
            if oldq and not q:
                x["root_lot_closed_at"] = i + 1
            x["holder"], x["remaining"] = h, q
    api.require(
        env.done
        and [p.status for p in env.states] == ["DONE", "DONE"]
        and [p.reward for p in env.states] == actual["scores"]
        and list(sim.cash) == shard["final_cash"],
        "archived official terminal changed",
    )
    api.require(len(active) == len(roots), "some enrolled edits not replayed")
    for x in active:
        x["allocation"] = dict(x["allocation"])
        x["receipts_first_10"] = [e for e in x["root_receipts"] if e["t"] <= x["root_t"] + 10]
        x["worker_new_harvest_first_10"] = sum(e["worker_new_harvest"] for e in x["first_10"])
        x["worker_first_own_payment"] = next(iter(x["same_worker_gross_receipts"]), None)
    return dict(
        seed=seed,
        seat=seat,
        map_sha256=game["map_sha256"],
        tape_sha256=api.sha(src),
        validation_trace_sha256=api.sha(cache),
        original_official_scores=actual["scores"],
        steps=sim.step_count,
        selected_edits=len(active),
        edits=active,
    )


def main():
    register = HERE / "validation_registration.json"
    reg = api.read(register)
    pack = api.sha(HERE / "candidate.py")
    api.require(reg["candidate_sha256"] == pack, "changed candidate")
    results = []
    for s in reg["validation"]:
        src = api.ROOT / s["receipt"]
        api.require(api.sha(src) == s["sha256"], "unsealed source receipt")
        receipt = api.read(src)
        for game in s["games"]:
            results.append(track(game, pack, receipt))
    summary = api.read(HERE / "validation.json")
    api.require(
        sum(x["selected_edits"] for x in results) == sum(n.get("edits", 0) for n in summary["maps"].values()),
        "factual count differs from frozen evaluation",
    )
    api.retain(
        HERE / "factual_after.json",
        dict(
            registration_sha256=api.sha(register),
            validation_sha256=api.sha(HERE / "validation.json"),
            candidate_sha256=pack,
            source_code_sha256=api.sha(Path(__file__)),
            games=results,
            scope="All held-out edits on original v1 order trajectories. Record horizon uses first ten real transitions, then follows root-owned cargo through friendly handoffs to its first exit or official terminal. Only actual confirmed payments become treasury; enemy capture or dropping inventory ends attribution. Candidate beyond its first simulated step is unobserved.",
        ),
    )


if __name__ == "__main__":
    main()
