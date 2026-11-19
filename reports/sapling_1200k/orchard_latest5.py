"""Audit the frozen purchase-return model on newly observed, map-disjoint Top Five games."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

from orchard_value import orchard_features, orchard_predict
from turning_point_ledger import CashEvents, FastCarbon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "master_family/deep_review"))
from recover_public import canonical


def digest(b):
    return hashlib.sha256(b).hexdigest()


def score_one(game, models, timeline=False):
    frames = game["steps"]
    board = frames[0][0]["observation"]
    sim = FastCarbon(game["configuration"]).reset(full_carbon=board["full_carbon"], cash=board["players"][0][0])
    sim.cash = CashEvents(sim.cash)
    purchases, active, skipped, missed = [], {}, Counter(), [{}, {}]
    cash_totals = [Counter(), Counter()]
    trace = []
    for turn, post in enumerate(frames[1:]):
        prev = frames[turn]
        board = prev[0]["observation"]
        moves = [frame["action"] or {} for frame in post]
        opportunity = {}
        for seat, player in enumerate(board["players"]):
            carbon = prev[seat]["observation"]["carbon"]
            for worker, (cell, _, role) in player[2].items():
                if role != "PLANTER":
                    continue
                info = orchard_features(board, carbon, seat, cell, turn, game["configuration"])
                if info is None:
                    continue
                kind, price, features = info
                predicted = orchard_predict(models[kind], features)
                opportunity[seat, worker] = (kind, price, predicted)
        sim.cash.events = []
        sim.step(*moves)
        after = post[0]["observation"]
        if list(sim.cash) != [p[0] for p in after["players"]]:
            raise ValueError(f"cash differs from recorded game at step {turn+1}")
        used = set()
        step_cash = [Counter(), Counter()]
        for kind, seat, amount, *details in sim.cash.events:
            cash_totals[seat][kind] += amount
            step_cash[seat][kind] += amount
            if kind in ("plant", "seize"):
                worker, cell = details
                tree, = [tid for tid in sim.player_trees[seat] if sim.trees[tid].pos == cell]
                used.add((seat, worker))
                previous_owner = active.pop((1-seat, tree), None)
                if previous_owner is not None:
                    previous_owner.update(last_owned_state=turn, end_reason="opponent_seizure")
                opp = opportunity.get((seat, worker))
                if opp is None or kind != opp[0]:
                    skipped[kind + "_expiry_or_public_kind_mismatch"] += 1
                    continue
                _, price, pred = opp
                row = {"turn":turn, "bought_state":turn+1, "last_owned_state":len(frames)-1,
                       "end_reason":"game_end", "seat":seat, "worker":worker, "tree":tree, "kind":kind,
                       "public_price":price, "paid":-amount, "predicted":pred,
                       "absorption":0., "collision_cash":0.}
                active[seat, tree] = row
                purchases.append(row)
            elif kind == "tree_absorption":
                tree = details[0]
                if abs(amount - float(after["trees"][tree][1])) > 1e-6:
                    raise ValueError("tree absorption disagrees with independent replay field")
                if (seat, tree) in active:
                    active[seat, tree]["absorption"] += amount
            elif kind == "tree_collision_cash" and (seat,details[2]) in active:
                active[seat,details[2]]["collision_cash"] += amount
        for key in list(active):
            if key[1] not in sim.player_trees[key[0]]:
                active.pop(key).update(last_owned_state=turn,end_reason="expiry")
        for (seat, worker), (kind, price, pred) in opportunity.items():
            if (seat,worker) in used:
                continue
            if board["players"][seat][0] < price:
                continue
            group = missed[seat].setdefault(kind, Counter())
            group["unbought_with_starting_cash"] += 1
            if moves[seat].get(worker) in ("UP","RIGHT","DOWN","LEFT"):
                group["expert_moved"] += 1
                group["expert_moved_and_high"] += int(pred-price > 50)
                group["expert_moved_high_with_90_reserve"] += int(pred-price > 50 and board["players"][seat][0] >= price+90)
            else:
                group["stayed_without_purchase"] += 1
        if sim.step_count != after["step"]:
            raise ValueError("turn mismatch")
        if timeline:
            trace.append({"state":turn+1,"cash":list(sim.cash),"components":step_cash})
    combined={kind:sum((Counter(side.get(kind,{})) for side in missed),Counter())
              for kind in ("plant","seize")}
    return {"steps_cash_exact":sim.step_count, "purchases":purchases, "skipped":skipped,
            "unbought":combined, "unbought_by_seat":missed, "score_by_seat":game["rewards"],
            "cash_totals_by_seat":cash_totals, "timeline":trace}


def summary(rows):
    if not rows:
        return {"n":0}
    n=len(rows)
    total = lambda name: sum(r[name] for r in rows)
    return {"n":n, "gross_absorption_mae":sum(abs(r["predicted"]-r["absorption"]) for r in rows)/n,
            "mean_observed_owned_states":sum(r["last_owned_state"]-r["bought_state"]+1 for r in rows)/n,
            "total_actual_cash_paid":total("paid"), "total_tree_absorption":total("absorption"),
            "total_tree_collision_cash":total("collision_cash"),
            "mean_absorption_minus_paid":(total("absorption")-total("paid"))/n,
            "mean_direct_cash_minus_paid":(total("absorption")+total("collision_cash")-total("paid"))/n,
            "repaid_in_absorption":sum(r["absorption"] >= r["paid"] for r in rows),
            "public_price_differs_from_settlement":sum(abs(r["public_price"]-r["paid"])>1e-6 for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="cards with id, seed, canonical_sha256")
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model_file=HERE/"orchard_learned.json"
    frozen=json.loads((HERE/"orchard_dev/build_manifest.json").read_bytes())["files_sha256"]
    for source in (model_file, HERE/"orchard_value.py"):
        if digest(source.read_bytes()) != frozen[str(source.relative_to(HERE.parents[1]))]:
            raise ValueError(f"orchard source differs from the played candidate: {source}")
    train_data=(HERE/"top5_h2h_data.json").read_bytes()
    trained=json.loads(model_file.read_bytes())
    if digest(train_data) != trained["manifest_sha256"]:
        raise ValueError("frozen model and training manifest mismatch")
    old=json.loads(train_data)["games"]
    used={key:{r[key] for r in old} for key in ("id","seed","canonical_sha256")}
    new=json.loads(args.manifest.read_bytes())
    cards=new if isinstance(new,list) else new["games"]
    games={}
    for card in cards:
        for key,seen in used.items():
            if card[key] in seen:
                raise ValueError(f"repeated {key}: {card[key]}")
            seen.add(card[key])
        path=args.replays/(card["id"]+".json.gz")
        env=json.loads(gzip.decompress(path.read_bytes()))
        if (env["statuses"]!=["DONE","DONE"] or digest(canonical(env))!=card["canonical_sha256"]
                or env["configuration"]["randomSeed"]!=card["seed"]):
            raise ValueError(f"game identity or terminal mismatch: {card['id']}")
        games[card["id"]]={"seed":card["seed"],"canonical_sha256":card["canonical_sha256"],
                           "cohort":card.get("cohort"),"seat_names":card.get("seat_names"),
                           **score_one(env,trained["model"],card.get("cohort")=="live_top5_mutual")}
    all_rows=[r for g in games.values() for r in g["purchases"]]
    pooled={}
    for kind in ("plant","seize"):
        rows=[r for r in all_rows if r["kind"]==kind]
        pooled[kind]={"all":summary(rows),
                      "low_predicted_net_below_minus25":summary([r for r in rows if r["predicted"]-r["public_price"] < -25]),
                      "high_predicted_net_above50":summary([r for r in rows if r["predicted"]-r["public_price"] > 50])}
    output={"frozen_model_sha256":digest(model_file.read_bytes()), "training_manifest_sha256":digest(train_data),
            "test_manifest_sha256":digest(args.manifest.read_bytes()), "pooled":pooled, "games":games,
            "definitions":"All recorded purchases followed through first continuous ownership. Target is gross tree absorption; a separate cash total adds collisions and subtracts the actual debited price. Cash and absorption are matched to the archived game step by step. Unbought counts are public opportunity scores, not observed alternative payoff; checks omit individual model logits, post-prefix cash, and joint-action safety vetoes. Multiple exposures by one planter are correlated."}
    args.out.write_text(json.dumps(output, indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
