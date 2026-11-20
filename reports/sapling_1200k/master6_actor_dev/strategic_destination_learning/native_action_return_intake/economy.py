"""Reconcile each realized branch with the official cash ledger, including prior 3/5/10 turns."""
import argparse
import json

import responses
import select_orders as sel

common = sel.common
ledger_source = sel.STUDY / "policy_window_returns/economy/reconcile.py"
ledger = responses.load_module("native_action_counterfactual_cash", ledger_source)


def track(seed):
    header, identity = sel.source_for(seed)
    dest = sel.HERE / "tickets" / str(seed)
    summary = sel.read_json(dest / "summary.json")
    root = summary["root"]
    ticket = next(row for row in sel.read_json(sel.HERE / "roots.json")["tickets"] if row["seed"] == seed)
    sel.need(summary["protocol"] == sel.sha(sel.HERE / "protocol.json") and
             all(summary[k] == v for k, v in identity.items()), "Unregistered ticket")
    games = {}
    for name, meta in summary["arms"].items():
        path = dest / (name + ".json.gz")
        sel.need(sel.sha(path) == meta["sha256"], "Branch changed")
        branch = ledger.load(path)
        history = header["rows"][:root] + branch["rows"]
        responses.auditor.replay(header, history, branch["terminal"])
        game = ledger.replay(header, history, branch["terminal"], name)
        sel.need(game["official_margin"] == meta["terminal"]["margin"], "Cash and official score differ")
        games[name] = game
    diff = ledger.ledger.comparison(games["alternative"], games["parent"])
    ledger.ledger.check_equal(diff["official_margin_delta"], summary["alternative_minus_parent"]["margin"],
                              "Not the scheduled paired return")
    seat = ticket["seat"]
    intervened = ticket["pivot_uid"]
    workers = header["rows"][root]["public"]["players"][seat][2]
    before = {
        "pivot_worker": workers[intervened],
        "actor_seat": seat,
        "root": root,
        "prior_public_and_orders": {
            str(n): {"public": header["rows"][root-n]["public"], "joint": header["rows"][root-n]["orders"]}
            for n in (3, 5, 10)
        },
        "public": header["rows"][root]["public"],
        "root_response": header["rows"][root]["orders"][1-seat],
    }
    paired = {
        "summary_sha256": sel.sha(dest / "summary.json"),
        "referee": summary["protocol"],
        "cash_ledger_sha256": sel.sha(ledger_source),
        "intervention": {**before, "pivot_uid": intervened, "orders": ticket["orders"]},
        "comparison": diff,
        "arms": {},
    }
    for name, game in games.items():
        short = ledger.shorten(game, root)
        paired["arms"][name] = {
            "margin": game["official_margin"], "terminal": game["terminal"],
            "cash_path_from_root": game["cash_margin_by_state"][root:],
            "largest_swings_from_root": short["pivots"]["from_intervention"],
            "turns_from_preceding_ten": game["transitions"][max(0, root-10):],
        }
    common.save(dest / "economy.json.gz", paired)
    return diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    comparison = track(args.seed)
    print(json.dumps({"margin": comparison["official_margin_delta"],
                      "components": comparison["margin_components"]}, ensure_ascii=False))
