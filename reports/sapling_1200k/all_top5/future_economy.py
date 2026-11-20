"""Query realized multi-step economic targets separately from a causal context.Game decision."""
import argparse
from functools import cached_property
from itertools import accumulate
import gzip
import json
from pathlib import Path

if __package__:
    from .context import Corpus, Game, HERE, digest, registered_context_receipts
    from .economy_extract import reconstruct
else:
    from context import Corpus, Game, HERE, digest, registered_context_receipts
    from economy_extract import reconstruct

HORIZONS = (10, 30)


def registered_indexes():
    """Discover the base and any published appendix by its independent context receipt."""
    indexes = []
    for receipt, _ in registered_context_receipts():
        index = receipt.with_name(receipt.name.removesuffix("_receipt.json") + "_index.tsv.gz")
        expected = json.loads(receipt.read_bytes())["index_sha256"]
        if digest(index.read_bytes()) != expected:
            raise ValueError(f"context receipt and index differ: {index.name}")
        indexes.append(index)
    return indexes


def open_game(game_id, replay_cache=None):
    found = [c for c in (Corpus(index=i) for i in registered_indexes()) if game_id in c.rows]
    if len(found) > 1:
        raise ValueError(f"game occurs in more than one registered batch: {game_id}")
    if not found:
        raise KeyError(game_id)
    corpus = found[0]
    if replay_cache is None:
        return corpus.open(game_id)
    # Same escrowed source; the unpacked cache avoids re-reading large Git tar blobs.
    item = corpus.rows[game_id]
    cached = Path(replay_cache) / (game_id + ".json.gz")
    if not cached.is_file():
        return corpus.open(game_id)
    blob = cached.read_bytes()
    if digest(blob) != item["replay_sha256"]:
        raise ValueError("unpacked replay digest differs from catalog")
    game = Game(json.loads(gzip.decompress(blob)))
    if game.turns != int(item["transitions"]):
        raise ValueError("catalog length differs from replay")
    return game


class FutureEconomy:
    """Lazy per-game target reader, for repeated (decision, perspective) queries."""

    def __init__(self, game):
        self.game = game

    @cached_property
    def _money(self):
        # The existing ledger checks every worker, tree and cash balance against the replay.
        # Entry i is cumulative cash after i joint transitions; outcomes never enter decision().
        ledger = reconstruct(self.game._frames, self.game._data["configuration"])[0]
        return [[list(accumulate((step[s][k] for step in ledger), initial=0.0))
                 for k in (0, 2)] for s in (0, 1)]

    @cached_property
    def _rosters(self):
        config = self.game._data["configuration"]
        result = [[], []]
        previously_staffed = [False, False]
        for t, frame in enumerate(self.game._frames):
            sides = frame[0]["observation"]["players"]
            rosters = [{uid for uid, unit in side[2].items() if unit[2] == "COLLECTOR"}
                       for side in sides]
            price = config["recCollectorCost"] + sum(map(len, rosters)) * config["recCollectorIncreaseCost"]
            for s, (cash, bases, workers, trees) in enumerate(sides):
                has_capacity = bool(bases) and len(workers) < config["workerLimit"]
                can_pay = cash >= price
                ticket = has_capacity and can_pay if t < self.game.turns else None
                previously_staffed[s] |= bool(rosters[s])
                result[s].append((rosters[s], cash, len(trees), price, can_pay, ticket, previously_staffed[s]))
        return result

    def _side(self, t, stop, side):
        states = self._rosters[side][t:stop + 1]
        counts = [len(state[0]) for state in states]
        last, cash, trees, price, can_pay, can_hire, _ = states[-1]
        start_ids = states[0][0]
        lost = next((i for i in range(1, len(states)) if counts[i-1] and not counts[i]), None)
        first_empty = next((i for i, n in enumerate(counts) if not n), None)
        production_gap = next((i for i, state in enumerate(states) if not state[0] and state[6]), None)
        recovered = None if production_gap is None else next(
            (i for i in range(production_gap + 1, len(states)) if counts[i]), None)
        can_refund_gap = next((i for i, state in enumerate(states)
                               if not state[0] and state[6] and state[5] is True), None)
        fundable_empty = next((i for i, state in enumerate(states)
                               if not state[0] and state[5] is True), None)
        bank, tree = self._money[side]
        return {
            "own_collector_delivery": round(bank[stop] - bank[t], 6),
            "tree_absorption": round(tree[stop] - tree[t], 6),
            "collectors_start": counts[0], "collectors_end": len(last),
            "original_collectors_surviving": len(start_ids & last),
            "sole_starting_collector_survived": bool(start_ids & last) if counts[0] == 1 else None,
            "min_collectors_including_start": min(counts),
            "first_loss_of_last_collector_offset": lost,
            "first_empty_offset": first_empty,
            "collector_owned_by_start": states[0][6],
            "first_empty_after_having_collector_offset": production_gap,
            "first_fundable_rehire_after_having_collector_offset": can_refund_gap,
            "first_collector_present_again_offset": recovered,
            "first_fundable_hire_while_empty_offset": fundable_empty,
            "cash_end": cash, "living_trees_end": trees, "hire_price_end": price,
            "cash_meets_hire_price_end": can_pay,
            "fundable_hire_with_base_and_capacity_end": can_hire,
        }

    def labels(self, t, seat, horizons=HORIZONS):
        """Full targets end at t+H. When unavailable, expose the observed suffix separately."""
        self.game._frame(t, seat)
        horizons = tuple(horizons)
        if not horizons or len(set(horizons)) != len(horizons) or any(
                type(h) is not int or h not in HORIZONS for h in horizons):
            raise ValueError("choose one or both of 10 and 30")
        windows = {}
        for h in horizons:
            stop = min(t + h, self.game.turns)
            complete = stop == t + h
            end = stop == self.game.turns
            kind = None
            if end:
                statuses = self.game._data["statuses"]
                if any(status != "DONE" for status in statuses):
                    kind = "other_terminal_status"
                else:
                    scheduled = self.game._data["configuration"]["episodeSteps"] - 1
                    kind = "scheduled" if stop == scheduled else "early"
            amounts = {name: self._side(t, stop, s) for name, s in (("own", seat), ("opponent", 1-seat))}
            windows[str(h)] = {"observed_steps": stop - t, "full_horizon": complete,
                               "terminal_at_observed_end": kind,
                               "target": amounts if complete else None,
                               "observed_to_terminal": amounts if not complete else None}
        return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_id")
    parser.add_argument("t", type=int)
    parser.add_argument("seat", type=int, choices=(0,1))
    parser.add_argument("--horizons", type=int, nargs="+", choices=HORIZONS, default=HORIZONS)
    parser.add_argument("--replay-cache", type=Path, help="directory of hash-verified unpacked .json.gz replays")
    parser.add_argument("--with-decision", action="store_true", help="also return context.Game.decision in a separate field")
    args = parser.parse_args()
    game = open_game(args.game_id, args.replay_cache)
    out = {"future_labels": FutureEconomy(game).labels(args.t,args.seat,args.horizons)}
    if args.with_decision:
        out["decision"] = game.decision(args.t,args.seat)
    print(json.dumps(out, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
