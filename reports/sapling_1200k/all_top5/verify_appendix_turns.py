"""Export every public team command and check all realized H10/H30 targets of an appendix."""
import argparse
import gzip
from pathlib import Path

import fetch
from all_turn_commands_1646.extract import extract, packed
from appendix_c2754b12.check_future import verify_game
from context import Corpus


def verify(directory, cache):
    directory = directory.resolve()
    manifest = fetch.load(directory / "manifest.json")
    index = directory.parent / f"context_{directory.name}_index.tsv.gz"
    corpus = Corpus(index=index)
    cards = manifest["games"]
    if set(corpus.rows) != {g["id"] for g in cards} or len(corpus.rows) != len(cards):
        raise ValueError("appendix membership differs from its signed context index")
    results = []
    for card in cards:
        gid = card["id"]
        stop = card["transitions"]
        window = {"entire_game": {"start": 0, "end": stop, "lookback": {"10": 0}}}
        source = extract(card, window, cache)
        if [row["t"] for row in source["transitions"]] != list(range(stop)):
            raise ValueError(f"incomplete physical command timeline for {gid}")
        data = gzip.compress(packed(source), compresslevel=6, mtime=0)
        output = f"{gid}_commands.json.gz"
        fetch.atomic(directory / output, data)
        evidence = verify_game(corpus, gid)
        evidence.update(command_log=output, command_log_sha256=fetch.sha(data), seats=card["seat_names"])
        results.append(evidence)
    fetch.dump(directory / "whole_game_checks.json", {
        "revision": manifest["revision"],
        "index_sha256": fetch.sha(index.read_bytes()),
        "games": results,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cache", type=Path, default=fetch.CACHE)
    args = parser.parse_args()
    verify(args.directory, args.cache)
