"""Price V4r's actual planted trees and count first-tenure losses from recorded commands."""
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tree_practice as practice


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = Path(__file__).resolve().parent / "tree_exam_v4r"
    result = json.loads((folder / "results.json").read_text())
    games = []
    for game in result["games"]:
        path = folder / Path(game["raw_file"]).name
        if sha(path) != game["raw_sha256"]:
            raise ValueError("raw game identity mismatch")
        with gzip.open(path, "rt") as f:
            raw = json.load(f)
        def replay(observation, configuration):
            return raw["actions"][observation["step"]][observation["player"]]
        match = practice.play(SimpleNamespace(agent=replay),
                              SimpleNamespace(make_agent=lambda: replay, __name__="archived_examiner"),
                              raw["seed"], raw["candidate_seat"])
        if abs(match["official_margin"] - game["margin"]) > 1.e-8:
            raise ValueError("replayed margin differs")
        games.append(dict(raw_sha256=game["raw_sha256"], seat=raw["candidate_seat"],
                          seed=raw["seed"], margin=match["official_margin"], summary=practice.summarize([match]),
                          trees=match["own_plants"]))
    print(json.dumps(dict(pack_sha256=raw["pack_sha256"], results_sha256=sha(folder/"results.json"),
                          evaluator_sha256=sha(Path(practice.__file__)), games=games), indent=2))


if __name__ == "__main__":
    main()
