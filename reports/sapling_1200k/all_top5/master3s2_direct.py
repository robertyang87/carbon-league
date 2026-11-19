"""Recover the frozen top-five's public matches against exactly Master3s2."""
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMINER = "claude-AlphaCarbon-Master3s2"


def main():
    batches = [HERE, *sorted(HERE.glob("appendix_*/manifest.json"))]
    cards = {}
    for batch in batches:
        manifest = batch if batch.is_file() else batch / "manifest.json"
        for card in json.loads(manifest.read_bytes())["games"]:
            if EXAMINER in card["seat_names"]:
                if card["id"] in cards:
                    raise ValueError(f"duplicate game {card['id']}")
                cards[card["id"]] = card

    rows = {}
    def read_game(game_id, source_sha, record):
        card = cards.get(game_id)
        if card is None:
            return
        if (game_id in rows or source_sha != card["sha256"]
                or record["rewards"] != card["score_by_seat"]):
            raise ValueError(f"score or identity mismatch in {game_id}")
        seat = 1 - card["seat_names"].index(EXAMINER)
        other = 1 - seat
        stages = []
        for stage in [*record["stages"], record["whole"]]:
            own, enemy = stage["seats"][seat], stage["seats"][other]
            stages.append({
                "from_to": [stage["a"], stage["b"]],
                "cash_margin_change": round(stage["margin0_delta"] * (1 - 2 * seat), 2),
                "bank": [own["bank"], enemy["bank"]], "tree": [own["tree"], enemy["tree"]],
                "captured_cargo": [own["cargo_acquired_from_enemy"], enemy["cargo_acquired_from_enemy"]],
            })
        rows[game_id] = {
            "id": game_id, "source_sha256": source_sha, "seed": card["seed"],
            "entrant": card["seat_names"][seat], "entrant_seat": seat,
            "score": [record["rewards"][seat], record["rewards"][other]],
            "margin": round(record["rewards"][seat] - record["rewards"][other], 2),
            "stages_then_whole": stages,
        }

    for shard in sorted((HERE / "economy_data").glob("*.jsonl.gz")):
        with gzip.open(shard, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                read_game(row["id"], row["replay_sha256"], row["record"])
    for manifest in batches[1:]:
        with gzip.open(manifest.with_name("cases.json.gz"), "rt") as stream:
            for row in json.load(stream)["rows"]:
                read_game(row["id"], row["source_sha256"], row["economy"])
    if rows.keys() != cards.keys():
        raise ValueError(f"missing games: {sorted(cards.keys() - rows.keys())}")
    output = {"examiner_name": EXAMINER, "games": [rows[g] for g in sorted(rows)]}
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
