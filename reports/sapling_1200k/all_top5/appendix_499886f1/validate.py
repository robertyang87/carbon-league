"""Check the eight additional archived appearances with the frozen full-game verifier."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import validation as audit


def run(cache):
    source = HERE / "manifest.json"
    manifest = json.loads(source.read_bytes())
    if not manifest["revision"].startswith("499886f1") or len(manifest["games"]) != 8 or not all(
            r["frame_verified"] for r in manifest["games"]):
        raise ValueError("expected the complete eight-game 499886f1 appendix")
    parent_ledger = HERE.parent / "validation_maps.jsonl.gz"
    if audit.sha(parent_ledger.read_bytes()) != "c70e0f7953ac7b4adc04e1a768b9a72c69f6057d6eca8f5f8cb58ffb9072f486":
        raise ValueError("the original 1,324-game map register changed")
    old = [json.loads(line) for line in gzip.decompress(parent_ledger.read_bytes()).splitlines()]
    games, maps = [], []
    for row in sorted(manifest["games"],key=lambda r:r["id"]):
        r = audit.check_one(row,cache,row)
        r["original_board"] = audit.official(row,cache)
        if r["verdict"] != "consistent" or r["original_board"]["first_divergence"] is not None:
            raise ValueError(f"inconsistent game {row['id']}: {r}")
        with gzip.open(cache / row["path"]) as f:
            cells=json.load(f)["steps"][0][0]["observation"]["full_carbon"]
        if audit.sha(audit.encoded(cells)) != r["initial_map_sha256"]:
            raise ValueError("source changed during verification")
        r["initial_terrain_u8_sha256"] = audit.terrain_sha256(cells)
        games.append(r)
        maps.append({k:r[k] for k in ("id","seed","initial_terrain_u8_sha256")})
    for key in ("id","seed","initial_terrain_u8_sha256"):
        if len({r[key] for r in old+maps}) != len(old)+len(maps):
            raise ValueError(f"repetition within appendix or with original registry: {key}")
    for name, records in (("validation_games",games),("validation_maps",maps)):
        audit.store(HERE/(name+".jsonl.gz"),gzip.compress(b"".join(audit.encoded(r)+b"\n" for r in records),mtime=0))
    summary = dict(revision=manifest["revision"],manifest_sha256=audit.sha(source.read_bytes()),
        verifier_sha256=audit.sha((HERE.parent/"validation.py").read_bytes()),
        base_map_registry_sha256=audit.sha(parent_ledger.read_bytes()),map_registry_sha256=audit.sha((HERE/"validation_maps.jsonl.gz").read_bytes()),
        cards=len(games),cohorts=dict(Counter(r["cohort"] for r in games)),
        frames=sum(r["frames"] for r in games),fastsim_transitions=sum(r["engine_steps_matched"] for r in games),
        original_board_transitions=sum(r["original_board"]["matched"] for r in games),
        terminals=dict(Counter(r["terminal_kind"] for r in games)),
        score_cards_agree=sum(r["card_scores_match"] for r in games),
        seed_alone_recreates_map=sum(r["seed_alone_recreates_map"] for r in games),
        distinct_union_match_ids=1332,distinct_union_seeds=1332,distinct_union_terrain_hashes=1332)
    audit.store(HERE/"validation.json", json.dumps(summary,ensure_ascii=False,indent=2).encode()+b"\n")
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path("/workspace/sapling-all-top5-20261119"))
    run(parser.parse_args().cache)
