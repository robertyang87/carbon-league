"""Validate this last index batch independently against both prior frozen corpora."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import validation as audit

REV = "4bfdabf588f631e74077d21fe97e267c2521e533"
PRIOR = "499886f1a673f71a8fc3454ce9fee998d65af0c9"
PARENTS = ((".", "c70e0f7953ac7b4adc04e1a768b9a72c69f6057d6eca8f5f8cb58ffb9072f486"),
           ("appendix_499886f1", "707d8c302ccb4ff7f8eb9106d503a6cfaa0208c5a42b3ae02f4ac5cdd5606e4e"))


def lines(path):
    return [json.loads(raw) for raw in gzip.decompress(path.read_bytes()).splitlines()]


def changed_index_ids(names, old):
    changed = subprocess.check_output(["git","diff","--name-only",PRIOR,REV,"--","data"],cwd=audit.ROOT).decode().splitlines()
    by_name, expected = {}, set()
    for path in changed:
        if not path.endswith("_matches.json"):
            continue
        rows = json.loads(subprocess.check_output(["git","show",REV+":"+path],cwd=audit.ROOT))
        ids = {f"{r['id']}-{r['game']}" for r in rows if r.get("state") == "done"
               and (r.get("name_a") in names or r.get("name_b") in names)} - old
        if ids:
            by_name[path] = len(ids)
            expected.update(ids)
    return by_name, expected


def run(cache):
    source = HERE / "manifest.json"
    manifest = json.loads(source.read_bytes())
    before = []
    for suffix, expected in PARENTS:
        d = HERE.parent / suffix
        if audit.sha((d/"validation_maps.jsonl.gz").read_bytes()) != expected:
            raise ValueError("prior frozen terrain register changed")
        before.extend(lines(d/"validation_games.jsonl.gz"))
    seen_ids = {r["id"] for r in before}
    by_file, expected = changed_index_ids({r["name"] for r in manifest["top5"]}, seen_ids)
    rows = manifest["games"]
    if (manifest["revision"] != REV or manifest["state"] != "complete"
            or len(before) != 1332 or len(rows) != len(expected) or {r["id"] for r in rows} != expected
            or not all(r["frame_verified"] for r in rows)):
        raise ValueError("whole frozen appendix must match the independent index delta")
    games, maps = [], []
    for row in sorted(rows,key=lambda r:r["id"]):
        result = audit.check_one(row,cache,row)
        result["original_board"] = audit.official(row,cache)
        if result["verdict"] != "consistent" or result["original_board"]["first_divergence"]:
            raise ValueError(f"inconsistent {row['id']}: {result}")
        with gzip.open(cache/row["path"]) as f:
            cells = json.load(f)["steps"][0][0]["observation"]["full_carbon"]
        if audit.sha(audit.encoded(cells)) != result["initial_map_sha256"]:
            raise ValueError("actual board changed")
        result["initial_terrain_u8_sha256"] = audit.terrain_sha256(cells)
        games.append(result)
        maps.append({k:result[k] for k in ("id","seed","initial_terrain_u8_sha256")})
    for name,records in (("validation_games",games),("validation_maps",maps)):
        audit.store(HERE/(name+".jsonl.gz"),gzip.compress(b"".join(audit.encoded(r)+b"\n" for r in records),mtime=0))
    all_games=before+games
    distinct={k:len({r[k] for r in all_games}) for k in
              ("id","environment_uuid","canonical_sha256","seed","initial_terrain_u8_sha256")}
    summary=dict(revision=REV,manifest_sha256=audit.sha(source.read_bytes()),
        prior_game_lists=[dict(relative_dir=name,terrain_registry_sha256=fingerprint) for name,fingerprint in PARENTS],
        verifier_sha256=audit.sha((HERE.parent/"validation.py").read_bytes()),independent_new_ids_by_score_index=by_file,
        map_registry_sha256=audit.sha((HERE/"validation_maps.jsonl.gz").read_bytes()),cards=len(games),
        cohorts=dict(Counter(r["cohort"] for r in games)),frames=sum(r["frames"] for r in games),
        fastsim_transitions=sum(r["engine_steps_matched"] for r in games),
        original_board_transitions=sum(r["original_board"]["matched"] for r in games),
        terminals=dict(Counter(r["terminal_kind"] for r in games)),score_cards_agree=sum(r["card_scores_match"] for r in games),
        seed_alone_recreates_map=sum(r["seed_alone_recreates_map"] for r in games),
        union_distinct=distinct, union_frames=sum(r["frames"] for r in all_games),
        union_transitions=sum(r["engine_steps_matched"] for r in all_games))
    audit.store(HERE/"validation.json",json.dumps(summary,ensure_ascii=False,indent=2).encode()+b"\n")
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path("/workspace/sapling-all-top5-20261119"))
    run(parser.parse_args().cache)
