"""Replay both engines for an index delta and verify its exact membership against older frozen ledgers."""
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import validation as audit

def lines(path):
    return [json.loads(raw) for raw in gzip.decompress(path.read_bytes()).splitlines()]


def changed_index_ids(names, old, previous, revision):
    changed = subprocess.check_output(["git","diff","--name-only",previous,revision,"--","data"],cwd=audit.ROOT).decode().splitlines()
    by_name, expected = {}, set()
    for path in changed:
        if not path.endswith("_matches.json"):
            continue
        rows = json.loads(subprocess.check_output(["git","show",revision+":"+path],cwd=audit.ROOT))
        ids = {f"{r['id']}-{r['game']}" for r in rows if r.get("state") == "done"
               and (r.get("name_a") in names or r.get("name_b") in names)} - old
        if ids:
            by_name[path] = len(ids)
            expected.update(ids)
    return by_name, expected


def run(cache, directory, revision, previous, parents, unscored=()):
    source = directory / "manifest.json"
    manifest = json.loads(source.read_bytes())
    before = []
    for suffix, expected in parents:
        d = HERE / suffix
        if audit.sha((d/"validation_maps.jsonl.gz").read_bytes()) != expected:
            raise ValueError("prior frozen terrain register changed")
        before.extend(lines(d/"validation_games.jsonl.gz"))
    seen_ids = {r["id"] for r in before}
    by_file, expected = changed_index_ids({r["name"] for r in manifest["top5"]}, seen_ids, previous, revision)
    if unscored:
        if not set(unscored) <= expected:
            raise ValueError("unscored cases absent from independent score index")
        cards = json.loads((directory / "unscored/cards.json").read_bytes())
        if set(cards) != set(unscored):
            raise ValueError("unscored game register differs")
        for gid, record in cards.items():
            card = record["card"]
            rewards = [card["scores"][s] for s in card["order"]]
            with gzip.open(directory / "unscored" / (gid + ".json.gz")) as stream:
                replay = json.load(stream)
            if (None not in rewards or replay["rewards"] != rewards
                    or any(value is None and status not in ("ERROR", "TIMEOUT", "INVALID")
                           for value, status in zip(rewards, replay["statuses"]))):
                raise ValueError("missing score is not supported by the original public game")
        expected -= set(unscored)
    rows = manifest["games"]
    if (manifest["revision"] != revision or manifest["state"] != "complete"
            or len(before) != manifest["baseline_games"] or len(seen_ids) != len(before) or len(rows) != len(expected) or {r["id"] for r in rows} != expected
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
        audit.store(directory/(name+".jsonl.gz"),gzip.compress(b"".join(audit.encoded(r)+b"\n" for r in records),mtime=0))
    all_games=before+games
    distinct={k:len({r[k] for r in all_games}) for k in
              ("id","environment_uuid","canonical_sha256","seed","initial_terrain_u8_sha256")}
    summary=dict(revision=revision,manifest_sha256=audit.sha(source.read_bytes()),
        prior_game_lists=[dict(relative_dir=name,terrain_registry_sha256=fingerprint) for name,fingerprint in parents],
        verifier_sha256=audit.sha((HERE/"validation.py").read_bytes()),independent_new_ids_by_score_index=by_file,
        map_registry_sha256=audit.sha((directory/"validation_maps.jsonl.gz").read_bytes()),cards=len(games),
        cohorts=dict(Counter(r["cohort"] for r in games)),frames=sum(r["frames"] for r in games),
        fastsim_transitions=sum(r["engine_steps_matched"] for r in games),
        original_board_transitions=sum(r["original_board"]["matched"] for r in games),
        terminals=dict(Counter(r["terminal_kind"] for r in games)),score_cards_agree=sum(r["card_scores_match"] for r in games),
        seed_alone_recreates_map=sum(r["seed_alone_recreates_map"] for r in games),
        union_distinct=distinct, union_frames=sum(r["frames"] for r in all_games),
        union_transitions=sum(r["engine_steps_matched"] for r in all_games))
    audit.store(directory/"validation.json",json.dumps(summary,ensure_ascii=False,indent=2).encode()+b"\n")
    print(json.dumps(summary,ensure_ascii=False))
