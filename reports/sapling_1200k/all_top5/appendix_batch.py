"""Seal an incremental replay batch, checking all prior cards and map identities."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import tarfile

import fetch


def prepare(revision, baselines, where, expected=None, unscored=None):
    where = Path(where)
    archive = "data/sapling_all_top5_20261119/" + where.name + ".tar"
    previous = [json.loads(path.read_bytes()) for path in baselines]
    old = previous[0]
    known = {game["id"]: game for batch in previous for game in batch["games"]}
    if len(known) != sum(len(batch["games"]) for batch in previous):
        raise ValueError("overlapping historical publications")
    fetch.REV = revision
    catalog = {}
    fetch.prepare(catalog, write=False, unscored=unscored)
    current = {g["id"]: g for g in catalog["games"]}
    if not known.keys() <= current.keys():
        raise ValueError("prior score references missing")
    for gid, g in known.items():
        if any(current[gid][k] != g[k] for k in ("seat_names", "score_by_seat", "cohort")):
            raise ValueError("history changed: " + gid)
    games = [current[i] for i in sorted(current.keys() - known.keys())]
    mutual = sum(g["cohort"] == "mutual" for g in games)
    if not games or expected is not None and (len(games), mutual) != expected:
        raise ValueError("unexpected scorecard delta")
    print("delta", [(g["id"],g["seat_names"]) for g in games], flush=True)
    obj = fetch.Retrieval(defaultdict(list), .40)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for g, result in zip(games, pool.map(obj.one, games)):
            if not result["frame_verified"]:
                raise ValueError((g["id"], result["source_attempt_errors"]))
            g.update(result)
    for field in ("seed", "environment_uuid", "canonical_sha256"):
        old_values = {g[field] for g in known.values()}
        new_values = {g[field] for g in games}
        if len(new_values) != len(games) or old_values & new_values:
            raise ValueError(f"overlapping {field}")
    dst = fetch.ROOT / archive
    dst.parent.mkdir(parents=True, exist_ok=True)
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for g in games:
            raw = (fetch.CACHE / g["path"]).read_bytes()
            if fetch.sha(raw) != g["sha256"]:
                raise ValueError(g["id"])
            entry = tarfile.TarInfo(g["path"])
            entry.mode, entry.size, entry.mtime = 0o644, len(raw), 0
            tar.addfile(entry, io.BytesIO(raw))
            g["archive_location"] = dict(archive=archive, member=g["path"])
    data = output.getvalue()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for g in games:
            if fetch.sha(tar.extractfile(g["path"]).read()) != g["sha256"]:
                raise ValueError("archive member changed " + g["id"])
    fetch.atomic(dst, data)
    scoped = {p: dict(git_blob=s["git_blob"], references=sum(p in g["score_sources"] for g in games))
              for p, s in catalog["index_sources"].items()}
    cov = dict(score_cards=len(games), frame_verified=len(games), score_only=0, mutual=mutual, external=len(games)-mutual, self_play=sum(g["self_play"] for g in games),
               recognized_score_index_paths=len(scoped), clean_terminal=sum(g["clean_terminal"] for g in games),
               verified_frames=sum(g["transitions"]+1 for g in games), verified_transitions=sum(g["transitions"] for g in games),
               registered_player_appearances=len(games)+mutual, sources=dict(Counter(g["source"]["method"] for g in games)),
               distinct_map_seeds=len(games), distinct_environment_uuids=len(games), distinct_replay_sha256=len(games))
    report = dict(schema=catalog["schema"], state="complete", revision=revision, ranking_as_of_utc=old["ranking_as_of_utc"],
                  top5=old["top5"], baseline_revision=old["revision"], baseline_games=len(known), accumulated_games=len(current),
                  scope="All newly archived appearances of the five frozen registered players, deduplicated against every frozen prior game; full index rescan.",
                  coverage=cov, index_sources=scoped, games=games,
                  new_archives=[dict(archive=archive,sha256=fetch.sha(data),bytes=len(data),members=len(games))])
    if len(previous)>1:
        report["baseline_revisions"]=[batch["revision"] for batch in previous]
    fetch.dump(where / "manifest.json", report)
    print(cov, report["new_archives"], flush=True)
