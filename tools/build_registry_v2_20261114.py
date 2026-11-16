#!/usr/bin/env python3
"""X10 fix (r4, strategy group -> war room -> DGX line, 2026-11-14):
build v2 of the Condition E registry
  /home/arm/junyang/carbon/data/k0_consumed_gen1r_shard1_registry.json

Defects in v1 (md5 7801a6d20f3311b9edc29f08283f662e):
  1. replay_id degenerate: 200/200 entries carry "selfplay". That string is
     the selfplay generator's constant document id field (and the shard
     index's id field) -- it was never a per-game identifier.
  2. No per-entry content hash, so the registry cannot bind an entry to the
     exact replay bytes it claims to register.

v2 truthful identity, per entry:
  - replay_id  = the replay file stem "selfplay-<game>" (unique per game);
  - id_field_in_replay = "selfplay" (the document's own id, kept verbatim);
  - open_seed  = the per-game discriminator inside the replay document
                 (result.open_seed; equals game and the index open_seed),
                 read back from each replay file at build time;
  - sha256 / md5 / bytes over the stored .json.gz file bytes, computed from
    disk at build time; sha256 asserted equal to the shard index's recorded
    sha256 for every entry (the index hash is over the same stored bytes --
    verified for game 994400 before this builder was written, asserted for
    all 200 here).

v1 is left byte-identical. This builder is one-shot: it refuses to
overwrite an existing v2.
"""
import gzip
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

W = "/home/arm/junyang/carbon/data"
V1 = os.path.join(W, "k0_consumed_gen1r_shard1_registry.json")
V1_MD5 = "7801a6d20f3311b9edc29f08283f662e"
V2 = os.path.join(W, "k0_consumed_gen1r_shard1_registry_v2.json")
SHARD = os.path.join(W, "gen1r_shard1")
INDEX = os.path.join(SHARD, "index.json")
CST = timezone(timedelta(hours=8))


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def file_hashes(p):
    raw = open(p, "rb").read()
    return raw, hashlib.sha256(raw).hexdigest(), hashlib.md5(raw).hexdigest()


def replay_internal(path):
    """Read the replay document's own id field and result.open_seed.

    Serialized layout (verified on selfplay-994400): the document opens
    with {"id": "selfplay", "steps": [... ~2.5MB ...], "identity": {...},
    "result": {...}}. So the id comes from the head of the decompressed
    stream and open_seed from a rolling tail buffer; the steps array is
    streamed through without being retained or parsed.
    """
    head = ""
    tail = ""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(262144)
            if not chunk:
                break
            if not head:
                head = chunk[:1024]
            tail = (tail + chunk)[-131072:]
    m_id = re.search(r'"id"\s*:\s*"([^"]+)"', head)
    m_seed = re.search(r'"open_seed"\s*:\s*(\d+)', tail)
    assert m_id, "no id field in replay head: %s" % path
    assert m_seed, "no open_seed in replay tail: %s" % path
    return m_id.group(1), int(m_seed.group(1))


def main():
    assert not os.path.exists(V2), "v2 already exists; refusing to overwrite"
    raw_v1 = open(V1, "rb").read()
    assert md5_bytes(raw_v1) == V1_MD5, "v1 md5 drifted: %s" % md5_bytes(raw_v1)
    v1 = json.loads(raw_v1)
    v1_games = v1["games"]
    assert len(v1_games) == 200
    assert all(g["replay_id"] == "selfplay" for g in v1_games), \
        "v1 defect premise changed: replay_id not uniformly 'selfplay'"

    idx_raw = open(INDEX, "rb").read()
    idx = json.loads(idx_raw)
    assert len(idx) == 200
    by_game = {e["game"]: e for e in idx}
    assert set(by_game) == set(g["game"] for g in v1_games), \
        "index game set != v1 registry game set"

    games = []
    for g in sorted(v1_games, key=lambda x: x["game"]):
        game = g["game"]
        path = g["file"]
        e = by_game[game]
        assert os.path.join(SHARD, e["file"]) == path, \
            "v1 file != index file for game %d" % game
        assert e["id"] == "selfplay" and e["open_seed"] == game
        raw, sha256, md5 = file_hashes(path)
        assert len(raw) == e["bytes"], "bytes mismatch game %d" % game
        assert sha256 == e["sha256"], \
            "disk sha256 != index sha256 for game %d" % game
        id_field, open_seed = replay_internal(path)
        assert id_field == "selfplay", \
            "replay id field changed for game %d: %r" % (game, id_field)
        assert open_seed == game, \
            "replay open_seed %d != game %d" % (open_seed, game)
        stem = os.path.basename(path).replace(".json.gz", "")
        assert stem == "selfplay-%d" % game
        games.append({
            "game": game,
            "file": path,
            "replay_id": stem,
            "id_field_in_replay": id_field,
            "open_seed": open_seed,
            "sha256": sha256,
            "md5": md5,
            "bytes": len(raw),
        })

    assert len(set(x["replay_id"] for x in games)) == 200
    assert len(set(x["sha256"] for x in games)) == 200
    assert [x["game"] for x in games] == list(range(994400, 994600))

    now = datetime.now(CST).isoformat(timespec="seconds")
    doc = {
        "header": {
            "version": 2,
            "purpose": v1["header"]["purpose"],
            "ruling": v1["header"]["ruling"],
            "rule": v1["header"]["rule"],
            "supersedes": {
                "path": V1,
                "md5": V1_MD5,
                "bytes": len(raw_v1),
                "note": "v1 is kept byte-identical for history. Per r4 item "
                        "X10 it must not be cited as evidence again; cite "
                        "this v2 instead.",
            },
            "defect_fixed": "r4 X10: v1 replay_id degenerate (200/200 "
                            "'selfplay' -- the generator's constant document "
                            "id, not a per-game identifier) and no per-entry "
                            "content hash.",
            "replay_id_definition": "replay_id is the replay file stem "
                                    "'selfplay-<game>'. The replay "
                                    "document's own id field is the constant "
                                    "'selfplay' (kept verbatim in "
                                    "id_field_in_replay); the per-game "
                                    "discriminator inside the document is "
                                    "result.open_seed, read back from every "
                                    "replay file at build time and equal to "
                                    "game and to the index open_seed.",
            "content_hash_definition": "sha256/md5/bytes are computed over "
                                       "the stored .json.gz file bytes at "
                                       "build time; sha256 is asserted "
                                       "equal to the shard index's recorded "
                                       "sha256 for all 200 entries.",
            "registered_at_v1": v1["header"]["registered_at"],
            "built_at": now,
            "built_by": "DGX line (X10 owner per strategy-group r4 "
                        "disposition, forwarded by the war room 2026-11-14)",
            "builder": "/home/arm/junyang/build_registry_v2.py",
            "source_index": INDEX,
            "source_index_md5": md5_bytes(idx_raw),
            "generation_identity": v1["header"]["generation_identity"],
            "count": 200,
        },
        "games": games,
    }
    with open(V2, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)

    back = json.load(open(V2))
    assert len(back["games"]) == 200
    raw_v2 = open(V2, "rb").read()
    # v1 must be untouched by this build
    assert md5_bytes(open(V1, "rb").read()) == V1_MD5
    print(json.dumps({
        "v2": V2,
        "v2_md5": md5_bytes(raw_v2),
        "v2_bytes": len(raw_v2),
        "count": len(back["games"]),
        "replay_id_unique": len(set(x["replay_id"] for x in back["games"])),
        "sha256_unique": len(set(x["sha256"] for x in back["games"])),
        "sha256_matches_index": 200,
        "open_seed_matches_game": 200,
        "v1_md5_after": V1_MD5,
        "built_at": now,
        "sample": back["games"][0],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
