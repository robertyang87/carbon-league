"""Check V4 elite supervision against raw worker identities on three matches."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from cnet2_feat import CANON, DIST
from sapling_tree_train import elite_events

GAMES = ("011008e4dee32c2fdbfa37044c3c2608-1", "01b56d406361dbb513d6d7417dc0184f-1",
         "02ab03aeb1de04e89d030d0e71b280ec-1")


def check(game, replays, shards):
    packed = (replays / (game + ".json.gz")).read_bytes()
    frames = json.loads(gzip.decompress(packed))["steps"]
    with np.load(shards / (game + ".npz"), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    masks = elite_events(data)
    transitions = len(frames) - 1
    counts = {key: Counter() for key in ("new_tree", "seized_tree", "harvest", "delivery")}
    for seat in (0, 1):
        # The final transition has no next decision row in the shard.
        for t in range(transitions-1):
            pre = frames[t][0]["observation"]["players"]
            own, foe = pre[seat], pre[1-seat]
            post = frames[t+1][0]["observation"]["players"][seat]
            actions = frames[t+1][seat]["action"] or {}
            own_after = {cell for cell, _ in post[3].values()}
            own_before = {cell for cell, _ in own[3].values()}
            foe_before = {cell for cell, _ in foe[3].values()}
            row = seat * transitions + t
            slots = {int(CANON[pos]) if seat else pos: s for s, pos in enumerate(data["tok_pos"][row, :10])
                     if data["tok_mask"][row, s]}
            for uid, (cell, cargo, kind) in own[2].items():
                slot = slots[cell]
                stay = actions.get(uid, "NONE") == "NONE"
                matching = post[2].get(uid)
                dest, load = matching[:2] if matching else (None, 0)
                safe = all(DIST[cell, pos] > 1 for pos, _, _ in foe[2].values())
                tree = kind == "PLANTER" and stay and cell in own_after
                physical = dict(new_tree=tree and cell not in (own_before | foe_before),
                                seized_tree=tree and cell in foe_before,
                                harvest=kind == "COLLECTOR" and stay and dest == cell and safe
                                        and cell not in own[1].values() and load > cargo,
                                delivery=kind == "COLLECTOR" and dest in own[1].values()
                                         and cargo > 0 and load == 0)
                for k, real in physical.items():
                    tag = bool(masks[k][row, slot])
                    counts[k]["observed"] += int(real)
                    counts[k]["tagged"] += int(tag)
                    counts[k]["false_positives"] += int(tag and not real)
                    counts[k]["false_negatives"] += int(real and not tag)
    return {"raw_gzip_sha256": hashlib.sha256(packed).hexdigest(), "event_counts": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    args = parser.parse_args()
    audit = {game: check(game, args.replays, args.shards) for game in GAMES}
    print(json.dumps(audit, indent=2))
    if any(counts["false_negatives"] or counts["false_positives"] for rep in audit.values()
           for counts in rep["event_counts"].values()):
        raise SystemExit("mismatch between observed and encoded actions")
