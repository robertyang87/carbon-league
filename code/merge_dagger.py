"""merge_dagger BASE_DS DAGGER_DIR[,DIR2...] OUT_DS [DAGGER_W] [BASE_FRAC]

env SLOT_W="1:2,3:2" multiplies the base-row sample weight of those teacher slots (DAgger rows untouched).
BASE_FRAC < 1 keeps that fraction of the base TRAIN games (crc of key; all base validation games are kept).

Concatenates a weighted cnet2 dataset (w.npy + uw.npy) with dagger_gen shards into a new dataset dir that
cnet2_train reads as one. DAgger rows get sample weight DAGGER_W (default 1.0) and keep their own unit weights
(relabelled planter slots 3, agreeing planter slots 1, collectors/bases 0), so they only supervise the planter head. 10% of the DAgger games are
validation, like the builder does for replay games -- R06/F02: a shard carrying orig_id (lb_relabel corpus shards) inherits its ORIGINAL
game's split (builder crc rule on the original replay name); only genuinely new games (selfplay shards) roll the crc
on their own shard name. A corpus-relabeled shard (lb_cp_*.npz) WITHOUT orig_id is refused outright (see the gate in main):
re-splitting it by its derived name can put a validation game's relabeled rows in train while the original game stays in val.
"""
import glob
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnet2_build import FIELDS  # noqa: E402


SLOT_W = {int(a): float(b) for a, b in (x.split(":") for x in os.environ.get("SLOT_W", "").split(",") if x)}


def _orig_of(D):
    """orig_id carried by a shard (the original replay file name), or None when absent/empty."""
    if "orig_id" not in D.files:
        return None
    o_ = D["orig_id"]
    return str(o_.item() if o_.shape == () else o_.flat[0]) or None


def main(base, dg, out, dw=1.0, frac=1.0):
    dw = float(dw); frac = float(frac)
    bm = json.load(open(os.path.join(base, "meta.json")))
    B = {k: np.load(os.path.join(base, k + ".npy"), mmap_mode="r") for k in list(FIELDS) + ["w", "uw"]}
    keep_g = [g for g in bm["games"] if g.get("val") or zlib.crc32(g["key"].encode()) % 1000 < frac * 1000]
    keep_ids = np.array(sorted({g["gi"] for g in keep_g}))
    gid = np.asarray(B["gid"])
    rows = np.nonzero(np.isin(gid, keep_ids))[0] if frac < 1 else np.arange(len(gid))
    nb = len(rows)
    shards = sorted(s for d in dg.split(",") for s in glob.glob(os.path.join(d, "shards", "*.npz")))
    # F02 gate: an lb_cp_*.npz shard (lb_relabel corpus mode) is DERIVED from an original replay game and
    # must inherit that game's split via orig_id. Legacy lb_cp shards predate orig_id; letting one fall
    # through to the crc roll on its derived name can move an original VALIDATION game's relabeled rows
    # into train while the original game itself stays in val -- a train/val leak by original game.
    # Refuse such shards up front, before anything is written. Genuinely new games (lb_sp_*/dg_*
    # self-play shards) carry no orig_id by design and are unaffected.
    bad = []
    for s in shards:
        if os.path.basename(s).startswith("lb_cp_"):
            with np.load(s) as D:
                if _orig_of(D) is None:
                    bad.append(s)
    if bad:
        sys.stderr.write(
            "merge_dagger: REFUSING to import relabeled-corpus shard(s) without orig_id:\n"
            + "".join(f"  - {s}\n" for s in bad)
            + "lb_cp_* shards derive from an original replay game; without orig_id they would be re-split\n"
              "by their derived file name and can leak an original validation game into train (R06/F02).\n"
              "Regenerate with the current lb_relabel (which writes orig_id) or backfill orig_id with the\n"
              "original replay file name, then re-run. No output was written.\n")
        sys.exit(1)
    lens = [int(np.load(s)["y"].shape[0]) for s in shards]
    nd = sum(lens)
    os.makedirs(out, exist_ok=True)
    fields = dict(FIELDS); fields["w"] = (np.float16, ()); fields["uw"] = (np.float16, B["uw"].shape[1:])
    mm = {k: np.lib.format.open_memmap(os.path.join(out, k + ".npy"), mode="w+", dtype=dt, shape=(nb + nd,) + tuple(sh))
          for k, (dt, sh) in fields.items()}
    for k in mm:                                   # base rows, in chunks
        for a in range(0, nb, 50000):
            b = min(a + 50000, nb)                 # the output is longer than the base: never slice past nb
            r = rows[a:b]
            mm[k][a:b] = B[k][r[0]:r[-1] + 1][r - r[0]]
            if k == "w" and SLOT_W:                # forgetting guard: up-weight the teacher rows of the given style slots
                t = np.asarray(B["teacher"][r[0]:r[-1] + 1][r - r[0]])
                f = np.ones(len(r), np.float32)
                for sl, m in SLOT_W.items():
                    f[t == sl] = m
                mm[k][a:b] = (np.asarray(mm[k][a:b], np.float32) * f).astype(np.float16)
    games = list(bm["games"]); g0 = len(games); o = nb
    for i, (s, n) in enumerate(zip(shards, lens)):
        D = np.load(s)
        for k in FIELDS:
            mm[k][o:o + n] = D[k] if k != "gid" else g0 + i
        mm["uw"][o:o + n] = D["uw"]; mm["w"][o:o + n] = dw
        key = os.path.basename(s)
        orig = _orig_of(D)     # R06: derived shard -> inherit the original game's split
        entry = {"file": s, "key": key, "names": ["dagger", "dagger"], "scores": [0, 0], "sides": [0],
                 "val": zlib.crc32((orig or key).encode()) % 10 == 0, "src": "dagger", "gi": g0 + i, "n": n}
        if orig:
            entry["orig"] = orig
        games.append(entry)
        o += n
    for k in mm:
        mm[k].flush()
    json.dump({"games": games, "total": nb + nd, "teachers": bm.get("teachers")}, open(os.path.join(out, "meta.json"), "w"))
    print(json.dumps({"base_rows": nb, "dagger_rows": nd, "dagger_games": len(shards), "base_games_kept": len(keep_g), "out": out}))


if __name__ == "__main__":
    main(*sys.argv[1:])
