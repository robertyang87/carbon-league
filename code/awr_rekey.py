"""awr_rekey CORPUS_DIR TAG -- make the game keys of an awr_gen corpus unique across corpora.

awr_gen names a game awr_<seed>_<k>. Two corpora generated from the same seed range (one per opponent) therefore share
every file name, and cnet2_build dedups games by file name: only the first directory would be built, and awr_weights
(keyed by game key) would hand those rows the weights of another corpus. This renames
  replays/awr_<seed>_<k>.json.gz -> replays/<TAG>_awr_<seed>_<k>.json.gz
and rewrites index.json, awr.jsonl and awr_raw.jsonl to match. Safe to run twice (already-tagged games are skipped).
New corpora: set AWR_TAG=<TAG> for awr_gen instead."""
import json
import os
import sys


def main():
    d, tag = sys.argv[1], sys.argv[2]
    pre = tag + "_"
    n = 0
    for fn in sorted(os.listdir(os.path.join(d, "replays"))):
        if fn.startswith("awr_") and fn.endswith(".json.gz"):
            os.rename(os.path.join(d, "replays", fn), os.path.join(d, "replays", pre + fn)); n += 1

    def fix(g, kf="key"):
        if g.get(kf, "").startswith("awr_"):
            g[kf] = pre + g[kf]
        if g.get("file", "").startswith("replays/awr_"):
            g["file"] = "replays/" + pre + g["file"][len("replays/"):]
        return g
    for name in ("awr.jsonl", "awr_raw.jsonl"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            rows = [fix(json.loads(l)) for l in open(p) if l.strip()]
            with open(p + ".tmp", "w") as f:
                for g in rows:
                    f.write(json.dumps(g) + "\n")
            os.replace(p + ".tmp", p)
    p = os.path.join(d, "index.json")
    if os.path.exists(p):
        idx = [fix(g, "id") for g in json.load(open(p))]
        json.dump(idx, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
        miss = sum(1 for g in idx if not os.path.exists(os.path.join(d, g["file"])))
        print(json.dumps({"dir": d, "tag": tag, "renamed": n, "index_games": len(idx), "missing_files": miss}))


if __name__ == "__main__":
    main()
