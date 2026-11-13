#!/bin/bash
# Harvest sparring-corpus games (Claude family esp. AlphaCarbon-2, plus
# NaMnSi-DeepSeek-RL and AgyCarbon — sparring/shadow material only) from the
# public replay snapshot repo into replays/accumulated/, deduped by match id.
# The snapshot repo is force-published with a rolling ~7d window, so each
# run clones fresh into a temp dir and copies out what we track.
set -u
cd ~/workspace/carbon-league
TMP=$(mktemp -d /tmp/replaysnap.XXXXXX)
ACC=replays/accumulated
mkdir -p "$ACC"
LOG=replays/accumulated/harvest.log

git clone --quiet --depth 1 https://github.com/JUZI1/carbon-league-replays.git "$TMP/snap" 2>/dev/null
if [ ! -f "$TMP/snap/index.json" ]; then
  echo "$(date -Is) CLONE_FAIL" >> "$LOG"
  rm -rf "$TMP"
  exit 1
fi

python3 - "$TMP/snap" "$ACC" <<'EOF'
import json, os, shutil, sys, datetime
snap, acc = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(snap, "index.json")))
targets = ("alphacarbon", "claude-opus", "deepseek", "agycarbon")  # name substrings, lowercase
new = 0
for m in idx:
    names = [m["strategies"][s]["name"].lower() for s in ("a", "b")]
    if not any(t in n for n in names for t in targets):
        continue
    src = os.path.join(snap, m["file"])
    dst = os.path.join(acc, os.path.basename(m["file"]))
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
        new += 1
    reg_path = os.path.join(acc, "index.json")
    reg = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    if m["id"] not in reg:
        reg[m["id"]] = {
            "file": os.path.basename(m["file"]),
            "a": m["strategies"]["a"]["name"], "b": m["strategies"]["b"]["name"],
            "winner": m.get("winner"), "scores": m.get("scores"),
        }
        json.dump(reg, open(reg_path, "w"), ensure_ascii=False, indent=1)
print(f"snapshot={len(idx)} new_harvested={new}")
with open(os.path.join(acc, "harvest.log"), "a") as f:
    f.write(f"{datetime.datetime.now().isoformat()} snapshot={len(idx)} new={new}\n")
EOF
rm -rf "$TMP"
