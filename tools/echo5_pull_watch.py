#!/usr/bin/env python3
"""Watch Echo5 placement; the moment it completes, pull full matches+replays, verify, manifest."""
import subprocess, json, os, sys, time, gzip, hashlib

REPO = "/home/arm/junyang/carbon-repo"
ENV = dict(os.environ, LADDER_BASE="https://117.72.210.134")
SID = "0d57bed456627f6586341c28959df780"
LABEL = "echo5"

def run(args, timeout=300):
    return subprocess.run(args, cwd=REPO, env=ENV, capture_output=True, text=True, timeout=timeout)

def mine_echo5():
    r = run(["python3", "code/ladder.py", "mine"], timeout=120)
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            x = json.loads(line)
        except Exception:
            continue
        if x.get("name") == "Echo5":
            return x
    return None

def ts():
    return time.strftime("%H:%M:%S")

print(f"[{ts()}] watcher start", flush=True)
placed_at = None
final = None
for i in range(40):
    x = mine_echo5()
    if x is None:
        print(f"[{ts()}] poll {i}: Echo5 not found in mine", flush=True)
    else:
        print(f"[{ts()}] poll {i}: games={x.get('games')} phase={x.get('ranking_phase')} status={x.get('status')} rating={x.get('rating')} rank={x.get('rank')}", flush=True)
        if (x.get("games") or 0) >= 40 or x.get("ranking_phase") != "placement":
            placed_at = ts()
            final = x
            break
    time.sleep(90)

if final is None:
    print("PLACEMENT_NOT_CONFIRMED_WITHIN_WINDOW", flush=True)
    sys.exit(2)

print(f"[{ts()}] placement complete at {placed_at}: {json.dumps(final)}", flush=True)

r = run(["python3", "code/ladder.py", "matches", SID, f"data/{LABEL}_matches.json"], timeout=280)
print("matches out:", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:], flush=True)

r = run(["python3", "code/ladder.py", "replays", f"data/{LABEL}_matches.json", f"data/{LABEL}_corpus"], timeout=900)
print("replays out:", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:], flush=True)

lst = json.load(open(os.path.join(REPO, f"data/{LABEL}_matches.json")))
games = lst if isinstance(lst, list) else lst.get("matches", [])
ids = set(str(g.get("id") or g.get("game_id") or g.get("replay_id")) for g in games)
cdir = os.path.join(REPO, f"data/{LABEL}_corpus")
files = sorted(os.listdir(cdir))
ok = bad = 0
lines = []
for fn in files:
    p = os.path.join(cdir, fn)
    try:
        with gzip.open(p, "rt") as f:
            d = json.load(f)
        assert "steps" in d
        ok += 1
    except Exception:
        bad += 1
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    lines.append(f"{h} data/{LABEL}_corpus/{fn}")
mpath = os.path.join(REPO, f"data/{LABEL}_corpus.manifest")
open(mpath, "w").write("\n".join(lines) + "\n")
fids = set(fn[:-len(".json.gz")].removesuffix("-1") for fn in files)
def md5_of(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()
print("=== ECHO5 PULL SUMMARY ===", flush=True)
print("placed_at_cst:", placed_at, flush=True)
print("final_mine:", json.dumps(final), flush=True)
print("list_games:", len(ids), "list_md5:", md5_of(os.path.join(REPO, f"data/{LABEL}_matches.json")), flush=True)
print("corpus_files:", len(files), "OK:", ok, "BAD:", bad, flush=True)
print("id_sets_equal:", ids == fids, "list_only:", len(ids - fids), "corpus_only:", len(fids - ids), flush=True)
print("manifest_lines:", len(lines), "manifest_md5:", md5_of(mpath), flush=True)
