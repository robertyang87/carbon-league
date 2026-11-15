import gh_api, os, re, base64
OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/recon_code"
os.makedirs(OUT, exist_ok=True)
FILES = ["code/rl2_ppo_rollout.py", "code/rl2_ppo_bench.py", "code/eval_v53.py", "code/rl2_ppo_adapters.py"]
texts = {}
for f in FILES:
    try:
        raw = gh_api.get_file("robertyang87", "carbon-league", f, ref="main")
        if isinstance(raw, dict):
            raw = base64.b64decode(raw["content"]).decode("utf-8", "replace")
        elif isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        texts[f] = raw
        open(os.path.join(OUT, os.path.basename(f)), "w").write(raw)
        print("FETCHED", f, len(raw), "chars", flush=True)
    except Exception as e:
        print("FETCH FAIL", f, type(e).__name__, e, flush=True)

ro = texts.get("code/rl2_ppo_rollout.py", "")
if ro:
    lines = ro.splitlines()
    print("=== rollout: def play_batch at", [i + 1 for i, l in enumerate(lines) if "def play_batch" in l], flush=True)
    for i, l in enumerate(lines):
        if re.search(r"dworth|extinct|\bterm\b|recs", l):
            print(f"RO {i+1}: {l.strip()[:150]}", flush=True)
ev = texts.get("code/eval_v53.py", "")
if ev:
    for i, l in enumerate(ev.splitlines()):
        if re.search(r"extinct|def play_game|term", l):
            print(f"EV {i+1}: {l.strip()[:150]}", flush=True)
bench = texts.get("code/rl2_ppo_bench.py", "")
if bench:
    print("=== BENCH FULL ===", flush=True)
    print(bench[:9000], flush=True)
print("DONE", flush=True)
