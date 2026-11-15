import gh_api, base64, os
OUT = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/recon_code"
def get(path):
    d = gh_api.get_file("robertyang87", "carbon-league", path, ref="main")
    return base64.b64decode(d["content"]).decode("utf-8", "replace")
s = get("code/cnet2_np.py")
open(os.path.join(OUT, "cnet2_np.py"), "w").write(s)
print("cnet2_np.py chars", len(s), flush=True)
lines = s.splitlines()
for i, l in enumerate(lines):
    if "act_tta" in l or "def encode" in l or "class Policy" in l or "SYMM" in l or "def act" in l:
        print(f"N {i+1}: {l.strip()[:160]}", flush=True)
i0 = next(i for i, l in enumerate(lines) if "def act_tta" in l)
print("=== act_tta body ===", flush=True)
print("\n".join(lines[i0:i0 + 45]), flush=True)
