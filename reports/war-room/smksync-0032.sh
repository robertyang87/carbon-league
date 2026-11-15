#!/bin/bash
# Final-smoke step 1: sync DGX repo to 0f704a9d (fetch + ff-only), verify HEAD + code md5s,
# trio check (no training), and inspect fastsim/CFG surface for the extinction construction.
# Marker: SMKSYNC0032A
MARK=SMKSYNC0032A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "SMKSYNC0032A DGX-BEGIN $(date '+%F %T %Z')"
hostname
echo '--- GIT SYNC ---'
cd /home/arm/junyang/carbon-repo
git rev-parse HEAD
timeout 60 git fetch origin 2>&1 | tail -3
git merge --ff-only 0f704a9da8da05e68976bfd1442467b52106bf5c 2>&1 | tail -3
echo "HEAD-AFTER $(git rev-parse HEAD)"
git status --porcelain | head -5
echo '--- CODE MD5 ---'
md5sum code/rl2_ppo_rollout.py code/eval_v53.py code/cnet2_search.py code/cnet2_np.py
echo '--- TRIO ---'
ps aux | grep -E 'cnet2_train|play_batch|bigeval|rl2_ppo' | grep -v grep || echo "no train/play processes"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader
echo '--- FASTSIM/CFG INSPECT ---'
cd /home/arm/junyang/carbon-repo/code && timeout 120 /home/arm/junyang/carbon/.venv/bin/python - <<'PYEOF'
import os, sys, json
sys.path.insert(0, os.getcwd())
import eval_v53
from fastsim import FastCarbon
import fastsim
print("fastsim file:", fastsim.__file__)
print("EPISODE_STEPS:", eval_v53.EPISODE_STEPS)
cfg = eval_v53.CFG
print("CFG type:", type(cfg).__name__)
try:
    items = {k: cfg[k] for k in sorted(cfg.keys())}
    print("CFG:", json.dumps(items, default=str)[:1500])
except Exception as e:
    print("CFG dump error:", repr(e))
sim = FastCarbon(cfg).reset(seed=991001)
st = sim.get_state()
def cnt(d, p):
    return sum(1 for v in d.values() if isinstance(v, dict) and v.get("player") == p)
print("reset state: cash", st["cash"],
      "workers", {p: cnt(st["workers"], p) for p in (0, 1)},
      "trees", {p: cnt(st["trees"], p) for p in (0, 1)},
      "bases", {p: bool(st["bases"].get(p)) for p in (0, 1)})
print("state keys:", sorted(st.keys()))
print("sim attrs:", [a for a in dir(sim) if not a.startswith("_")])
for a in ("workers", "trees", "bases", "cash", "carbon", "state", "_state"):
    if hasattr(sim, a):
        v = getattr(sim, a)
        print("attr", a, type(v).__name__, (len(v) if hasattr(v, "__len__") else v))
# is get_state() a live view? mutate a copy marker and re-read
try:
    n0 = len(st["workers"])
    if st["workers"]:
        k0 = next(iter(st["workers"]))
        st["workers"].pop(k0)
        st2 = sim.get_state()
        print("get_state live-view test: before", n0, "after pop+reget", len(st2["workers"]))
except Exception as e:
    print("live-view test error:", repr(e))
PYEOF
echo "SMKSYNC0032A DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
