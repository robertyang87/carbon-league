#!/bin/bash
echo S0V1_BEGIN
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
set +e
echo "== HOST =="; hostname; date '+%F %T %Z'
echo "== LEFTOVER SWEEP =="
for p in $(pgrep -f 'git (fetch|pull|remote|clone)'); do echo "LEFTOVER PID $p: $(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null)"; kill $p 2>/dev/null && echo "killed $p"; done
echo "-- git/ssh procs now:"; ps aux | grep -E 'git|ssh' | grep -v grep | head -8
echo "== REPO PULL =="
cd /home/arm/junyang/carbon-repo || { echo NOREPO; exit 1; }
echo "branch: $(git rev-parse --abbrev-ref HEAD) head_before: $(git rev-parse HEAD)"
timeout 60 git fetch origin main 2>&1 | tail -2
timeout 60 git pull --ff-only origin main 2>&1 | tail -2
echo "head_after: $(git rev-parse HEAD)"
git merge-base --is-ancestor b5f8dd46e09b911fd2d7e8c9074599d78d1fb25c HEAD && echo ANCESTOR-B5F8-OK || echo ANCESTOR-B5F8-MISS
echo "== MD5 STEP0 =="
md5sum code/eval_v53.py code/bigeval.py code/cnet2_search.py code/rl2_ppo_rollout.py
echo "== GPU/ENV =="
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | head -2
ls -d /home/arm/junyang/carbon-repo/.venv /home/arm/junyang/carbon/.venv /home/arm/junyang/venv 2>/dev/null
echo "== ASSETS =="
find /home/arm/junyang -maxdepth 4 -name 'cnet2_m3s2r1L.npz' 2>/dev/null
find /home/arm/junyang -maxdepth 3 -type d -name 'ds_m3s2*' 2>/dev/null
echo "== CODE LIST =="
ls code/ | head -60
echo "== ROLLOUT HELP =="
timeout 30 python3 code/rl2_ppo_rollout.py --help 2>&1 | head -30
echo "== TRAINER/MODEL GREP =="
grep -ln 'cross_entropy\|log_softmax' code/*.py 2>/dev/null | head
grep -n 'class .*Net\|class .*Model' code/*.py 2>/dev/null | head
DGXEOF
echo "ssh_rc=$?"
echo S0V1_END
