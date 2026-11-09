#!/bin/bash
# bringup.sh HOST QUEUE  -- one command from a fresh cloud box to a running exam queue:
# deploy code + venv, push all weights and the exam-only opponents, push exam baselines, smoke-check that one game
# reproduces the Mac bit-for-bit (v17te seed 992001 side 1 -> 1365.1), start QUEUE (a script in code/cloud), start pull.sh.
set -euo pipefail
H=$1; Q=$2; K=~/.ssh/carbon_aliyun_ed25519; S="ssh -i $K -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$H"
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT/code"
bash cloud/deploy.sh "$H"
rsync -az -e "ssh -i $K" "$ROOT/data/weights/" "root@$H:carbon/data/weights/"
rsync -az -e "ssh -i $K" exp/opp_1cz_v9.py exp/opp_1cz_v5.py exp/opp_wb532.py "root@$H:carbon/code/exp/"   # exam-only, never in a submission
rsync -az -e "ssh -i $K" "$ROOT/data/bigeval_live/" "root@$H:carbon/data/bigeval_live/"
$S 'cd ~/carbon/code; OMP_NUM_THREADS=1 BC_W=64 ../.venv/bin/python -c "
import eval_v53
r=eval_v53.play_game((\"t\",\"exp/submission_cnet2_v17te_nc.py\",\"exp/submission_cnet2_v13.py\",992001,1))
print(\"SMOKE\", r[\"margin\"])"' </dev/null | tee /dev/stderr | grep -q "SMOKE 1365.1" || { echo "SMOKE MISMATCH: not starting the queue"; exit 3; }
scp -q -i $K "cloud/$Q" "root@$H:/root/$Q"
$S "setsid nohup bash /root/$Q > /dev/null 2>&1 < /dev/null & sleep 3; pgrep -fa 'bash /root/$Q'" </dev/null
nohup bash cloud/pull.sh "$H" root 300 > /dev/null 2>&1 &
echo "UP $H queue $Q, pull.sh pid $!"
