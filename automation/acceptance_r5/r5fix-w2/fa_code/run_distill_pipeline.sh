#!/bin/bash
# Durable distill pipeline: wait for dataset -> train (checkpointed,
# resumable) -> offline fidelity eval.  Launched via setsid so VM /
# session teardown does not kill it; each stage logs to its own file.
cd ~/workspace/carbon-league || exit 1
rm -f distill_pipeline_done.txt
while true; do
  python3 -c "import numpy as np; d=np.load('bc_data/distill_dataset.npz'); _=d['W_x'].shape" 2>/dev/null && break
  sleep 30
done
echo "dataset ready $(date)" >> train_distill.log
./venv/bin/python train_distill.py >> train_distill.log 2>&1
echo "train exit=$? $(date)" >> train_distill.log
python3 distill_offline_eval.py >> offline_eval.log 2>&1
echo "offline exit=$? $(date)" >> offline_eval.log
echo done > distill_pipeline_done.txt
