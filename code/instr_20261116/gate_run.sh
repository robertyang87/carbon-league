#!/bin/bash
# Equivalence gate runner: gate_run.sh <off|on>
# One formal round (it=51) resumed from state_frozen_it50.pt, CPU, no servers,
# identical argv except --game-log and --out. Clock pinned by gate_pin_time.py.
set -u
MODE="$1"
if [ "$MODE" = "off" ]; then GL=0; else GL=1; fi
OUT=/root/carbon/gate_$MODE
cd /root/carbon/code_instr
export CNET2_TEACHER=2 RL2_SNAP_TEMP=0.5 CNET2_TEMP_KIND=0.15,0.3,0.3
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
POOL='[["SNAP:cnet2:/root/carbon/data/weights/cnet2_m3s2r1L.npz#T2", 1.1], ["SNAP:cnet2:/root/carbon/data/weights/cnet2_m3s2g1L.npz#T2", 0.55], ["SNAP:cnet2:/root/carbon/data/weights/cnet2_m7bsr1L.npz#T2", 0.55], ["SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj260wr1_fromstudio.npz#T2", 0.5], ["SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj260r1.npz#T2", 0.45], ["SNAP:cnet2:/root/carbon/data/weights/cnet2_m6r1L.npz#T2", 0.45], ["SNAP:cnet2:/root/carbon/data/weights/cnet2_m7bsv2r1.npz#T2", 0.425], ["SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj260g1.npz#T2", 0.35], ["SNAP:cnet2:/root/carbon/data/weights/cnet2_m7bsv2g1.npz#T2", 0.3], ["SNAP:cnet2:/root/carbon/data/weights/cnet2_caragor1_ep19.npz#T2", 0.275]]'
nice -n 10 /root/carbon/.venv/bin/python /root/carbon/code_instr/gate_pin_time.py /root/carbon/code_instr/rl2_ppo_train.py \
  --adapter cnet2 --device cpu --server 0 --reward u --value-head tanh \
  --init /root/carbon/data/weights/cnet2_m3s2r1L.npz \
  --pool-authoritative --pool "$POOL" \
  --self-prior 0.5 --snap-prior 1 --snap-every 10 --max-snaps 2 --vwarm 0 \
  --workers 7 --lockstep 16 --dec-frac 0.25 --epochs 1 --mb 768 --pre-mb 4096 \
  --games 96 --lr 1e-5 --kl 0.05 --gamma 1.0 --rscale 1.0 --ckpt-every 1 \
  --eval-every 5 --eval-pairs 32 --eval-opps exp/opp_master2r.py,exp/opp_m3echo.py \
  --seed0 3000000 --lam 0.95 --iters 1 \
  --out "$OUT" --resume /root/carbon/data/rl3/ppou2_m/state_frozen_it50.pt \
  --ref /root/carbon/data/rl3/ppou2_m/ckpt_37.npz --game-log $GL
echo "GATE_RUN_${MODE}_RC=$?"
