#!/bin/bash
# ppo_run.sh OUT WORKERS GAMES DEVICE [extra args]  -- league PPO from b128v5 (style 7).
# Training pool = own nets only. Held-out exams (v13, b160, 1C-Zero v5/v9, Master2 stand-in) are never in it.
cd "$(dirname "$0")/.."
OUT=$1; W=$2; G=$3; DEV=$4; shift 4
export CNET2_TEACHER=7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64
POOL='[["exp/submission_cnet2_v14.py",2.0],["exp/submission_cnet2_v12.py",1.0],["exp/cnet2_v5t5g.py",1.0],["exp/cnet2_v5t6g.py",1.0],["exp/cnet2_style_t0.py",0.7],["exp/cnet2_style_t3.py",0.7],["exp/cnet2_b128v3_t1_packed.py",0.7]]'
mkdir -p "$OUT"
exec ../.venv/bin/python rl2_ppo_train.py --adapter cnet2 --init ../data/weights/cnet2_b128v5.npz --out "$OUT" \
  --device "$DEV" --workers "$W" --games "$G" --lockstep 2 --rscale 3.33e-4 --sep-value 1 --ratio unit \
  --kl 0.1 --lr 1e-5 --temp 1.0 --eval-every 5 --eval-pairs 16 --eval-opps exp/submission_cnet2_v13.py \
  --ckpt-every 5 --snap-every 10 --c-dep 0.1 --c-lost 0.5 --shape-iters 100 --pool "$POOL" "$@"
