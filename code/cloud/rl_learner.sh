#!/bin/bash
# rl_learner.sh OUT GAMES [extra]  -- Mac learner (MPS) for the split loop; cloud actor plays.
cd "$(dirname "$0")/.."
OUT=$1; G=$2; shift 2
export CNET2_TEACHER=7 BC_W=64
POOL='[["exp/submission_cnet2_v14.py",2.0],["exp/submission_cnet2_v12.py",1.0],["exp/cnet2_v5t6g.py",1.0],["exp/cnet2_style_t0.py",0.7],["exp/cnet2_style_t3.py",0.7],["exp/cnet2_b128v3_t1_packed.py",0.7],["exp/cnet2_v6t5g.py",2.5],["exp/cnet2_v5t5g.py",0.5]]'
mkdir -p "$OUT/remote_in"
exec ../.venv/bin/python rl2_ppo_train.py --adapter cnet2 --init ../data/weights/cnet2_b128v5.npz --out "$OUT" \
  --remote "$OUT/remote_in" --device mps --workers 1 --games "$G" --overlap 1 --rscale 3.33e-4 --sep-value 1 --ratio unit \
  --kl 0.1 --lr 1e-5 --temp 1.0 --eval-every 0 --ckpt-every 5 --snap-every 10 \
  --c-dep 0.1 --c-lost 0.5 --shape-iters 100 --pool "$POOL" "$@"
