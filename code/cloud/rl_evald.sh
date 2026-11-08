#!/bin/bash
# rl_evald.sh [EVERY] [WORKERS] -- cloud: evaluate every EVERY-th learner weight (rlc/in/live/w_<it>.npz)
# vs v13 (seeds 992000, 200 games) and WB-532 (seeds 995000, 200 games); paired baselines c_v12g_* already exist.
cd ~/carbon/code; RLC=${RLC:-rlc2}; E=${1:-5}; W=${2:-24}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=$W CNET2_TEACHER=7
mkdir -p ../data/$RLC/eval
while true; do
  for f in $(ls ../data/$RLC/in/live/ 2>/dev/null | sed -n 's/^w_\([0-9]*\)\.npz$/\1/p' | sort -n); do
    [ $((f % E)) -eq 0 ] && [ "$f" -gt 0 ] || continue
    [ -f ../data/$RLC/eval/done_$f ] && continue
    cp ../data/$RLC/in/live/w_$f.npz ../data/$RLC/eval/w_$f.npz || continue
    ok=1
    rm -f ../data/bigeval_live/rl_${f}_v13.jsonl ../data/bigeval_live/rl_${f}_wb532.jsonl ../data/bigeval_live/rl_${f}_m2s.jsonl   # retries start clean
    RL_W=$(pwd)/../data/$RLC/eval/w_$f.npz ../.venv/bin/python bigeval.py exp/cnet2_rlg.py exp/submission_cnet2_v13.py 992000 200 rl_${f}_v13 >> ../data/$RLC/eval/evald.log 2>&1 || ok=0
    RL_W=$(pwd)/../data/$RLC/eval/w_$f.npz ../.venv/bin/python bigeval.py exp/cnet2_rlg.py exp/opp_wb532.py 995000 200 rl_${f}_wb532 >> ../data/$RLC/eval/evald.log 2>&1 || ok=0
    RL_W=$(pwd)/../data/$RLC/eval/w_$f.npz ../.venv/bin/python bigeval.py exp/cnet2_rlg.py exp/cnet2_m2stand.py 994000 50 rl_${f}_m2s >> ../data/$RLC/eval/evald.log 2>&1 || ok=0
    # only a complete, error-free exam is marked done; otherwise it is retried next pass and logged
    if [ $ok = 1 ]; then touch ../data/$RLC/eval/done_$f; else echo "$(date +%T) it=$f INCOMPLETE, will retry" >> ../data/$RLC/eval/evald.log; fi
  done
  sleep 30
done
