#!/bin/zsh
# gate_watch.sh RUNDIR PREFIX -- gate every rl_it{10,20,...}.npz once (64 games vs v9, 2 workers)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=${EVO_WORKERS:-2}
RUN=$1; PFX=$2
while true; do
  for f in $(ls $RUN/rl_it*0.npz 2>/dev/null | sort -V); do
    it=$(basename $f .npz); tag=${PFX}_${it}
    grep -q "\"tag\": \"$tag\"" ../data/gate_results.jsonl 2>/dev/null && continue
    ../.venv/bin/python gate.py $f $tag 32 >> ../data/gate_watch.log 2>&1
  done
  sleep 120
done
