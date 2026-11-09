#!/bin/bash
# q7: next batch, so neither host idles. Runs after this host's current queue finishes (WAIT_FILE).
# REVERSE=1 runs the list bottom-up (the DGX), so the two hosts meet in the middle; sync_hosts.sh shares results.
# usage: bash q7.sh ROOT_DIR WAIT_FILE [REVERSE]
R=$1; W=$2; REV=${3:-0}
cd $R/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=$(( $(nproc) > 32 ? $(nproc) - 4 : $(nproc) ))
B=../.venv/bin/python; P=../data/q7.progress
while [ ! -f "$W" ]; do sleep 30; done
run() {  # run PACK OPP SEED N TAG
  local f=../data/bigeval_live/$5.jsonl
  if [ -f "$f" ] && [ "$(grep -c '"margin": -\?[0-9]' "$f")" -ge "$4" ]; then echo "$(date +%T) skip $5" >> $P; return; fi
  [ -f "$f" ] && { echo "$(date +%T) skip $5 (other run in progress)" >> $P; return; }
  $B bigeval.py exp/$1.py exp/$2.py $3 $4 $5 >> ../data/q7.log 2>&1; echo "$(date +%T) $5" >> $P
}
JOBS=(
"submission_cnet2_v17d2_nc cnet2_wsstand 999500 100 c_v17d2_ws"
"submission_cnet2_v17d2_nc cnet2_m2s2stand 996200 200 c_v17d2_m2s2b"
"submission_cnet2_v17c_nc cnet2_m2s2stand 996200 200 c_v17c_m2s2b"
)
if [ "$REV" = 1 ]; then for ((i=${#JOBS[@]}-1; i>=0; i--)); do run ${JOBS[$i]}; done; else for j in "${JOBS[@]}"; do run $j; done; fi
echo DONE > ../data/q7.done
