#!/bin/bash
# Backlog filler: keeps a host busy with low-priority exams whenever its scheduled work (queues, exam chains) leaves
# cores idle, and gets out of the way the moment scheduled work runs.
#   usage: filler.sh CODE_DIR WORKERS            backlog file: CODE_DIR/../data/backlog.txt (re-read every pass, so
#                                                jobs can be appended at any time without restarting the filler)
# Backlog line:  TAG SEED N PACK OPP [REQUIRED_FILE]      (# comments allowed)
#   TAG must contain "_bk" (that is how the filler tells its own games from scheduled ones).
#   PACK / OPP are exp/ module names without .py; REQUIRED_FILE (relative to CODE_DIR) must exist before the job runs
#   (e.g. weights that a training run will produce later). Finished jobs (>= N result rows) are skipped, so the same
#   tag can sit in two hosts' backlogs: whichever finishes it first wins, and sync_hosts brings the file across.
# Priority: any bigeval process WITHOUT "_bk" in its command line is scheduled work; while one runs, the filler's games
# are SIGSTOPped and resumed afterwards. Only use packs without clock caps (_nc / pure net): pausing then changes nothing.
cd "$1" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=$2
B=../.venv/bin/python; Q=../data/backlog.txt; P=../data/backlog.progress; touch $Q ../data/backlog.fail
sched() { ps -eo args | grep "[b]igeval.py exp" | grep -v "_bk" | grep -q .; }   # not grep -vq: wrong on BSD grep
tree() { for p in $(pgrep -f "[b]igeval.py exp.*_bk"); do echo $p; pgrep -P $p; done; }   # macOS spawns pool workers
sig() { local t; t=$(tree); [ -n "$t" ] && kill -$1 $t 2>/dev/null; }                     # with a different cmdline
FP=$$
( while true; do
    if sched; then sig STOP; else sig CONT; fi
    sleep 10; kill -0 $FP 2>/dev/null || { sig CONT; exit; }    # (BSD pgrep never matches its own ancestors)
  done ) &
done_rows() { local f="../data/bigeval_live/$1.jsonl"; if [ -f "$f" ]; then grep -c '"margin": -\?[0-9]' "$f"; else echo 0; fi; }
while true; do
  ran=0
  while read -r tag seed n pack opp req; do
    case "$tag" in ''|\#*) continue;; esac
    [[ "$tag" == *_bk* ]] || { echo "$(date +%T) bad tag (needs _bk): $tag" >> $P; continue; }
    [ "$(done_rows $tag)" -ge "$n" ] && continue
    [ -n "$req" ] && [ ! -f "$req" ] && continue
    [ "$(grep -cx "$tag" ../data/backlog.fail)" -ge 2 ] && continue   # failed twice: leave it
    while sched; do sleep 30; done                       # don't start a new job while scheduled work runs
    echo "$(date +%T) start $tag" >> $P
    rm -f "../data/bigeval_live/$tag.jsonl"
    $B bigeval.py exp/$pack.py exp/$opp.py $seed $n $tag >> ../data/backlog.log 2>&1 < /dev/null
    r=$(done_rows $tag); echo "$(date +%T) end $tag $r/$n" >> $P; [ "$r" -lt "$n" ] && echo "$tag" >> ../data/backlog.fail
    ran=1; break   # re-read the backlog after every job
  done < $Q
  [ $ran = 0 ] && sleep 60
done
