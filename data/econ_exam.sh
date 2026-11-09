#!/bin/bash
# Pure-net economy exam on the Mac vs the Master2s stand-in (seeds 996000, 200 games, ledger lx6 with rec60/tree50).
# usage: bash econ_exam.sh NAME WEIGHTS_BASENAME TEACHER TEACHER_LATE LATE_STEP [TEACHER_LATE2 LATE_STEP2]
#   e.g. econ_exam.sh dg7e3s257 cnet2_b128dg7_ep3 2 5 100 7 240     (weights pulled from the DGX if missing)
cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=4
N=$1; W=$2; T=$3; TL=$4; LS=$5; TL2=$6; LS2=$7
[ -f ../data/weights/$W.npz ] || rsync -az "dgx-via-spot3:junyang/carbon/data/weights/$W.npz" ../data/weights/ || exit 1
python3 - "$N" "$W" "$T" "$TL" "$LS" "$TL2" "$LS2" <<'PY' || exit 1
import sys
N, W, T, TL, LS, TL2, LS2 = sys.argv[1:]
s = open("exp/cnet2_dg5e5s57_240.py").read()
s = s.replace("'cnet2_b128dg5_ep5.npz'", repr(W + ".npz"))
extra = f", 'TEACHER_LATE2': {TL2}, 'LATE_STEP2': {LS2}" if TL2 else ""
s = s.replace("'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240}", f"'TEACHER': {T}, 'TEACHER_LATE': {TL}, 'LATE_STEP': {LS}{extra}}}")
s = s.replace('"""b128dg5_ep5', f'"""{N} ({W}, styles {T}/{TL}@{LS}' + (f"/{TL2}@{LS2}" if TL2 else "") + ') was b128dg5_ep5')
assert W in s and f"'TEACHER': {T}," in s
open(f"exp/cnet2_{N}.py", "w").write(s)
PY
../.venv/bin/python bigeval.py exp/cnet2_$N.py exp/cnet2_m2s2stand.py 996000 200 m2_${N}_lx6 >> ../data/econ_exam.log 2>&1
echo "EXAM_DONE $N $(wc -l < ../data/bigeval_live/m2_${N}_lx6.jsonl)" >> ../data/econ_exam.log
