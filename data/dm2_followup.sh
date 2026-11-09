#!/bin/bash
# dm2 follow-up on the Mac (pure net): dm2 epoch E with TEACHER_LATE = 2 (the M2 slot) vs dg5 epoch 5 with
# TEACHER_LATE = 7, same 200 stand-in seeds, ledger with rec259_275 / rec276. Prints paired margin, strand, recruits 259-275 and from 276 (M2: ~2.5 and 0),
# enemy-seized last 30, deposits, tree-steps.   usage: bash dm2_followup.sh E
cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=4
E=$1; LS=${2:-240}; TAG=${3:-}
rsync -az "dgx-via-spot3:junyang/carbon/data/weights/cnet2_b128dm2_ep$E.npz" ../data/weights/ || exit 1
sed "s/dg2/dm2_ep$E/g; s/'TEACHER_LATE': 7/'TEACHER_LATE': 2/; s/'LATE_STEP': 240/'LATE_STEP': $LS/" exp/cnet2_dg2s57_240.py > exp/cnet2_dm2e${E}s57_${LS}.py
grep -q "'TEACHER_LATE': 2" exp/cnet2_dm2e${E}s57_${LS}.py && grep -q "'LATE_STEP': $LS" exp/cnet2_dm2e${E}s57_${LS}.py || { echo "wrapper not switched"; exit 1; }
[ -f ../data/bigeval_live/m2_dg5e5n_m2s2_lx5.jsonl ] || ../.venv/bin/python bigeval.py exp/cnet2_dg5e5s57_240.py exp/cnet2_m2s2stand.py 996000 200 m2_dg5e5n_m2s2_lx5 >> ../data/dm2_followup.log 2>&1
../.venv/bin/python bigeval.py exp/cnet2_dm2e${E}s57_${LS}.py exp/cnet2_m2s2stand.py 996000 200 m2_dm2e${E}s${LS}n_m2s2_lx5 >> ../data/dm2_followup.log 2>&1
cd ../data
python3 - "$E" "$LS" <<'EOF'
import json, sys, statistics as S
def rows(t): return {(g['seed'], g['side']): g for g in map(json.loads, open(f'bigeval_live/{t}.jsonl')) if g.get('margin') is not None}
A = rows(f'm2_dm2e{sys.argv[1]}s{sys.argv[2]}n_m2s2_lx5'); B = rows('m2_dg5e5n_m2s2_lx5'); K = [k for k in A if k in B]
P = lambda g, k: g['lx']['pol'].get(k, 0) or 0
d = lambda f: (lambda x: f"{S.mean(x):+.1f} ± {S.stdev(x)/len(x)**.5:.1f}")([f(A[k]) - f(B[k]) for k in K])
m = lambda R, f: S.mean(f(R[k]) for k in K)
print(f"dm2 ep{sys.argv[1]} (late slot 2 from step {sys.argv[2]}) vs dg5 ep5 (late slot 7 from 240), n {len(K)}")
for nm, f in [("paired margin", lambda g: g['margin']), ("strand", lambda g: g.get('strand') or 0),
              ("recruits steps 259-275", lambda g: P(g, 'rec259_275')), ("recruits from step 276", lambda g: P(g, 'rec276')), ("enemy-seized last 30", lambda g: P(g, 'late30')),
              ("haul after step 240", lambda g: P(g, 'haul240')), ("deposits", lambda g: g['dep']), ("tree-steps", lambda g: P(g, 'treesteps'))]:
    print(f"  {nm:26s} {d(f):>14s}   (dm2 {m(A, f):.1f} / dg5 {m(B, f):.1f})")
EOF
