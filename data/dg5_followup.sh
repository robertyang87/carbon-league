#!/bin/bash
# dg5 follow-up on the Mac (pure net is cheap here, keeps the DGX CPU free): for each chosen epoch E, pull the
# checkpoint, play 200 games vs the Master2s stand-in on the same seeds as m2_dg2n_m2s2_lx3 (new collection ledger),
# then print the economy table next to dg2: margin, deposits, haul per car-step, cell carbon when collecting,
# tree-steps, tree income, planters recruited, zero-yield share, trees per planter.
# usage: bash dg5_followup.sh E1 [E2 ...]
cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=4
for e in "$@"; do
  rsync -az "dgx-via-spot3:junyang/carbon/data/weights/cnet2_b128dg5_ep$e.npz" ../data/weights/ || exit 1
  sed "s/dg2/dg5_ep$e/g" exp/cnet2_dg2s57_240.py > exp/cnet2_dg5e${e}s57_240.py
  grep -q "agent.last_stats" exp/cnet2_dg5e${e}s57_240.py || sed -i '' 's/^    a = _AG(observation, configuration)$/    a = _AG(observation, configuration)\
    agent.last_stats = _AG.stats/' exp/cnet2_dg5e${e}s57_240.py
  ../.venv/bin/python bigeval.py exp/cnet2_dg5e${e}s57_240.py exp/cnet2_m2s2stand.py 996000 200 m2_dg5e${e}n_m2s2_lx3 >> ../data/dg5_followup.log 2>&1
done
cd ../data
python3 - "$@" <<'EOF'
import json, sys, statistics as S
def rows(t): return {(g['seed'], g['side']): g for g in map(json.loads, open(f'bigeval_live/{t}.jsonl')) if g.get('margin') is not None}
B = rows('m2_dg2n_m2s2_lx3')
P = lambda g, k: g['lx']['pol'].get(k, 0) or 0
print("ckpt | paired margin vs dg2 | deposits | haul/car-step | cell C | tree-steps | tree income | planters | zero-yield | trees/planter")
for e in sys.argv[1:]:
    A = rows(f'm2_dg5e{e}n_m2s2_lx3'); K = [k for k in A if k in B]
    d = lambda f: (lambda x: f"{S.mean(x):+.0f}±{S.stdev(x)/len(x)**.5:.0f}")([f(A[k]) - f(B[k]) for k in K])
    ratio = lambda R, a, b: sum(P(R[k], a) for k in K) / max(1, sum(P(R[k], b) for k in K))
    zy = lambda R: S.mean(1 - P(R[k], 'pla_ok') / max(1, P(R[k], 'pla_rec')) for k in K)
    tp = lambda R: S.mean(P(R[k], 'planted') / max(1, P(R[k], 'pla_rec')) for k in K)
    print(f"ep{e} | {d(lambda g: g['margin'])} | {d(lambda g: g['dep'])} | {ratio(A,'haul100','colsteps100'):.2f} vs {ratio(B,'haul100','colsteps100'):.2f} "
          f"| {ratio(A,'cc100_sum','cc100_n'):.1f} vs {ratio(B,'cc100_sum','cc100_n'):.1f} | {d(lambda g: P(g,'treesteps'))} | {d(lambda g: g['tinc'])} "
          f"| {S.mean(P(A[k],'pla_rec') for k in K):.1f} vs {S.mean(P(B[k],'pla_rec') for k in K):.1f} | {zy(A):.2f} vs {zy(B):.2f} | {tp(A):.2f} vs {tp(B):.2f}")
EOF
