"""econ_table BASE A [A ...] -- pure-net economy table, paired on (seed, side) vs BASE (tags m2_<name>_lx6).
Main readings for opening changes (opponent-independent): trees/planter (target >= 0.8; M2 makes 1.0 vs us)
and zero-yield share (target <= 0.35); mid-game deposits (100-240) must not fall. Neither exam opponent reproduces
the opening pressure of the real Master2s / M2 (ladder t50 3 vs 4.8 / 3.0 vs 5.2), so margin and t50 move little here.
Columns: paired margin and deposits (mean ± SE), trees at step 50 ours/opp, zero-yield planter share,
planter share of recruits before step 60, planters recruited, haul per car-step to 100, trees per planter."""
import json, os, sys, statistics as S
PRE, DIR = os.environ.get('ET_PREFIX', 'm2'), os.environ.get('ET_DIR', 'bigeval_live')   # v9 + dgx_live for the 1C-v9 group
def rows(t): return {(g['seed'], g['side']): g for g in map(json.loads, open(f'{DIR}/{PRE}_{t}_lx6.jsonl')) if g.get('margin') is not None}
P = lambda g, k, s='pol': g['lx'][s].get(k, 0) or 0
B = rows(sys.argv[1])
def desc(R, K):
    m = lambda f: S.mean(f(R[k]) for k in K)
    return (f"tree50 {m(lambda g: P(g,'tree50')):.1f}/{m(lambda g: P(g,'tree50','opp')):.1f}  "
            f"zero-yield {m(lambda g: 1 - P(g,'pla_ok') / max(1, P(g,'pla_rec'))):.2f}  "
            f"rec<60 planter share {sum(P(R[k],'rec60_p') for k in K) / max(1, sum(P(R[k],'rec60_p') + P(R[k],'rec60_c') for k in K)):.2f}  "
            f"planters {m(lambda g: P(g,'pla_rec')):.1f}  haul/car-step {sum(P(R[k],'haul100') for k in K) / max(1, sum(P(R[k],'colsteps100') for k in K)):.2f}  "
            f"dep 0-100 {m(lambda g: P(g,'dep100')):.0f}  dep 100-240 {m(lambda g: P(g,'dep240') - P(g,'dep100')):.0f}  trees/planter {m(lambda g: P(g,'planted') / max(1, P(g,'pla_rec'))):.2f}  dep {m(lambda g: g['dep']):.0f}  margin {m(lambda g: g['margin']):.0f}")
print(f"{sys.argv[1]:14s} n {len(B)}  {desc(B, list(B))}")
for a in sys.argv[2:]:
    A = rows(a); K = [k for k in A if k in B]
    d = lambda f: (lambda x: f"{S.mean(x):+.0f} ± {S.stdev(x)/len(x)**.5:.0f}")([f(A[k]) - f(B[k]) for k in K])
    print(f"{a:14s} n {len(K)}  paired margin {d(lambda g: g['margin'])}, deposits {d(lambda g: g['dep'])}, dep 0-100 {d(lambda g: P(g,'dep100'))}, dep 100-240 {d(lambda g: P(g,'dep240') - P(g,'dep100'))}\n{'':16s}{desc(A, K)}")
