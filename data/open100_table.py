"""open100_table OPP BASE A [A ...] -- opening exam (step 100) table, paired on (seed, side) vs BASE.
Pass lines (revised 2026-11-09): tree income >= the opponent's, planted/seized >= 2, neighbour carbon of planted trees >= 75
(4-neighbour carbon before the planting step; check its scale vs the ladder-replay ledger); 850 absolute only vs an M2 stand-in.
Reference, ladder, steps 0-100: us vs M2/M3 tree income ~650 vs ~1080; planted/seized 6.0/4.8 vs 10.7/2.7;
zero-yield 51% vs 42%; planter distance to nearest enemy tree 3.3 vs 4.4."""
import json, sys, statistics as S
OPP = sys.argv[1]
def rows(t): return {(g['seed'], g['side']): g for g in map(json.loads, open(f'bigeval_live/o100_{OPP}_{t}.jsonl')) if g.get('margin') is not None}
P = lambda g, k, s='pol': g['lx'][s].get(k, 0) or 0
def desc(R, K, s='pol'):
    m = lambda f: S.mean(f(R[k]) for k in K)
    sm = lambda k: sum(P(R[q], k, s) for q in K)
    ti = (lambda g: g['tinc']) if s == 'pol' else (lambda g: g['otinc'])
    return (f"tree income {m(ti):.0f}  tree-steps {m(lambda g: P(g,'treesteps',s)):.0f}  planted/seized {m(lambda g: P(g,'planted',s)):.1f}/{m(lambda g: P(g,'seized',s)):.1f}"
            f" (ratio {sm('planted') / max(1, sm('seized')):.2f})  zero-yield deaths {sm('pla_dead0') / max(1, sm('pla_rec')):.0%}"
            f"  nb-carbon planted/seized {sm('pc4_sum') / max(1, sm('pc4_n')):.0f}/{sm('sc4_sum') / max(1, sm('sc4_n')):.0f}"
            f"  planter deaths clash-standing/clash-moving/enemy-car/own-car {m(lambda g: P(g,'pd_clash_stay',s)):.1f}/{m(lambda g: P(g,'pd_clash_move',s)):.1f}/{m(lambda g: P(g,'pd_ecar',s)):.1f}/{m(lambda g: P(g,'pd_owncar',s)):.1f}"
            f"  cash+cargo@100 {m(lambda g: P(g,'cash100',s) + P(g,'cargo100',s)):.0f}  planters {m(lambda g: P(g,'pla_rec',s)):.1f}  dist to enemy tree {sm('pet_sum') / max(1, sm('pet_n')):.2f}  t50 {m(lambda g: P(g,'tree50',s)):.1f}")
B = rows(sys.argv[2])
print(f"opponent {OPP}")
print(f"  {sys.argv[2]:12s} n {len(B)}  cash margin@100 {S.mean(g['margin'] for g in B.values()):.0f}\n      us:  {desc(B, list(B))}\n      opp: {desc(B, list(B), 'opp')}")
for a in sys.argv[3:]:
    A = rows(a); K = [k for k in A if k in B]
    d = lambda f: (lambda x: f"{S.mean(x):+.0f} ± {S.stdev(x)/len(x)**.5:.0f}")([f(A[k]) - f(B[k]) for k in K])
    print(f"  {a:12s} n {len(K)}  paired: plant spend {d(lambda g: g.get('pspend', 0))} (ours {S.mean(A[k].get('pspend', 0) for k in K):.0f} vs base {S.mean(B[k].get('pspend', 0) for k in K):.0f}), cash+cargo margin {d(lambda g: P(g,'cash100') + P(g,'cargo100') - P(g,'cash100','opp') - P(g,'cargo100','opp'))} (main), cash {d(lambda g: g['margin'])}, tree income {d(lambda g: g['tinc'])}, tree-steps {d(lambda g: P(g,'treesteps'))}, deposits {d(lambda g: g['dep'])}"
          f"\n      us:  {desc(A, K)}\n      opp: {desc(A, K, 'opp')}")

def verdict(R, K):
    sm = lambda k, s='pol': sum(P(R[q], k, s) for q in K)
    ti, oti = S.mean(R[k]['tinc'] for k in K), S.mean(R[k]['otinc'] for k in K)
    ratio = sm('planted') / max(1, sm('seized')); pc = sm('pc4_sum') / max(1, sm('pc4_n'))
    gap = pc - sm('pc4_sum', 'opp') / max(1, sm('pc4_n', 'opp'))   # landing-spot gap: the only placement reading that matters when land is contested
    life = sm('life50_sum') / max(1, sm('life50_n')); lost = S.mean(P(R[k], 'seized', 'opp') for k in K); t100 = S.mean(P(R[k], 'tree100') for k in K)
    ladder = f" | LADDER-GAP LINES: planted-tree life {'ok' if life >= 24 else 'NO'} ({life:.1f}, >= 24), seized from us {'ok' if lost <= 4 else 'NO'} ({lost:.1f}, <= 4), trees standing at 100 {'ok' if t100 >= 4 else 'NO'} ({t100:.1f}, >= 4)"
    return ladder + f" | PASS-LINES: tree income vs opp {'ok' if ti >= oti else 'NO'} ({ti:.0f}/{oti:.0f}), planted/seized {'ok' if ratio >= 2 else 'NO'} ({ratio:.2f}), nb-carbon planted {'ok' if pc >= 75 else 'NO'} ({pc:.0f}), nb-carbon gap us - opp {gap:+.0f}"
for a in sys.argv[3:]:
    A = rows(a); K = [k for k in A if k in B]; srch = [A[k] for k in K if A[k].get('srch')]
    pk = f"  planter in search {sum(g.get('pk_in', 0) for g in srch) / max(1, len(srch)):.1f} steps/game, cut {sum(g.get('pk_cut', 0) for g in srch) / max(1, len(srch)):.1f}, planter share of replaced moves {sum(g.get('pk_rep', 0) for g in srch) / max(1, sum(g.get('chg') or 0 for g in srch)):.0%}, planted within 3 steps of a replacement {sum(g.get('pk_rep3', 0) for g in srch) / max(1, sum(g.get('pk_rep', 0) for g in srch)):.0%}, cash covered planting at the replacement {sum(g.get('pk_repc', 0) for g in srch) / max(1, sum(g.get('pk_rep', 0) for g in srch)):.0%}, searched {S.mean(g.get('srch') or 0 for g in srch):.0f} / shallow {S.mean(g.get('shal') or 0 for g in srch):.0f} steps/game, worst step {max(g.get('wms') or 0 for g in srch):.0f} ms" if srch else ""
    print(f"  {a:12s} {verdict(A, K)}{pk}")
