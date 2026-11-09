"""ro_count TAG -- the five M-likeness numbers for the rule opener (opponent side of TAG) in a 100-step exam,
plus the dg5 side for reference. Targets (M vs us): planted 9-10, seized ~6, planter clashes ~11, eaten by enemy cars <= 2, own car ~0."""
import json, sys, statistics as S
R = [g for g in map(json.loads, open(f'bigeval_live/{sys.argv[1]}.jsonl')) if g.get('margin') is not None]
o = lambda k, s='opp': S.mean(g['lx'][s].get(k, 0) or 0 for g in R)
print(f"{sys.argv[1]} n {len(R)}: RULE planted {o('planted'):.1f} seized {o('seized'):.1f} recruited {o('pla_rec'):.1f} dead {o('pla_dead'):.1f} "
      f"(zero-yield {o('pla_dead0'):.1f}) t50 {o('tree50'):.1f} tree income {S.mean(g['otinc'] for g in R):.0f} cash+cargo {S.mean(g['lx']['opp']['cash100'] + g['lx']['opp']['cargo100'] for g in R):.0f}"
      f" | dg5 planted {o('planted','pol'):.1f} seized {o('seized','pol'):.1f} t50 {o('tree50','pol'):.1f} tree income {S.mean(g['tinc'] for g in R):.0f}")
