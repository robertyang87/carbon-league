"""planter_check TAG [TAG...] -- planter/tree ledger (ours / opponent): planters that succeeded at least once,
successes per successful planter, share of trees planted (vs seized), mean life of our own planted trees.
v16b ladder baseline: 46% / 1.70 / 57% / 28 steps; targets (Master2s level): >=54% / >=1.85 / >=65% / >=33."""
import json, sys
for tag in sys.argv[1:]:
    R = [json.loads(l) for l in open(f'../data/bigeval_live/{tag}.jsonl')]
    R = [r for r in R if r.get('lx') and 'pla_ok' in r['lx']['pol']]
    if not R: print(tag, 'no planter ledger (run predates the field)'); continue
    def agg(side):
        S = lambda k: sum(r['lx'][side][k] for r in R)
        return (100 * S('pla_ok') / max(1, S('pla_rec')), S('pla_succ') / max(1, S('pla_ok')),
                100 * S('planted') / max(1, S('planted') + S('seized')), S('life_sum') / max(1, S('life_n')))
    a, b = agg('pol'), agg('opp')
    print(f"{tag} n{len(R)}  ok% {a[0]:.0f}/{b[0]:.0f}  succ/ok {a[1]:.2f}/{b[1]:.2f}  planted% {a[2]:.0f}/{b[2]:.0f}  own-tree life {a[3]:.0f}/{b[3]:.0f}")
