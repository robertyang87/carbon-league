"""tree_curve TAG [TAG...] -- mean tree count (ours / opponent) at steps 50 75 100 150 200 250, plus family targets"""
import json, sys
T = {"Master2s": {50: 6, 200: 4.5, 250: 4.5}, "无始": {100: 5}}
for tag in sys.argv[1:]:
    R = [json.loads(l) for l in open(f'../data/bigeval_live/{tag}.jsonl')]
    R = [r for r in R if r.get('lx') and r['lx']['pol'].get('trees')]
    if not R: print(tag, 'no tree curve (run predates the field)'); continue
    n = len(R)
    me = [sum(r['lx']['pol']['trees'][i] for r in R) / n for i in range(6)]
    op = [sum(r['lx']['opp']['trees'][i] for r in R) / n for i in range(6)]
    eff = lambda side: sum(r['lx'][side].get('plants', 0) for r in R) / max(1, sum(r['lx'][side].get('pla_rec', 0) for r in R))
    print(f"{tag} n{n}  " + "  ".join(f"t{s}:{a:.1f}/{b:.1f}" for s, a, b in zip((50, 75, 100, 150, 200, 250), me, op))
          + f" | trees per planter {eff('pol'):.2f} / {eff('opp'):.2f} (target >= 0.5; v16 0.34, Master2s 0.76, 无始 0.54)")
print("targets:", T)
