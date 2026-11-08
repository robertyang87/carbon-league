"""stand_check [--vs OPP] TAG [TAG...] -- the stand-in (side B of bigeval) ledger vs the real bot's ladder means.
Targets must come from the real bot against the SAME opponent the stand-in faced (seizures depend on the opponent):
  all  = Master2s vs everyone (122 games);  v15 = Master2s vs v15 (4 replayed games: lost 550, last-15 ~400)."""
import json, sys
TARGETS = {"all": {"col75": 8.2, "colsteps": 2234, "lost": 868, "late15": 551, "dep": 8473},
           "v15": {"col75": 8.2, "colsteps": 2234, "lost": 550, "late15": 400, "dep": 8473}}
args = sys.argv[1:]
vs = "all"
if args and args[0] == "--vs":
    vs, args = args[1], args[2:]
REAL = TARGETS[vs]
print("targets: Master2s vs", vs, REAL)
for tag in args:
    R = [json.loads(l) for l in open(f'../data/bigeval_live/{tag}.jsonl') if '"lx"' in l]
    R = [r for r in R if r.get('margin') is not None]
    n = len(R)
    if n == 0:
        print(tag, 'no ledger rows (run predates the lx fields)'); continue
    m = lambda f: sum(f(r) for r in R) / n
    got = {"col75": m(lambda r: r['lx']['opp']['col75']), "colsteps": m(lambda r: r['lx']['opp']['colsteps']),
           "lost": m(lambda r: r['olost']), "late15": m(lambda r: r['lx']['opp']['late15']), "dep": m(lambda r: r['odep'])}
    ok = all(abs(got[k] / REAL[k] - 1) <= 0.15 for k in ("col75", "colsteps", "lost"))
    print(f"{tag}: n {n} stand-in wins {sum(r['margin'] < 0 for r in R)} mean margin(us) {m(lambda r: r['margin']):+.0f} | " +
          " ".join(f"{k} {got[k]:.0f}/{REAL[k]} ({100*(got[k]/REAL[k]-1):+.0f}%)" for k in REAL) + f" | LIKE={ok}")
