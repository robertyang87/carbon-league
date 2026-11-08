"""stand_check [--vs OPP] TAG [TAG...] -- the stand-in (side B of bigeval) ledger vs the real bot's ladder means.
Targets must come from the real bot against the SAME opponent the stand-in faced (seizures depend on the opponent):
  all  = Master2s vs everyone (122 games);  v15 = Master2s vs v15 (4 replayed games: lost 550, last-15 ~400)."""
import json, sys
TARGETS = {"all": {"col75": 8.2, "colsteps": 2234, "lost": 868, "late15": 551, "dep": 8473},
           "v15": {"col75": 8.2, "colsteps": 2234, "lost": 550, "late15": 400, "dep": 8473},
           # 无始 (140 games): the hard part is MANY trees and FEW losses; >400 lost means it is just a 1C net
           "ws_all": {"col75": 6.1, "colsteps": 1929, "lost": 222, "late15": 191, "dep": 8498, "treesteps": 1299},
           "ws_v15": {"col75": 6.5, "colsteps": 1904, "lost": 223, "late15": 212, "dep": 8610, "treesteps": 1394}}
KEYS = {"all": ("col75", "colsteps", "lost", "late15"), "v15": ("col75", "colsteps", "lost", "late15"),
        "ws_all": ("col75", "colsteps", "lost", "late15", "treesteps"), "ws_v15": ("col75", "colsteps", "lost", "late15", "treesteps")}
args = sys.argv[1:]
vs = "all"
if args and args[0] == "--vs":
    vs, args = args[1], args[2:]
REAL = TARGETS[vs]
print("targets:", vs, REAL)
for tag in args:
    R = [json.loads(l) for l in open(f'../data/bigeval_live/{tag}.jsonl') if '"lx"' in l]
    R = [r for r in R if r.get('margin') is not None]
    n = len(R)
    if n == 0:
        print(tag, 'no ledger rows (run predates the lx fields)'); continue
    m = lambda f: sum(f(r) for r in R) / n
    got = {"col75": m(lambda r: r['lx']['opp']['col75']), "colsteps": m(lambda r: r['lx']['opp']['colsteps']),
           "lost": m(lambda r: r['olost']), "late15": m(lambda r: r['lx']['opp']['late15']), "dep": m(lambda r: r['odep']),
           "treesteps": m(lambda r: r['lx']['opp'].get('treesteps', 0))}
    ok = all(abs(got[k] / REAL[k] - 1) <= 0.15 for k in KEYS[vs])   # late15 / treesteps are what the exams test
    print(f"{tag}: n {n} stand-in wins {sum(r['margin'] < 0 for r in R)} mean margin(us) {m(lambda r: r['margin']):+.0f} | " +
          " ".join(f"{k} {got[k]:.0f}/{REAL[k]} ({100*(got[k]/REAL[k]-1):+.0f}%)" for k in REAL) + f" | LIKE={ok}")
