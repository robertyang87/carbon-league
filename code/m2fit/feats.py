"""shared feature code for the whole-game M2 imitation (extractor + m2_bot use the same functions)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cnet2_feat as F
S = 15; N = 225
D = F.DIST
DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]
MV = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
def _st(p, mv):
    r, c = divmod(p, S); dr, dc = MV[mv]; return ((r + dr) % S) * S + (c + dc) % S
NBR = np.array([[_st(p, m) for m in DIRS] for p in range(N)])
_r, _c = np.divmod(np.arange(N), S)
_dr = np.abs(_r[:, None] - _r[None]); _dc = np.abs(_c[:, None] - _c[None])
CHEB = np.maximum(np.minimum(_dr, S - _dr), np.minimum(_dc, S - _dc))
RADC = 7

def board(P, me, car):
    """per-step board summary shared by all unit decisions"""
    op = 1 - me
    car = np.asarray(car, float)
    W, E = P[me][2], P[op][2]
    b = dict(car=car, nb=car[NBR].sum(1), me=me)
    b["obase"] = [int(x) for x in P[me][1].values()]
    b["ebase"] = [int(x) for x in P[op][1].values()]
    b["otree"] = [int(p) for p, _ in P[me][3].values()]
    b["etree"] = [int(p) for p, _ in P[op][3].values()]
    b["ocol"] = {u: (int(w[0]), float(w[1])) for u, w in W.items() if w[2] == "COLLECTOR"}
    b["ecol"] = [(int(w[0]), float(w[1])) for w in E.values() if w[2] == "COLLECTOR"]
    b["epl"] = [int(w[0]) for w in E.values() if w[2] == "PLANTER"]
    b["opl"] = [int(w[0]) for w in W.values() if w[2] == "PLANTER"]
    big = 30
    dmin = lambda L: D[:, L].min(1) if len(L) else np.full(N, big)
    b["dobase"] = dmin(b["obase"]); b["debase"] = dmin(b["ebase"])
    isb = np.zeros(N, bool); isb[b["obase"] + b["ebase"]] = True; b["isbase"] = isb
    tr = np.zeros(N, int)
    for p in b["otree"] + b["etree"]:
        tr[p] = 1
    b["istree"] = tr
    b["tree4"] = tr[NBR].sum(1)
    et = np.zeros(N, int); et[b["etree"]] = 1
    b["etree8"] = (CHEB[:, b["etree"]] == 1).sum(1) if b["etree"] else np.zeros(N, int)
    b["ecol_pos"] = np.array([p for p, _ in b["ecol"]], int); b["ecol_c"] = np.array([c for _, c in b["ecol"]], float)
    b["decol"] = dmin(list(b["ecol_pos"]))
    b["depl"] = dmin(b["epl"])
    nzc = car > 0
    b["cands_base"] = np.nonzero(nzc & ~isb)[0]
    cash = float(P[me][0]); own_tr = len(P[me][3])
    live = sum(1 for q in (0, 1) for _, a in P[q][3].values() if a < 50)
    b["ctx"] = dict(cash=cash, own_tr=own_tr, en_tr=len(P[op][3]), price=20.0 if own_tr == 0 else 20.0 + 5.0 * 1.235 ** live,
                    n_col=len(b["ocol"]), n_pl=len(b["opl"]), en_col=len(b["ecol"]), en_pl=len(b["epl"]), n_wk=len(W),
                    ecash=float(P[op][0]), cargo_all=sum(c for _, c in b["ocol"].values()))
    return b

CNAMES = ["c_car", "c_dist", "c_isbase", "c_here", "c_dbase", "c_trip", "c_slack", "c_nb", "c_carrel", "c_rank", "c_rate", "c_raterank",
          "c_tree4", "c_etree8", "c_ecol", "c_ethr", "c_eprey", "c_ocol", "c_ocloser", "c_istree",
          "step", "left", "cargo", "car_here", "dbase_me", "n_col", "best_car", "best_rate", "cash"]

def col_cands(b, u, step):
    """candidate next stops for own collector u: cells within RADC with carbon (and its own cell), plus own bases"""
    pos, cargo = b["ocol"][u]
    car = b["car"]; d = D[pos]
    c = b["cands_base"]; c = c[d[c] <= RADC]
    if not b["isbase"][pos] and pos not in c:
        c = np.r_[c, pos]
    cells = np.r_[c, np.array(b["obase"], int)].astype(int)
    nb_ = len(c)
    isbase = np.r_[np.zeros(nb_, int), np.ones(len(b["obase"]), int)]
    cc = np.where(isbase == 1, 0.0, car[cells]); dist = d[cells]
    dbase = b["dobase"][cells]
    left = 300 - step
    rate = cc / (dist + 1.0)
    ncar = np.where(isbase == 1, -1, cc)
    rank = (-ncar).argsort().argsort(); rrank = (-np.where(isbase == 1, -1, rate)).argsort().argsort()
    E, EC = b["ecol_pos"], b["ecol_c"]
    if len(E):
        de = D[np.ix_(cells, E)]
        ecol = de.min(1)
        thr = np.where(EC[None] < cargo, de, 30).min(1); prey = np.where(EC[None] > cargo, de, 30).min(1)
    else:
        ecol = thr = prey = np.full(len(cells), 30)
    oth = [p for v, (p, _) in b["ocol"].items() if v != u]
    if oth:
        do = D[np.ix_(cells, oth)]; ocol = do.min(1); ocl = (do < dist[:, None]).sum(1)
    else:
        ocol = np.full(len(cells), 30); ocl = np.zeros(len(cells), int)
    n = len(cells); ctx = b["ctx"]
    bestc = ncar[:nb_].max() if nb_ else 0.0; bestr = rate[:nb_].max() if nb_ else 0.0
    X = np.stack([cc, dist, isbase, (dist == 0).astype(int), dbase, dist + dbase, left - dist - dbase, b["nb"][cells], cc - car[pos], rank, rate, rrank,
                  b["tree4"][cells], b["etree8"][cells], ecol, thr, prey, ocol, ocl, b["istree"][cells],
                  np.full(n, step), np.full(n, left), np.full(n, cargo), np.full(n, car[pos] if not b["isbase"][pos] else 0.0), np.full(n, b["dobase"][pos]),
                  np.full(n, ctx["n_col"]), np.full(n, bestc), np.full(n, bestr), np.full(n, ctx["cash"])], 1).astype(np.float32)
    return cells, X
