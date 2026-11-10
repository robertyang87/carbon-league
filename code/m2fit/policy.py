"""target + option features shared by the extractor (m2fit/extract_opt.py) and the bot (m2_bot.py)"""
import json, os
import numpy as np
from m2fit.feats import board, col_cands, D, NBR, DIRS, CNAMES
from m2fit.stage import row_feats
from mfit.extract_tgt import cand_feats, best_at
CI = {n: k for k, n in enumerate(CNAMES)}
ECON = True   # economic veto: never collect next to enemy trees when the absorption exceeds the haul
OPTS = [None] + DIRS


class Tree:
    def __init__(self, d):
        self.__dict__.update(d)
        self.cls = d["classes"]
        self._a = (np.asarray(self.left), np.asarray(self.right), np.asarray(self.feat), np.asarray(self.thr), np.asarray(self.val))

    def p(self, x, c="1"):
        n = 0
        while self.left[n] != -1:
            n = self.left[n] if x[self.feats[self.feat[n]]] <= self.thr[n] else self.right[n]
        return dict(zip(self.cls, self.val[n])).get(c, 0.0)

    def pm(self, X, cols, c="1"):
        if c not in self.cls:
            return np.zeros(len(X))
        k = self.cls.index(c)
        Z = X[:, [cols.index(f) for f in self.feats]]
        left, right, feat, thr, val = self._a
        node = np.zeros(len(X), int)
        act = left[node] != -1
        while act.any():
            i = np.nonzero(act)[0]; n = node[i]
            node[i] = np.where(Z[i, feat[n]] <= thr[n], left[n], right[n])
            act = left[node] != -1
        return val[node, k]


def load_trees(here, files=("m2_bot_ctrees.json", "m2_bot_ptrees.json", "m2_bot_otrees.json")):
    T = {}
    for fn in files:
        p = os.path.join(here, fn)
        if os.path.exists(p):
            T.update({k: Tree(v) for k, v in json.load(open(p)).items()})
    return T


def col_target(T, b, u, step, taken, stay_p=0.5, home_p=0.5, force_home=True, tgtw=0.0, sidew=0.0, prev=None, hyst=0.0, compw=0.0, avoid=None):
    pos, cargo = b["ocol"][u]
    cells, X = col_cands(b, u, step)
    rf = row_feats(X)
    left = 300 - step
    on_carbon = rf["on_base"] == 0 and rf["car_here"] > 0
    sp = T["c_stay"].p(rf, "True") if on_carbon else 0.0
    hp = T["c_home"].p(rf, "True")
    home = False; tg = None
    absorb_bad = ECON and rf["here_etree8"] > 0 and 0.3 * rf["car_here"] < 0.1 * cargo * rf["here_etree8"]
    if on_carbon and sp >= stay_p and not absorb_bad:
        tg = pos
    else:
        home = hp >= home_p and cargo > 0
        if force_home and cargo > 0 and left <= rf["dbase_me"] + 2:
            home = True
        if not home:
            m = (X[:, CI["c_isbase"]] == 0) & (X[:, CI["c_here"]] == 0)
            if taken:
                m &= np.array([int(c) not in taken for c in cells])
            if m.any():
                Xm = X[m]; cm = cells[m]
                p = T["c_tgt"].pm(Xm, CNAMES)
                gain = Xm[:, CI["c_car"]] * 0.657 / (Xm[:, CI["c_dist"]] + 3.0)      # 3 stays: 1-0.7^3 of the cell
                if ECON:
                    et = Xm[:, CI["c_etree8"]]
                    p = p - 1.0 * ((et > 0) & (0.3 * Xm[:, CI["c_car"]] < 0.1 * (cargo + 0.3 * Xm[:, CI["c_car"]]) * et))
                key = p + tgtw * gain / 20.0 + sidew * (b["dobase"][cm] - b["debase"][cm]) / 10.0 + 1e-4 * Xm[:, CI["c_rate"]] / 100 - 1e-7 * Xm[:, CI["c_dist"]]
                if avoid:
                    key = key - 1.0 * np.isin(cm, list(avoid))
                if compw:
                    key = key - compw * (Xm[:, CI["c_ecol"]] < Xm[:, CI["c_dist"]])
                if prev is not None and hyst:
                    key = key + hyst * (cm == prev)
                tg = int(cm[np.argmax(key)])
            elif cargo > 0:
                home = True
            else:
                tg = pos
    if force_home and cargo > 0 and left <= rf["dbase_me"] + 1:
        home = True
    if home:
        bs = b["obase"]
        tg = min(bs, key=lambda q: (D[pos, q], q)) if bs else pos
    return tg, home, rf, sp, hp


ONAMES_C = ["o_stay", "o_appr", "o_tgt_here", "o_car", "o_obase", "o_ebase", "o_e_eq1", "o_e_lt1", "o_e_gt1", "o_e_at_eq", "o_e_at_lt", "o_e_at_gt",
            "o_epl1", "o_epl_at", "o_own1", "o_etree8", "o_tree", "cargo", "step", "left", "home", "stay_p", "home_p", "dtg", "car_here", "e_near"]


def col_opts(b, u, tg, home, sp, hp, step):
    pos, cargo = b["ocol"][u]
    car = b["car"]; EP, EC = b["ecol_pos"], b["ecol_c"]
    obase = set(b["obase"]); ebase = set(b["ebase"])
    epl = np.array(b["epl"], int)
    oth = np.array([p for v, (p, _) in b["ocol"].items() if v != u] + b["opl"], int)
    rows = []
    e_near = int(len(EP) and (D[EP, pos] <= 2).any())
    for d in OPTS:
        nxt = pos if d is None else int(NBR[pos][DIRS.index(d)])
        if len(EP):
            dn = D[EP, nxt]; n1 = dn <= 1; at = dn == 0
            eq = np.abs(EC - cargo) < 1e-6; lt = EC < cargo - 1e-6; gt = EC > cargo + 1e-6
            e = [int((n1 & eq).sum()), int((n1 & lt).sum()), int((n1 & gt).sum()), int((at & eq).sum()), int((at & lt).sum()), int((at & gt).sum())]
        else:
            e = [0] * 6
        rows.append([int(d is None), D[nxt, tg] - D[pos, tg], int(tg == pos), car[nxt] if nxt not in obase else 0.0, int(nxt in obase), int(nxt in ebase)] + e +
                    [int(len(epl) and (D[epl, nxt] <= 1).any()), int(len(epl) and (D[epl, nxt] == 0).any()), int(len(oth) and (D[oth, nxt] <= 1).any()),
                     b["etree8"][nxt], b["istree"][nxt], cargo, step, 300 - step, int(home), sp, hp, D[pos, tg],
                     car[pos] if pos not in obase else 0.0, e_near])
    return np.array(rows, float)


def pl_target(T, ctx, arr, age, pos, ptaken):
    arr["best_at"][pos] = best_at(arr, pos)
    cells, cl = cand_feats(pos, ctx, arr, dict(age=age))
    names = list(cl.keys()); Xc = np.stack([cl[k] for k in names], 1).astype(float)
    p = T["p_tgt"].pm(Xc, names)
    key = p - 1e-4 * cl["c_dist"] + 1e-7 * cl["c_nb"]
    if ptaken:
        key = key - 10 * np.array([int(c) in ptaken for c in cells])
    k = int(np.argmax(key))
    return (int(cells[k]), float(p[k])) if len(cells) else (pos, 0.0)


ONAMES_P = ["o_stay", "o_appr", "o_tgt_here", "tgt_p", "o_epl_at", "o_epl1", "o_depl", "o_depl_ch", "o_ecol_at", "o_ecol1", "o_free", "o_etree", "o_otree",
            "o_base", "o_nb", "o_own1", "step", "cash", "price", "afford", "n_pl", "en_pl", "en_tr", "own_tr", "depl_here", "free_here", "etree_here", "nb_here", "dtg"]


def pl_opts(arr, ctx, pos, tg, tp, b):
    occ, nb = arr["occ"], arr["nb"]
    epl = np.array(arr["epl"], int); ecol = np.array(arr["ecol"], int)
    oth = np.array([p for p, _ in b["ocol"].values()] + [p for p in b["opl"] if p != pos], int)
    dep = lambda c: D[epl, c].min() if len(epl) else 30
    rows = []
    for d in OPTS:
        nxt = pos if d is None else int(NBR[pos][DIRS.index(d)])
        rows.append([int(d is None), D[nxt, tg] - D[pos, tg], int(tg == pos), tp,
                     int(len(epl) and (D[epl, nxt] == 0).any()), int(len(epl) and (D[epl, nxt] <= 1).any()), dep(nxt), dep(nxt) - dep(pos),
                     int(len(ecol) and (D[ecol, nxt] == 0).any()), int(len(ecol) and (D[ecol, nxt] <= 1).any()),
                     int(occ[nxt] == 0), int(occ[nxt] == 2), int(occ[nxt] == 1), int(occ[nxt] == 3), nb[nxt], int(len(oth) and (D[oth, nxt] <= 1).any()),
                     ctx["step"], ctx["cash"], ctx["price"], int(ctx["cash"] >= ctx["price"]), ctx["n_pl"], ctx["en_pl"], ctx["en_tr"], ctx["own_tr"],
                     dep(pos), int(occ[pos] == 0), int(occ[pos] == 2), nb[pos], D[pos, tg]])
    return np.array(rows, float)
