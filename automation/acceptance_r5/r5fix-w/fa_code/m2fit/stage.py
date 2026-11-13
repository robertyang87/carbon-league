"""staged collector decisions derived from the candidate matrix:
   row features (stay?, home?) and non-base target candidates"""
import numpy as np
from m2fit.feats import CNAMES
I = {n: k for k, n in enumerate(CNAMES)}
RNAMES = ["car_here", "cargo", "step", "left", "dbase_me", "n_col", "cash", "on_base",
          "here_tree4", "here_etree8", "here_ethr", "here_ecol", "here_ocol",
          "b1", "b2", "b3", "b5", "b7", "r_best", "r_best_ex", "cnt50_3", "here_rank", "slack_home", "dbest_rate"]

def row_feats(X):
    """X: candidate matrix of one collector-step -> dict of row features"""
    nb = X[:, I["c_isbase"]] == 0
    here = (X[:, I["c_here"]] == 1) & nb
    o = nb & ~here
    car = X[:, I["c_car"]]; d = X[:, I["c_dist"]]; rate = X[:, I["c_rate"]]
    x0 = X[0]
    f = dict(car_here=x0[I["car_here"]], cargo=x0[I["cargo"]], step=x0[I["step"]], left=x0[I["left"]], dbase_me=x0[I["dbase_me"]],
             n_col=x0[I["n_col"]], cash=x0[I["cash"]], on_base=float(x0[I["dbase_me"]] == 0))
    if here.any():
        h = X[here][0]
        f.update(here_tree4=h[I["c_tree4"]], here_etree8=h[I["c_etree8"]], here_ethr=h[I["c_ethr"]], here_ecol=h[I["c_ecol"]], here_ocol=h[I["c_ocol"]],
                 here_rank=h[I["c_rank"]])
    else:
        f.update(here_tree4=0, here_etree8=0, here_ethr=30, here_ecol=30, here_ocol=30, here_rank=99)
    for k in (1, 2, 3, 5, 7):
        m = o & (d <= k); f[f"b{k}"] = car[m].max() if m.any() else 0.0
    f["r_best"] = rate[nb].max() if nb.any() else 0.0
    f["r_best_ex"] = rate[o].max() if o.any() else 0.0
    f["cnt50_3"] = float((o & (d <= 3) & (car >= 50)).sum())
    f["slack_home"] = f["left"] - f["dbase_me"]
    f["dbest_rate"] = f["car_here"] - f["r_best_ex"]
    return f

def rows(R):
    import pandas as pd
    return pd.DataFrame([row_feats(r["X"]) for r in R])
