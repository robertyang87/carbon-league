"""cnet2_agent -- serving wrapper for the cnet2 policy.

make_agent(weights, teacher=1, shield=False, nocollide=False, endgame=False)
  teacher: style id fed to the teacher embedding (cnet2_feat.TEACHERS index)
  shield:    a collector carrying cargo never picks a move whose destination
             a lighter-or-equal enemy collector can reach, if a safe move exists
  nocollide: avoid destinations already reserved by an earlier own unit
             (except own base cells in the last 15 moves)
  pshield:   a planter avoids destinations any enemy worker can reach, if possible
  rguard:    if no own collector exists (or is being recruited), a base recruits a collector
  v10rec:    base actions from v10's rule recruit gate (nn_agent.recruit_choice), policy picks C vs P
  endgame:   a loaded collector whose remaining moves <= distance+1 to the
             nearest own base takes the best move that reduces the distance
These are decode-time overrides (the autoregressive prefix sees the final choice).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnet2_feat as F  # noqa: E402
import cnet2_np as NP  # noqa: E402


def load_weights(src):
    if isinstance(src, dict):
        return src
    d = np.load(src, allow_pickle=False)
    return {k: d[k] for k in d.files}


def make_hook(x, shield, nocollide, endgame, pshield=False, rguard=False, v10rec=False, step=0):
    left = x["left"]
    ncol = int(sum(1 for s in range(10) if x["tok_mask"][s] and x["tok_kind"][s] == 0))
    rec = {"c": 0}

    def hook(s, lg, pf, a):
        if s >= 10 and rguard and lg[1] > -1e8:
            # never sit without a collector: recruit one if none exists or is being recruited
            if ncol + rec["c"] == 0 and a != 1 and left > 10:
                rec["c"] += 1
                return 1
            if a == 1:
                rec["c"] += 1
        if s >= 10 and v10rec:
            import nn_agent
            nw = 10 - pf.slots
            npl = int(sum(1 for t in range(10) if x["tok_mask"][t] and x["tok_kind"][t] == 1)) + pf.rec_pl
            blocked = pf.reserved[int(x["tok_pos"][s])] > 0
            r = nn_agent.recruit_choice(lg[:3], pf.cash, nw, 10, npl, x["price"], step, blocked)
            if r == 2:
                pf.rec_pl += 1
            return r
        if s >= 10:
            if nocollide and a > 0 and pf.reserved[int(x["tok_pos"][s])] > 0:
                return 0
            return None
        pos = int(x["tok_pos"][s])
        isc = x["tok_kind"][s] == 0
        cargo = float(x["tok_cargo"][s])
        order = [int(m) for m in np.argsort(-lg)]
        dests = F.DEST[pos]
        aux = pf.a
        cand = order
        if endgame and isc and cargo > 0:
            d0 = aux["dob"][pos]
            if d0 > 0 and left <= d0 + 1:
                closer = [m for m in order if aux["dob"][dests[m]] < d0]
                cand = closer + [m for m in order if m not in closer]
        ok = list(cand)
        if shield and isc and cargo > 0:
            safe = [m for m in ok if not aux["en_min"][dests[m]] <= cargo]
            if safe:
                ok = safe + [m for m in ok if m not in safe]
        if pshield and not isc:
            # planter: avoid cells any enemy worker can reach (planter-planter kills both, collector kills planter)
            safe = [m for m in ok if not (np.isfinite(aux["en_min"][dests[m]]) or aux["en_pl"][dests[m]] > 0)]
            if safe:
                ok = safe + [m for m in ok if m not in safe]
        if nocollide:
            pile = left <= 15
            free = [m for m in ok if pf.reserved[dests[m]] == 0 or (pile and isc and aux["own_base"][dests[m]])]
            if free:
                ok = free + [m for m in ok if m not in free]
        if cand is order and ok[0] == a:
            return None
        return ok[0]
    return hook


def make_agent(weights, teacher=1, shield=False, nocollide=False, endgame=False, pshield=False, rguard=False,
               v10rec=False, tta=1):
    pol = NP.Policy(load_weights(weights))
    st = {"f": None}
    rules = shield or nocollide or endgame or pshield or rguard or v10rec

    def agent(obs, config=None):
        me = obs["player"]
        if st["f"] is None or obs["step"] == 0 or st["f"].seat != me:
            st["f"] = F.Feat(me)
        f = st["f"]
        x = f.observe(obs)
        hook = make_hook(x, shield, nocollide, endgame, pshield, rguard, v10rec, obs["step"]) if rules else None
        if tta > 1:
            y, _ = NP.act_tta(pol, x, f.aux, teacher=teacher, k=tta, hook=hook)
        else:
            y, _, _ = pol.act(x, f.aux, teacher=teacher, hook=hook)
        return f.to_actions(y)
    return agent
