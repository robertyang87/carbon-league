"""build_probe: create probe_v53.py = base_v53.py + read-only instrumentation
for the avoidance-tax measurement (structure roadmap, case C step 1).

ZERO-POLLUTION CONTRACT: every insertion is one of
  (a) new attributes/methods that decision code never reads,
  (b) duplicated penalty-component arrays built in parallel locals
      (base arrays C0/C1/Cr are never written by probe code),
  (c) extra run_dp() calls whose results go only into probe locals,
  (d) per-line recordings placed BEFORE the base line they mirror
      (the base line itself is byte-identical),
  (e) end-of-act snapshot/dump after all decisions are made.
No base line is modified or reordered. Verified empirically by the
parity phase of eval_probe.py (identical action sequences, same seed).

Measurement definitions (also in avoid_tax_report.md):
  DP-layer tax (per collector-turn, free-planning collectors):
    regret_all = max_a q_all(a) - q_all(a_pen), where q_all is the DP
      action-value vector recomputed with BOTH safety components removed
      (minreach avoid1/avoid2 layer + cnt2 soft-risk layer), a_pen is the
      argmax under the penalised q5. regret_mr = same with only the
      minreach layer removed (soft kept). Soft marginal = regret_all -
      regret_mr. burden = q_all(a_pen) - q5(a_pen) (penalty mass carried
      by the chosen plan). Role units (camp/guard/rover/shadow/mark)
      are accumulated separately ('role' keys).
  Action-layer tax (per collector-turn with any lethal/tie/endgame
    safety charge): let_loss = regret of the final q5 argmax when the
      -1e6 lethal charges are restored; all_loss = same restoring
      lethal + tie (tie_cost + c, camp_occ) + endgame-scaled charges.
  Headline tax per game = sum over turns of (regret_all over free
    collectors) + let_loss.
  Enemy strike rate (per transition, pair level): an opportunity is an
    (enemy collector e, my collector m) pair with ec < mc whose chosen
    destination dm satisfies dm in N1[e.pos] (e could reach the cell m
    picked). strike: e's realised destination == dm. approach: e's
    destination != dm but within distance 1 of dm. Buckets by ec.
  Interception loss (ground truth): my collector absent next turn;
    cause attributed by the lightest previous-turn cargo among units
    now standing on its destination cell (enemy collector lighter ->
    clean interception; equal -> tie; own lighter collector -> own
    collision; other/no-attribution separated).
"""

import sys

SRC = "base_v53.py"
DST = "probe_v53.py"

src = open(SRC).read()


def sub_once(text, anchor, repl, label):
    n = text.count(anchor)
    if n != 1:
        print(f"ANCHOR FAIL [{label}]: count={n}")
        sys.exit(1)
    return text.replace(anchor, repl, 1)


def insert_line_before(text, needle, new_line_body, label):
    """Insert a line (indentation copied from the matched line) before the
    unique line containing `needle`."""
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if needle in ln]
    if len(idx) != 1:
        print(f"LINE ANCHOR FAIL [{label}]: count={len(idx)}")
        sys.exit(1)
    i = idx[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i, indent + new_line_body)
    return "\n".join(lines)


# ---------------------------------------------------------------- helpers
HELPERS = '''
def _pb_new_stats():
    return {
        'dp_turns_free': 0, 'dp_turns_role': 0,
        'regret_mr_free': 0.0, 'regret_all_free': 0.0,
        'regret_mr_role': 0.0, 'regret_all_role': 0.0,
        'chg_mr_free': 0, 'chg_all_free': 0, 'chg_mr_role': 0, 'chg_all_role': 0,
        'burden_free': 0.0, 'burden_role': 0.0,
        'regret_final': 0.0, 'final_turns': 0,
        'let_turns': 0, 'let_charged_actions': 0, 'let_reroute': 0, 'let_loss': 0.0,
        'all_reroute': 0, 'all_loss': 0.0,
        'en_turns': 0, 'en_gone': 0,
        'opp_pairs': 0, 'strike_pairs': 0, 'approach_pairs': 0,
        'tie_opp_pairs': 0, 'tie_strike_pairs': 0,
        'opp_turns': 0, 'strike_turns': 0, 'approach_turns': 0,
        'buckets': {},
        'loss_clean': 0.0, 'loss_clean_n': 0,
        'loss_tie': 0.0, 'loss_tie_n': 0,
        'loss_own': 0.0, 'loss_own_n': 0,
        'loss_other': 0.0, 'loss_other_n': 0,
        'loss_unknown': 0.0, 'loss_unknown_n': 0,
        'loss_enbase': 0.0, 'loss_enbase_n': 0,
        'deaths_c': 0, 'my_c_turns': 0, 'exposure100': 0,
    }


def _pb_bucket(x):
    if x <= 0:
        return '0'
    if x <= 80:
        return '1-80'
    if x <= 150:
        return '81-150'
    if x <= 250:
        return '151-250'
    return '251+'


'''

src = sub_once(src, "class Bot:", HELPERS + "class Bot:", "helpers")

# ---------------------------------------------------------------- __init__
src = sub_once(
    src,
    "        self.belief = None\n        self.last_dest = {}\n",
    "        self.belief = None\n        self.last_dest = {}\n"
    "        self._pb_prev = None\n"
    "        self._pb_mydest = {}\n"
    "        self._pb_rec = {}\n"
    "        self._pb_stats = _pb_new_stats()\n",
    "init",
)

# ---------------------------------------------------------------- methods
METHODS = '''    # ------------------------------------------------------------------
    # probe_v53 instrumentation (read-only; never feeds decisions)
    def _pb_transition(self, workers, me, en_bases):
        prev = self._pb_prev
        if not prev:
            return
        S = self._pb_stats
        en_base_set = set(en_bases)
        mydest = self._pb_mydest
        myC_prev = {}
        enC_prev = {}
        for wid_, w_ in prev.items():
            if w_[2] == 'C':
                if w_[3] == me:
                    myC_prev[wid_] = w_
                else:
                    enC_prev[wid_] = w_
        S['my_c_turns'] += len(myC_prev)
        for w_ in myC_prev.values():
            if w_[1] >= 100:
                S['exposure100'] += 1
        # (b) enemy strike behaviour vs my collectors' chosen destinations
        for ew, ev in enC_prev.items():
            epos, ec = ev[0], ev[1]
            cur = workers.get(ew)
            if cur is None:
                S['en_gone'] += 1
                continue
            S['en_turns'] += 1
            edest = cur[0]
            bk = S['buckets'].setdefault(_pb_bucket(ec), [0, 0, 0])
            t_opp = t_str = t_app = False
            for mw, mv in myC_prev.items():
                dm = mydest.get(mw)
                if dm is None or dm not in N1[epos]:
                    continue
                mc = mv[1]
                if ec < mc:
                    S['opp_pairs'] += 1
                    bk[0] += 1
                    t_opp = True
                    if edest == dm:
                        S['strike_pairs'] += 1
                        bk[1] += 1
                        t_str = True
                    elif DIST[edest][dm] <= 1:
                        S['approach_pairs'] += 1
                        bk[2] += 1
                        t_app = True
                elif ec == mc:
                    S['tie_opp_pairs'] += 1
                    if edest == dm:
                        S['tie_strike_pairs'] += 1
            if t_opp:
                S['opp_turns'] += 1
            if t_str:
                S['strike_turns'] += 1
            if t_app:
                S['approach_turns'] += 1
        # (c) my collector losses by attributed cause
        for mw, mv in myC_prev.items():
            if mw in workers:
                continue
            S['deaths_c'] += 1
            mc = mv[1]
            dm = mydest.get(mw)
            if dm is None:
                S['loss_unknown'] += mc
                S['loss_unknown_n'] += 1
                continue
            if dm in en_base_set:
                S['loss_enbase'] += mc
                S['loss_enbase_n'] += 1
                continue
            prev_cargo_en = [prev[wid_][1] for wid_, v in workers.items()
                             if v[0] == dm and v[3] != me and v[2] == 'C'
                             and wid_ in prev]
            prev_cargo_my = [prev[wid_][1] for wid_, v in workers.items()
                             if v[0] == dm and v[3] == me and v[2] == 'C'
                             and wid_ in prev]
            any_here = any(v[0] == dm for v in workers.values())
            if prev_cargo_en and min(prev_cargo_en) < mc:
                S['loss_clean'] += mc
                S['loss_clean_n'] += 1
            elif prev_cargo_en and min(prev_cargo_en) == mc:
                S['loss_tie'] += mc
                S['loss_tie_n'] += 1
            elif prev_cargo_my and min(prev_cargo_my) < mc:
                S['loss_own'] += mc
                S['loss_own_n'] += 1
            elif any_here:
                S['loss_other'] += mc
                S['loss_other_n'] += 1
            else:
                S['loss_unknown'] += mc
                S['loss_unknown_n'] += 1

    def flow(self, wnd):'''

src = sub_once(src, "    def flow(self, wnd):", METHODS, "methods")

# ------------------------------------------------------- transition hook
src = sub_once(
    src,
    "        bel.update(workers, trees, visible, obs['carbon'], step == 0, self.last_dest)",
    "        self._pb_transition(workers, me, en_bases)\n"
    "        bel.update(workers, trees, visible, obs['carbon'], step == 0, self.last_dest)",
    "transition-hook",
)

# ------------------------------------------------------- per-turn rec reset
src = sub_once(
    src,
    "        corder = sorted(myC, key=lambda w: (-workers[w][1], -est[workers[w][0]]))",
    "        self._pb_rec = {}\n"
    "        corder = sorted(myC, key=lambda w: (-workers[w][1], -est[workers[w][0]]))",
    "rec-reset",
)

# ------------------------------------------- penalty component arrays
SOFT_BLOCK = """            # soft risk for the next two arrivals
            if enC:
                for z, lst in cnt2.items():
                    n2 = sum(1 for x in lst if x <= c)
                    if n2:
                        pen = P['soft_w'] * (c + 30.0) * (1.0 if n2 >= 3 else (0.25 if n2 == 2 else 0.06))
                        C0[z] += pen
                        C1[z] += 0.5 * pen
"""
COMPONENTS = SOFT_BLOCK + """            # probe: penalty components rebuilt in parallel (base arrays untouched)
            _pb_avC1 = None; _pb_avCr = None; _pb_sfC0 = None; _pb_sfC1 = None
            if enC and c > 0:
                _pb_avC1 = np.zeros(N); _pb_avCr = np.zeros(N)
                for z in range(N):
                    if minreach[z] <= c and z not in my_base_set:
                        _pb_avC1[z] = P['avoid1'] * (c + 30.0)
                        _pb_avCr[z] = P['avoid2'] * (c + 30.0)
            if enC:
                _pb_sfC0 = np.zeros(N); _pb_sfC1 = np.zeros(N)
                for z, lst in cnt2.items():
                    n2 = sum(1 for x in lst if x <= c)
                    if n2:
                        _pen = P['soft_w'] * (c + 30.0) * (1.0 if n2 >= 3 else (0.25 if n2 == 2 else 0.06))
                        _pb_sfC0[z] = _pen; _pb_sfC1[z] = 0.5 * _pen
"""
src = sub_once(src, SOFT_BLOCK, COMPONENTS, "components")

# ------------------------------------------------------- DP tax accounting
EG_BLOCK = """            if endgame and c > 0:
                # delivered cargo counts fully when stepping into a base
                for a in range(1, 5):
                    if dests[a] in my_base_set:
                        q5[a] = float(c)
                if p in my_base_set:
                    q5[0] = float(c)
"""
DP_TAX = EG_BLOCK + """            # probe: DP-layer avoidance tax (read-only counterfactual DPs)
            if _pb_avC1 is not None or _pb_sfC0 is not None:
                _zN = np.zeros(N)
                _avC1 = _pb_avC1 if _pb_avC1 is not None else _zN
                _avCr = _pb_avCr if _pb_avCr is not None else _zN
                _sfC0 = _pb_sfC0 if _pb_sfC0 is not None else _zN
                _sfC1 = _pb_sfC1 if _pb_sfC1 is not None else _zN
                def _pb_q5(C0x, C1x, Crx):
                    Vx, _ = run_dp(G, lam, C0x, C1x, Crx, base_val, term, False)
                    if Hh >= 2:
                        arrx = gamma * Vx[0] - C0x
                        qsx = G[0][0][p] + gamma * Vx[1][p] - C0x[p] - lam
                    else:
                        arrx = gamma * term[0] - C0x
                        qsx = G[0][0][p] + gamma * term[0][p] - C0x[p]
                    qx = [float(qsx)] + [float(arrx[zz]) - lam for zz in NB[p]]
                    if endgame and c > 0:
                        for a_ in range(1, 5):
                            if dests[a_] in my_base_set:
                                qx[a_] = float(c)
                        if p in my_base_set:
                            qx[0] = float(c)
                    return qx
                _a_pen = max(range(5), key=lambda a_: q5[a_])
                _q_mr = _pb_q5(C0, C1 - _avC1, Cr - _avCr)
                _q_all = _pb_q5(C0 - _sfC0, C1 - _avC1 - _sfC1, Cr - _avCr)
                _a_mr = max(range(5), key=lambda a_: _q_mr[a_])
                _a_all = max(range(5), key=lambda a_: _q_all[a_])
                _role = (wid in self.camp or wid in self.guard or wid in self.rover
                         or wid in self.shadow or wid in getattr(self, 'mark', {}))
                _k = 'role' if _role else 'free'
                _S = self._pb_stats
                _S['dp_turns_' + _k] += 1
                _S['regret_mr_' + _k] += _q_mr[_a_mr] - _q_mr[_a_pen]
                _S['regret_all_' + _k] += _q_all[_a_all] - _q_all[_a_pen]
                _S['burden_' + _k] += _q_all[_a_pen] - q5[_a_pen]
                if _a_mr != _a_pen:
                    _S['chg_mr_' + _k] += 1
                if _a_all != _a_pen:
                    _S['chg_all_' + _k] += 1
                self._pb_rec[wid] = (_q_all, _a_pen)
"""
src = sub_once(src, EG_BLOCK, DP_TAX, "dp-tax")

# ------------------------------------------------------- lethal init
src = sub_once(
    src,
    "            for a in range(5):\n                z = dests[a]\n                q5[a] += bn[a]\n",
    "            _pb_let = [0.0] * 5; _pb_tie = [0.0] * 5; _pb_eg = [0.0] * 5\n"
    "            for a in range(5):\n                z = dests[a]\n                q5[a] += bn[a]\n",
    "lethal-init",
)

# ------------------------------------------------------- branch recordings
src = insert_line_before(
    src, "q5[a] -= (0.7 if a == 0 else 0.45) * (c + 30.0)",
    "_pb_eg[a] += (0.7 if a == 0 else 0.45) * (c + 30.0)", "eg-record")
src = insert_line_before(
    src, "q5[a] -= 1e6 + (500.0 if a == 0 else 0.0)",
    "_pb_let[a] += 1e6 + (500.0 if a == 0 else 0.0)", "let-record")
src = insert_line_before(
    src, "q5[a] -= tie_cost + c",
    "_pb_tie[a] += tie_cost + c", "tie-record")
src = insert_line_before(
    src, "q5[a] -= P['camp_occ']",
    "_pb_tie[a] += P['camp_occ']", "campocc-record")

# ------------------------------------------------------- lethal accounting
LET_ACCT = """            # probe: action-layer lethal/tie counterfactual (read-only)
            if any(_pb_let) or any(_pb_tie) or any(_pb_eg):
                _S = self._pb_stats
                _ch_b = max(range(5), key=lambda a_: q5[a_])
                _q_nl = [q5[a_] + _pb_let[a_] for a_ in range(5)]
                _ch_nl = max(range(5), key=lambda a_: _q_nl[a_])
                _S['let_turns'] += 1
                _S['let_charged_actions'] += sum(1 for a_ in range(5) if _pb_let[a_] > 0)
                if _ch_nl != _ch_b:
                    _S['let_reroute'] += 1
                    _S['let_loss'] += _q_nl[_ch_nl] - _q_nl[_ch_b]
                _q_nt = [q5[a_] + _pb_let[a_] + _pb_tie[a_] + _pb_eg[a_] for a_ in range(5)]
                _ch_nt = max(range(5), key=lambda a_: _q_nt[a_])
                if _ch_nt != _ch_b:
                    _S['all_reroute'] += 1
                    _S['all_loss'] += _q_nt[_ch_nt] - _q_nt[_ch_b]
            Q[wid] = q5
            # claim the cells this collector plans to harvest
"""
src = sub_once(src, "            Q[wid] = q5\n            # claim the cells this collector plans to harvest\n", LET_ACCT, "lethal-acct")

# ------------------------------------------------------- final cross-check
src = sub_once(
    src,
    "        actions = {}\n        self.last_dest = {}\n",
    "        # probe: final-action regret cross-check (read-only)\n"
    "        for _wid, (_q_all, _a_pen) in self._pb_rec.items():\n"
    "            if _wid in choice:\n"
    "                _S = self._pb_stats\n"
    "                _S['final_turns'] += 1\n"
    "                _S['regret_final'] += max(_q_all) - _q_all[choice[_wid]]\n"
    "        actions = {}\n        self.last_dest = {}\n",
    "final-xcheck",
)

# ------------------------------------------------------- snapshot + dump
src = sub_once(
    src,
    "        return actions\n",
    "        # probe: snapshot for next-turn transition stats + final dump\n"
    "        self._pb_prev = {wid: (w[0], w[1], w[2], w[3]) for wid, w in workers.items()}\n"
    "        self._pb_mydest = dict(self.last_dest)\n"
    "        if step == 299:\n"
    "            _pp = os.environ.get('PROBE_STATS_PATH')\n"
    "            if _pp:\n"
    "                try:\n"
    "                    import json as _json\n"
    "                    with open(_pp, 'w') as _f:\n"
    "                        _json.dump(self._pb_stats, _f)\n"
    "                except Exception:\n"
    "                    pass\n"
    "        return actions\n",
    "snapshot",
)

open(DST, "w").write(src)
print("probe_v53.py written:", len(src), "bytes")
