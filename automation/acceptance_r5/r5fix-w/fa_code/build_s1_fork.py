"""build_s1_fork: derive fork_s1_treeval.py from base_v53.py by surgical
replacements (each asserted to hit exactly once), then emit ablation
variants by flipping the s1_* PARAMS switches.

Changes vs base:
  * header docstring (derivation note)
  * PARAMS += s1_sim / s1_tax / s1_haz switches
  * Bot state: taxF (enemy-cargo zone EMA), haz (seizure counters),
    _s1prev (my-tree threat snapshot) -- in __init__ and in act() reset
  * per-step S1 stats update after the plant-price line
  * TV/SV block replaced by campaign-level valuation (pool simulation +
    tax term + hazard survival), keeping base closed-form as *_OLD
    (used for calibration logging and as the all-switches-off fallback)
  * PRED_LOG instrumentation at plant/seize commit sites
  * CARBON_DEBUG env read removed (per v53_verify_report recommendation)
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "base_v53.py")).read()


def sub(old, new, name):
    global src
    n = src.count(old)
    assert n == 1, f"{name}: anchor found {n} times"
    src = src.replace(old, new)


# --- A. header -------------------------------------------------------------
assert src.startswith('"""Carbon League strategy.')
HEADER = '''"""Carbon League strategy -- fork-s1-treeval (muse-spark1-3).

NOT ORIGINAL. Derived from the public strategy claude-opus-v53 by author
"Claude" (source: https://132.232.176.153/source/97898ce7047c0bb1dd7a13f02eff0dac),
local base copy base_v53.py (byte-identical to reference_claude_opus_v53.py).
Structural change in this version (structure line S1, tree-campaign
valuation; roadmap lesion P4 / plan D):
  * plant/seize valuation (TV/SV) replaces the static snapshot formula
    with an env-faithful L-step pool simulation (time-varying
    co-absorber count from known tree ages), an enemy-cargo tax term
    (0.1 x EMA of enemy cargo mass inside the tree's 9-cell zone), and
    a seizure-hazard survival discount (pstay-style threat counters).
  * three independent switches in PARAMS: s1_sim / s1_tax / s1_haz;
    all off reproduces base decisions exactly.
  * the CARBON_DEBUG env read of the original is removed.
The original docstring follows.
"""
'''
sub('"""Carbon League strategy.', HEADER + '"""Carbon League strategy.', "header")

# --- B. PARAMS switches ----------------------------------------------------
sub("    guard_w=0.5, guard_min=20.0,\n",
    "    guard_w=0.5, guard_min=20.0,\n    s1_sim=1, s1_tax=1, s1_haz=1,\n",
    "params")

# --- C. Bot.__init__ state --------------------------------------------------
sub("        self.ehist = {}\n",
    "        self.ehist = {}\n"
    "        self.taxF = [0.0] * N\n"
    "        self.haz = dict(th_s=0, th_n=0, ca_s=0, ca_n=0)\n"
    "        self._s1prev = {}\n",
    "init_state")

# --- D. act() step-0 reset --------------------------------------------------
sub("self.eval_ec = 0.0\n        bel = self.belief",
    "self.eval_ec = 0.0; self.taxF = [0.0] * N; "
    "self.haz = dict(th_s=0, th_n=0, ca_s=0, ca_n=0); self._s1prev = {}\n"
    "        bel = self.belief",
    "act_reset")

# --- E. per-step S1 stats update -------------------------------------------
STATS = '''
        # --- S1 stats: enemy-cargo tax-flow EMA + seizure-hazard exposure.
        # taxF[q] = EMA (half-life ~20 steps) of the enemy cargo mass in
        # q's 9-cell zone -- the per-step taxable mass for a tree at q.
        # haz counters: for each of my trees, one exposure step per turn,
        # classified threatened (enemy planter within DIST 3) or calm,
        # and whether the enemy seized it during that step.
        _zm = [0.0] * N
        for _wid in enC:
            _w = workers[_wid]
            if _w[1] > 0:
                _zm[_w[0]] += _w[1]
                for _z in NB8[_w[0]]:
                    _zm[_z] += _w[1]
        _al = 0.034
        _F = self.taxF
        for _i in range(N):
            _F[_i] = (1.0 - _al) * _F[_i] + _al * _zm[_i]
        _enp_pos = [workers[_w][0] for _w in enP]
        _cur_mine = {}
        for _tid, _t in trees.items():
            if _t[2] == me:
                _d = min([DIST[_t[0]][_p] for _p in _enp_pos] or [99])
                _cur_mine[_tid] = _d <= 3
        for _tid, _thr in self._s1prev.items():
            if _thr:
                self.haz['th_n'] += 1
                if _tid in trees and trees[_tid][2] != me:
                    self.haz['th_s'] += 1
            else:
                self.haz['ca_n'] += 1
                if _tid in trees and trees[_tid][2] != me:
                    self.haz['ca_s'] += 1
        self._s1prev = _cur_mine
'''
sub("        price = 20.0 if my_tree_n == 0 else 20.0 + 5.0 * 1.235 ** n_alive_trees\n",
    "        price = 20.0 if my_tree_n == 0 else 20.0 + 5.0 * 1.235 ** n_alive_trees\n"
    + STATS,
    "stats_update")

# --- F. TV/SV block ----------------------------------------------------------
start_marker = "        # ---------------- planters ------------------------------------"
end_marker = "        plant_tasks = []"
i0 = src.index(start_marker)
i1 = src.index(end_marker)
assert i0 < i1
NEWBLOCK = '''        # ---------------- planters ------------------------------------
        # S1 campaign-level tree valuation (see header). Base closed-form
        # values are computed first as TV_OLD / SV_OLD / PIN_OLD: they are
        # the all-switches-off fallback and the calibration baseline for
        # PRED_LOG. The new model adds, per tree/candidate:
        #   sim : L-step pool simulation with time-varying co-absorber
        #         counts (adjacent trees age out at 50 on known ages);
        #         pools do not regenerate while flagged (env phases 5/7),
        #         so no regen term exists inside an absorption horizon.
        #   tax : 0.1 * taxF[q] per step (enemy cargo in the 9-cell zone),
        #         survival-weighted like the absorption it accompanies.
        #   haz : survival discount (1-h)^t with h from threat counters
        #         (threatened: enemy planter within DIST 3 of the tree).
        L = max(0, min(50, R - 1))
        TV = [-1e9] * N
        SV = {}
        TV_OLD = [-1e9] * N
        SV_OLD = {}
        PIN = [0.0] * N
        PIN_OLD = [0.0] * N
        TOT_OLD = [0.0] * N
        if L > 0:
            for q in range(N):
                if q in tree_at or q in base_cells:
                    continue
                tot = 0.0
                for n in NB[q]:
                    if n in tree_at or n in base_cells:
                        continue
                    v = est[n]
                    if v <= 0:
                        continue
                    m = adj_cnt[n]; mine = adj_mine[n]
                    ea = v * (1.0 - (1.0 - ABS * (m + 1)) ** L)
                    if m:
                        eb = v * (1.0 - (1.0 - ABS * m) ** L)
                        tot += ea * (mine + 1) / (m + 1) - eb * mine / m
                        tot += P['steal_w'] * (eb * (m - mine) / m - ea * (m - mine) / (m + 1))
                    else:
                        tot += ea
                    PIN_OLD[q] += ea / (m + 1)
                TOT_OLD[q] = tot
                TV_OLD[q] = tot * P['hold_p'] - P['cell_loss_w'] * est[q]
        for tid, t in trees.items():
            Lr = max(0, min(50 - t[1], R - 1))
            tot = 0.0
            for n in NB[t[0]]:
                if n in tree_at or n in base_cells:
                    continue
                m = max(1, adj_cnt[n])
                tot += est[n] * (1.0 - (1.0 - ABS * m) ** Lr) / m
            SV_OLD[tid] = tot
        if not (P['s1_sim'] or P['s1_tax'] or P['s1_haz']):
            TV = TV_OLD
            SV = SV_OLD
            PIN = PIN_OLD
        elif L > 0:
            Hz = L
            v0a = np.asarray(est, dtype=np.float64)
            if P['s1_haz']:
                lam_th = min(0.05, (self.haz['th_s'] + 0.5) / (self.haz['th_n'] + 100.0))
                lam_ca = min(0.05, (self.haz['ca_s'] + 0.1) / (self.haz['ca_n'] + 400.0))
                denp = [99] * N
                for _p in [workers[_w][0] for _w in enP]:
                    _row = DIST[_p]
                    for _q in range(N):
                        if _row[_q] < denp[_q]:
                            denp[_q] = _row[_q]
                hvec = np.array([lam_th if d <= 3 else lam_ca for d in denp])
                Smat = (1.0 - hvec)[:, None] ** np.arange(Hz)[None, :]
            else:
                Smat = np.ones((N, Hz))
            taxF = np.asarray(self.taxF, dtype=np.float64) * 0.1
            tax_t = taxF * Smat.sum(axis=1) if P['s1_tax'] else np.zeros(N)
            if P['s1_sim']:
                mB = np.zeros((N, Hz)); myB = np.zeros((N, Hz))
                for tid, t in trees.items():
                    rem = 50 - t[1]
                    if rem <= 0:
                        continue
                    if rem > Hz:
                        rem = Hz
                    for n in NB[t[0]]:
                        if n in tree_at or n in base_cells:
                            continue
                        mB[n, :rem] += 1.0
                        if t[2] == me:
                            myB[n, :rem] += 1.0
                enB = mB - myB
                _sv = np.ones((N, Hz + 1))
                _sv[:, 1:] = np.cumprod(1.0 - ABS * mB, axis=1)
                PB = _sv[:, :Hz]
                _sv2 = np.ones((N, Hz + 1))
                _sv2[:, 1:] = np.cumprod(1.0 - ABS * (mB + 1.0), axis=1)
                PA = _sv2[:, :Hz]
                ABv = (ABS * v0a)[:, None]
                co = ABv * PA
                pool_m = (ABv * (PA - PB) * myB).sum(axis=1) \\
                    + P['steal_w'] * (ABv * (PB - PA) * enB).sum(axis=1)
                own_s = (co[NBR] * Smat[:, None, :]).sum(axis=(1, 2))
                marg = own_s.copy()
                for q in range(N):
                    if q in tree_at or q in base_cells:
                        continue
                    for n in NB[q]:
                        if n in tree_at or n in base_cells:
                            continue
                        marg[q] += pool_m[n]
                TVa = (marg + tax_t) * P['hold_p'] - P['cell_loss_w'] * v0a
                TV = [(-1e9 if (q in tree_at or q in base_cells) else float(TVa[q])) for q in range(N)]
                PIN = (own_s + tax_t).tolist()
                credB = ABv * PB
                for tid, t in trees.items():
                    Lr = max(0, min(50 - t[1], R - 1))
                    if Lr <= 0:
                        SV[tid] = 0.0
                        continue
                    qq = t[0]
                    Sq = Smat[qq, :Lr]
                    tot = 0.0
                    for n in NB[qq]:
                        if n in tree_at or n in base_cells:
                            continue
                        tot += float((credB[n, :Lr] * Sq).sum())
                    if P['s1_tax']:
                        tot += float(taxF[qq] * Sq.sum())
                    SV[tid] = tot
            else:
                Sbar = Smat.mean(axis=1) if P['s1_haz'] else np.ones(N)
                TVa = (np.asarray(TOT_OLD) * Sbar + tax_t) * P['hold_p'] - P['cell_loss_w'] * v0a
                TV = [(-1e9 if (q in tree_at or q in base_cells) else float(TVa[q])) for q in range(N)]
                PIN = (np.asarray(PIN_OLD) * Sbar + tax_t).tolist()
                for tid, t in trees.items():
                    Lr = max(0, min(50 - t[1], R - 1))
                    if Lr <= 0:
                        SV[tid] = 0.0
                        continue
                    qq = t[0]
                    Sq = Smat[qq, :Lr]
                    sbar = float(Sq.mean()) if P['s1_haz'] else 1.0
                    tot = SV_OLD[tid] * sbar
                    if P['s1_tax']:
                        tot += float(taxF[qq] * Sq.sum())
                    SV[tid] = tot
        else:
            TV = TV_OLD
            SV = SV_OLD
            PIN = PIN_OLD
'''
src = src[:i0] + NEWBLOCK + src[i1:]

# --- G. PRED_LOG at commit sites --------------------------------------------
sub("""                if q5[0] >= max(q5[1:]) and best_t[0] == p:
                    if best_t[1] == 0 and cash - spend >= price:
                        spend += price
                    elif best_t[1] == 1 and cash - spend >= 20:
                        spend += 20.0
""",
    """                if q5[0] >= max(q5[1:]) and best_t[0] == p:
                    if best_t[1] == 0 and cash - spend >= price:
                        spend += price
                        PRED_LOG.append({'step': step, 'kind': 'plant', 'cell': best_t[0],
                                         'pin_new': round(float(PIN[best_t[0]]), 2),
                                         'pin_old': round(float(PIN_OLD[best_t[0]]), 2),
                                         'price': price})
                    elif best_t[1] == 1 and cash - spend >= 20:
                        spend += 20.0
                        _tid = tree_at.get(best_t[0])
                        PRED_LOG.append({'step': step, 'kind': 'seize', 'cell': best_t[0], 'tid': _tid,
                                         'pin_new': round(float(SV.get(_tid, 0.0)), 2),
                                         'pin_old': round(float(SV_OLD.get(_tid, 0.0)), 2),
                                         'price': 20.0})
""",
    "pred_log")

# --- H. PRED_LOG global + agent() hygiene ------------------------------------
sub("\n_BOT = Bot()\n", "\nPRED_LOG = []\n\n_BOT = Bot()\n", "pred_global")
sub("""    if observation['step'] == 0:
        _BOT = Bot()
""",
    """    if observation['step'] == 0:
        _BOT = Bot()
        PRED_LOG.clear()
""",
    "pred_clear")
sub("""    except Exception:
        if os.environ.get('CARBON_DEBUG'):
            raise
        return {}
""",
    """    except Exception:
        return {}
""",
    "debug_removal")

out = os.path.join(HERE, "fork_s1_treeval.py")
open(out, "w").write(src)
print("wrote", out, len(src), "bytes")

SW = "s1_sim=1, s1_tax=1, s1_haz=1"
assert src.count(SW) == 1
VARIANTS = {
    "fork_s1_off.py": (0, 0, 0),
    "fork_s1_nosim.py": (0, 1, 1),
    "fork_s1_notax.py": (1, 0, 1),
    "fork_s1_nohaz.py": (1, 1, 0),
    "fork_s1_simonly.py": (1, 0, 0),
    "fork_s1_taxonly.py": (0, 1, 0),
    "fork_s1_hazonly.py": (0, 0, 1),
}
for fn, (a, b, c) in VARIANTS.items():
    v = src.replace(SW, f"s1_sim={a}, s1_tax={b}, s1_haz={c}")
    open(os.path.join(HERE, fn), "w").write(v)
    print("wrote", fn)
