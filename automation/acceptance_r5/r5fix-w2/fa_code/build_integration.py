"""build_integration: assemble fork-v0.2 (integration line) from
fork_v0_1.py by assertive string patches, and generate the attribution /
scan variants used by eval_integration.py.

Components stacked on the fork-v0.1 chassis (rho 0.08):
  * campfix (S4 verdict): eval_T 30->12, camp verdicts read the
    got/expect window ratio instead of self.flow(), downgrade after ONE
    bad eval. Code paths gated by PARAMS camp_fix (eval_T is a plain
    param: the campfix-off attribution variant also restores 30).
  * S1 salvage (a): seizure-hazard survival haircut on the SV *seize*
    usage only (PARAMS s1_seize_haz). Hazard machinery ported from S1
    (threat-tiered Beta posteriors over per-tree exposure counters).
    TV planting and guard SV are NOT discounted.
  * S1 salvage (b): cargo-tax term rebuilt on dwell-filtered flow
    (PARAMS s1_dwell_tax): an enemy collector's cargo feeds a cell's
    taxable-mass EMA only after >= 3 consecutive steps inside that
    cell's 9-zone (ehist positions t-2, t-1 + current). Term enters
    SV only (standing trees), never TV.

Every replacement is asserted to occur exactly once. Variants are
produced by PARAMS value edits on the assembled text (also asserted).
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

OLD_HEADER = '''"""Carbon League strategy -- fork-v0.1 (muse-spark1-3).

NOT ORIGINAL. Derived from the public strategy claude-opus-v53 by author
"Claude" (source: https://132.232.176.153/source/97898ce7047c0bb1dd7a13f02eff0dac),
local base copy base_v53.py (byte-identical to reference_claude_opus_v53.py).
Change in this version (single parameter, hygiene-grade tuning):
  * PARAMS rho (collector cargo hold cost in the whole-map DP)
    0.02 -> 0.08.  rho_early unchanged (0.05).
Local paired-seed test vs unmodified base (same map, both sides):
  8 games, 6 wins, mean margin +1184 (see v53_tuning_report.md).
The original docstring follows.
"""'''

NEW_HEADER = '''"""Carbon League strategy -- fork-v0.2 (muse-spark1-3).

NOT ORIGINAL. Derived from the public strategy claude-opus-v53 by author
"Claude" (source: https://132.232.176.153/source/97898ce7047c0bb1dd7a13f02eff0dac),
local base copy base_v53.py (byte-identical to reference_claude_opus_v53.py),
via fork-v0.1 (this project). Integration of the structure campaign's
surviving items; local benchmark / distillation teacher only, NOT submitted.
Changes vs claude-opus-v53:
  * PARAMS rho (collector cargo hold cost in the whole-map DP)
    0.02 -> 0.08 (from fork-v0.1; paired +1184 vs base, 8 games).
  * Camp evaluation repair (structure line S4, verdict "repair, don't
    cut"): PARAMS eval_T 30 -> 12; camp verdicts (the periodic blockade
    judgement and the rammed-posts give-up check) use the got/expect
    window ratio the code already computes -- enemy deliveries vs
    eff_rate x enemy collector-steps -- instead of self.flow(), whose
    denominator is MY OWN harvest rate (sensor confound); downgrade
    after ONE bad eval (eval_bad >= 1, was >= 2). Gated by PARAMS
    camp_fix=1 for attribution.
  * S1 salvage (a): seizure-hazard survival haircut applied ONLY to
    the seize-task value (the SV seize usage). Hazard machinery ported
    from fork_s1 (threat-tiered Beta posteriors over per-tree exposure
    counters: threatened = enemy planter within DIST 3). TV planting
    values and guard SV are NOT discounted. Single-item evidence:
    seize-valuation bias +16.6 -> +3.4 (s1_treeval_report.md).
    Gated by PARAMS s1_seize_haz=1.
  * S1 salvage (b): cargo-tax term rebuilt on dwell-filtered flow.
    An enemy collector's cargo feeds a cell's taxable-mass EMA (alpha
    0.034, as in S1) only when the collector has dwelled inside that
    cell's 9-cell zone for >= 3 consecutive steps (ehist positions
    t-2/t-1 plus current) -- pass-through traffic, which reroutes the
    moment a tree is planted (S1 feedback-effect finding), is excluded.
    The term (0.1 * dtaxF * remaining tree life) enters SV only
    (standing trees: seize/guard), never TV planting.
    Gated by PARAMS s1_dwell_tax=1.
Local paired-seed tests: see integration_report.md.
The original docstring follows.
"""'''

STATS_BLOCK = '''
        # --- integration salvage stats: dwell-filtered cargo-tax flow
        # EMA + seizure-hazard exposure counters (ported from S1).
        # dtaxF[q] = EMA (alpha 0.034, half-life ~20 steps) of the enemy
        # cargo mass in q's 9-cell zone, counting ONLY collectors that
        # have dwelled inside that zone for >= 3 consecutive steps
        # (ehist positions t-2, t-1 plus the current one). haz counters:
        # per own tree one exposure step per turn, classed threatened
        # (enemy planter within DIST 3) or calm, and whether the enemy
        # seized it during that step.
        _zm = [0.0] * N
        for _wid in enC:
            _w = workers[_wid]
            if _w[1] > 0:
                _h = self.ehist.get(_wid, ())
                if len(_h) >= 2:
                    _z = set(NB8[_h[-2]]); _z.add(_h[-2])
                    _z2 = set(NB8[_h[-1]]); _z2.add(_h[-1])
                    _z &= _z2
                    _z2 = set(NB8[_w[0]]); _z2.add(_w[0])
                    _z &= _z2
                    for _q in _z:
                        _zm[_q] += _w[1]
        _al = 0.034
        _F = self.dtaxF
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

PATCHES = [
    # header
    (OLD_HEADER, NEW_HEADER),
    # PARAMS: integration switches after the guard_w line
    ("    guard_w=0.5, guard_min=20.0,\n",
     "    guard_w=0.5, guard_min=20.0,\n"
     "    camp_fix=1, s1_seize_haz=1, s1_dwell_tax=1,\n"),
    # PARAMS: campfix eval_T 30 -> 12
    ("eval_T=30,", "eval_T=12,"),
    # __init__: salvage state
    ("        self.ehist = {}\n        self.myh_hist = []",
     "        self.ehist = {}\n"
     "        self.dtaxF = [0.0] * N\n"
     "        self.haz = dict(th_s=0, th_n=0, ca_s=0, ca_n=0)\n"
     "        self._s1prev = {}\n"
     "        self.myh_hist = []"),
    # step-0 reset line: append salvage state resets
    ("self.eval_on = 0; self.eval_d0 = 0.0; self.eval_lost0 = 0; self.eval_ec = 0.0",
     "self.eval_on = 0; self.eval_d0 = 0.0; self.eval_lost0 = 0; self.eval_ec = 0.0; "
     "self.dtaxF = [0.0] * N; self.haz = dict(th_s=0, th_n=0, ca_s=0, ca_n=0); self._s1prev = {}"),
    # stats block right after the tree price line
    ("        price = 20.0 if my_tree_n == 0 else 20.0 + 5.0 * 1.235 ** n_alive_trees\n",
     "        price = 20.0 if my_tree_n == 0 else 20.0 + 5.0 * 1.235 ** n_alive_trees\n"
     + STATS_BLOCK),
    # SV: dwell-tax term (SV only)
    ("                tot += est[n] * (1.0 - (1.0 - ABS * m) ** Lr) / m\n"
     "            SV[tid] = tot",
     "                tot += est[n] * (1.0 - (1.0 - ABS * m) ** Lr) / m\n"
     "            if P['s1_dwell_tax'] and Lr > 0:\n"
     "                tot += 0.1 * self.dtaxF[t[0]] * Lr\n"
     "            SV[tid] = tot"),
    # seize tasks: survival haircut on the seize usage of SV only
    ("        seize_tasks = []\n"
     "        for tid, t in trees.items():\n"
     "            if t[2] != me:\n"
     "                w = (1.0 + P['steal_w']) * SV[tid] - 20.0\n"
     "                if w >= P['seize_min']:\n"
     "                    seize_tasks.append((w, t[0]))",
     "        seize_tasks = []\n"
     "        if P['s1_seize_haz']:\n"
     "            _lam_th = min(0.05, (self.haz['th_s'] + 0.5) / (self.haz['th_n'] + 100.0))\n"
     "            _lam_ca = min(0.05, (self.haz['ca_s'] + 0.1) / (self.haz['ca_n'] + 400.0))\n"
     "        for tid, t in trees.items():\n"
     "            if t[2] != me:\n"
     "                _svv = SV[tid]\n"
     "                if P['s1_seize_haz']:\n"
     "                    _Lr = max(0, min(50 - t[1], R - 1))\n"
     "                    if _Lr > 0:\n"
     "                        _d = min([DIST[t[0]][workers[_w][0]] for _w in enP] or [99])\n"
     "                        _h = _lam_th if _d <= 3 else _lam_ca\n"
     "                        if _h > 0.0:\n"
     "                            _svv *= (1.0 - (1.0 - _h) ** _Lr) / (_Lr * _h)\n"
     "                w = (1.0 + P['steal_w']) * _svv - 20.0\n"
     "                if w >= P['seize_min']:\n"
     "                    seize_tasks.append((w, t[0]))"),
    # campfix: rammed-posts give-up check reads got/expect
    ("            got = sum(self.deliv_hist[-wnd:])\n"
     "            expect = P['eff_rate'] * sum(self.ec_hist[-wnd:])\n"
     "            if self.flow(wnd) > P['r2_kill']:",
     "            got = sum(self.deliv_hist[-wnd:])\n"
     "            expect = P['eff_rate'] * sum(self.ec_hist[-wnd:])\n"
     "            _kr = got / max(1.0, expect) if P['camp_fix'] else self.flow(wnd)\n"
     "            if _kr > P['r2_kill']:"),
    # campfix: periodic verdict keeps got/expect; downgrade after 1 bad
    ("                    ratio = got / max(1.0, expect)\n"
     "                    ratio = self.flow(P['eval_T'])\n",
     "                    ratio = got / max(1.0, expect)\n"
     "                    if not P['camp_fix']:\n"
     "                        ratio = self.flow(P['eval_T'])\n"),
    ("                        if self.eval_bad >= 2:",
     "                        if self.eval_bad >= (1 if P['camp_fix'] else 2):"),
]


def apply_patches(src):
    for i, (old, new) in enumerate(PATCHES):
        n = src.count(old)
        assert n == 1, f"patch {i}: anchor occurs {n} times (need 1)"
        src = src.replace(old, new)
    return src


def set_param(src, key, value):
    """Replace `key=<old>` inside the PARAMS dict with key=value.

    The search starts at the PARAMS definition so docstring mentions
    of the same `key=value` text are never touched."""
    i = src.index("PARAMS = dict(")
    head, tail = src[:i], src[i:]
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(key) + r"=[^,\s]+")
    tail, n = pat.subn(f"{key}={value}", tail, count=1)
    assert n == 1, f"param {key}: no match in PARAMS"
    return head + tail


def write(name, src):
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        f.write(src)
    print(f"wrote {name} ({len(src)} bytes)")


def main():
    with open(os.path.join(HERE, "fork_v0_1.py")) as f:
        src = f.read()
    out = apply_patches(src)
    write("fork_v0_2.py", out)

    # parity variant: every integration switch off, eval_T restored --
    # must decide exactly like fork_v0_1.
    off = out
    off = set_param(off, "camp_fix", "0")
    off = set_param(off, "s1_seize_haz", "0")
    off = set_param(off, "s1_dwell_tax", "0")
    off = set_param(off, "eval_T", "30")
    write("fork_v0_2_off.py", off)

    # attribution variants (vs fork_v0_1 in batch B)
    v = out
    v = set_param(v, "camp_fix", "0")
    v = set_param(v, "eval_T", "30")
    write("fork_v0_2_nocampfix.py", v)

    v = out
    v = set_param(v, "s1_seize_haz", "0")
    v = set_param(v, "s1_dwell_tax", "0")
    write("fork_v0_2_nos1.py", v)

    v = out
    v = set_param(v, "s1_seize_haz", "0")
    write("fork_v0_2_nohaz.py", v)

    v = out
    v = set_param(v, "s1_dwell_tax", "0")
    write("fork_v0_2_notax.py", v)


if __name__ == "__main__":
    main()
