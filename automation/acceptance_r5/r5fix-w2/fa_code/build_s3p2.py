"""build_s3p2: derive fork_s3_phase2.py (+ ablation variants) from
fork_s3_phase.py by anchored, P6-only surgery.

Guarantees checked at build time:
  * every replacement anchor matches exactly once;
  * a unified diff vs fork_s3_phase.py is written to
    s3p2_vs_s3_diff.txt and every hunk is asserted disjoint from the
    P5 hunt-feedback regions of the v1 file (death attribution,
    feedback eval/level scaling, hunter snapshot) -- P5 code is not
    modified, only P6 regions and pure insertions change.
"""

import difflib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "fork_s3_phase.py")
DST = os.path.join(HERE, "fork_s3_phase2.py")

src = open(SRC).read()
lines = src.split("\n")

# ---------------- 1. header docstring ----------------
hdr_end = src.index('The original docstring follows.\n"""') + len(
    'The original docstring follows.\n"""')
new_header = '''"""Carbon League strategy -- fork-s3-phase2 (muse-spark1-3).

NOT ORIGINAL. Derived from the public strategy claude-opus-v53 by author
"Claude" (source: https://132.232.176.153/source/97898ce7047c0bb1dd7a13f02eff0dac),
local base copy base_v53.py (byte-identical to reference_claude_opus_v53.py),
via fork_s3_phase.py (S3 v1). This v2 recalibrates the P6 sprint phase
against the failure modes measured in s3_phase_report.md; the P5 hunt
feedback loop (switch s3_hunt) is byte-identical to fork_s3_phase.py
(diff-verified, see s3p2_vs_s3_diff.txt):
  * mode calibration -- v1's fixed +/-400 margin band (a step-250
    snapshot decided the mode for the whole window) is replaced by a
    normalised margin: ratio = diff / (R * s3p2_rate), with
    s3p2_rate = 38/step calibrated to the measured post-step-250
    deposit rate (48 samples, mean 37.6/step). Modes are re-evaluated
    every step with enter/exit hysteresis (s3p2_pro/gam/bank
    enter/exit), so no mode is for life.
  * banking mode (new, ratio <= s3p2_bank_enter = -0.85; at window
    entry that is a deficit of ~-1600, between v1's two observed deep
    cases: gamble-flip at -1238 / ratio -0.65, gamble-disaster at
    -1851 / ratio -0.97) -- deep-behind no longer loiters on a
    relaxed lam: lam gets the tightest floor of all modes
    (s3p2_bank_lam0 -> s3p2_bank_lam1 over the window) and phase
    recruits are closed. All-out conversion of in-transit cargo.
  * protect fade -- v1's flat lam x1.3 leaked precisely the biggest
    leads (+1851 / +1366 entries). The extra tightening is now full
    up to ratio s3p2_pro_fade_lo and fades linearly to x1.0 at
    s3p2_pro_fade_hi: the larger the lead, the less extra tightening.
  * recruit payback gate widened (switch s3p2_recgate; =0 restores
    the v1 bars for ablation) -- v1 fired once in 12 phase games
    (3 recruits) while the background rate said the bet is positive
    (28 window recruits, 14 paid back, paybacks 328-1930 vs cost 30):
    bars s3p2_payback = 0.8 / s3p2_gamble_payback = 0.6, cap
    s3p2_rec_cap = 6.
The original docstring follows.
"""'''
src = new_header + src[hdr_end:]
lines = src.split("\n")

# ---------------- 2. params: pure insertion after the s3 line --------
anchor = [i for i, l in enumerate(lines) if l.startswith("    s3_phase=1, s3_hunt=1,")]
assert len(anchor) == 1, anchor
new_params = ("    s3p2_phase=1, s3p2_rate=38.0, s3p2_pro_enter=0.30, s3p2_pro_exit=0.18, "
              "s3p2_gam_enter=-0.30, s3p2_gam_exit=-0.18, s3p2_bank_enter=-0.85, s3p2_bank_exit=-0.68, "
              "s3p2_pro_fade_lo=0.55, s3p2_pro_fade_hi=0.95, s3p2_bank_lam0=1.7, s3p2_bank_lam1=2.4, "
              "s3p2_recgate=1, s3p2_payback=0.8, s3p2_gamble_payback=0.6, s3p2_rec_min_R=17, s3p2_rec_cap=6,")
lines.insert(anchor[0] + 1, new_params)

# ---------------- 3. _s3p2_reset method + call sites -----------------
stats_anchor = [i for i, l in enumerate(lines)
                if "hunt_levels=[], hunt_deaths=0, hunt_death_cargo=0.0)" in l]
assert len(stats_anchor) == 1
method = ["",
          "    def _s3p2_reset(self):",
          '        """S3 phase-v2 state: hysteresis mode + per-mode step counts."""',
          "        self.s3p2_mode = 0",
          "        self.s3p2_stats = dict(steps_bank=0, steps_protect=0, steps_gamble=0,",
          "                               steps_neutral=0, transitions=0)"]
lines[stats_anchor[0] + 1:stats_anchor[0] + 1] = method

init_call = [i for i, l in enumerate(lines)
             if l == "        self._s3_reset()" ]
assert len(init_call) == 1, init_call
lines.insert(init_call[0] + 1, "        self._s3p2_reset()")

act_call = [i for i, l in enumerate(lines)
            if l == "            self._s3_reset()"]
assert len(act_call) == 1, act_call
lines.insert(act_call[0] + 1, "            self._s3p2_reset()")

# ---------------- 4. phase block replacement -------------------------
start = [i for i, l in enumerate(lines)
         if "S3 sprint phase (roadmap P6)" in l]
assert len(start) == 1
start = start[0] - 1  # the '# ----' comment line above? no: comment is on same block start
# the block comment starts at the line containing '# ---- S3 sprint phase'
start = [i for i, l in enumerate(lines)
         if l.strip().startswith("# ---- S3 sprint phase")][0]
end = [i for i in range(start, len(lines))
       if "self.s3_stats['phase_steps'] += 1" in lines[i]]
assert len(end) == 1
end = end[0]

new_block = '''        # ---- S3 sprint phase v2 (roadmap P6, recalibrated after
        # s3_phase_report.md): R in [15,50] gets its own posture. The
        # score margin is normalised by what can still be earned in
        # the remaining steps (R * s3p2_rate, rate calibrated to the
        # measured post-250 deposit rate) and the mode is re-evaluated
        # every step with enter/exit hysteresis -- v1's fixed +/-400
        # band made the step-250 snapshot decide the mode for life.
        # Deep-behind gets a dedicated banking mode (tightest return
        # thresholds, no fresh recruits) instead of v1's gamble
        # loitering, which was the single biggest phase loss in v1;
        # protect's extra tightening fades as the lead grows (v1's
        # flat x1.3 leaked exactly the biggest leads).
        s3_phase_on = bool(P['s3p2_phase']) and not endgame and P['s3_R1'] <= R <= P['s3_R0']
        s3_lam_mult = 1.0
        s3_mode = 0
        if s3_phase_on:
            def _preal(wids, blist):
                num = 0.0; den = 0.0
                for w_ in wids:
                    c_ = workers[w_][1]
                    if c_ <= 0:
                        continue
                    d_ = min(DIST[workers[w_][0]][b_] for b_ in blist)
                    p_ = 1.0 if d_ <= R - 2 else max(0.0, 1.0 - (d_ - (R - 2)) / 10.0)
                    num += c_ * p_; den += c_
                return num / den if den > 0 else 1.0
            pr_me = _preal(myC, my_bases)
            diff = (cash + sum(workers[w_][1] for w_ in myC) * pr_me) \\
                - (players[1 - me][0] + sum(workers[w_][1] for w_ in enC) * _preal(enC, en_bases))
            prog = (P['s3_R0'] - R) / float(P['s3_R0'] - P['s3_R1'])
            s3_lam_mult = 1.0 + (P['s3_lam_top'] - 1.0) * prog * (1.25 - 0.75 * pr_me)
            ratio = diff / max(1.0, R * P['s3p2_rate'])
            m = self.s3p2_mode
            if m == 2 and ratio > P['s3p2_bank_exit']:
                m = 0
            if m != 2 and ratio <= P['s3p2_bank_enter']:
                m = 2
            if m == 0:
                if ratio >= P['s3p2_pro_enter']:
                    m = 1
                elif ratio <= P['s3p2_gam_enter']:
                    m = -1
            elif m == 1 and ratio < P['s3p2_pro_exit']:
                m = 0
            elif m == -1 and ratio > P['s3p2_gam_exit']:
                m = 0
            if m != self.s3p2_mode:
                self.s3p2_stats['transitions'] += 1
            self.s3p2_mode = m
            s3_mode = m
            if m == 2:
                bank_lam = P['s3p2_bank_lam0'] + (P['s3p2_bank_lam1'] - P['s3p2_bank_lam0']) * prog
                s3_lam_mult = max(s3_lam_mult, bank_lam)
                self.s3p2_stats['steps_bank'] += 1
            elif m == 1:
                fade = min(1.0, max(0.0, (P['s3p2_pro_fade_hi'] - ratio)
                                    / (P['s3p2_pro_fade_hi'] - P['s3p2_pro_fade_lo'])))
                s3_lam_mult *= 1.0 + (P['s3_protect_lam'] - 1.0) * fade
                self.s3p2_stats['steps_protect'] += 1
            elif m == -1:
                s3_lam_mult *= P['s3_gamble_lam']
                self.s3p2_stats['steps_gamble'] += 1
            else:
                self.s3p2_stats['steps_neutral'] += 1
            self.s3_stats['phase_steps'] += 1'''.split("\n")
lines[start:end + 1] = new_block

# ---------------- 5. recruit gate replacement ------------------------
gstart = [i for i, l in enumerate(lines)
          if "# S3 sprint phase: a collector whose empty-collector" in l]
assert len(gstart) == 1
gstart = gstart[0]
gend = [i for i in range(gstart, gstart + 30)
        if lines[i].strip() == "s3_rec_ok = True"]
assert len(gend) == 1
gend = gend[0]
new_gate = '''                    # S3 phase v2 payback gate: protect and banking
                    # modes take no fresh risk; with s3p2_recgate on,
                    # the bars are the widened v2 ones (v1 fired once
                    # in 12 phase games against a positive background
                    # rate); s3p2_recgate=0 restores the v1 bars/cap.
                    if P['s3p2_recgate']:
                        pb_n, pb_g = P['s3p2_payback'], P['s3p2_gamble_payback']
                        rec_min_R, rec_cap = P['s3p2_rec_min_R'], P['s3p2_rec_cap']
                    else:
                        pb_n, pb_g = P['s3_payback'], P['s3_gamble_payback']
                        rec_min_R, rec_cap = P['s3_rec_min_R'], P['s3_rec_cap']
                    if s3_phase_on and s3_mode not in (1, 2) and R >= rec_min_R \\
                            and self.s3_recruits_done < rec_cap \\
                            and nwork < P['max_units'] and budget >= 30:
                        pb_s3 = pb_g if s3_mode < 0 else pb_n
                        if Wfree[b] >= 30.0 * pb_s3:
                            s3_rec_ok = True'''.split("\n")
lines[gstart:gend + 1] = new_gate

out = "\n".join(lines)
open(DST, "w").write(out)

# ---------------- variants -------------------------------------------
def variant(name, subs):
    t = out
    for a, b in subs:
        assert t.count(a) == 1, (name, a, t.count(a))
        t = t.replace(a, b)
    open(os.path.join(HERE, name), "w").write(t)

variant("fork_s3p2_honly.py", [("s3p2_phase=1,", "s3p2_phase=0,")])
variant("fork_s3p2_off.py", [("s3p2_phase=1,", "s3p2_phase=0,"),
                             ("s3_hunt=1,", "s3_hunt=0,")])
variant("fork_s3p2_norec.py", [("s3p2_recgate=1,", "s3p2_recgate=0,")])

# ---------------- P5 diff self-proof ---------------------------------
orig_lines = open(SRC).read().split("\n")
new_lines = out.split("\n")
diff = list(difflib.unified_diff(orig_lines, new_lines, lineterm="", n=1))
open(os.path.join(HERE, "s3p2_vs_s3_diff.txt"), "w").write("\n".join(diff))

# P5 regions in v1 (1-based, inclusive): death attribution,
# feedback eval + level application, hunter snapshot.
P5_REGIONS = [(1353, 1360), (1370, 1405), (1588, 1593)]
hunks = []
cur = None
for l in diff:
    mm = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", l)
    if mm:
        cur = (int(mm.group(1)), int(mm.group(2) or 1))
        hunks.append(cur)
bad = []
for (s, n) in hunks:
    e = s + max(n, 1) - 1
    for (a, b) in P5_REGIONS:
        if s <= b and e >= a:
            bad.append(((s, e), (a, b)))
print(f"diff hunks (v1 coords): {hunks}")
print(f"P5 regions: {P5_REGIONS}")
assert not bad, f"P5 REGION TOUCHED: {bad}"
print("P5 self-proof OK: no diff hunk overlaps a P5 region.")
print("wrote", DST, len(out), "bytes")
