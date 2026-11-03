"""build_s2: generate the S2 (structure line A) derivative files from
base_v53.py by anchored text patches. Every anchor is asserted unique.

Generated files:
  switcher_harvest.py  sparring bot: base + forced planter cut at step 100
                       (Alpha-killer #1, alpha_analysis 6.1)
  switcher_sprint.py   sparring bot: base + planter cut at step 150,
                       rho/endgame/recruit sprint (killer #3, 6.3)
  switcher_treewar.py  sparring bot: base + planting-budget boost in
                       steps 50-100 (killer #2, 6.2)
  fork_s2_posture.py   base + opponent-archetype classifier + posture
                       overlays + anti-jitter switching (the S2 product)
  fork_s2_no_tree.py   ablation: anti-tree overlay disabled
  fork_s2_no_harv.py   ablation: anti-harvest overlay disabled
  fork_s2_off.py       ablation/control: classifier off (pure base)

Common patches (all files):
  * __init__ / step-0 reset call self._s2_reset()
  * retire branch honours self._cut_p (forced planter cut): keep_p -> 0
    and the retire elif fires regardless of stock/tree conditions
  * a hook self._s2_update(...) is called once per act(), right before
    the enemy-reach maps are built -- i.e. before every PARAMS consumer
    (endgame local, rho local, tree values, hunt, camp, recruiting).
  * _s2_update restores self.P from a pristine copy each step, then
    applies the file's schedule/overlay, so overlays are purely
    functional on (step, posture). No global-random use anywhere in
    the added code (eval/record reproducibility protocol preserved).
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "base_v53.py")).read()


def patch(text, old, new, name):
    n = text.count(old)
    assert n == 1, f"anchor {name!r} found {n} times (want 1)"
    return text.replace(old, new)


A_INIT = "        self.eval_ec = 0.0\n"
A_KEEP = "        keep_p = P['early_planters'] if self.block_ok else P['retire_keep']"
A_RETIRE = ("        elif step >= P['retire_from'] and stock_low and "
            "n_en_trees <= P['retire_entrees'] and R >= P['retire_min_left'] "
            "and len(myP) > keep_p and cash >= 30 and myC:")
A_ACT = "    def act(self, obs, conf):"
A_HOOK = "        # enemy collector reach maps"

NEW_KEEP = ("        keep_p = 0 if getattr(self, '_cut_p', False) else "
            "(P['early_planters'] if self.block_ok else P['retire_keep'])")
NEW_RETIRE = ("        elif (getattr(self, '_cut_p', False) or "
              "(step >= P['retire_from'] and stock_low and "
              "n_en_trees <= P['retire_entrees'] and "
              "R >= P['retire_min_left'])) and len(myP) > keep_p "
              "and cash >= 30 and myC:")


def common(text, header, methods, hook_call, init_extra):
    text = patch(text, A_INIT, A_INIT + init_extra, "init")
    # step-0 reset: the giant one-line reset ends a line that starts with
    # the Belief construction; append the s2 reset at that line's end.
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines)
            if "self.belief = Belief(me, my_bases + en_bases)" in ln]
    assert len(hits) == 1, f"reset line hits {len(hits)}"
    lines[hits[0]] = lines[hits[0]] + "; self._s2_reset()"
    text = "\n".join(lines)
    text = patch(text, A_KEEP, NEW_KEEP, "keep_p")
    text = patch(text, A_RETIRE, NEW_RETIRE, "retire")
    text = patch(text, A_ACT, methods + "\n" + A_ACT, "methods")
    text = patch(text, A_HOOK, hook_call + "\n" + A_HOOK, "hook")
    return header + text


# ----------------------------------------------------------------------
# switcher bots
# ----------------------------------------------------------------------
SW_METHODS = '''
    # ---------------- S2 switcher schedule (sparring only) ------------
    def _s2_reset(self):
        self._P0 = dict(PARAMS)
        self._cut_p = False

    def _s2_update(self, step, R):
        P = self.P
        P.clear(); P.update(self._P0)
        self._cut_p = False
__BODY__
'''

SW_BODIES = {
    "harvest": """        if step >= 100:
            # killer #1 (alpha_analysis 6.1): all planters cut at step
            # 100, pure 10C/0P harvest for the rest of the game
            self._cut_p = True
            P['early_planters'] = 0
            P['retire_keep'] = 0
            P['recruit_until'] = 30
""",
    "sprint": """        if step >= 150:
            # killer #3 (alpha_analysis 6.3): cut planters at step 150,
            # pure collecting + endgame sprint (earlier delivery,
            # longer endgame phase, recruiting kept alive to R=22)
            self._cut_p = True
            P['early_planters'] = 0
            P['retire_keep'] = 0
            P['rho'] = 0.05
            P['endgame'] = 22
            P['recruit_until'] = 22
""",
    "treewar": """        if 50 <= step <= 100:
            # killer #2 (alpha_analysis 6.2): planting-budget boost in
            # the step 50-100 tree-efficiency window: cheaper thresholds,
            # a third planter, cheaper seizes
            P['plant_min'] = 5.0
            P['plant_ratio'] = 1.15
            P['early_planters'] = 3
            P['seize_min'] = 5.0
""",
}

SW_HEADERS = {
    "harvest": "# switcher_harvest: SPARRING ONLY. base_v53 + forced planter cut at step 100 (pure harvest), replicating Alpha-killer #1 (alpha_analysis 6.1). Not a submission candidate.\n",
    "sprint": "# switcher_sprint: SPARRING ONLY. base_v53 + planter cut at step 150 and endgame sprint, replicating Alpha-killer #3 (alpha_analysis 6.3). Not a submission candidate.\n",
    "treewar": "# switcher_treewar: SPARRING ONLY. base_v53 + planting-budget boost in steps 50-100, replicating Alpha-killer #2 (alpha_analysis 6.2). Not a submission candidate.\n",
}


def build_switcher(mode):
    methods = SW_METHODS.replace("__BODY__", SW_BODIES[mode])
    out = common(SRC, SW_HEADERS[mode], methods,
                 "        self._s2_update(step, R)",
                 "        self._s2_reset()\n")
    path = os.path.join(HERE, f"switcher_{mode}.py")
    open(path, "w").write(out)
    return path


# ----------------------------------------------------------------------
# posture layer
# ----------------------------------------------------------------------
POSTURE_METHODS = '''
    # ---------------- S2 posture layer (plan A) ------------------------
    def _s2_reset(self):
        self._P0 = dict(PARAMS)
        self._cut_p = False
        self.posture = 'balanced'
        self.post_cand = 'balanced'
        self.post_cand_run = 0
        self.post_last_switch = -99
        self.post_switches = 0
        self.post_trace = []
        self._pt_en_prev = set()
        self._pt_my_prev = set()
        self.feat_first_en_tree = None
        self.feat_first_en_seize = None
        self.feat_en_plants = 0
        self.feat_en_seizes = 0
        self.feat_en_hist = []
        self.feat_siege_hist = []
        self._ckpt_done = set()

    def _feat_snapshot(self, step, n_en_trees):
        return dict(step=step, first_tree=self.feat_first_en_tree,
                    first_seize=self.feat_first_en_seize,
                    en_plants=self.feat_en_plants,
                    en_seizes=self.feat_en_seizes,
                    en_trees=n_en_trees,
                    deliv20=round(sum(self.deliv_hist[-20:]), 1),
                    siege30=sum(self.feat_siege_hist[-30:]))

    def _classify(self, step, n_enC, n_enP, n_en_trees):
        if step < 35:
            return 'balanced'
        hist = self.feat_en_hist[-15:]
        enP_recent = sum(h[1] for h in hist) / max(1, len(hist))
        # harvest: no planters left/ever, full collector roster, and
        # (almost) no planting history -- the step-100/150 cut signature
        if n_enP == 0 and n_enC >= 6 and self.feat_en_plants <= 3:
            return 'harvest'
        # tree: heavy planting + standing forest + planter presence;
        # the >=2.6 recent-planter clause is the 3-planter window boost
        if enP_recent >= 2.6 and self.feat_en_plants >= 8 and n_en_trees >= 5:
            return 'tree'
        if (self.feat_en_plants >= 14 and n_en_trees >= 8
                and enP_recent >= 1.8):
            return 'tree'
        # blockade: sustained camping at my bases while the enemy's
        # delivery tempo stays low (they invest in campers, not cargo).
        # Gates tightened after the smoke game: base-family opponents
        # camp as a matter of course and early-game delivery is low for
        # everyone, so require step>=110 + denser siege + lower tempo.
        if (step >= 110
                and sum(self.feat_siege_hist[-30:]) >= 24
                and sum(self.deliv_hist[-20:]) <= 150.0):
            return 'blockade'
        return 'balanced'

    def _apply_posture(self):
        P = self.P
        if self.posture == 'tree' and POSTURE_CFG['anti_tree']:
            # anti-tree: seize budget/speed up, contest cells, guard harder
            P['seize_min'] = 3.5
            P['steal_w'] = 0.95
            P['plant_min'] = 9.0
            P['plant_ratio'] = 1.35
            P['guard_w'] = 0.85
            P['guard_min'] = 12.0
        elif self.posture == 'harvest' and POSTURE_CFG['anti_harvest']:
            # anti-harvest: cut own planters, sprint phase early,
            # tighter cargo protection, hunt loaded collectors harder
            self._cut_p = True
            P['early_planters'] = 0
            P['retire_keep'] = 0
            P['rho'] = 0.05
            P['endgame'] = 26
            P['hunt_min'] = 100.0
            P['avoid1'] = 0.7
            P['avoid2'] = 0.2
            P['recruit_until'] = 28
        elif self.posture == 'blockade' and POSTURE_CFG['anti_block']:
            # anti-blockade: stop feeding the camp war, add gate guards
            P['campers'] = 0
            P['sb_guards'] = 3

    def _s2_update(self, step, R, workers, trees, enC, enP, my_bases, me):
        P = self.P
        P.clear(); P.update(self._P0)
        self._cut_p = False
        # --- enemy behaviour features (always tracked, even when the
        # classifier is off, so ablation traces stay comparable)
        en_tids = {tid for tid, t in trees.items() if t[2] != me}
        my_tids = {tid for tid, t in trees.items() if t[2] == me}
        for tid in en_tids - self._pt_en_prev:
            if tid in self._pt_my_prev:
                self.feat_en_seizes += 1
                if self.feat_first_en_seize is None:
                    self.feat_first_en_seize = step
            else:
                self.feat_en_plants += 1
                if self.feat_first_en_tree is None:
                    self.feat_first_en_tree = step
        self._pt_en_prev = en_tids
        self._pt_my_prev = my_tids
        self.feat_en_hist.append((len(enC), len(enP)))
        siege_now = 0
        for wid in enC:
            w = workers[wid]
            if w[1] <= P['camper_max'] and any(
                    DIST[w[0]][b] <= 2 for b in my_bases):
                siege_now = 1
                break
        self.feat_siege_hist.append(siege_now)
        cand = self._classify(step, len(enC), len(enP), len(en_tids))
        for ck in (60, 100):
            if step >= ck and ck not in self._ckpt_done:
                self._ckpt_done.add(ck)
                self.post_trace.append(dict(
                    kind='ckpt', step=ck, cand=cand, posture=self.posture,
                    feats=self._feat_snapshot(step, len(en_tids))))
        if not POSTURE_CFG['on']:
            return
        # --- anti-jitter: candidate must persist, dwell between
        # switches, at most max_switches posture changes per game
        if cand == self.post_cand:
            self.post_cand_run += 1
        else:
            self.post_cand = cand
            self.post_cand_run = 1
        # switch budget counts only moves INTO a non-balanced posture;
        # retreating to balanced is always allowed (still dwell-bound)
        budget_ok = (self.post_cand == 'balanced'
                     or self.post_switches < POSTURE_CFG['max_switches'])
        if (self.post_cand != self.posture
                and self.post_cand_run >= POSTURE_CFG['confirm_steps']
                and step - self.post_last_switch >= POSTURE_CFG['min_dwell']
                and budget_ok
                and step >= POSTURE_CFG['start_step']):
            self.post_trace.append(dict(
                kind='switch', step=step, frm=self.posture,
                to=self.post_cand,
                feats=self._feat_snapshot(step, len(en_tids))))
            self.posture = self.post_cand
            self.post_last_switch = step
            if self.post_cand != 'balanced':
                self.post_switches += 1
        self._apply_posture()
'''

POSTURE_HOOK = ("        self._s2_update(step, R, workers, trees, enC, "
                "enP, my_bases, me)")

POSTURE_VARIANTS = {
    "fork_s2_posture.py": dict(on=1, anti_tree=1, anti_harvest=1,
                               anti_block=1),
    "fork_s2_no_tree.py": dict(on=1, anti_tree=0, anti_harvest=1,
                               anti_block=1),
    "fork_s2_no_harv.py": dict(on=1, anti_tree=1, anti_harvest=0,
                               anti_block=1),
    "fork_s2_off.py": dict(on=0, anti_tree=1, anti_harvest=1,
                           anti_block=1),
}


def build_posture(fname, cfg):
    full = dict(on=cfg["on"], anti_tree=cfg["anti_tree"],
                anti_harvest=cfg["anti_harvest"],
                anti_block=cfg["anti_block"], confirm_steps=5,
                min_dwell=20, max_switches=2, start_step=40)
    cfg_txt = "POSTURE_CFG = " + repr(full) + "\n\n\n"
    header = (f"# {fname}: base_v53 + S2 posture layer (plan A). "
              f"POSTURE_CFG={full}. Derived from base_v53.py by "
              f"build_s2.py; not a submission candidate.\n")
    # POSTURE_CFG must be defined before Bot uses it at runtime; method
    # bodies resolve it at call time, so injecting it right before
    # "class Bot" is sufficient and keeps the diff small.
    text = patch(SRC, "class Bot:", cfg_txt + "class Bot:", "cfg")
    out = common(text, header, POSTURE_METHODS, POSTURE_HOOK,
                 "        self._s2_reset()\n")
    path = os.path.join(HERE, fname)
    open(path, "w").write(out)
    return path


if __name__ == "__main__":
    made = []
    for mode in ("harvest", "sprint", "treewar"):
        made.append(build_switcher(mode))
    for fname, cfg in POSTURE_VARIANTS.items():
        made.append(build_posture(fname, cfg))
    for p in made:
        print("built", os.path.basename(p), os.path.getsize(p))
