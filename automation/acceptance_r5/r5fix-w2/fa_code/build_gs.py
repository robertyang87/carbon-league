"""build_gs: generate the guard/steal single-knob variants from
base_v53.py by anchored text patches (same protocol as build_s2.py;
every anchor asserted unique; base_v53.py itself is never modified).

Context: S2's posture layer was convicted as a net negative, but its
report (s2_posture_report.md sec.7 item 1) prescribes a retest of the
OPPOSITE response to a harvest-type opponent (one that cuts all its
planters): instead of cutting our own trees, keep the forest and
press the seize side -- the opponent's standing trees can no longer
be replaced or re-seized back once its planters are gone.  This build
tests the two halves as strictly single-knob variants, gated on the
corrected S2 detection rule (sec.6 case 2 / sec.7 item 2):

    trigger = enemy planter count == 0 for >= 12 consecutive steps
    (single condition; the planting-history gate is dropped)

Implementation note (arming): both sides start the game with zero
planters (first planter appears around step 55-70), so a raw
"P == 0 for 12 steps" counter would fire at about step 12 of every
game.  The counter is therefore armed only after the enemy has
fielded at least one planter; the 12-step sustained-absence condition
itself is unchanged.  The eval harness records the full enemy-planter
trajectory so the raw (unarmed) rule's would-be trigger step can be
reconstructed and reported alongside.

Once triggered, the override latches for the rest of the game.

Variants (exactly one PARAMS knob each, applied only after trigger):
  fork_gs_seize.py  "press the seize":  seize_min 8.0 -> 3.5
                    (the seize-task admission gate, L1001-1002; the
                    same endpoint S2's bundled anti-tree overlay used
                    for this knob, here isolated)
  fork_gs_guard.py  "guard the forest": guard_w 0.5 -> 0.85
                    (guard-task weight L1010 + collector intercept
                    bonus L1160; same endpoint as S2's anti-tree)
  fork_gs_both.py   both knobs (built only for the A+B interaction
                    check; used iff A and B are each positive alone)

The per-step hook restores self.P from a pristine copy before
applying the override (base never mutates self.P after __init__,
verified by grep), so an untriggered game is decision-identical to
base_v53.  No added code consumes global random.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "base_v53.py")).read()


def patch(text, old, new, name):
    n = text.count(old)
    assert n == 1, f"anchor {name!r} found {n} times (want 1)"
    return text.replace(old, new)


A_INIT = "        self.eval_ec = 0.0\n"
A_ACT = "    def act(self, obs, conf):"
A_HOOK = "        # enemy collector reach maps"

GS_METHODS = '''
    # ------------- GS guard/steal response (single-knob) -----------
    def _gs_reset(self):
        self._P0 = dict(PARAMS)
        self.gs_armed = False
        self.gs_run = 0
        self.gs_max_run = 0
        self.gs_trigger_step = None
        self.gs_trace = []

    def _gs_update(self, step, enP):
        P = self.P
        P.clear(); P.update(self._P0)
        if len(enP) > 0:
            self.gs_armed = True
            self.gs_run = 0
        elif self.gs_armed:
            self.gs_run += 1
            if self.gs_run > self.gs_max_run:
                self.gs_max_run = self.gs_run
        if self.gs_trigger_step is None and self.gs_run >= 12:
            self.gs_trigger_step = step
            self.gs_trace.append(dict(kind='trigger', step=step))
        if self.gs_trigger_step is not None:
__OVERRIDE__
'''

OVERRIDES = {
    "fork_gs_seize.py": (
        "            P['seize_min'] = 3.5\n",
        "seize_min 8.0 -> 3.5 (seize gate only)"),
    "fork_gs_guard.py": (
        "            P['guard_w'] = 0.85\n",
        "guard_w 0.5 -> 0.85 (guard weight only)"),
    "fork_gs_both.py": (
        "            P['seize_min'] = 3.5\n"
        "            P['guard_w'] = 0.85\n",
        "seize_min 8.0 -> 3.5 AND guard_w 0.5 -> 0.85"),
}

HEADERS = {
    "fork_gs_seize.py": (
        "# fork_gs_seize: base_v53 + GS response variant A (press the "
        "seize).\n# Trigger: enemy planters == 0 for >=12 consecutive "
        "steps (armed after the\n# enemy first fields a planter; "
        "latching). Knob: seize_min 8.0 -> 3.5 only.\n# Derived from "
        "base_v53.py by build_gs.py; single-variable validation for "
        "the S2\n# report sec.7 prescription. Not a submission "
        "candidate.\n"),
    "fork_gs_guard.py": (
        "# fork_gs_guard: base_v53 + GS response variant B (guard the "
        "forest).\n# Trigger: enemy planters == 0 for >=12 consecutive "
        "steps (armed after the\n# enemy first fields a planter; "
        "latching). Knob: guard_w 0.5 -> 0.85 only.\n# Derived from "
        "base_v53.py by build_gs.py; single-variable validation for "
        "the S2\n# report sec.7 prescription. Not a submission "
        "candidate.\n"),
    "fork_gs_both.py": (
        "# fork_gs_both: base_v53 + GS variants A+B combined "
        "(interaction check only).\n# Trigger: enemy planters == 0 for "
        ">=12 consecutive steps (armed, latching).\n# Knobs: seize_min "
        "8.0 -> 3.5 AND guard_w 0.5 -> 0.85.\n# Derived from "
        "base_v53.py by build_gs.py. Not a submission candidate.\n"),
}


def build(fname):
    override, _desc = OVERRIDES[fname]
    methods = GS_METHODS.replace("__OVERRIDE__", override.rstrip("\n"))
    text = SRC
    text = patch(text, A_INIT, A_INIT + "        self._gs_reset()\n",
                 "init")
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines)
            if "self.belief = Belief(me, my_bases + en_bases)" in ln]
    assert len(hits) == 1, f"reset line hits {len(hits)}"
    lines[hits[0]] = lines[hits[0]] + "; self._gs_reset()"
    text = "\n".join(lines)
    text = patch(text, A_ACT, methods + "\n" + A_ACT, "methods")
    text = patch(text, A_HOOK,
                 "        self._gs_update(step, enP)\n" + A_HOOK, "hook")
    out = HEADERS[fname] + text
    path = os.path.join(HERE, fname)
    open(path, "w").write(out)
    return path


if __name__ == "__main__":
    for fname in OVERRIDES:
        p = build(fname)
        print("built", os.path.basename(p), os.path.getsize(p))
