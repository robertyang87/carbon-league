"""build_activation: lineage-regression activation scan variants.

Each variant is base_v53.py with exactly ONE change (see lineage_diff
report section 3, items D1-D4 and D6):

  fork_d6mark.py    PARAMS mark 0 -> 1            (D6)
  fork_d6rovers.py  PARAMS rovers 0 -> 1          (D6)
  fork_d6pextra.py  PARAMS p_extra 0 -> 1         (D6)
  fork_d6campocc.py PARAMS camp_occ 0.0 -> 30.0   (D6; report gives no
                    activation magnitude for camp_occ -- 30.0 = collector
                    replacement cost, the natural price of an even trade)
  fork_d2nopost.py  PARAMS recruit_post 20 -> 0   (D2: bool() gate closes
                    the late_post exception, restoring v18 hard stop)
  fork_d3keep25.py  PARAMS camp_keep 100.0 -> 25.0 (D3, v18 value)
  fork_d4tie3.py    PARAMS camp_tie 40.0 -> 3.0    (D4, v18 value)
  fork_d1floor.py   restore v18's two deleted camp late-floor lines
                    after `self.ncamp_mid = ncamp` (D1; v18 L974-975)

Every replacement is asserted to hit exactly once; diffs are printed for
eyeball verification.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = open(os.path.join(HERE, "base_v53.py")).read()

PARAM_EDITS = {
    "fork_d6mark.py": ("mark=0, mark_min", "mark=1, mark_min",
                       "PARAMS mark 0 -> 1 (D6 activation)"),
    "fork_d6rovers.py": ("rovers=0, rover_d", "rovers=1, rover_d",
                         "PARAMS rovers 0 -> 1 (D6 activation)"),
    "fork_d6pextra.py": ("p_extra=0, p_extra_trees", "p_extra=1, p_extra_trees",
                         "PARAMS p_extra 0 -> 1 (D6 activation)"),
    "fork_d6campocc.py": ("camp_occ=0.0, mark=0", "camp_occ=30.0, mark=0",
                          "PARAMS camp_occ 0.0 -> 30.0 (D6 activation)"),
    "fork_d2nopost.py": ("recruit_post=20", "recruit_post=0",
                         "PARAMS recruit_post 20 -> 0 (D2: v18 hard stop)"),
    "fork_d3keep25.py": ("camp_keep=100.0", "camp_keep=25.0",
                         "PARAMS camp_keep 100.0 -> 25.0 (D3: v18 value)"),
    "fork_d4tie3.py": ("camp_tie=40.0", "camp_tie=3.0",
                       "PARAMS camp_tie 40.0 -> 3.0 (D4: v18 value)"),
}

D1_ANCHOR = "        self.ncamp_mid = ncamp\n"
D1_INSERT = ("        if R <= P['camp_end_from'] and len(myC) >= 2:\n"
             "            ncamp = max(ncamp, P['camp_end'])\n")


def build(fname, text, note):
    stem = os.path.splitext(fname)[0]
    header = (f"# {stem}: base_v53 (public claude-opus-v53) with {note}. "
              f"Lineage-regression activation scan variant, not for submission.\n")
    out = os.path.join(HERE, fname)
    with open(out, "w") as f:
        f.write(header + text)
    subprocess.run([sys.executable, "-m", "py_compile", out], check=True)
    print(f"built {fname}")


def main():
    only = sys.argv[1:] or None
    for fname, (old, new, note) in PARAM_EDITS.items():
        if only and fname not in only:
            continue
        assert BASE.count(old) == 1, f"{fname}: anchor {old!r} count={BASE.count(old)}"
        build(fname, BASE.replace(old, new), note)
    if not only or "fork_d1floor.py" in only:
        assert BASE.count(D1_ANCHOR) == 1, f"D1 anchor count={BASE.count(D1_ANCHOR)}"
        build("fork_d1floor.py", BASE.replace(D1_ANCHOR, D1_ANCHOR + D1_INSERT),
              "v18 camp late-floor lines restored (D1)")


if __name__ == "__main__":
    main()
