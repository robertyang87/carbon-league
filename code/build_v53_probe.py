#!/usr/bin/env python3
"""build_v53_probe -- create base_v53_probe.py: a byte-level copy of
base_v53.py with FOUR read-only probe insertions (list appends of
already-computed locals; no randomness consumed, no control flow
changed):

  1. module global PROBE_RECORDS = [] next to _BOT.
  2. collector loop, right after the DP action values q5 are built
     (incl. the endgame delivery fix, BEFORE gate/runme/bonus/role
     overrides and lethal penalties): record ('C', wid, max(q5)) --
     the teacher's pure DP valuation of that collector's state.
  3. planter retiree branch: record ('P', wid, max(q5)).
  4. planter main branch: record ('P', wid, max(q5)) -- the teacher's
     task-value (TV/SV) valuation of that planter's state.

The diff between base_v53.py and base_v53_probe.py must show exactly
these insertions and nothing else (checked by the builder).
"""

import sys

SRC = "base_v53.py"
DST = "base_v53_probe.py"

src = open(SRC).read()

EDITS = []

# 1. global record list (module-level _BOT only; agent() re-assigns an
#    indented _BOT, so anchor at line start)
old = "\n_BOT = Bot()\n"
assert src.count(old) == 1
EDITS.append((old, "\n_BOT = Bot()\nPROBE_RECORDS = []\n"))

# 2. collector DP value (capture point: after endgame fix block,
#    before the gate block)
old = ("                if p in my_base_set:\n"
       "                    q5[0] = float(c)\n"
       "            if P['gate'] and gate_out")
assert src.count(old) == 1
EDITS.append((old,
              "                if p in my_base_set:\n"
              "                    q5[0] = float(c)\n"
              "            PROBE_RECORDS.append(('C', wid, float(max(q5))))\n"
              "            if P['gate'] and gate_out"))

# 3. planter retiree branch
old = ("                Q[wid] = q5\n"
       "                continue\n")
assert src.count(old) == 1
EDITS.append((old,
              "                Q[wid] = q5\n"
              "                PROBE_RECORDS.append(('P', wid, float(max(q5))))\n"
              "                continue\n"))

# 4. planter main branch (Q assignment right before the defence section)
old = ("            Q[wid] = q5\n"
       "\n"
       "        # ---------------- defence / attack bonuses")
assert src.count(old) == 1
EDITS.append((old,
              "            Q[wid] = q5\n"
              "            PROBE_RECORDS.append(('P', wid, float(max(q5))))\n"
              "\n"
              "        # ---------------- defence / attack bonuses"))

out = src
for old, new in EDITS:
    out = out.replace(old, new, 1)

open(DST, "w").write(out)
print(f"wrote {DST}: {len(src)} -> {len(out)} bytes")

# verify: removing the inserted lines reproduces the source exactly
stripped = out
for line in ("PROBE_RECORDS = []\n",
             "            PROBE_RECORDS.append(('C', wid, float(max(q5))))\n",
             "                PROBE_RECORDS.append(('P', wid, float(max(q5))))\n",
             "            PROBE_RECORDS.append(('P', wid, float(max(q5))))\n"):
    assert stripped.count(line) == 1, line
    stripped = stripped.replace(line, "", 1)
assert stripped == src, "probe insertions are not the only diff"
print("diff check OK: probe file == source + 4 insertions")
