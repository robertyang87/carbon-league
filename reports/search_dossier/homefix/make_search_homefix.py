#!/usr/bin/env python3
"""Build the HOMEFIX derived cnet2_search source (composite package:
D1 search layer + endgame return-home rule layer, exit-overlay form).

Anchored-transform method, same as iter1's make_search_cmp3.py: the
base is the D1 package's cmp3 search source (iter2/build_d1, md5
b8c77acacf936ee383fa5ab973e4ff73) and the delta is EXACTLY:

  1. This header block (inserted after the module docstring).
  2. Import of the rule layer (cnet2_homefix as HF) after the
     cnet2_comparator import.
  3. The three final-action exits of the constructed agent() -- the
     no-search early return, the search-path return, and the
     exception-fallback return -- pass their joint action through
     HF.apply_homefix(obs, ...) before returning. Nothing else in the
     agent changes: candidate generation, the comparator, the guard
     and every rule layer above the exit see no trace of the rule.

There are NO other changes: no DEFAULTS delta, no CONF keys (the
rule's only parameter, HOME_BUFFER=2, is hard-pinned in
cnet2_homefix per REGISTRATION_COMPOSITE_RULE_LAYER_20261116.md
section 1; the shipped composite's CONF stays byte-identical to the
D1 package's CONF), the actor module and the weights are untouched.

Every anchor must match exactly once or the build stops.
"""
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.expanduser("~/workspace/search-group/iter2/build_d1/cnet2_search.py")
OUT = os.path.join(HERE, "build_comp", "cnet2_search.py")
PIN_BASE_MD5 = "b8c77acacf936ee383fa5ab973e4ff73"

base = open(BASE, "rb").read().decode("utf-8")
got = hashlib.md5(base.encode("utf-8")).hexdigest()
assert got == PIN_BASE_MD5, f"base search md5 mismatch: {got} != {PIN_BASE_MD5}"

HEADER = '''# =====================================================================
# HOMEFIX DERIVED SOURCE -- cnet2_search_cmp3 (D1, md5
# b8c77acacf936ee383fa5ab973e4ff73) plus the endgame return-home rule
# layer in EXIT OVERLAY form (generator: make_search_homefix.py;
# registered by reports/REGISTRATION_COMPOSITE_RULE_LAYER_20261116.md
# section 1). Everything not listed below is byte-identical to the
# cmp3 base.
# Changes vs the cmp3 base:
#   1. This header block (inserted after the module docstring).
#   2. Import of the rule layer (cnet2_homefix as HF) after the
#      cnet2_comparator import.
#   3. The three final-action exits of the constructed agent() pass
#      the joint action through HF.apply_homefix(obs, ...) before
#      returning: the no-search early return (acts), the search-path
#      return (out), and the exception-fallback return (acts). The
#      rule is a deterministic pure function of the observation and
#      the final action; candidate generation and the comparator do
#      NOT carry it. HOME_BUFFER=2 is hard-pinned in cnet2_homefix;
#      this source adds no CONF keys and no DEFAULTS entries.
# =====================================================================
'''


def sub_once(text, old, new):
    assert text.count(old) == 1, f"anchor not unique ({text.count(old)}): {old[:80]!r}"
    return text.replace(old, new)


text = base

# (2) rule-layer import after the comparator import
text = sub_once(
    text,
    "from cnet2_comparator import Candidate, Comparator, HeuristicLeafAdapter, StatsCollector  # noqa: E402\n",
    "from cnet2_comparator import Candidate, Comparator, HeuristicLeafAdapter, StatsCollector  # noqa: E402\n"
    "import cnet2_homefix as HF  # noqa: E402\n")

# (3a) no-search early return (16-space indent, returns acts)
text = sub_once(
    text,
    "                agent.last_stats = dict(stats)\n                return acts\n",
    "                agent.last_stats = dict(stats)\n                return HF.apply_homefix(obs, acts)\n")

# (3b) search-path return (12-space indent, returns out)
text = sub_once(
    text,
    "            agent.last_stats = dict(stats)\n            return out\n",
    "            agent.last_stats = dict(stats)\n            return HF.apply_homefix(obs, out)\n")

# (3c) exception-fallback return (12-space indent, returns acts)
text = sub_once(
    text,
    "            agent.last_stats = dict(stats)\n            return acts\n",
    "            agent.last_stats = dict(stats)\n            return HF.apply_homefix(obs, acts)\n")

# (1) header block after the module docstring (first closing ''')
end = text.index('"""', text.index('"""') + 3) + 3
text = text[:end] + "\n" + HEADER + text[end:]

open(OUT, "w").write(text)

diff = list(difflib.unified_diff(base.splitlines(), text.splitlines(), lineterm=""))
added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
print(f"wrote {OUT}: {len(text)} chars, md5 {hashlib.md5(text.encode()).hexdigest()}")
print(f"diff vs cmp3 base: +{added} -{removed} lines")
sys.exit(0)
