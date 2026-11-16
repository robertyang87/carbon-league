#!/usr/bin/env python3
"""Build deliverables A/B for KataGo probe-2 Part 1 from the Echo3 pack body.
A: verbatim copy, only CONF "SEARCH": False -> True.
B: A + embedded cnet2_search gains a flat SIMS budget key +
   CONF CLOCK="sims"/SIMS=200/FIX_TRUNC=True + PROVISIONAL header comment.
   Counting point (PREREG_SEARCH.md S1, locked): inside leaf() itself --
   1 sim = 1 leaf evaluation; stats["sims"] records measured leaf deltas.
The _SRC line is re-encoded with repr() (verified byte-identical round-trip).
"""
import ast, hashlib, sys

BASE = '/home/hatch/workspace/katago-probe/impl/echo3_ref.py'
OUTA = '/home/hatch/workspace/katago-probe2/modules/echo3_search_pack.py'
OUTB = '/home/hatch/workspace/katago-probe2/modules/echo3_sims_pack.py'

raw = open(BASE, 'rb').read()
md5 = hashlib.md5(raw).hexdigest()
assert md5 == '7b7620499411a088cf4dc17c0e929bd7' and len(raw) == 4576790, md5
text = raw.decode('utf-8')
lines = text.split('\n')

# ---------- A ----------
assert lines[26].count('"SEARCH": False') == 1, lines[26][:80]
a_lines = list(lines)
a_lines[26] = a_lines[26].replace('"SEARCH": False', '"SEARCH": True')
a_text = '\n'.join(a_lines)
open(OUTA, 'w').write(a_text)
print('A written', len(a_text.encode()), hashlib.md5(a_text.encode()).hexdigest())

# ---------- B ----------
b_lines = list(a_lines)
# (1) PROVISIONAL comment above CONF (line index 26 -> insert before it)
prov = ('# PROVISIONAL (KataGo probe-2 Part 1 prebuild, 2026-11-15): SEARCH=True, CLOCK="sims",\n'
        '# SIMS=200 and FIX_TRUNC=True below are placeholders; the final SIMS value is locked by\n'
        '# the construction-period calibration ladder (PREREG_SEARCH.md S1). Counting unit is\n'
        '# registered: 1 sim = one leaf evaluation; the counting point is the leaf() entry in the\n'
        '# embedded cnet2_search (stats["leaf_evals"]); stats["sims"]/stats["sims_steps"] record\n'
        '# measured per-step leaf deltas. Sweep budget enforcement is prospective: each kept\n'
        '# root candidate costs exactly one leaf evaluation at the horizon.')
b_lines.insert(26, prov)
conf_idx = 27
assert '"SEARCH": True' in b_lines[conf_idx]
b_lines[conf_idx] = b_lines[conf_idx].replace(
    '"START_DL": None}',
    '"START_DL": None, "CLOCK": "sims", "SIMS": 200, "FIX_TRUNC": True}')
assert '"CLOCK": "sims", "SIMS": 200, "FIX_TRUNC": True' in b_lines[conf_idx]

# (2) embedded cnet2_search patches (decoded domain, then repr re-encode)
src_idx = 7
assert b_lines[src_idx].startswith('_SRC = ')
S = ast.literal_eval(b_lines[src_idx][len('_SRC = '):])
s = S['cnet2_search']
old1 = 'PROJ=1.3, CLOCK="wall", HARD=0.6,\n'
new1 = 'PROJ=1.3, CLOCK="wall", HARD=0.6, SIMS=None,\n'
assert s.count(old1) == 1
s = s.replace(old1, new1)
old2 = '        fix_n = cfg["FIX_SIMS"][min(step // 50, len(cfg["FIX_SIMS"]) - 1)] if fixed else 0\n'
new2 = ('        fix_n = (cfg["SIMS"] if cfg.get("SIMS") is not None else\n'
        '                 cfg["FIX_SIMS"][min(step // 50, len(cfg["FIX_SIMS"]) - 1)]) if fixed else 0\n')
assert s.count(old2) == 1
s = s.replace(old2, new2)

# (2b) counting point: leaf() entry increments stats["leaf_evals"]
old_leaf = '''    def leaf(sim, me, left):
        m = [float(sim.cash[0]), float(sim.cash[1])]
'''
new_leaf = '''    def leaf(sim, me, left):
        stats["leaf_evals"] = stats.get("leaf_evals", 0) + 1  # SIMS COUNTING POINT (PREREG_SEARCH.md S1: 1 sim = 1 leaf evaluation)
        m = [float(sim.cash[0]), float(sim.cash[1])]
'''
assert s.count(old_leaf) == 1, "leaf anchor"
s = s.replace(old_leaf, new_leaf)

# (2c) _search: capture leaf baseline right after searched += 1
old_s0 = '''        stats["searched"] += 1
        tok, ctx = pol.last_enc
        uid2slot = {u: s for s, u in enumerate(f.own_uid) if u is not None}
        # ---- root candidate joint actions (b128 re-decode, prefix forced to the default)
'''
new_s0 = '''        stats["searched"] += 1
        _le0 = stats.get("leaf_evals", 0)
        tok, ctx = pol.last_enc
        uid2slot = {u: s for s, u in enumerate(f.own_uid) if u is not None}
        # ---- root candidate joint actions (b128 re-decode, prefix forced to the default)
'''
assert s.count(old_s0) == 1, "S0 anchor"
s = s.replace(old_s0, new_s0)

# (2d) _mcts: capture leaf baseline + record measured leaf delta
old_m0 = '''        stats["searched"] += 1
        tok, ctx = pol.last_enc
        lg0 = st["lg0"]
'''
new_m0 = '''        stats["searched"] += 1
        _le0 = stats.get("leaf_evals", 0)
        tok, ctx = pol.last_enc
        lg0 = st["lg0"]
'''
assert s.count(old_m0) == 1, "M0 anchor"
s = s.replace(old_m0, new_m0)
old_m1 = '''        stats["sims"] = stats.get("sims", 0) + sims_done
        stats.setdefault("sims_steps", []).append((step, sims_done))   # per-step budget actually used (calibrates FIX_SIMS)
'''
new_m1 = '''        _used = stats.get("leaf_evals", 0) - _le0   # measured at the counting point: leaf evaluations this call
        stats["sims"] = stats.get("sims", 0) + _used
        stats.setdefault("sims_steps", []).append((step, _used))   # per-step budget actually used (calibrates FIX_SIMS)
'''
assert s.count(old_m1) == 1, "M1 anchor"
s = s.replace(old_m1, new_m1)

# (3) sweep-path sims clock: candidate cap + depth-gate exemption + counter
old_s1 = """        nc = len(cand)
        bl = [(mi, ci) for mi in range(len(models)) for ci in range(nc)]
        nb = len(bl)
"""
new_s1 = """        nc = len(cand)
        fixed = cfg["CLOCK"] == "sims"
        if fixed:       # sims clock (sweep): cap candidates at the sim budget (1 candidate = 1 leaf evaluation)
            fix_n = (cfg["SIMS"] if cfg.get("SIMS") is not None else
                     cfg["FIX_SIMS"][min(step // 50, len(cfg["FIX_SIMS"]) - 1)])
            keep = max(1, -(-fix_n // len(models)))   # whole candidates per opponent model
            if keep < nc:
                cand = cand[:keep]; cmeta = cmeta[:keep]; nc = len(cand)
        bl = [(mi, ci) for mi in range(len(models)) for ci in range(nc)]
        nb = len(bl)
"""
assert s.count(old_s1) == 1, "S1 anchor"
s = s.replace(old_s1, new_s1)
old_s2 = """            if last_dt is not None and now - t0 + cfg["PROJ"] * last_dt > cfg["BUDGET"]:
                break
            if last_dt is None and now - t0 > 0.45 * cfg["BUDGET"]:
                break
            if last_dt is not None and time.time() - tw + cfg["PROJ"] * last_wall > cfg["HARD"]:
                break
"""
new_s2 = """            if not fixed and last_dt is not None and now - t0 + cfg["PROJ"] * last_dt > cfg["BUDGET"]:
                break
            if not fixed and last_dt is None and now - t0 > 0.45 * cfg["BUDGET"]:
                break
            if not fixed and last_dt is not None and time.time() - tw + cfg["PROJ"] * last_wall > cfg["HARD"]:
                break
"""
assert s.count(old_s2) == 1, "S2 anchor"
s = s.replace(old_s2, new_s2)
# S3a: shallow exit -- record the leaf delta at this exit (leaves not yet called -> 0)
old_s3a = """        if depth < min(cfg["MIN_DEPTH"], H):
            stats["shallow"] += 1
            return acts
"""
new_s3a = """        if depth < min(cfg["MIN_DEPTH"], H):
            stats["shallow"] += 1
            if fixed:       # sims clock: record leaf evaluations this step (none performed on this exit)
                _used = stats.get("leaf_evals", 0) - _le0
                stats["sims"] = stats.get("sims", 0) + _used
                stats.setdefault("sims_steps", []).append((step, _used))
            return acts
"""
assert s.count(old_s3a) == 1, "S3a anchor"
s = s.replace(old_s3a, new_s3a)
# S3b: normal exit -- record AFTER the sc line, the only place leaf() is called in _search
old_s3b = """        else:
            sc = sc[0]
        j = int(np.argmax(sc))
"""
new_s3b = """        else:
            sc = sc[0]
        if fixed:       # sims clock: record leaf evaluations actually performed this step (counting point: leaf())
            _used = stats.get("leaf_evals", 0) - _le0
            stats["sims"] = stats.get("sims", 0) + _used
            stats.setdefault("sims_steps", []).append((step, _used))
        j = int(np.argmax(sc))
"""
assert s.count(old_s3b) == 1, "S3b anchor"
s = s.replace(old_s3b, new_s3b)
S['cnet2_search'] = s
S['cnet2_search'] = s
b_lines[src_idx] = '_SRC = ' + repr(S)

b_text = '\n'.join(b_lines)
open(OUTB, 'w').write(b_text)
print('B written', len(b_text.encode()), hashlib.md5(b_text.encode()).hexdigest())
