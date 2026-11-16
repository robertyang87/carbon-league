#!/usr/bin/env python3
"""Gate-3 mechanical assertions: pack vs original zero-difference outside
SEARCH/sims items. Programmatic asserts, not human hunk review.
1) CONF dicts compared key-by-key (ast-literal).
2) Embedded DEFAULTS dicts compared key-by-key.
3) Weight payloads byte-identical (raw strings + decoded bytes md5).
4) Embedded module sources: 3 of 4 byte-identical; cnet2_search changed-line
   set must be covered by the sims whitelist patterns.
5) Pack-level lines outside _SRC/CONF/comment: identical sequence.
"""
import ast, base64, difflib, hashlib, re, sys, zlib

ORIG = '/home/hatch/workspace/katago-probe/impl/echo3_ref.py'
A = '/home/hatch/workspace/katago-probe2/modules/echo3_search_pack.py'
B = '/home/hatch/workspace/katago-probe2/modules/echo3_sims_pack.py'
ok = []
def chk(cond, msg):
    ok.append((bool(cond), msg))
    print(('PASS' if cond else 'FAIL'), '-', msg)
    return cond

def load(path):
    text = open(path).read()
    conf = src = None
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'CONF':
                conf = ast.literal_eval(node.value)
            elif node.targets[0].id == '_SRC':
                src = ast.literal_eval(node.value)
    return text, conf, src

def get_defaults(src_search):
    for node in ast.parse(src_search).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'DEFAULTS':
            v = node.value
            if isinstance(v, ast.Call):   # DEFAULTS = dict(k=v, ...)
                return {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords}
            return ast.literal_eval(v)
    raise AssertionError('DEFAULTS not found')

def payloads(text):
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id != '_SRC':
            try:
                v = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(v, str):
                out[node.targets[0].id] = v
    return out

to, co, so = load(ORIG)
ta, ca, sa = load(A)
tb, cb, sb = load(B)

# 1) CONF key-by-key
chk(set(ca) == set(co), 'A CONF key set == orig')
chk(all(ca[k] == co[k] for k in co if k != 'SEARCH') and ca['SEARCH'] is True and co['SEARCH'] is False,
    'A CONF values == orig except SEARCH False->True')
chk(set(cb) == set(co) | {'CLOCK', 'SIMS', 'FIX_TRUNC'}, 'B CONF key set == orig + {CLOCK,SIMS,FIX_TRUNC}')
chk(all(cb[k] == co[k] for k in co if k != 'SEARCH') and cb['SEARCH'] is True,
    'B CONF shared values == orig except SEARCH')
chk(cb['CLOCK'] == 'sims' and cb['SIMS'] == 200 and cb['FIX_TRUNC'] is True, 'B CONF sims keys as built')

# 2) DEFAULTS key-by-key
do, da, db = get_defaults(so['cnet2_search']), get_defaults(sa['cnet2_search']), get_defaults(sb['cnet2_search'])
chk(da == do, 'A embedded DEFAULTS == orig (all keys/values)')
chk(set(db) == set(do) | {'SIMS'} and db['SIMS'] is None and all(db[k] == do[k] for k in do),
    'B embedded DEFAULTS == orig + {SIMS: None}')

# 3) weight payloads
po, pa, pb = payloads(to), payloads(ta), payloads(tb)
chk(set(po) == set(pa) == set(pb), 'payload variable sets equal')
for k in sorted(po):
    chk(po[k] == pa[k] == pb[k], f'payload {k} byte-identical across orig/A/B')
w = zlib.decompress(base64.b64decode(po['_B1']))
chk(hashlib.md5(w).hexdigest() == '444cc54c0d88da5656ff4c7eef1ec8eb', 'decoded _B1 md5 as audited')

# 4) embedded sources: module identity + changed-line whitelist for cnet2_search
for k in ('cnet2_feat', 'cnet2_np', 'fastsim'):
    chk(so[k] == sa[k] == sb[k], f'embedded module {k} byte-identical')
WL = [re.compile(p) for p in (
    r'SIMS=None', r'fix_n = \(cfg\["SIMS"\]', r'cfg\["FIX_SIMS"\]\[min\(step // 50',
    r'fixed = cfg\["CLOCK"\] == "sims"', r'sims clock \(sweep\)', r'fix_n = cfg\["SIMS"\]',
    r'keep = max\(1, -\(-fix_n // len\(models\)\)\)', r'if keep < nc:',
    r'cand = cand\[:keep\]; cmeta = cmeta\[:keep\]; nc = len\(cand\)',
    r'if not fixed and last_dt', r'sims clock: record leaf evaluations',
    r'leaf_evals', r'_le0', r'_used',
    r'\+ sims_done', r'append\(\(step, sims_done\)\)',
)]
sm = difflib.SequenceMatcher(None, so['cnet2_search'].split('\n'), sb['cnet2_search'].split('\n'), autojunk=False)
bad = []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal':
        continue
    for j in range(j1, j2):
        line = sb['cnet2_search'].split('\n')[j]
        if not any(p.search(line) for p in WL):
            bad.append(line)
    for i in range(i1, i2):  # removed lines must be the pre-image of a whitelisted change
        line = so['cnet2_search'].split('\n')[i]
        if not any(p.search(line) for p in WL) and 'fix_n = cfg["FIX_SIMS"]' not in line \
                and 'if last_dt' not in line and 'SIMS' not in line and 'CLOCK="wall", HARD=0.6,' not in line:
            bad.append('ORIG: ' + line)
chk(not bad, f'cnet2_search changed-line set fully whitelisted (violations: {bad[:3]})')
chk(so['cnet2_search'] == sa['cnet2_search'], 'A embedded cnet2_search == orig')

# 5) pack-level lines outside _SRC / CONF / PROVISIONAL comment
def skeleton(text):
    out = []
    in_prov = False
    for ln in text.split('\n'):
        if ln.startswith('# PROVISIONAL'):
            in_prov = True
            continue
        if in_prov and ln.startswith('#'):
            continue
        in_prov = False
        if ln.startswith('_SRC = ') or ln.startswith('CONF = '):
            continue
        out.append(ln)
    return out
chk(skeleton(to) == skeleton(ta) == skeleton(tb), 'pack skeleton (non-_SRC/CONF/comment lines) identical')

print('\n%d/%d assertions passed' % (sum(1 for c, _ in ok if c), len(ok)))
sys.exit(0 if all(c for c, _ in ok) else 1)
