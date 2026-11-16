#!/usr/bin/env python3
"""Gate-3 mechanical audit: Part 3 arm packs vs the base A pack
(echo3_search_pack.py, ea20a271; base per strategy-group ruling 2026-11-15
13:22 -- the sims pack is permanently closed and is NOT the base).
Programmatic asserts only. Changed-line classes inside the embedded
cnet2_search: terminal / leaf / MARGIN / leaf-wiring. Any changed line
outside the registered edit blocks = class 'other' = FAIL.
Also: constructive proof (re-applying build_variant.build to the base source
reproduces each variant's embedded source byte-for-byte) and the A<->B
single-variable assertion (sources differ only by the value-leaf block).
"""
import ast, base64, difflib, hashlib, json, sys, zlib
sys.path.insert(0, '/home/hatch/workspace/katago-probe2/part3/scripts')
import build_variant as BV

BASE = BV.BASE_PACK
ARMS = {'A': BV.OUT['A'], 'B': BV.OUT['B']}
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
            if isinstance(v, ast.Call):
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

def skeleton(text):
    return [ln for ln in text.split('\n') if not ln.startswith('_SRC = ')]

ts, cs, ss = load(BASE)
ps = payloads(ts)
chk(hashlib.md5(ts.encode()).hexdigest() == BV.BASE_MD5, 'base = A pack ea20a271 (13:22 ruling)')
chk('leaf_evals' not in ss['cnet2_search'] and 'SIMS' not in get_defaults(ss['cnet2_search'])
    and 'CLOCK' not in cs and 'SIMS' not in cs,
    'base carries no sims hooks/keys (counting-point constraint void per 13:22 ruling)')

CLS = {}
for ln in BV.HELPERS.split('\n'):
    if ln: CLS[ln] = 'terminal'
for ln in BV.TERMINAL_BRANCH.split('\n'):
    if ln: CLS[ln] = 'terminal'
for ln in BV.VALUE_BLOCK.split('\n'):
    if ln: CLS[ln] = 'leaf'
CLS['    def leaf(sim, me, left, fm=None):'] = 'leaf'
CLS['    def leaf(sim, me, left):'] = 'leaf'
_mline = [l for l in ss['cnet2_search'].split('\n') if BV.OLD_MARGIN in l][0]
CLS[_mline] = 'MARGIN'
CLS[_mline.replace(BV.OLD_MARGIN, BV.NEW_MARGIN)] = 'MARGIN'
for old, new in ((BV.OLD_SEARCH_CALL, BV.NEW_SEARCH_CALL), (BV.OLD_MCTS_CALL, BV.NEW_MCTS_CALL)):
    for ln in old.split('\n'):
        if ln: CLS[ln] = 'leaf-wiring'
    for ln in new.split('\n'):
        if ln: CLS[ln] = 'leaf-wiring'

arm_src = {}
for arm, path in ARMS.items():
    tv, cv, sv = load(path)
    arm_src[arm] = sv['cnet2_search']
    chk(skeleton(tv) == skeleton(ts), f'{arm}: pack skeleton outside _SRC line byte-identical to base A pack')
    chk(cv == cs, f'{arm}: CONF dict identical to base A pack')
    chk(cv['TEACHER'] == 2 and cv['GUARD'] is True and cv['GUARD_NARROW'] is True and cv['ROLL'] == 'self'
        and cv['ABORT_DL'] == 0.95 and cv['TEACHER_LATE'] == 2 and cv['LATE_STEP'] == 240
        and cv['SEARCH'] is True and cv['BUDGET'] == 0.7 and cv['HARD'] == 0.85 and cv['MODE'] == 'sweep',
        f'{arm}: CONF registered spot values (TEACHER=2/GUARD/GUARD_NARROW/ROLL/ABORT_DL/TEACHER_LATE/LATE_STEP/SEARCH/BUDGET/HARD/MODE)')
    pv = payloads(tv)
    chk(set(pv) == set(ps) and all(pv[k] == ps[k] for k in ps), f'{arm}: weight payloads byte-identical')
    for k in ('cnet2_feat', 'cnet2_np', 'fastsim'):
        chk(sv[k] == ss[k], f'{arm}: embedded module {k} byte-identical')
    dv, ds_ = get_defaults(sv['cnet2_search']), get_defaults(ss['cnet2_search'])
    chk(set(dv) == set(ds_) and all(dv[k] == ds_[k] for k in ds_ if k != 'MARGIN')
        and dv['MARGIN'] == 625.5 and ds_['MARGIN'] == 20.0 and dv['GUARD_CASH'] == 40.0,
        f'{arm}: embedded DEFAULTS identical except MARGIN 20.0 -> 625.5 (GUARD_CASH=40 kept)')
    chk(BV.build(ss['cnet2_search'], arm) == sv['cnet2_search'],
        f'{arm}: constructive reproduction (base source + registered edits == variant source, byte-exact)')
    a_lines, b_lines = ss['cnet2_search'].split('\n'), sv['cnet2_search'].split('\n')
    sm = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    counts, other = {}, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        for ln in a_lines[i1:i2] + b_lines[j1:j2]:
            if not ln:
                continue
            c = CLS.get(ln)
            if c is None:
                other.append(ln)
            else:
                counts[c] = counts.get(c, 0) + 1
    chk(not other, f'{arm}: every changed line classified leaf/terminal/MARGIN/leaf-wiring (other: {other[:3]})')
    print(f'   {arm}: changed-line class counts {counts}')
    chk('leaf_evals' not in sv['cnet2_search'], f'{arm}: no sims counting hook introduced')

w = zlib.decompress(base64.b64decode(ps['_B1']))
chk(hashlib.md5(w).hexdigest() == '444cc54c0d88da5656ff4c7eef1ec8eb', 'decoded _B1 weights md5 as audited (444cc54c)')

chk(arm_src['A'].replace(BV.VALUE_BLOCK, '') == arm_src['B'],
    'A<->B: embedded sources differ ONLY by the value-leaf block (A minus block == B, byte-exact)')
ta, ca_, sa_ = load(ARMS['A']); tb, cb_, sb_ = load(ARMS['B'])
chk(ca_ == cb_ and all(sa_[k] == sb_[k] for k in sa_ if k != 'cnet2_search'),
    'A<->B: CONF and other embedded modules identical')

print('\n%d/%d assertions passed' % (sum(1 for c, _ in ok if c), len(ok)))
print(json.dumps({'A_pack_md5': hashlib.md5(ta.encode()).hexdigest(),
                  'B_pack_md5': hashlib.md5(tb.encode()).hexdigest()}))
sys.exit(0 if all(c for c, _ in ok) else 1)
