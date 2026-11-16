#!/usr/bin/env python3
"""Gate-3 mechanical re-audit for the Part 3 refit arms (strategy-group
ruling 2026-11-15 13:34 sec.3 (iii) + supplementary MARGIN ruling).
Mirrors audit_variant.py assertion for assertion, with the refit
literals: DEFAULTS MARGIN == MARGIN' in BOTH arms; constructive
reproduction via build_variant_refit.build; A'<->B single-variable
proof re-established on the rebuilt pair (A' minus the refit value
block == B(') byte-exact). Extra refit-specific proofs:
  * if arm B was rebuilt (MARGIN' != 625.5): B' vs the accepted B pack
    (fbe9beff) differs ONLY in the MARGIN line of the embedded source;
  * A' vs the accepted A pack (d8d7ebc8): every changed line belongs to
    the value-leaf block (old or new) or the MARGIN line;
  * if MARGIN' == 625.5: the B arm file is the accepted fbe9beff pack,
    byte-untouched.
"""
import ast
import base64
import difflib
import hashlib
import json
import os
import sys
import zlib

sys.path.insert(0, '/home/hatch/workspace/katago-probe2/part3/scripts')
import build_variant as BV      # noqa: E402
import build_variant_refit as BVR  # noqa: E402

BASE = BV.BASE_PACK
MOD = '/home/hatch/workspace/katago-probe2/part3/modules'
OLD_A = os.path.join(MOD, 'echo3_part3_a_pack.py')
OLD_B = os.path.join(MOD, 'echo3_part3_b_pack.py')
OLD_A_MD5 = 'd8d7ebc82a564c9510e9aeae015ce4cd'
OLD_B_MD5 = 'fbe9beffd1683ddd7995a1282aafc572'
NEW_A = os.path.join(MOD, 'echo3_part3_a_pack_v2.py')
NEW_B = os.path.join(MOD, 'echo3_part3_b_pack_v2.py')
B_REBUILT = os.path.exists(NEW_B)
ARMS = {'A': NEW_A, 'B': NEW_B if B_REBUILT else OLD_B}
MP = BVR.MARGIN_PRIME

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


def md5t(t):
    return hashlib.md5(t.encode()).hexdigest()


def changed_lines(a_text, b_text):
    a_lines, b_lines = a_text.split('\n'), b_text.split('\n')
    sm = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        out.extend(ln for ln in a_lines[i1:i2] + b_lines[j1:j2] if ln)
    return out


ts, cs, ss = load(BASE)
ps = payloads(ts)
chk(md5t(ts) == BV.BASE_MD5, 'base = A pack ea20a271 (13:22 ruling)')
chk('leaf_evals' not in ss['cnet2_search'] and 'SIMS' not in get_defaults(ss['cnet2_search'])
    and 'CLOCK' not in cs and 'SIMS' not in cs,
    'base carries no sims hooks/keys')

CLS = {}
for ln in BV.HELPERS.split('\n'):
    if ln: CLS[ln] = 'terminal'
for ln in BV.TERMINAL_BRANCH.split('\n'):
    if ln: CLS[ln] = 'terminal'
for ln in BVR.VALUE_BLOCK.split('\n'):
    if ln: CLS[ln] = 'leaf'
for ln in BV.OLD_LEAF_HEAD.split('\n'):
    if ln: CLS.setdefault(ln, 'leaf')
CLS['    def leaf(sim, me, left, fm=None):'] = 'leaf'
CLS['    def leaf(sim, me, left):'] = 'leaf'
_mline = [l for l in ss['cnet2_search'].split('\n') if BV.OLD_MARGIN in l][0]
CLS[_mline] = 'MARGIN'
CLS[_mline.replace(BV.OLD_MARGIN, BVR.NEW_MARGIN)] = 'MARGIN'
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
        f'{arm}: CONF registered spot values')
    pv = payloads(tv)
    chk(set(pv) == set(ps) and all(pv[k] == ps[k] for k in ps), f'{arm}: weight payloads byte-identical')
    for k in ('cnet2_feat', 'cnet2_np', 'fastsim'):
        chk(sv[k] == ss[k], f'{arm}: embedded module {k} byte-identical')
    dv, ds_ = get_defaults(sv['cnet2_search']), get_defaults(ss['cnet2_search'])
    chk(set(dv) == set(ds_) and all(dv[k] == ds_[k] for k in ds_ if k != 'MARGIN')
        and dv['MARGIN'] == MP and ds_['MARGIN'] == 20.0 and dv['GUARD_CASH'] == 40.0,
        f"{arm}: embedded DEFAULTS identical except MARGIN 20.0 -> {MP} (GUARD_CASH=40 kept)")
    chk(BVR.build(ss['cnet2_search'], arm) == sv['cnet2_search'],
        f'{arm}: constructive reproduction (base source + refit registered edits == variant source, byte-exact)')
    other = [ln for ln in changed_lines(ss['cnet2_search'], sv['cnet2_search'])
             if CLS.get(ln) is None]
    chk(not other, f'{arm}: every changed line classified leaf/terminal/MARGIN/leaf-wiring (other: {other[:3]})')
    counts = {}
    for ln in changed_lines(ss['cnet2_search'], sv['cnet2_search']):
        counts[CLS[ln]] = counts.get(CLS[ln], 0) + 1
    print(f'   {arm}: changed-line class counts {counts}')
    chk('leaf_evals' not in sv['cnet2_search'], f'{arm}: no sims counting hook introduced')

w = zlib.decompress(base64.b64decode(ps['_B1']))
chk(hashlib.md5(w).hexdigest() == '444cc54c0d88da5656ff4c7eef1ec8eb', 'decoded _B1 weights md5 as audited (444cc54c)')

chk(arm_src['A'].replace(BVR.VALUE_BLOCK, '') == arm_src['B'],
    "A'<->B: embedded sources differ ONLY by the value-leaf block (A' minus refit block == B, byte-exact)")
ta, ca_, sa_ = load(ARMS['A']); tb, cb_, sb_ = load(ARMS['B'])
chk(ca_ == cb_ and all(sa_[k] == sb_[k] for k in sa_ if k != 'cnet2_search'),
    "A'<->B: CONF and other embedded modules identical")
dva = get_defaults(sa_['cnet2_search']); dvb = get_defaults(sb_['cnet2_search'])
chk(dva['MARGIN'] == dvb['MARGIN'] == MP,
    f"A'<->B: both arms carry the shared MARGIN' = {MP} (no two-variable form)")

# ---- refit-specific: deltas vs the accepted (pre-refit) arms
toa, _, soa = load(OLD_A)
chk(md5t(toa) == OLD_A_MD5, 'accepted A pack on disk is still d8d7ebc8 (not overwritten)')
_bline = [l for l in ss['cnet2_search'].split('\n') if BV.OLD_MARGIN in l][0]
allowed_a = set(BV.VALUE_BLOCK.split('\n')) | set(BVR.VALUE_BLOCK.split('\n'))
allowed_a |= {_bline,
              _bline.replace(BV.OLD_MARGIN, BV.NEW_MARGIN),
              _bline.replace(BV.OLD_MARGIN, BVR.NEW_MARGIN)}
diff_a = changed_lines(soa['cnet2_search'], arm_src['A'])
bad_a = [ln for ln in diff_a if ln not in allowed_a]
chk(not bad_a, f"A' vs accepted A: all {len(diff_a)} changed lines are value-block or MARGIN lines (strays: {bad_a[:3]})")

tob, _, sob = load(OLD_B)
chk(md5t(tob) == OLD_B_MD5, 'accepted B pack on disk is still fbe9beff (not overwritten)')
if B_REBUILT:
    diff_b = changed_lines(sob['cnet2_search'], arm_src['B'])
    bad_b = [ln for ln in diff_b if 'MARGIN=' not in ln]
    chk(not bad_b and len(diff_b) == 2,
        f"B' vs accepted B: exactly the MARGIN line changed ({len(diff_b)} lines, both MARGIN lines; strays: {bad_b[:3]})")
    chk(skeleton(tb) == skeleton(tob) and payloads(tb) == payloads(tob),
        "B' vs accepted B: skeleton + payloads identical outside _SRC")
else:
    chk(ARMS['B'] == OLD_B and md5t(tb) == OLD_B_MD5,
        "MARGIN' == 625.5: B arm is the accepted fbe9beff pack, byte-untouched")

print('\n%d/%d assertions passed' % (sum(1 for c, _ in ok if c), len(ok)))
print(json.dumps({'A_prime_pack_md5': md5t(ta),
                  'B_pack_md5': md5t(tb),
                  'B_rebuilt': B_REBUILT,
                  'margin_prime': MP}))
sys.exit(0 if all(c for c, _ in ok) else 1)
