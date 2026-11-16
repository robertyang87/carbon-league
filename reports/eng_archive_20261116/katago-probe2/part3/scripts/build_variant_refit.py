#!/usr/bin/env python3
"""Part 3 refit step 4 (strategy-group ruling 2026-11-15 13:34, sec.3
disposition (iii), as amended by the supplementary ruling on MARGIN):
rebuild the Part 3 arms on base A (echo3_search_pack.py, ea20a271)
replaying build_variant.py's registered edit set verbatim (terminal
corrections + leaf wiring, imported from build_variant; build()
below mirrors BV.build line for line), with only the registered
literals changed:

  * arm A's value leaf carries the PACK-space affine pair (a', b')
    from fit_pack_results.json (fit on the deployed pack's own z,
    through the pack's embedded feat/np + decoded int8 weights);
  * DEFAULTS MARGIN = MARGIN' = 1.0 x pack-space residual SD (0.1
    rounding) in BOTH arms -- MARGIN is a shared mechanism parameter
    (the registered design's single variable is the leaf only); the
    supplementary ruling forbids the two-variable form (A' at MARGIN',
    B at 625.5). Arm B's leaf block stays byte-identical; if MARGIN'
    == 625.5 exactly, arm B is naturally untouched and no B v2 file
    is produced.

Outputs: modules/echo3_part3_a_pack_v2.py (+ modules/
echo3_part3_b_pack_v2.py iff MARGIN' != 625.5); the accepted
d8d7ebc8 / fbe9beff files are never overwritten. Per-arm extracted
search sources + manifest under work/refit/.
"""
import ast
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_variant as BV  # noqa: E402  (registered edit set, reused)

BASE = "/home/hatch/workspace/katago-probe2"
BASE_PACK = BV.BASE_PACK
BASE_PACK_MD5 = BV.BASE_MD5
FIT = os.path.join(BASE, "part3", "data", "fit_pack_results.json")
MOD_DIR = os.path.join(BASE, "part3", "modules")
WORK = os.path.join(BASE, "part3", "work", "refit")
OUT = {"A": os.path.join(MOD_DIR, "echo3_part3_a_pack_v2.py"),
       "B": os.path.join(MOD_DIR, "echo3_part3_b_pack_v2.py")}
OLD_MARGIN_VALUE = 625.5


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fit():
    fit = json.load(open(FIT))
    fit_md5 = md5_file(FIT)
    pk = fit["pack_affine"]
    mp = fit["margin_prime"]
    a, b = pk["a"], pk["b"]
    sigma = mp["sigma_pack_unrounded"]
    margin = mp["value"]
    assert abs(pk["resid_std"] - sigma) < 1e-12
    assert margin == round(1.0 * sigma, 1), (margin, sigma)
    return a, b, sigma, margin, fit_md5


A_PRIME, B_PRIME, SIGMA_PACK, MARGIN_PRIME, FIT_MD5 = load_fit()

VALUE_BLOCK = f'''        # Part 3 arm A: value leaf -- the ONLY difference between the arms. A
        # non-terminal leaf is priced by the subject net's own value head:
        # M_eq = A_AFF * z + B_AFF, z the value-head pre-activation on this leaf
        # state (branch Feat history continued on a private copy -- the leaf must
        # not advance branch state). Affine pair REFIT IN PACK Z SPACE on the
        # same Part 2 held-out set (strategy-group ruling 2026-11-15 13:34
        # sec.3 (iii)); it supersedes the fp-space pair registered in Part 2b.
        # A_AFF = {A_PRIME!r}, B_AFF = {B_PRIME!r}; evidence
        # fit_pack_results.json md5 {FIT_MD5}. DEFAULTS MARGIN = {MARGIN_PRIME!r}
        # (1.0 x pack-space residual SD {SIGMA_PACK!r}) -- a shared mechanism
        # parameter carried identically by both arms. The heuristic body below
        # is the arm-B leaf, unreachable in this arm.
        if fm is None:
            raise RuntimeError("value leaf needs the branch Feat (fm)")
        fc_ = feat_copy(fm)
        xm_ = fc_.observe(sim.obs(me))
        ctx_ = encode_b(pol, stack_x([xm_]), np.array([st.get("T_now", T)]))[1][0]
        z_ = float(ctx_ @ pol.w["value.weightT"][:, 0] + pol.w["value.bias"][0])
        return {A_PRIME!r} * z_ + ({B_PRIME!r})
'''

NEW_MARGIN = f'HUNT=True, MARGIN={MARGIN_PRIME!r}, BUDGET=0.38'


def build(s_base, arm):
    s = s_base
    s = BV.rep1(s, BV.OLD_MARGIN, NEW_MARGIN, 'margin')
    leaf_head = ('    def leaf(sim, me, left, fm=None):\n'
                 + BV.TERMINAL_BRANCH
                 + (VALUE_BLOCK if arm == 'A' else '')
                 + '        m = [float(sim.cash[0]), float(sim.cash[1])]\n')
    s = BV.rep1(s, BV.OLD_LEAF_HEAD, BV.HELPERS + leaf_head, 'leaf-head')
    s = BV.rep1(s, BV.OLD_SEARCH_CALL, BV.NEW_SEARCH_CALL, 'search-call')
    s = BV.rep1(s, BV.OLD_MCTS_CALL, BV.NEW_MCTS_CALL, 'mcts-call')
    return s


def main():
    raw = open(BASE_PACK, 'rb').read()
    md5 = hashlib.md5(raw).hexdigest()
    assert md5 == BASE_PACK_MD5, f'base A pack md5 drift: {md5}'
    text = raw.decode('utf-8')
    lines = text.split('\n')
    src_idx = next(i for i, ln in enumerate(lines) if ln.startswith('_SRC = '))
    S = ast.literal_eval(lines[src_idx][len('_SRC = '):])
    s_base = S['cnet2_search']
    assert hashlib.md5(s_base.encode()).hexdigest() == BV.BASE_SEARCH_MD5
    man = {"base_pack": BASE_PACK, "base_pack_md5": md5,
           "fit": FIT, "fit_md5": FIT_MD5,
           "a_prime": A_PRIME, "b_prime": B_PRIME,
           "sigma_pack": SIGMA_PACK, "margin_prime": MARGIN_PRIME,
           "arms": {}}
    arms = ["A"] + (["B"] if MARGIN_PRIME != OLD_MARGIN_VALUE else [])
    if MARGIN_PRIME == OLD_MARGIN_VALUE:
        man["b_arm"] = ("untouched: MARGIN' == 625.5 exactly; the "
                        "accepted fbe9beff pack stands as arm B")
    for arm in arms:
        s_new = build(s_base, arm)
        compile(s_new, 'cnet2_search.py', 'exec')
        S2 = dict(S)
        S2['cnet2_search'] = s_new
        out_lines = list(lines)
        out_lines[src_idx] = '_SRC = ' + repr(S2)
        out_text = '\n'.join(out_lines)
        compile(out_text, OUT[arm], 'exec')
        open(OUT[arm], 'w').write(out_text)
        wdir = os.path.join(WORK, arm)
        os.makedirs(wdir, exist_ok=True)
        open(os.path.join(wdir, 'cnet2_search.py'), 'w').write(s_new)
        man["arms"][arm] = {
            "pack": OUT[arm],
            "pack_md5": hashlib.md5(out_text.encode()).hexdigest(),
            "pack_bytes": len(out_text.encode()),
            "search_md5": hashlib.md5(s_new.encode()).hexdigest(),
            "search_bytes": len(s_new.encode())}
        print(arm, man["arms"][arm])
    mp = os.path.join(WORK, 'build_manifest_refit.json')
    open(mp, 'w').write(json.dumps(man, indent=1))
    print('manifest ->', mp)


if __name__ == "__main__":
    main()
