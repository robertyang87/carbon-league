#!/usr/bin/env python3
"""Build Part 3 aligned-substrate packs.

BASE (per strategy-group ruling 2026-11-15 13:22, superseding the earlier sims
base): deliverable A of Part 1 -- echo3_search_pack.py (ea20a271), the Echo3
pack with ONLY CONF SEARCH flipped to True, native wall-clock BUDGET=0.7, no
sims hooks. The sims pack (Part 1 deliverable B, 8e6f7a65) is permanently
closed: its calibration ladder found no lockable SIMS value.

Substrate corrections applied to BOTH arms (registered prerequisites,
PREREG_SEARCH.md S3):
  (a) terminal: _extinct/_terminal helpers ported from cnet2_search (R02) +
      step-limit clause; leaf() scores terminal states at true value (F01 for
      extinction, exact cash difference at the 300-step limit). The pack's
      embedded search predates R02 and had NO terminal branch at all.
  (b) MARGIN: 20.0 -> 625.5 in embedded DEFAULTS (PROVISIONAL proposal = Part
      2b affine fit residual SD 625.5417; pending strategy-group acceptance).
Arm difference (single variable): ONLY the non-terminal leaf evaluation.
  arm A: value leaf  M_eq = 3687.9177926610573 * z + (-638.5907028300794)
         (Part 2b registered affine pair; z = value-head pre-activation via
         the branch Feat history)
  arm B: original heuristic leaf body, untouched.
Both leaf call sites gain the branch Feat argument (fs[..][0]) -- wiring only,
identical in both arms. Everything else (CONF, payloads, other embedded
modules, pack skeleton) is byte-identical to the base A pack.
"""
import ast, hashlib, json, os

BASE_PACK = '/home/hatch/workspace/katago-probe2/modules/echo3_search_pack.py'
BASE_MD5 = 'ea20a2714209e1a58b3e435b41d943bb'
BASE_SEARCH_MD5 = 'adbe660db6a9342212f953d976fa5554'
OUT = {'A': '/home/hatch/workspace/katago-probe2/part3/modules/echo3_part3_a_pack.py',
       'B': '/home/hatch/workspace/katago-probe2/part3/modules/echo3_part3_b_pack.py'}
WORK = '/home/hatch/workspace/katago-probe2/part3/work'

OLD_MARGIN = 'HUNT=True, MARGIN=20.0, BUDGET=0.38'
NEW_MARGIN = 'HUNT=True, MARGIN=625.5, BUDGET=0.38'

HELPERS = '''    def _extinct(sim, p):
        """original env (carbon.py compute_next_board): a side with no workers, no trees and
        (no conversion centre, or cash below the cheapest recruit) is DONE and the game ends
        on the spot. FastCarbon deliberately does not model this (see its header), so the
        search layer must recognise the terminal itself (R02, ported from cnet2_search --
        this pack's embedded search predates R02 and had no terminal branch at all)."""
        if any(w.player == p for w in sim.workers.values()):
            return False
        if any(t.player == p for t in sim.trees.values()):
            return False
        if not sim.bases[p]:
            return True
        cheapest = min(sim.config.get("recCollectorCost", 30), sim.config.get("recPlanterCost", 30))
        return float(sim.cash[p]) < cheapest

    def _terminal(sim):
        # Part 3 construction prerequisite (a) (PREREG_SEARCH.md S3): the 300-step limit
        # is a terminal too -- a state at episodeSteps already has a final score; it is
        # not a position to be priced by a leaf estimate.
        return (sim.step_count >= sim.config.get("episodeSteps", 300)
                or _extinct(sim, 0) or _extinct(sim, 1))

'''

TERMINAL_BRANCH = '''        if _terminal(sim):
            # Terminal states score at true value, never by a leaf estimate
            # (prerequisite (a)). Extinction: F01 original scoring (carbon.py
            # compute_next_board) -- an extinct side scores (step - episodeSteps - 1);
            # a survivor keeps the cash the state carries (no caller in this pack
            # passes a pre-step cash, so the state's own cash is used, as in
            # cnet2_search for a state that was already terminal when handed over).
            # Step limit with both sides alive: the exact final cash difference
            # (verified on recorded games: 300-step finals equal the final cash
            # difference exactly). Extinction takes precedence where both apply.
            if _extinct(sim, 0) or _extinct(sim, 1):
                pen = float(sim.step_count) - float(sim.config.get("episodeSteps", 300)) - 1.0
                rew = [pen if _extinct(sim, p) else float(sim.cash[p]) for p in (0, 1)]
                return rew[me] - rew[1 - me]
            return float(sim.cash[me]) - float(sim.cash[1 - me])
'''

VALUE_BLOCK = '''        # Part 3 arm A: value leaf -- the ONLY difference between the arms. A
        # non-terminal leaf is priced by the subject net's own value head:
        # M_eq = A_AFF * z + B_AFF, z the value-head pre-activation on this leaf
        # state (branch Feat history continued on a private copy -- the leaf must
        # not advance branch state). Affine pair registered in Part 2b (fit on
        # the Part 2 held-out set; evidence fit_affine_results.json md5
        # bccbf8ed28dc77d0cafe25cd15dc83b2): A_AFF = 3687.9177926610573,
        # B_AFF = -638.5907028300794. DEFAULTS MARGIN = 625.5 is PROVISIONAL
        # (proposal: the Part 2b fit residual SD 625.5417), pending strategy-group
        # acceptance -- not a registered value yet. The heuristic body below is
        # the arm-B leaf, unreachable in this arm.
        if fm is None:
            raise RuntimeError("value leaf needs the branch Feat (fm)")
        fc_ = feat_copy(fm)
        xm_ = fc_.observe(sim.obs(me))
        ctx_ = encode_b(pol, stack_x([xm_]), np.array([st.get("T_now", T)]))[1][0]
        z_ = float(ctx_ @ pol.w["value.weightT"][:, 0] + pol.w["value.bias"][0])
        return 3687.9177926610573 * z_ + (-638.5907028300794)
'''

OLD_LEAF_HEAD = '''    def leaf(sim, me, left):
        m = [float(sim.cash[0]), float(sim.cash[1])]
'''

OLD_SEARCH_CALL = '        sc = np.array([leaf(s_, me, lft) for s_ in sims]).reshape(len(models), nc)\n'
NEW_SEARCH_CALL = '        sc = np.array([leaf(s_, me, lft, fs[i][0]) for i, s_ in enumerate(sims)]).reshape(len(models), nc)\n'

OLD_MCTS_CALL = '''            for (c, i), s_ in zip(batch, sims):
                vals[c].append((i, leaf(s_, me, lft))); n[c] += 1
'''
NEW_MCTS_CALL = '''            for j, ((c, i), s_) in enumerate(zip(batch, sims)):
                vals[c].append((i, leaf(s_, me, lft, fs[j][0]))); n[c] += 1
'''


def rep1(s, old, new, tag):
    n = s.count(old)
    assert n == 1, f'anchor {tag}: count={n}'
    return s.replace(old, new)


def build(s_base, arm):
    s = s_base
    s = rep1(s, OLD_MARGIN, NEW_MARGIN, 'margin')
    leaf_head = ('    def leaf(sim, me, left, fm=None):\n'
                 + TERMINAL_BRANCH
                 + (VALUE_BLOCK if arm == 'A' else '')
                 + '        m = [float(sim.cash[0]), float(sim.cash[1])]\n')
    s = rep1(s, OLD_LEAF_HEAD, HELPERS + leaf_head, 'leaf-head')
    s = rep1(s, OLD_SEARCH_CALL, NEW_SEARCH_CALL, 'search-call')
    s = rep1(s, OLD_MCTS_CALL, NEW_MCTS_CALL, 'mcts-call')
    return s


def main():
    raw = open(BASE_PACK, 'rb').read()
    md5 = hashlib.md5(raw).hexdigest()
    assert md5 == BASE_MD5, f'base A pack md5 drift: {md5}'
    text = raw.decode('utf-8')
    lines = text.split('\n')
    src_idx = next(i for i, ln in enumerate(lines) if ln.startswith('_SRC = '))
    S = ast.literal_eval(lines[src_idx][len('_SRC = '):])
    s_base = S['cnet2_search']
    assert hashlib.md5(s_base.encode()).hexdigest() == BASE_SEARCH_MD5
    man = {'base_pack': BASE_PACK, 'base_pack_md5': md5,
           'base_ruling': 'strategy-group 2026-11-15 13:22: base = Part 1 deliverable A (wall clock); sims pack B permanently closed',
           'arms': {}}
    for arm in ('A', 'B'):
        s_new = build(s_base, arm)
        compile(s_new, 'cnet2_search.py', 'exec')
        S2 = dict(S); S2['cnet2_search'] = s_new
        out_lines = list(lines)
        out_lines[src_idx] = '_SRC = ' + repr(S2)
        out_text = '\n'.join(out_lines)
        compile(out_text, OUT[arm], 'exec')
        open(OUT[arm], 'w').write(out_text)
        wdir = os.path.join(WORK, arm.lower())
        os.makedirs(wdir, exist_ok=True)
        open(os.path.join(wdir, 'cnet2_search.py'), 'w').write(s_new)
        man['arms'][arm] = {'pack': OUT[arm],
                            'pack_md5': hashlib.md5(out_text.encode()).hexdigest(),
                            'pack_bytes': len(out_text.encode()),
                            'search_md5': hashlib.md5(s_new.encode()).hexdigest(),
                            'search_bytes': len(s_new.encode())}
        print(arm, man['arms'][arm])
    open(os.path.join(WORK, 'build_manifest.json'), 'w').write(json.dumps(man, indent=1))
    print('manifest ->', os.path.join(WORK, 'build_manifest.json'))


if __name__ == '__main__':
    main()
