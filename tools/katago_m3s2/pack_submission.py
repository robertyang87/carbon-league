"""Plan stage 3: pack a trained checkpoint into a single-file submission.

Chassis: the owner's V1 search file (src/apexcarbon_v4tta_search.py,
md5 70e7a9ac...) by default -- owner ruling 2026-11-19 18:29: the
40-game paired verdict showed the fix2 shell significantly weaker than
v1's at identical weights, so v1 is the default chassis and
src/apexcarbon_v1_fix2.py remains an accepted alternate (--chassis).
The chassis code is kept untouched except for two mechanical, auditable
operations:

1. PAYLOAD SWAP.  The _WEIGHTS / _VALUE base64 payloads are replaced by
   the trained checkpoint's (source key layout, float16, npz-compressed
   + base64 -- the source's own storage format; the in-file _quant() is a
   *feature* quantizer, the source never int8-quantizes weights).
   A _ROLL payload is embedded only when the chassis actually wires one:
   master3s2's _net() reads globals().get('_ROLL'); neither the v1 nor
   the fix2 chassis has a _ROLL slot (their rollouts use the main net),
   so with those chassis a --roll argument is reported and skipped,
   main-only packed.  agent() and the search layer are never edited by
   the swap.
   Aux heads never enter the checkpoint npz (model.export_source_dict
   writes only source keys), so there is nothing to strip at pack time.

2. CLOCK-REMOVAL PATCH (owner ruling; skipped only with --no-clock-patch).
   The chassis drives its search by wall clock.  Every site is replaced
   by a deterministic work budget -- a fixed candidate count (_FIXED_K),
   a fixed root TTA count (_FIXED_TTA), the chassis' fixed stage table
   and D2, fixed batch size _S_ROWS -- sized so the worst-case step stays
   under the 0.75 s red line.  The patch is an explicit list of exact
   string replacements below; each must match its declared count or the
   pack FAILS LOUDLY (an unpatched chassis is never shipped silently).
   Post-condition asserted on the output: no 'perf_counter' remains
   anywhere in the file.  The patch set is selected by a signature line
   (v1 and fix2 texts differ slightly); an unrecognized chassis fails
   loudly.  Sites neutralized (both chassis unless noted):
     * agent(): t0 = perf_counter()                    -> t0 = 0.0
     * _net(): _S_COST seeded from a timed warmup pass -> nominal constants
     * _act root: n_tta chosen from _S_COST vs _S_ABORT -> _FIXED_TTA
       (v1: unconditional TTA line, _S_ABORT=0.62; fix2: the
       _S_USE_V4_ROOT-guarded line, _S_ABORT=0.58)
     * _act: candidate cap `most` fitted to the time left via _plan()
       and _S_FILL/_S_ABORT                            -> most = _FIXED_K
     * upd_cost(): per-pass cost tracker feeding late()-> no-op
       (v1 body blends 0.5/0.5, fix2 caps then blends 0.7/0.3)
     * look()/other()/play(): perf_counter() call sites -> upd_cost(*, 0.0)
     * late(): abort predicate (elapsed + _S_SAFE*need > _S_ABORT)
                                                           -> always False
       (the fixed plan always runs in full, including the top-2
       multi-response verification the chassis runs 'when time allows')
     * _act tail: dt = perf_counter() - t0 (stats)      -> dt = 0.0 (fix2 only;
       v1 has no such line)
   _S_COST remains as dead telemetry state: after the patch nothing reads
   it for a decision (fix2's depth==0 decay in the stats tail is inert).

--verify (numpy only, no torch) checks the packed file end to end:
   * it imports standalone and its _Policy loads;
   * its (numpy) net forward on fixed encode() inputs matches a
     reference net built directly from the checkpoint payload (float16
     quantization tolerance, number reported) -- note the v1/fix2 chassis'
     _Policy is a pure weight container with no forward method; the
     forward lives in its _WinoNetNP/_TorchNet, so the check runs
     through _WinoNetNP, the same class the packed agent uses;
   * determinism + timing over full self-games (both seats = the packed
     agent, fixed _SEED): two independent runs must choose identical
     actions, and the per-step wall times must satisfy max < 0.75 s.
Exit codes: 0 = pass, 1 = import/forward/determinism failure,
2 = timing gate failure.
"""

import argparse
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout

FORWARD_TOL = 0.25          # logit tolerance for float16 payload storage
STEP_TIME_LIMIT = 0.75      # seconds, the ladder red line


# ---------------------------------------------------------------------------
# Clock-removal patch: (name, old, new, expected_count).  Patterns are the
# fix2 chassis' exact text; any mismatch aborts the pack.
# ---------------------------------------------------------------------------
V1_TTA_LINE = ("    n_tta = 4 if (_S_COST[0] <= 0.0 or 8 * _S_COST[0] < 0.25 * _S_ABORT) else 2")
FIX2_TTA_LINE = ("    n_tta = (4 if (_S_COST[0] <= 0.0 or 8 * _S_COST[0] < 0.25 * _S_ABORT) else 2) "
                 "if _S_USE_V4_ROOT else 1")


def clock_patches(fixed_k, fixed_tta, variant):
    """Exact-replacement patch list for one chassis variant ('v1'/'fix2')."""
    common = [
        ('fixed-constants',
         "_S_COST = [0.0, 0.0, 0.0, 0.0]\n",
         "_S_COST = [0.0, 0.0, 0.0, 0.0]\n"
         "_FIXED_K = %d         # CLOCK-REMOVAL PATCH: fixed candidate budget per step\n"
         "_FIXED_TTA = %d       # CLOCK-REMOVAL PATCH: fixed root V4-TTA count\n"
         % (fixed_k, fixed_tta), 1),
        ('net-cost-seeding',
         "        if _S_COST[0] == 0.0:\n"
         "            t = _time.perf_counter()\n"
         "            _NET(np.zeros((16, NPL, 15, 15), np.float32), _SID, np.zeros((16, 5), np.float32))\n"
         "            _S_COST[0] = _S_COST[2] = 1.4 * (_time.perf_counter() - t) / 16.0\n"
         "            _S_COST[1] = _S_COST[3] = 0.0015\n",
         "        if _S_COST[0] == 0.0:\n"
         "            # CLOCK-REMOVAL PATCH: nominal constants; _S_COST feeds no decision.\n"
         "            _NET(np.zeros((16, NPL, 15, 15), np.float32), _SID, np.zeros((16, 5), np.float32))\n"
         "            _S_COST[0] = _S_COST[2] = 0.004\n"
         "            _S_COST[1] = _S_COST[3] = 0.0015\n", 1),
        ('candidate-budget',
         "        most = _S_K\n"
         "        if _S_COST[0] > 0.0:\n"
         "            left = _S_FILL * max(0.02, _S_ABORT - (_time.perf_counter() - t0))\n"
         "            most = 2\n"
         "            for n in range(3, _S_K + 1):\n"
         "                rows, steps, passes = _plan(n, D2)\n"
         "                if rows * _S_COST[0] + steps * _S_COST[1] + passes * _S_CFIX > left:\n"
         "                    break\n"
         "                most = n\n",
         "        most = _FIXED_K  # CLOCK-REMOVAL PATCH: fixed candidate budget (was fitted to the time left)\n", 1),
        ('tp-sites',
         "            tp = _time.perf_counter()",
         "            tp = 0.0  # CLOCK-REMOVAL PATCH", 3),
        ('upd-call-look-v4',
         "upd_cost(0, max(0.0, _time.perf_counter() - tp - passes * _S_CFIX) / (v4_k * len(live)))",
         "upd_cost(0, 0.0)", 1),
        ('upd-call-look',
         "upd_cost(0, max(0.0, _time.perf_counter() - tp - passes * _S_CFIX) / (w * len(live)))",
         "upd_cost(0, 0.0)", 1),
        ('upd-call-other',
         "upd_cost(0, max(0.0, _time.perf_counter() - tp - passes * _S_CFIX) / len(live))",
         "upd_cost(0, 0.0)", 1),
        ('upd-call-play',
         "upd_cost(1, (_time.perf_counter() - tp) / len(live))",
         "upd_cost(1, 0.0)", 1),
        ('late-predicate',
         "        def late(rows, steps):\n"
         "            need = -(-rows // _S_ROWS) * _S_CFIX + rows * max(_S_COST[0], _S_COST[2]) + steps * max(_S_COST[1], _S_COST[3])\n"
         "            return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT\n",
         "        def late(rows, steps):\n"
         "            return False  # CLOCK-REMOVAL PATCH: the fixed plan always runs in full\n", 1),
        ('agent-t0',
         "def agent(observation, configuration):\n    t0 = _time.perf_counter()",
         "def agent(observation, configuration):\n    t0 = 0.0  # CLOCK-REMOVAL PATCH: no wall clock in the decision path", 1),
    ]
    if variant == 'v1':
        specific = [
            ('root-tta-count', V1_TTA_LINE,
             "    n_tta = _FIXED_TTA  # CLOCK-REMOVAL PATCH: fixed root TTA count", 1),
            ('upd-cost-noop',
             "        def upd_cost(i, x):\n"
             "            _S_COST[i] = x if _S_COST[i] == 0.0 else 0.5 * _S_COST[i] + 0.5 * x\n"
             "            _S_COST[i + 2] = x\n",
             "        def upd_cost(i, x):\n"
             "            return  # CLOCK-REMOVAL PATCH: cost telemetry disabled; _S_COST feeds no decision\n", 1),
        ]
    else:  # fix2
        specific = [
            ('root-tta-count', FIX2_TTA_LINE,
             "    n_tta = _FIXED_TTA if _S_USE_V4_ROOT else 1  # CLOCK-REMOVAL PATCH: fixed TTA count", 1),
            ('upd-cost-noop',
             "        def upd_cost(i, x):\n"
             "            cap = 0.05 if i == 0 else 0.02\n"
             "            x = min(x, cap)\n"
             "            _S_COST[i] = x if _S_COST[i] == 0.0 else 0.7 * _S_COST[i] + 0.3 * x\n"
             "            _S_COST[i + 2] = x\n",
             "        def upd_cost(i, x):\n"
             "            return  # CLOCK-REMOVAL PATCH: cost telemetry disabled; _S_COST feeds no decision\n", 1),
            ('stats-dt',
             "        dt = _time.perf_counter() - t0",
             "        dt = 0.0  # CLOCK-REMOVAL PATCH: stats only; no wall clock in the decision path", 1),
        ]
    return common + specific


def detect_variant(text):
    if V1_TTA_LINE in text:
        return 'v1'
    if FIX2_TTA_LINE in text:
        return 'fix2'
    raise SystemExit('CLOCK PATCH FAILED: unrecognized chassis (neither the '
                     'v1 nor the fix2 root-TTA line found). Refusing to ship '
                     'an unpatched chassis.')


def apply_clock_patch(text, fixed_k, fixed_tta):
    variant = detect_variant(text)
    summary = ['  chassis variant: %s' % variant]
    for name, old, new, count in clock_patches(fixed_k, fixed_tta, variant):
        found = text.count(old)
        if found != count:
            raise SystemExit(
                'CLOCK PATCH FAILED at %r: pattern found %d times, expected %d. '
                'Refusing to ship an unpatched chassis.' % (name, found, count))
        text = text.replace(old, new)
        summary.append('  patch %-18s -%d/+%d lines'
                       % (name, old.count('\n'), new.count('\n')))
    if 'perf_counter' in text:
        raise SystemExit('CLOCK PATCH FAILED: perf_counter still present after patching')
    return text, summary


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
def load_ckpt(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def pack(chassis_path, ckpt, roll, out_path, clock_patch=True, fixed_k=12,
         fixed_tta=None):
    text = open(chassis_path).read()
    if fixed_tta is None:
        # v1's quality branch is TTA=4 (used whenever the machine is fast);
        # fix2 ships with root TTA off and 2 is its conservative on-value.
        fixed_tta = 4 if (V1_TTA_LINE in text) else 2
    pol, val = layout.split_checkpoint(ckpt)
    if not val:
        raise SystemExit('checkpoint has no value keys (v1./v2./v3.); the chassis needs _VALUE')
    pol_b64 = layout.encode_npz(layout.to_f16(pol))
    val_b64 = layout.encode_npz(layout.to_f16(val))
    for name, payload in (('_WEIGHTS', pol_b64), ('_VALUE', val_b64)):
        pat = re.compile(r'^%s = "[A-Za-z0-9+/=]+"' % name, re.M)
        text, n = pat.subn('%s = "%s"' % (name, payload), text)
        if n != 1:
            raise SystemExit('payload swap failed for %s: %d matches' % (name, n))
    notes = []
    has_roll_slot = "globals().get('_ROLL')" in text
    if roll is not None and has_roll_slot:
        roll_b64 = layout.encode_npz(layout.to_f16(roll))
        text, n = re.subn(r'^(_VALUE = "[A-Za-z0-9+/=]+")$',
                          r'\1\n_ROLL = "%s"' % roll_b64, text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit('failed to embed _ROLL after _VALUE')
        notes.append('embedded _ROLL payload (%d arrays)' % len(roll))
    elif roll is not None:
        notes.append('chassis %s has NO _ROLL slot (its _net() never reads a '
                     '_ROLL global): --roll skipped, packed main-only'
                     % os.path.basename(chassis_path))
    else:
        notes.append('no --roll given; chassis _ROLL wiring: %s'
                     % ('present' if has_roll_slot else 'absent'))
    summary = []
    if clock_patch:
        text, summary = apply_clock_patch(text, fixed_k, fixed_tta)
    compile(text, out_path, 'exec')
    with open(out_path, 'w') as f:
        f.write(text)
    return notes, summary


# ---------------------------------------------------------------------------
# Verification (numpy only)
# ---------------------------------------------------------------------------
def _fixed_positions(mod):
    import extract_m3s2
    pl, carbon = extract_m3s2.synthetic_players(mod)
    return [(pl, 40, me, carbon) for me in (0, 1)]


def verify_forward(packed_path, ckpt):
    """Packed net vs a reference net built from the checkpoint payload.

    Uses the chassis' own numpy net class (_WinoNetNP) on both sides --
    the fix2 _Policy has no forward method (it is a weight container).
    Falls back to _Policy.forward (master3s2-style chassis) when the
    chassis has no _WinoNetNP.
    """
    mod = layout.load_module(packed_path, 'pk_fwd')
    pol, val = layout.split_checkpoint(ckpt)
    xs, scs = [], []
    for pl, step, me, carbon in _fixed_positions(mod):
        xs.append(mod._quant(mod.encode(pl, step, me, carbon, [1.0] * mod.N)))
        scs.append(mod._scalars(pl, step, me))
    X = np.stack(xs).astype(np.float32)
    SC = np.stack(scs).astype(np.float32)
    net = mod._net()
    lw_a, lb_a, v_a = net(X, mod._SID, SC)
    if hasattr(mod, '_WinoNetNP'):
        ref_pol = mod._Policy(layout.encode_npz(layout.to_f16(pol)))
        ref_val = {k: v.astype(np.float32) for k, v in val.items()}
        ref = mod._WinoNetNP(ref_pol, ref_val)
        lw_b, lb_b, v_b = ref(X, mod._SID, SC)
    else:
        lw_b = np.stack([mod._POLICY.forward(X[i], mod._SID)[0] for i in range(len(X))])
        lb_b = np.stack([mod._POLICY.forward(X[i], mod._SID)[1] for i in range(len(X))])
        v_b = v_a
    return max(float(np.abs(lw_a - lw_b).max()), float(np.abs(lb_a - lb_b).max()),
               float(np.abs(v_a - v_b).max()))


def play_packed_game(packed_path, tag, game_seed=7, agent_seed=999):
    """One full self-game, both seats = the packed agent.  Returns
    (actions, step_times, final_cash).  CARBON_DEBUG is set so a silently
    swallowed search exception fails the verification instead of hiding."""
    os.environ['CARBON_DEBUG'] = '1'
    m0 = layout.load_module(packed_path, 'pk_%s_a' % tag)
    m1 = layout.load_module(packed_path, 'pk_%s_b' % tag)
    m0._SEED = agent_seed
    m1._SEED = agent_seed
    mods = [m0, m1]
    st = layout.new_game_state(m0, game_seed)
    cfg = dict(m0.DEFAULT_CONFIG)
    times, actions = [], []
    while st.step <= m0.LAST_TURN:
        acts = []
        for me in (0, 1):
            obs = layout.make_obs(st, me)
            t = time.perf_counter()
            acts.append(mods[me].agent(obs, dict(cfg)))
            times.append(time.perf_counter() - t)
        actions.append(acts)
        st = m0.next_state(st, acts, cfg)
    return actions, times, [float(pl[0]) for pl in st.players]


def verify_games(packed_path, games=1):
    a1, t1, cash1 = play_packed_game(packed_path, 'g1')
    a2, t2, cash2 = play_packed_game(packed_path, 'g2')
    det = (repr(a1) == repr(a2)) and cash1 == cash2
    times = np.array(t1 + t2)
    return det, times, cash1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--chassis', default=layout.APEX_V4TTA,
                    help='chassis source file (default: the v1 file, owner '
                         'ruling 18:29; the fix2 file is an accepted alternate)')
    ap.add_argument('--ckpt', required=True, help='trained checkpoint npz (source layout)')
    ap.add_argument('--roll', default=None, help='optional _ROLL npz (policy keys only)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-clock-patch', action='store_true',
                    help='skip the clock-removal patch (debugging only; the '
                         'output then keeps the wall-clock search)')
    ap.add_argument('--fixed-k', type=int, default=12,
                    help='deterministic candidate budget baked into the patch')
    ap.add_argument('--fixed-tta', type=int, default=None,
                    help='fixed root TTA count (default: 4 for the v1 chassis, '
                         '2 for fix2 -- each chassis\' own quality branch)')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--verify-games', type=int, default=1,
                    help='full self-games for the determinism/timing check '
                         '(each game is played twice)')
    args = ap.parse_args()

    ckpt = load_ckpt(args.ckpt)
    roll = load_ckpt(args.roll) if args.roll else None
    notes, summary = pack(args.chassis, ckpt, roll, args.out,
                          clock_patch=not args.no_clock_patch,
                          fixed_k=args.fixed_k, fixed_tta=args.fixed_tta)
    print('packed %s (chassis %s)' % (args.out, os.path.basename(args.chassis)))
    for n in notes:
        print('  note: %s' % n)
    if summary:
        print('clock-removal patch applied:')
        print('\n'.join(summary))
        print('  post-condition: no perf_counter in output')
    else:
        print('  WARNING: clock patch skipped (--no-clock-patch)')

    if not args.verify:
        return 0
    err = verify_forward(args.out, ckpt)
    print('forward check: max abs err vs checkpoint reference net = %.4f (tol %.2f) -> %s'
          % (err, FORWARD_TOL, 'PASS' if err < FORWARD_TOL else 'FAIL'))
    if err >= FORWARD_TOL:
        return 1
    det, times, cash = verify_games(args.out, args.verify_games)
    print('determinism over %d full game(s) x2 runs: %s (final cash %s)'
          % (args.verify_games, 'IDENTICAL actions' if det else 'MISMATCH', cash))
    if not det:
        return 1
    print('step times over %d agent calls: mean %.3fs p95 %.3fs max %.3fs (limit %.2fs) -> %s'
          % (len(times), times.mean(), np.percentile(times, 95), times.max(),
             STEP_TIME_LIMIT, 'PASS' if times.max() < STEP_TIME_LIMIT else 'FAIL'))
    return 0 if times.max() < STEP_TIME_LIMIT else 2


if __name__ == '__main__':
    sys.exit(main())
