"""Unit alignment for the value-scale bridge (ruling section 3.1).

Reconciles B(U(d)) against _terminal(d) on a margin grid over
[-6000, 6000] (extra-dense near 0, where _terminal's +-0.5 jump lives):
outside the clamp zone (|d| <= 3000) the error must be < 1e-3; inside
the clamp zone the deviation is quantified separately (B saturates at
_terminal(+/-3000) = +/-1.5 by design).  Also checks monotonicity of B,
the saturation values, the exact zero, and the inversion residual
|U(U^-1(u)) - u|.  With --packed FILE, additionally reconciles the
bridge twin injected into a packed chassis against bridge.py.

numpy only.  Exit code 0 = all checks pass, 1 = a check failed.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import bridge


def margin_grid():
    base = np.linspace(-6000.0, 6000.0, 48001)
    near = np.concatenate([
        np.linspace(-2.0, 2.0, 8001),
        np.array([-1e-6, -1e-4, -1e-3, -0.01, -0.1, 0.0,
                  1e-6, 1e-4, 1e-3, 0.01, 0.1])])
    return np.unique(np.concatenate([base, near]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--packed', default=None,
                    help='optional packed chassis file: also reconcile its '
                         'injected _bridge_u_values against bridge.py')
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        print('%-34s %s  %s' % (name, 'PASS' if cond else 'FAIL', detail))
        ok = ok and bool(cond)

    d = margin_grid()
    u = bridge.score_utility(d)
    b = bridge.bridge(u)
    t = bridge.terminal_value(d)
    err = np.abs(b - t)

    inside = np.abs(d) <= 3000.0
    i = int(np.argmax(err[inside]))
    check('grid |d|<=3000: max |B(U(d))-T(d)|', err[inside].max() < 1e-3,
          'max %.3e at d=%.4f (grid %d pts, tol 1e-3)'
          % (err[inside].max(), d[inside][i], len(d)))

    zone = ~inside
    check('clamp zone 3000<|d|<=6000: B saturates',
          np.all(np.abs(b[zone]) == 1.5),
          'B == +/-1.5 on all %d zone pts; deviation vs _terminal: '
          'max %.4f, mean %.4f (expected: _terminal keeps rising to '
          '+/-2.5 at |d|=6000 while B is clamped)'
          % (zone.sum(), err[zone].max(), err[zone].mean()))

    check('monotone: B(U(d)) in d', np.all(np.diff(b) >= 0.0),
          'min diff %.3e over the sorted grid' % np.diff(b).min())
    ug = np.linspace(-1.0, 1.0, 20001)
    bg = bridge.bridge(ug)
    check('monotone: B(u) in u', np.all(np.diff(bg) >= 0.0),
          'min diff %.3e over %d u pts' % (np.diff(bg).min(), len(ug)))

    check('saturation values',
          bridge.bridge(1.0) == 1.5 and bridge.bridge(-1.0) == -1.5
          and bridge.bridge(0.9999999) == 1.5 and bridge.bridge(-0.9999999) == -1.5,
          'B(1)=%.4f B(-1)=%.4f B(0.9999999)=%.4f'
          % (bridge.bridge(1.0), bridge.bridge(-1.0), bridge.bridge(0.9999999)))
    check('zero: B(0) == 0 exactly, jump preserved',
          bridge.bridge(0.0) == 0.0
          and abs(bridge.bridge(bridge.score_utility(1e-6)) - 0.5) < 1e-3
          and abs(bridge.bridge(bridge.score_utility(-1e-6)) + 0.5) < 1e-3,
          'B(U(1e-6))=%.6f B(U(-1e-6))=%.6f'
          % (bridge.bridge(bridge.score_utility(1e-6)),
             bridge.bridge(bridge.score_utility(-1e-6))))

    uu = np.linspace(-1.0, 1.0, 20001)
    rec = bridge.margin_from_u(uu)
    res = np.abs(bridge.score_utility(rec) - uu)
    unsat = np.abs(uu) < bridge._U_AT_CLAMP
    check('inversion residual (unsaturated)', res[unsat].max() < 1e-9,
          'max |U(U^-1(u))-u| %.3e for |u| < U(3000)=%.8f; saturated band '
          'pins the margin to +/-3000 by design'
          % (res[unsat].max(), bridge._U_AT_CLAMP))

    if args.packed:
        import layout
        mod = layout.load_module(args.packed, 'pk_bridge_verify')
        if not hasattr(mod, '_bridge_u_values'):
            check('packed bridge present', False,
                  '%s has no _bridge_u_values (packed without --value-scale u?)'
                  % args.packed)
        else:
            bp = np.asarray(mod._bridge_u_values(u), dtype=np.float64)
            dp = np.abs(bp - b)
            check('packed twin vs bridge.py', dp.max() < 1e-6,
                  'max |packed B - bridge B| %.3e over the grid' % dp.max())
            up = np.asarray(mod._u_of_margin(d), dtype=np.float64)
            check('packed _u_of_margin vs U', np.abs(up - u).max() < 1e-12,
                  'max err %.3e' % np.abs(up - u).max())

    print('BRIDGE VERIFY: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
