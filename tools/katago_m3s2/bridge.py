"""Value-scale bridge for the KataGo warm-start line (stage 2+).

Strategy ruling: reports/RULING_KATAGO_VALUE_SCALE_BRIDGE_20261119.md.

Two scales of the same final cash margin d are in play:

  * U(d) = 0.6*tanh(d/600) + 0.4*clip(d/2500, -1, 1), range (-1, 1) --
    the plan's Score Utility (train.py's score_utility).  The trained
    value head outputs this scale.
  * _terminal(m) = clip(m/3000, -2.5, 2.5) + 0.5*sign(m), range [-3, 3] --
    the chassis' native critic scale (master3s2.py / v1 / fix2 / fix3,
    identical text in all four).  The chassis' search arithmetic (backup, fusion,
    veto thresholds) was tuned on this scale, and terminal boards are
    scored with it directly.

The bridge is the exact composition

    B(u) = _terminal(U^{-1}(u)),

applied ONLY where the value head's output enters search leaf
evaluation: pack_submission.py injects a self-contained twin of this
module's inversion into the packed chassis' look(), and vec_selfplay.py
applies bridge() to deep-plan leaf values when the acting checkpoint is
a trained (U-scale) one.  Gen-0 / native checkpoints are never bridged.
No fitted or approximate bridge is permitted (ruling section 2.1): U is
a deterministic, strictly increasing function of the margin, so the
inverse is computed exactly, through the margin domain:

  * No elementary closed form exists on the interior: on |d| <= 2500,
    U mixes tanh(d/600) with a linear term, which is transcendental in
    d.  The inverse is therefore computed by bisection on the strictly
    increasing U (robust everywhere, including the derivative kink at
    |d| = 2500 and the near-flat tail), polished by guarded Newton steps
    (a step is accepted only if it stays inside the bisection bracket
    and reduces |U(d) - u|).
  * Saturation IS closed-form: the recovered margin is clamped to
    +/-3000 (the registered clamp, ruling section 2.2).  A u at or
    beyond U(+/-3000) = +/-(0.6*tanh(5) + 0.4) ~= +/-0.99994552 has its
    pre-image at or beyond the clamp and saturates: B = _terminal(+/-
    3000) = +/-1.5 exactly.  Nothing is extrapolated past the clamp.
    Inputs outside [-1, 1] (a regression head can overshoot slightly)
    are clipped to [-1, 1] first and likewise saturate.
  * The zero jump is preserved exactly, not smoothed: _terminal jumps
    by +/-0.5 at m = 0, and B inherits the jump through the exact
    composition -- B(0) = 0 exactly (u == 0 is special-cased to the
    exact pre-image d = 0), while u -> 0+/- gives B -> +/-0.5.  Because
    the inversion is solved per point (no sampled table at any
    resolution), the ruling's zero-density requirement (section 2.3) is
    met by construction.

Verification: verify_bridge.py reconciles B(U(d)) against _terminal(d)
on a dense margin grid (extra-dense near 0), per ruling section 3.1.
"""

import numpy as np

MARGIN_CLAMP = 3000.0          # registered clamp on the recovered margin
_U_AT_CLAMP = 0.6 * float(np.tanh(5.0)) + 0.4   # U(3000) ~= 0.99994552


def score_utility(d):
    """U(d) = 0.6*tanh(d/600) + 0.4*clip(d/2500, -1, 1), elementwise."""
    d = np.asarray(d, dtype=np.float64)
    return 0.6 * np.tanh(d / 600.0) + 0.4 * np.clip(d / 2500.0, -1.0, 1.0)


def terminal_value(m):
    """_terminal(m) = clip(m/3000, -2.5, 2.5) + 0.5*sign(m), elementwise.

    Vectorized twin of the chassis' scalar _terminal (identical formula).
    """
    m = np.asarray(m, dtype=np.float64)
    return np.clip(m / 3000.0, -2.5, 2.5) + 0.5 * np.sign(m)


def _u_prime(d):
    """U'(d); the clip term's derivative is 0.4/2500 for |d| < 2500, else 0."""
    t = np.tanh(d / 600.0)
    return 0.001 * (1.0 - t * t) + np.where(np.abs(d) < 2500.0, 0.00016, 0.0)


def _invert_core(u, xp):
    """Exact U^{-1} on the clamped margin domain, shared numpy/torch core.

    u: an xp float64 array, already clipped to [-1, 1].  Returns the
    recovered margin, saturated to +/-MARGIN_CLAMP.  Only xp operations
    that exist under both numpy and torch are used (tanh / where / sign /
    clip-or-clamp are resolved by the callers into this module's helpers
    below via the xp shim functions _clip/_abs).
    """
    clip = xp.clip if hasattr(xp, 'clip') else xp.clamp
    lo = xp.full_like(u, -MARGIN_CLAMP)
    hi = xp.full_like(u, MARGIN_CLAMP)
    # Bisection: U is strictly increasing, so U(mid) < u keeps the root
    # bracketed; 60 iterations shrink the 6000-wide bracket below float64
    # resolution of the margin everywhere the slope is non-negligible.
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        um = 0.6 * xp.tanh(mid / 600.0) + 0.4 * clip(mid / 2500.0, -1.0, 1.0)
        right = um < u
        lo = xp.where(right, mid, lo)
        hi = xp.where(right, hi, mid)
    d = 0.5 * (lo + hi)
    # Guarded Newton polish (interior only; saturated entries are pinned
    # to the clamp and skipped, the flat tail makes Newton steps there
    # meaningless).  A step is accepted only inside the bracket and only
    # if it reduces the residual.
    for _ in range(3):
        t = xp.tanh(d / 600.0)
        ud = 0.6 * t + 0.4 * clip(d / 2500.0, -1.0, 1.0)
        res = ud - u
        der = 0.001 * (1.0 - t * t) + xp.where(
            xp.abs(d) < 2500.0, xp.full_like(d, 0.00016), xp.zeros_like(d))
        step = xp.where(der > 0.0, res / xp.where(der > 0.0, der, xp.ones_like(der)),
                        xp.zeros_like(d))
        cand = d - step
        tc = xp.tanh(cand / 600.0)
        uc = 0.6 * tc + 0.4 * clip(cand / 2500.0, -1.0, 1.0)
        ok = (cand >= lo) & (cand <= hi) & (xp.abs(uc - u) <= xp.abs(res))
        d = xp.where(ok, cand, d)
    # Saturation and the exact zero (see module docstring).
    d = xp.where(u >= _U_AT_CLAMP, xp.full_like(d, MARGIN_CLAMP), d)
    d = xp.where(u <= -_U_AT_CLAMP, xp.full_like(d, -MARGIN_CLAMP), d)
    d = xp.where(u == 0.0, xp.zeros_like(d), d)
    return d


def margin_from_u(u):
    """U^{-1}(u) with the recovered margin clamped to +/-3000.

    Accepts scalars or numpy arrays; u is clipped to [-1, 1] first, and
    |u| >= U(3000) saturates to exactly +/-3000 (no extrapolation).
    """
    a = np.asarray(u, dtype=np.float64)
    scalar = a.ndim == 0
    d = _invert_core(np.clip(np.atleast_1d(a), -1.0, 1.0), np)
    return float(d[0]) if scalar else d


def bridge(u):
    """B(u) = _terminal(U^{-1}(u)): U-scale value -> chassis _terminal scale."""
    a = np.asarray(u, dtype=np.float64)
    scalar = a.ndim == 0
    d = _invert_core(np.clip(np.atleast_1d(a), -1.0, 1.0), np)
    b = np.clip(d / 3000.0, -2.5, 2.5) + 0.5 * np.sign(d)
    return float(b[0]) if scalar else b


def bridge_torch(u):
    """Torch path of bridge(), sharing _invert_core with the numpy path.

    u: a torch tensor (any device, any float dtype); computed in float64
    and cast back to the input dtype.  Leaf batches are tiny, so the
    float64 bisection cost is negligible next to the net forward.
    """
    import torch
    a = u.to(torch.float64)
    d = _invert_core(torch.clamp(a, -1.0, 1.0), torch)
    b = torch.clamp(d / 3000.0, -2.5, 2.5) + 0.5 * torch.sign(d)
    return b.to(u.dtype)
