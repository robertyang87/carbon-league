"""One map-balanced policy-gradient update on sampled, whole-team commands."""
from collections import Counter
import math

import numpy as np


LIMITS = dict(draws_per_root=8, max_l2_step=0.5, max_halvings=6, max_root_kl=0.02,
              ratio_low=0.8, ratio_high=1.2, min_surrogate_gain=1e-10)


def _support(distribution, root, delta):
    """Evaluate the entire categorical distribution, including execution aliases."""
    values = distribution(root, delta)
    if not values:
        raise ValueError('empty executed-command distribution')
    out = {}
    for key, (p, grad) in values.items():
        p = float(p)
        g = np.asarray(grad, dtype=np.float64)
        if not isinstance(key, str) or not math.isfinite(p) or p < 0 or p > 1 + 1e-12:
            raise ValueError('invalid executed-command probability')
        if g.shape != (5,) or not np.isfinite(g).all():
            raise ValueError('invalid gradient of command probability')
        if p > 0:
            out[key] = (p, g)
    if abs(math.fsum(p for p, _ in out.values()) - 1.) > 2e-6:
        raise ValueError('incomplete executed-command support')
    return out


def _prepare(roots, distribution):
    if not roots:
        raise ValueError('no training roots')
    maps = Counter()
    root_ids = set()
    seed_terrain = {}
    prepared = []
    zero = np.zeros(5, dtype=np.float32)
    for root in roots:
        terrain = str(root.get('terrain') or root.get('terrain_sha256') or root.get('map') or root['seed'])
        seed = int(root['seed'])
        seat = int(root['seat'])
        t = int(root['t'])
        if seat not in (0, 1) or t < 0 or seed_terrain.setdefault(seed, terrain) != terrain:
            raise ValueError('invalid physical root identity')
        ident = (terrain, seat, t)
        if ident in root_ids:
            raise ValueError('duplicate map/seat/step')
        root_ids.add(ident)
        maps[terrain] += 1
        q = _support(distribution, root, zero)
        s = root['samples']
        if len(s) != LIMITS['draws_per_root']:
            raise ValueError('expected exactly eight independent with-replacement draws per root')
        keys = [v['key'] for v in s]
        util = np.array([v['u'] for v in s], dtype=np.float64)
        if not np.isfinite(util).all() or np.any((util < 0) | (util > 1)):
            raise ValueError('official U outside [0,1]')
        for v in s:
            p = float(v['old_p'])
            if v['key'] not in q or not math.isfinite(p) or p <= 0 or not math.isclose(p, q[v['key']][0], rel_tol=1e-5, abs_tol=1e-12):
                raise ValueError('sample propensity disagrees with frozen executable distribution')
        # The other seven independently drawn commands form the baseline.
        # Identical realised actions/outcomes retain their full multiplicity.
        a = (len(s) * util - math.fsum(util)) / (len(s)-1)
        prepared.append({'root':root, 'terrain':terrain, 'id':{'seed':seed, 'terrain':terrain, 'seat':seat, 't':t},
                         'old':q, 'keys':keys, 'adv':a, 'u':util})
    for row in prepared:
        row['w'] = 1./(len(maps)*maps[row['terrain']])
    return prepared, maps


def train(roots, distribution):
    """Return JSON-safe frozen-step results and a five-element float32-compatible delta.

    distribution(root, delta) must return all executed team command keys mapped
    to (probability, gradient of probability) for the actual FP32 addition.
    Only the direction at the original actor uses derivatives. Each trial is
    re-decoded in full, including proposals that change the greedy prefix.
    """
    rows, maps = _prepare(list(roots), distribution)
    gradient = np.zeros(5)
    signs = Counter()
    baseline = []
    for r in rows:
        old = r['old']
        score = np.stack([old[key][1]/old[key][0] for key in r['keys']])
        gradient += r['w'] * (r['adv'][:, None] * score).mean(0)
        positive = int((r['adv'] > 1e-14).sum())
        negative = int((r['adv'] < -1e-14).sum())
        signs.update(positive=positive, negative=negative, zero=len(r['keys'])-positive-negative)
        baseline.append(dict(r['id'], map_root_weight=r['w'], mean_observed_u=float(r['u'].mean()),
                             draws=[dict(key=k,old_p=old[k][0],u=float(v),advantage=float(a))
                                    for k,v,a in zip(r['keys'],r['u'],r['adv'])],
                             old_support={k:p for k,(p,_) in sorted(old.items())}))
    norm = float(np.linalg.norm(gradient))
    if not np.isfinite(norm):
        raise ValueError('nonfinite map-balanced policy gradient')
    base = dict(settings=LIMITS.copy(), map_count=len(maps), roots=len(rows), samples=sum(len(r['keys']) for r in rows),
                gradient=gradient.tolist(), gradient_l2=norm, advantage_signs=dict(signs), baseline=baseline,
                trials=[], delta=[0.]*5, decision='KEEP_ZERO')
    if norm <= 1e-14:
        base['reason'] = 'zero_map_balanced_gradient'
        return base
    for h in range(LIMITS['max_halvings']+1):
        delta = np.asarray((LIMITS['max_l2_step'] * 2.**(-h)) * gradient/norm, dtype=np.float32)
        records = []
        total_gain = 0.
        weighted_clip = 0.
        any_missing = False
        peak_kl = 0.
        for r in rows:
            old = r['old']
            new = _support(distribution, r['root'], delta)
            ratios = {k:(new[k][0]/v[0] if k in new else 0.) for k,v in old.items()}
            missing = sorted(k for k in old if k not in new)
            any_missing |= bool(missing)
            kl = None if missing else math.fsum(p*(math.log(p)-math.log(new[k][0])) for k,(p,_) in old.items())
            if kl is not None:
                if kl < -3e-6:
                    raise ValueError('inconsistent complete support KL')
                peak_kl = max(peak_kl, kl)
            rho = np.array([ratios[k] for k in r['keys']])
            clip_rho = np.clip(rho, LIMITS['ratio_low'], LIMITS['ratio_high'])
            clipped = float(np.mean(rho != clip_rho))
            adv = r['adv']
            gain = float(np.mean(np.minimum(rho*adv,clip_rho*adv)-adv))
            total_gain += r['w']*gain
            weighted_clip += r['w']*clipped
            records.append(dict(r['id'], kl_old_new=kl, missing_old_keys=missing,
                                new_only_mass=math.fsum(p for k,(p,_) in new.items() if k not in old),
                                new_support={k:p for k,(p,_) in sorted(new.items())},
                                all_old_support_ratio_min=min(ratios.values()), all_old_support_ratio_max=max(ratios.values()),
                                sample_ratios=rho.tolist(), sample_clip_rate=clipped, ppo_gain=gain))
        result=dict(halvings=h, delta=delta.tolist(), l2=float(np.linalg.norm(delta.astype(np.float64))),
                    max_root_kl=None if any_missing else peak_kl, any_old_support_lost=any_missing,
                    map_weighted_ppo_gain=total_gain, map_weighted_sample_clip_rate=weighted_clip,
                    sample_ratio_min=min(min(r['sample_ratios']) for r in records),
                    sample_ratio_max=max(max(r['sample_ratios']) for r in records), roots=records)
        reasons=[]
        if any_missing:
            reasons.append('old_positive_probability_lost')
        if peak_kl > LIMITS['max_root_kl']:
            reasons.append('root_kl_exceeds_limit')
        if total_gain <= LIMITS['min_surrogate_gain']:
            reasons.append('insufficient_sample_ppo_gain')
        result['rejections']=reasons
        base['trials'].append(result)
        if not reasons:
            base.update(delta=delta.tolist(),decision='UPDATE',reason='first_predeclared_step_passing_train_trust_gate')
            return base
    base['reason']='no_predeclared_step_passed'
    return base
