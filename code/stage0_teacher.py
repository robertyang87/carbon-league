"""Stage 0 root-candidate teacher prototype (v1, numpy / cnet2_np).

Spec: CARBONFORMER_RESPONSE_20261113 section 2 + katago_architecture_design.md
section 2 (2026-11-13 revision):
  - K<=16 complete joint candidates (greedy + forced-prefix single-unit deviations)
  - M=8 belief scenarios, common random numbers across candidates
  - opponent root action generated BEFORE seeing our candidates
  - H=2 synchronous steps, then fixed bare-net leaf value
  - soft target q ∝ P_A^0.5 * exp((Q - maxQ) / 0.05), Q in U scale
  - quality-gate prototype outputs: target entropy, agreement rate with the
    bare-net argmax (greedy) arm

v1 simplifications (documented, cost-identical; full treatment is Stage 1):
  - scenario = opponent-action sample (temp 1.0) + continuation RNG stream.
    Full belief-state resampling (belief.py) is NOT wired in yet.
  - key units ranked by planter-cargo, then smallest top1-top2 logit gap
    (not cnet2_search.key_units); base slots included at lower priority.
  - branch Feats start fresh (history features zeroed inside the H=2
    continuation); the root position itself uses the live tracked Feat.
  - leaf value comes from the existing value head (tanh(M/3000)) mapped to
    U via M = 3000*atanh(clip(v, +-0.999)); terminal leaves use the cash
    difference as the margin approximation (extinction penalty not applied).

Q is in U units: U = 0.5*W + 0.5*sigmoid(M/1000), W in {0, 0.5, 1}.
"""

import math
import time

import numpy as np

import cnet2_feat as F
from fastsim import FastCarbon

NEG = -1e9


def value_to_u(v):
    """Existing value-head scalar v = tanh(M/3000) -> (U, M)."""
    m = 3000.0 * math.atanh(max(-0.999, min(0.999, float(v))))
    w = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
    u = 0.5 * w + 0.5 / (1.0 + math.exp(-m / 1000.0))
    return u, m


def joint_logp(logits, labels):
    """Sum of per-slot log-probs of a joint action under the decoder's
    teacher-forced logits (complete autoregressive joint logp, D02 form)."""
    lp = 0.0
    for s in range(F.OWN):
        a = int(labels[s])
        if a < 0:
            continue
        lg = logits[s].astype(np.float64)
        z = lg - lg.max()
        p = np.exp(z)
        p = p / p.sum()
        lp += math.log(max(p[a], 1e-12))
    return lp


def _extinct_state(st, p):
    """Same rule as rl2_ppo_rollout._extinct, on FastCarbon.get_state() output."""
    if any(w["player"] == p for w in st["workers"].values()):
        return False
    if any(t["player"] == p for t in st["trees"].values()):
        return False
    if not st["bases"].get(p):
        return True
    return st["cash"][p] < 30


def _leaf_u(sim, me, pol, teacher, counter):
    """Bare-net leaf value at the current sim state, from `me`'s seat, in U."""
    feat = F.Feat(me)
    x = feat.observe(sim.obs(me))
    counter[0] += 1
    _, _, v = pol.act(x, feat.aux, teacher=teacher)
    return value_to_u(v)[0]


def gen_candidates(pol, x, aux, teacher, K=16, counter=None):
    """Greedy joint action + forced-prefix single-unit deviations.
    Returns (arms, priors): arms are label arrays [OWN]; priors are the
    normalised joint policy probs over the arm set."""
    counter[0] += 1
    labels, logits, _ = pol.act(x, aux, teacher=teacher)
    greedy = np.asarray(labels, dtype=np.int64)
    arms = [greedy]
    arm_logits = [logits]
    gaps = []
    for s in range(F.OWN):
        if int(greedy[s]) < 0:
            continue
        lg = logits[s].astype(np.float64)
        top = np.partition(lg, -2)[-2:]
        gap = float(top[1] - top[0])
        kind = int(x["tok_kind"][s])
        cargo = float(x["tok_cargo"][s]) if s < 10 else 0.0
        prio = 0 if (s < 10 and kind == 1 and cargo > 0) else (1 if s < 10 else 2)
        gaps.append((prio, gap, s))
    gaps.sort()
    for _, _, s in gaps[:6]:
        lg = logits[s]
        legal = [m for m in range(5) if lg[m] > NEG / 2 and m != int(greedy[s])]
        for m in legal:
            fz = greedy.copy()
            fz[:s] = greedy[:s]
            fz[s] = m
            counter[0] += 1
            lab, lg2, _ = pol.act(x, aux, teacher=teacher, forced=fz)
            lab = np.asarray(lab, dtype=np.int64)
            if not any(np.array_equal(lab, a) for a in arms):
                arms.append(lab)
                arm_logits.append(lg2)
            if len(arms) >= K:
                break
        if len(arms) >= K:
            break
    lps = np.array([joint_logp(lg, a) for lg, a in zip(arm_logits, arms)])
    priors = np.exp(lps - lps.max())
    priors = priors / priors.sum()
    return arms, priors


def evaluate_root(sim, me, pol, x_me, aux_me, x_opp, aux_opp, feat_me, feat_opp,
                  teacher=1, K=16, M=8, H=2, seed_base=7000):
    """Root-candidate teacher evaluation at one position.

    sim: FastCarbon at the root position (not mutated). feat_me / feat_opp are
    seat/own_uid shims (SimpleNamespace with .seat and .own_uid, captured at
    the root step) used only for to_actions conversion.
    Returns dict with arms, priors P_A, Q (mean U over scenarios), target q,
    entropy, agreement with greedy, act-call count and wall ms.
    """
    from eval_v53 import CFG
    t0 = time.time()
    counter = [0]

    def act(px, paux, **kw):
        counter[0] += 1
        return pol.act(px, paux, teacher=teacher, **kw)

    # ---- opponent root actions FIRST (one per scenario; before candidates) ----
    opp_actions = []
    for m in range(M):
        rng_m = np.random.default_rng(seed_base + m)
        lab, _, _ = act(x_opp, aux_opp, sample_rng=rng_m, temp=1.0)
        opp_actions.append(F.Feat.to_actions(feat_opp, np.asarray(lab, dtype=np.int64)))

    # ---- candidate arms ----
    arms, priors = gen_candidates(pol, x_me, aux_me, teacher, K=K, counter=counter)
    Q = np.zeros((len(arms), M))
    root_full = sim.full_observation(0)
    for ci, arm in enumerate(arms):
        a_me = F.Feat.to_actions(feat_me, arm)
        for m in range(M):
            clone = FastCarbon(CFG)
            clone.load_observation(root_full)
            a_opp = opp_actions[m]
            clone.step(*((a_me, a_opp) if me == 0 else (a_opp, a_me)))
            done = False
            for _ in range(H):
                st = clone.get_state()
                if _extinct_state(st, 0) or _extinct_state(st, 1) or clone.step_count >= 299:
                    done = True
                    break
                f0 = F.Feat(0)
                f1 = F.Feat(1)
                x0 = f0.observe(clone.obs(0))
                x1 = f1.observe(clone.obs(1))
                l0, _, _ = act(x0, f0.aux)
                l1, _, _ = act(x1, f1.aux)
                clone.step(*(f0.to_actions(np.asarray(l0, dtype=np.int64)),
                             f1.to_actions(np.asarray(l1, dtype=np.int64))))
            if done or clone.step_count >= 299:
                st = clone.get_state()
                mg = float(st["cash"][me] - st["cash"][1 - me])
                w = 1.0 if mg > 0 else (0.0 if mg < 0 else 0.5)
                Q[ci, m] = 0.5 * w + 0.5 / (1.0 + math.exp(-mg / 1000.0))
            else:
                Q[ci, m] = _leaf_u(clone, me, pol, teacher, counter)
    q_mean = Q.mean(axis=1)
    wq = np.exp((q_mean - q_mean.max()) / 0.05) * np.sqrt(np.maximum(priors, 1e-12))
    q = wq / wq.sum()
    ent = float(-(q * np.log(np.maximum(q, 1e-12))).sum())
    return {
        "n_arms": len(arms), "priors": priors, "Q_mean": q_mean, "Q_std": Q.std(axis=1),
        "q": q, "entropy": ent, "agree_greedy": bool(int(np.argmax(q)) == 0),
        "greedy_q_rank": int((q > q[0]).sum()),
        "n_act_calls": counter[0],
        "wall_ms": round((time.time() - t0) * 1000, 1),
    }
