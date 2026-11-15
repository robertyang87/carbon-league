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
  - leaf value comes from the Stage 0 utility head (E[U], sigmoid, [0,1])
    via Policy.act(want_u=True); terminal leaves use F01/F04 scoring
    (eval_v53: extinct side steps_played-300-1, survivor pre-step cash).
  - guard: evaluate_root runs guard-on by default (production config
    GUARD=True, GUARD_NARROW=True, GUARD_CASH=40.0, LOWW=2) through the
    reusable BankruptcyGuard (cnet2_search); guard=None runs the old
    guard-off path. Each candidate arm / scenario branch holds its own
    guard snapshot -- never a shared global guard across branches.

Q is in U units: U = 0.5*W + 0.5*sigmoid(M/1000), W in {0, 0.5, 1}.
"""

import math
import time

import numpy as np

import cnet2_feat as F
from cnet2_search import BankruptcyGuard, CachePolicy
from fastsim import FastCarbon

NEG = -1e9

# A3: production guard config, frozen (same as producer selfplay CONF and
# the exam CONF): guard on, narrow bankruptcy-risk trigger, cash floor 40.
TEACHER_GUARD_CFG = dict(GUARD=True, GUARD_NARROW=True, GUARD_CASH=40.0, LOWW=2)


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
    """Bare-net leaf value at the current sim state, from `me`'s seat, in U
    (Stage 0 utility head, E[U] in [0,1])."""
    feat = F.Feat(me)
    x = feat.observe(sim.obs(me))
    counter[0] += 1
    _, _, _, u = pol.act(x, feat.aux, teacher=teacher, want_u=True)
    return float(u)


def _terminal_u(sim, me, pre, episode_steps=300):
    """F01/F04 terminal scoring, verbatim eval_v53 (L493-494): an extinct side
    scores (steps_played - episodeSteps - 1); a survivor keeps the cash it
    carried INTO the terminal step (pre-step state `pre`); a plain time-out
    finish scores final cash. Returns U in [0,1] from `me`'s seat."""
    st = sim.get_state()
    extinct = [_extinct_state(st, 0), _extinct_state(st, 1)]
    steps_played = sim.step_count
    cash_fin = [float(st["cash"][0]), float(st["cash"][1])]
    pre_cash = [float(pre["cash"][0]), float(pre["cash"][1])]
    score = [(steps_played - episode_steps - 1) if extinct[p]
             else (pre_cash[p] if any(extinct) else cash_fin[p])
             for p in (0, 1)]
    mg = score[me] - score[1 - me]
    w = 1.0 if mg > 0 else (0.0 if mg < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-mg / 1000.0))


def gen_candidates(pol, x, aux, teacher, K=16, counter=None, guard=None, own_uid=None):
    """Greedy joint action + forced-prefix single-unit deviations.
    Returns (arms, priors): arms are label arrays [OWN]; priors are the
    normalised joint policy probs over the arm set.
    guard: BankruptcyGuard or None. When set, the greedy arm and every
    deviation arm pass through the guard (production posture: the net's
    move IS the guarded move); pol is wrapped to CachePolicy if needed
    (the guard's re-decode path needs enc_for)."""
    if guard is not None and not isinstance(pol, CachePolicy):
        _cfg = pol.cfg
        pol = CachePolicy(pol.w)
        pol.cfg = _cfg
        if _cfg is not None:
            pol.G = _cfg["G"]
    counter[0] += 1
    labels, logits, _ = pol.act(x, aux, teacher=teacher)
    if guard is not None:
        guard.reset_step()
        labels = guard.apply(x, aux, np.asarray(labels, dtype=np.int64), logits,
                             pol, own_uid, teacher)
    greedy = np.asarray(labels, dtype=np.int64)
    arms = [greedy]
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
            if guard is not None:
                guard.reset_step()
                lab = guard.apply(x, aux, lab, lg2, pol, own_uid, teacher)
            if not any(np.array_equal(lab, a) for a in arms):
                arms.append(lab)
            if len(arms) >= K:
                break
        if len(arms) >= K:
            break
    # S05 (Codex review 2026-11-13): the guard may have rewritten an arm's
    # labels, so the decode logits captured along the PRE-guard path no
    # longer score the arm actually returned. Re-score every final arm by
    # teacher-forcing its exact labels, verify each chosen action is legal
    # under its own final prefix (masked logit above NEG/2), and drop
    # deviation arms that fail; priors are the normalised joint policy
    # probs of the FINAL arms. The greedy arm must always survive: it is
    # produced by a real decode / guard re-decode, so a legality failure
    # there means the two decoders disagree -- fail loudly.
    scored = []
    for i, arm in enumerate(arms):
        counter[0] += 1
        _, lg2, _ = pol.act(x, aux, teacher=teacher,
                            forced=np.asarray(arm, dtype=np.int64))
        legal = all(int(arm[s]) < 0 or float(lg2[s][int(arm[s])]) > NEG / 2
                    for s in range(F.OWN) if x["tok_mask"][s])
        if not legal:
            if i == 0:
                raise RuntimeError("greedy arm illegal under its own final prefix")
            continue
        scored.append((arm, joint_logp(lg2, arm)))
    arms = [a for a, _ in scored]
    lps = np.array([lp for _, lp in scored])
    priors = np.exp(lps - lps.max())
    priors = priors / priors.sum()
    return arms, priors


def _is_terminal(sim):
    """Terminal after a step: either seat extinct, or the step cap reached."""
    st = sim.get_state()
    return _extinct_state(st, 0) or _extinct_state(st, 1) or sim.step_count >= 299


def evaluate_root(sim, me, pol, x_me, aux_me, x_opp, aux_opp, feat_me, feat_opp,
                  teacher=1, K=16, M=8, H=2, seed_base=7000, guard="on",
                  guard_snaps=None):
    """Root-candidate teacher evaluation at one position.

    sim: FastCarbon at the root position (not mutated). feat_me / feat_opp are
    seat/own_uid shims (SimpleNamespace with .seat and .own_uid, captured at
    the root step) used only for to_actions conversion.
    pol is wrapped to CachePolicy when it is a bare NP.Policy (the guard's
    re-decode path needs enc_for; numerics are unchanged).
    guard: "on" (default, production config TEACHER_GUARD_CFG), None/False
    (guard-off, legacy path), or a cfg dict merged over TEACHER_GUARD_CFG.
    guard_snaps: optional (snap_me, snap_opp) BankruptcyGuard snapshots
    carrying each seat's REAL guard history up to the root (S04: GUARD_NARROW
    depends on lostw, so a fresh guard at the root misstates the production
    continuation whenever a unit was lost earlier). Either entry may be
    None (fresh for that seat). Branch guards restore from the root
    snapshots, so branches never mutate the caller's history.
    Depth: the root step plus H continuation steps (H=2 -> 3 plies total).
    Raises ValueError on an already-terminal root position.
    Returns dict with arms, priors P_A, Q (mean U over scenarios), target q,
    entropy, agreement with greedy, act-call count, wall ms, and guard
    trigger totals.
    """
    from eval_v53 import CFG
    t0 = time.time()
    counter = [0]
    ep_steps = int(CFG.get("episodeSteps", 300))
    _root_st = sim.get_state()
    if _is_terminal(sim):
        raise ValueError("evaluate_root called on an already-terminal root "
                         "position (extinct, or step limit reached)")
    if not isinstance(pol, CachePolicy):
        _cfg = pol.cfg
        pol = CachePolicy(pol.w)
        pol.cfg = _cfg
        if _cfg is not None:
            pol.G = _cfg["G"]
    if guard == "on":
        gcfg = TEACHER_GUARD_CFG
    elif isinstance(guard, dict):
        gcfg = dict(TEACHER_GUARD_CFG)
        gcfg.update(guard)
    elif guard in (None, False):
        gcfg = None
    else:
        raise ValueError(f"guard must be 'on', None, or a cfg dict, got {guard!r}")
    g_me_root = BankruptcyGuard(gcfg) if gcfg is not None else None
    g_opp_root = BankruptcyGuard(gcfg) if gcfg is not None else None
    if guard_snaps is not None and gcfg is not None:
        _sm, _so = guard_snaps
        if _sm is not None:
            g_me_root.restore(_sm)
        if _so is not None:
            g_opp_root.restore(_so)
    gtot = {"guard_moves": 0, "guard_recruit": 0}

    def act(px, paux, **kw):
        counter[0] += 1
        return pol.act(px, paux, teacher=teacher, **kw)

    # ---- opponent root actions FIRST (one per scenario; before candidates) ----
    opp_actions = []
    for m in range(M):
        rng_m = np.random.default_rng(seed_base + m)
        lab, lg, _ = act(x_opp, aux_opp, sample_rng=rng_m, temp=1.0)
        if g_opp_root is not None:
            g_opp_root.reset_step()
            lab = g_opp_root.apply(x_opp, aux_opp, np.asarray(lab, dtype=np.int64), lg,
                                   pol, feat_opp.own_uid, teacher, gtot)
        opp_actions.append(F.Feat.to_actions(feat_opp, np.asarray(lab, dtype=np.int64)))

    # ---- candidate arms ----
    arms, priors = gen_candidates(pol, x_me, aux_me, teacher, K=K, counter=counter,
                                  guard=g_me_root, own_uid=feat_me.own_uid)
    me_snap = g_me_root.snapshot() if g_me_root is not None else None
    opp_snap = g_opp_root.snapshot() if g_opp_root is not None else None
    Q = np.zeros((len(arms), M))
    root_full = sim.full_observation(0)
    for ci, arm in enumerate(arms):
        a_me = F.Feat.to_actions(feat_me, arm)
        for m in range(M):
            clone = FastCarbon(CFG)
            clone.load_observation(root_full)
            a_opp = opp_actions[m]
            pre = clone.get_state()  # state before the root step (survivor scoring)
            clone.step(*((a_me, a_opp) if me == 0 else (a_opp, a_me)))
            g_me = BankruptcyGuard(gcfg) if gcfg is not None else None
            g_opp = BankruptcyGuard(gcfg) if gcfg is not None else None
            if g_me is not None:
                g_me.restore(me_snap)
            if g_opp is not None:
                g_opp.restore(opp_snap)
            # S01 (Codex review 2026-11-13): terminal status is checked after
            # EVERY step, including the root step and the last continuation
            # step. The old loop only checked at the top of the next
            # iteration, so an extinction caused by the final step fell
            # through to the network leaf value. `pre` always holds the
            # state before the step that caused termination.
            done = _is_terminal(clone)
            for _ in range(H):
                if done:
                    break
                f0 = F.Feat(0)
                f1 = F.Feat(1)
                x0 = f0.observe(clone.obs(0))
                x1 = f1.observe(clone.obs(1))
                l0, lg0, _ = act(x0, f0.aux)
                l1, lg1, _ = act(x1, f1.aux)
                # guards are seat-indexed: g_me holds the root snapshot of
                # seat `me`, g_opp of seat 1-me -- when me == 1 they must
                # NOT be applied to l0/l1 in me/opp order (state leakage
                # across seats; strategy-group review 2026-11-13).
                g_seat = [None, None]
                if g_me is not None:
                    g_seat[me] = g_me
                    g_seat[1 - me] = g_opp
                if g_seat[0] is not None:
                    g_seat[0].reset_step()
                    l0 = g_seat[0].apply(x0, f0.aux, np.asarray(l0, dtype=np.int64), lg0,
                                         pol, f0.own_uid, teacher, gtot)
                if g_seat[1] is not None:
                    g_seat[1].reset_step()
                    l1 = g_seat[1].apply(x1, f1.aux, np.asarray(l1, dtype=np.int64), lg1,
                                         pol, f1.own_uid, teacher, gtot)
                pre = clone.get_state()  # state before this continuation step
                clone.step(*(f0.to_actions(np.asarray(l0, dtype=np.int64)),
                             f1.to_actions(np.asarray(l1, dtype=np.int64))))
                done = _is_terminal(clone)
            if done:
                Q[ci, m] = _terminal_u(clone, me, pre, episode_steps=ep_steps)
            else:
                Q[ci, m] = _leaf_u(clone, me, pol, teacher, counter)
    q_mean = Q.mean(axis=1)
    wq = np.exp((q_mean - q_mean.max()) / 0.05) * np.sqrt(np.maximum(priors, 1e-12))
    q = wq / wq.sum()
    ent = float(-(q * np.log(np.maximum(q, 1e-12))).sum())
    return {
        "arms": arms, "n_arms": len(arms), "priors": priors, "Q_mean": q_mean, "Q_std": Q.std(axis=1),
        "q": q, "entropy": ent, "agree_greedy": bool(int(np.argmax(q)) == 0),
        "greedy_q_rank": int((q > q[0]).sum()),
        "n_act_calls": counter[0],
        "wall_ms": round((time.time() - t0) * 1000, 1),
        "guard": dict(gtot),
    }
