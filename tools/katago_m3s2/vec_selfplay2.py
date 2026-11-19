"""Phase-1 lockstep self-play generator for the KataGo warm-start line.

Derived from tools/katago_m3s2/vec_selfplay.py (the production generator;
this file never modifies it).  The per-game logic -- opponent selection,
per-game RNG stream, fast/deep move construction, target computation,
record layout -- is transplanted from vec_selfplay.py function by
function with the text unchanged wherever the change of evaluation
transport allows it.  What changes is ONLY the transport:

  * vec_selfplay.py: N worker processes, each plays one game at a time
    and blocks on a synchronous RPC to a single inference server for
    EVERY forward (batch 1 for every fast move; the server batches only
    whatever happens to be queued when it wakes).
  * vec_selfplay2.py (this file): ONE driver process advances a pool of
    games in lockstep.  Each game is a coroutine (generator) that yields
    an evaluation request (tag, X, SC) exactly where vec_selfplay.py
    called client.forward(...), and receives (lw, lb, v) back.  The
    scheduler collects the outstanding request of every game in the
    pool, concatenates all requests per net tag into ONE batch, and
    issues a single batched forward per tag per round.  Simulation
    (encode / next_state / sampling) stays per-game, in the worker-side
    code, exactly as in vec_selfplay.py.

Equivalence discipline (see gate_vec2.py):
  * Per-game seeds are derived exactly as in vec_selfplay.py:
    game seed = --seed + 1000003 * gid (mod 2**32 inside play_game_g,
    the single point where the seed enters any RNG), one
    numpy RandomState per game, drawn in the same order as
    vec_selfplay.py's play_game.  The scheduler never touches a game's
    RNG; deep-move internals keep their own (step, seat)-seeded
    RandomStates, unchanged.
  * Batching therefore cannot change WHAT is sampled given identical
    forward outputs; it can only change forward numerics (batched GPU
    kernels vs batch-1 kernels, BF16 autocast).  That residue is what
    the equivalence gate measures; it is not hidden.
  * Shard schema is byte-identical to vec_selfplay.py: write_shard and
    the record construction below are verbatim, keys / dtypes / field
    order unchanged (SHARD_KEYS).

Fixed search configuration, opponent pool, value-scale bridge and the
observation convention are inherited from vec_selfplay.py unchanged;
see its module docstring (playout-cap randomization 75% fast / 25%
deep at --deep-frac 0.25, opponent pool 50% mirror / 30% master3s2 /
20% hall-of-fame, deep plan K=24 / D2=6 / tau=1.0, bridge applied to
deep-plan leaf values under --value-scale u).

CLI is aligned with vec_selfplay.py (--src/--ckpt/--hof/--games/
--out/--seed/--deep-frac/--device/--dtype/--threads/--value-scale/
--smoke) except: --workers is replaced by --pool-games (games in
flight in the lockstep pool, default 32) and --batch-cap (max rows per
single GPU forward chunk inside a round, default 2048; chunking is a
memory guard only -- a round still issues one logical batch per tag).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# bridge / layout live next to vec_selfplay.py in tools/katago_m3s2/;
# they are imported lazily (see _load_deps) so this module stays
# importable -- and its scheduler unit-testable -- without them.
bridge = None
layout = None


def _load_deps():
    global bridge, layout
    if bridge is None:
        import bridge as _bridge
        bridge = _bridge
    if layout is None:
        import layout as _layout
        layout = _layout
    return bridge, layout


DEEP_TAU = 1.0
DEEP_K = 24
DEEP_D2 = 6
# the chassis' own candidate temperature table (_S_GEN in master3s2.py)
GEN_TABLE = ((2.5, 8), (1.0, 3), (3.5, 8), (0.0, 6), (5.0, 8))
AUX_HORIZONS = (15, 40, 80)

# Shard schema: the exact key list (and order) of vec_selfplay.py's
# write_shard.  gate_vec2.py checks produced shards against this.
SHARD_KEYS = ['x', 'sc', 'tw', 'tb', 'wmask', 'bmask', 'carbon', 'margin',
              'aux_delta', 'win', 'searched']


# --------------------------------------------------------------------------
# Move generation helpers -- verbatim from vec_selfplay.py.
# --------------------------------------------------------------------------
import inspect


def _accepts(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _onehot_logits(l):
    """Logits whose softmax is the exact one-hot argmax distribution:
    0.0 at each row's argmax, -1e30 elsewhere (exp(-1e30) is exactly 0.0,
    so sampling from them at temperature 1 IS deterministic greedy
    selection -- with no division by a temperature anywhere)."""
    l = np.asarray(l)
    out = np.full(l.shape, -1e30, dtype=np.float64)
    np.put_along_axis(out, np.argmax(l, axis=-1)[..., None], 0.0, axis=-1)
    if np.issubdtype(l.dtype, np.floating):
        out = out.astype(l.dtype, copy=False)
    return out


def sample(mod, players, me, lw, lb, rng, step, temp=1.0, fixed=None, x_feat=None):
    """The chassis' _sample, across the two source API variants:
    fix2 takes temp/x_feat kwargs; master3s2 takes temperature through the
    module-global _TEMP (its _search sets _TEMP around calls the same way).

    temp <= 0 (greedy) is handled HERE, as an exact argmax: the logits
    are replaced by one-hot logits (see _onehot_logits) and the chassis
    is called at temperature 1, so a zero temperature never reaches a
    division or the _TEMP global.  (Stage-1 smoke failure, 2026-11-19:
    temperature-zero handling that leaked into divisions -- lw0 / 0.0 in
    the GEN_TABLE loop, and _TEMP = 0 in master3s2's _probs softmax at
    source line 566 -- produced NaN probabilities and killed every
    worker.)  For 0 < temp the chassis' own temperature path is used
    unchanged.
    """
    if temp <= 0:
        lw, lb = _onehot_logits(lw), _onehot_logits(lb)
        temp = 1.0
    if _accepts(mod._sample, 'temp'):
        kw = {'temp': temp, 'fixed': fixed}
        if x_feat is not None and _accepts(mod._sample, 'x_feat'):
            kw['x_feat'] = x_feat
        return mod._sample(players, me, lw, lb, rng, step, **kw)
    old = mod._TEMP
    mod._TEMP = temp
    try:
        return mod._sample(players, me, lw, lb, rng, step, fixed=fixed)
    finally:
        mod._TEMP = old


def probs(mod, l, temp=1.0):
    """The chassis' _probs at a temperature, across the same two variants.

    temp <= 0 is the exact argmax one-hot, computed here (identical to
    both chassis variants' own temp <= 0 branch) so _TEMP is never set
    to 0; see sample().
    """
    if temp <= 0:
        p = np.zeros_like(np.asarray(l, dtype=np.float64))
        p[np.argmax(l)] = 1.0
        return p
    if _accepts(mod._probs, 'temp'):
        return mod._probs(l, temp)
    old = mod._TEMP
    mod._TEMP = temp
    try:
        return mod._probs(l)
    finally:
        mod._TEMP = old


def policy_targets_from_probs(mod, players, me, lw, lb):
    """Fast-move targets: the net's own per-unit probabilities."""
    workers, _, bases, _ = mod.parse(players)
    tw = np.zeros((225, 5), np.float32)
    tb = np.zeros((225, 3), np.float32)
    for wid, w in workers.items():
        if w[3] == me:
            tw[w[0]] = probs(mod, lw[w[0]], 1.0)
    for bid, pos in bases[me]:
        tb[pos] = probs(mod, lb[pos], 1.0)
    return tw, tb


def masks_for(mod, players, me):
    workers, _, bases, _ = mod.parse(players)
    wm = np.zeros(225, np.uint8)
    bm = np.zeros(225, np.uint8)
    for w in workers.values():
        if w[3] == me:
            wm[w[0]] = 1
    for _, pos in bases[me]:
        bm[pos] = 1
    return wm, bm


# --------------------------------------------------------------------------
# Move generation as coroutines.  Each `yield (tag, X, SC)` is exactly a
# vec_selfplay.py `client.forward(tag, X, SC)` call site; the value sent
# back into the generator is the (lw, lb, v) triple for that request.
# Bodies are vec_selfplay.py's fast_action / deep_move with only the
# transport swapped.
# --------------------------------------------------------------------------
def fast_action_g(mod, tag, st, me, rng, prev):
    x = mod._quant(mod.encode(st.players, st.step, me, st.carbon,
                              [1.0] * mod.N, prev))
    sc = mod._scalars(st.players, st.step, me)
    lw, lb, _ = yield (tag, x[None], sc[None])
    act = sample(mod, st.players, me, lw[0], lb[0], rng, st.step, temp=1.0)
    return act, (x, sc, lw[0], lb[0])


def deep_move_g(mod, my_tag, opp_tag, st, me, rng, tau=DEEP_TAU,
                k_max=DEEP_K, d2=DEEP_D2, value_scale='terminal'):
    """Fixed-plan search for one move.  Returns (action, targets, root_info).

    Deterministic given the nets: candidate counts, opponent responses and
    rollout depth are constants; every rng is seeded from (step, seat,
    pair index) exactly like the chassis seeds its search rngs.  No clock
    is consulted anywhere.
    """
    op = 1 - me
    players, step = st.players, st.step
    seed0 = int(1000003 * (step + 1) + 17 * me) % (2 ** 31)
    srng = np.random.RandomState(seed0)
    full = [1.0] * mod.N
    x0 = mod._quant(mod.encode(players, step, me, st.carbon, full, None))
    xo = mod._quant(mod.encode(players, step, op, st.carbon, full, None))
    sc0 = mod._scalars(players, step, me)
    lw0, lb0, _ = yield (my_tag, x0[None], sc0[None])
    lwo, lbo, _ = yield (opp_tag, xo[None], mod._scalars(players, step, op)[None])
    lw0, lb0, lwo, lbo = lw0[0], lb0[0], lwo[0], lbo[0]

    cands, keys = [], set()

    def add(a):
        if len(cands) >= k_max:
            return False
        k = tuple(sorted(a.items()))
        if k in keys:
            return False
        keys.add(k)
        cands.append(a)
        return True

    act0 = sample(mod, players, me, lw0, lb0, srng, step, temp=1.0)
    add(act0)
    add(sample(mod, players, me, lw0, lb0, srng, step, temp=0.0))
    for temp, cap in GEN_TABLE:
        new = dups = 0
        if temp == 1.0:
            lwt = lw0
        elif temp <= 0.0:
            # Greedy table entry: one-hot argmax targets.  lw0 / temp here
            # used to divide by zero, turning every logit into +-inf and
            # the sampled probabilities into NaN (the stage-1 smoke
            # failure: all workers died on their first deep move).
            lwt = _onehot_logits(lw0)
        else:
            lwt = lw0 / temp
        for _ in range(4 * cap):
            if new >= cap or len(cands) >= k_max or dups >= 8:
                break
            if add(sample(mod, players, me, lwt, lb0, srng, step, temp=1.0)):
                new += 1
                dups = 0
            else:
                dups += 1
    for a_d in mod._deviations(players, me, lw0, act0, {}, 6):
        add(a_d)

    opp_resp = [sample(mod, players, op, lwo, lbo, np.random.RandomState(seed0 + 7),
                       step, temp=0.0, x_feat=xo),
                sample(mod, players, op, lwo, lbo, np.random.RandomState(seed0 + 38),
                       step, temp=1.0, x_feat=xo)]

    cfg = mod.DEFAULT_CONFIG
    # boards[ci][mi] = State after (cand, opp_resp); roll forward d2 plies.
    boards = {}
    for ci, cand in enumerate(cands):
        for mi, oresp in enumerate(opp_resp):
            acts = [cand, oresp] if me == 0 else [oresp, cand]
            boards[ci, mi] = mod.next_state(st, acts, cfg)
    depth = min(d2, mod.LAST_TURN - step)
    for ply in range(2, depth + 1):
        live = [(k, b) for k, b in boards.items() if b.step <= mod.LAST_TURN]
        if not live:
            break
        Xm, Xo, SCm = [], [], []
        for (ci, mi), b in live:
            Xm.append(mod._quant(mod.encode(b.players, b.step, me, b.carbon, full, None)))
            Xo.append(mod._quant(mod.encode(b.players, b.step, op, b.carbon, full, None)))
            SCm.append(mod._scalars(b.players, b.step, me))
        lwm, lbm, _ = yield (my_tag, np.stack(Xm), np.stack(SCm))
        lwp, lbp, _ = yield (opp_tag, np.stack(Xo), np.stack(SCm))
        for j, ((ci, mi), b) in enumerate(live):
            r1 = np.random.RandomState(seed0 + 7919 * ply + 31 * mi + 1)
            r2 = np.random.RandomState(seed0 + 7919 * ply + 31 * mi + 2)
            am = sample(mod, b.players, me, lwm[j], lbm[j], r1, b.step, temp=1.0)
            ao = sample(mod, b.players, op, lwp[j], lbp[j], r2, b.step, temp=1.0)
            boards[ci, mi] = mod.next_state(b, [am, ao] if me == 0 else [ao, am], cfg)
    # leaf values from my perspective
    vals = {}
    term, leaf = [], []
    for k, b in boards.items():
        if b.step > mod.LAST_TURN:
            vals[k] = mod._terminal(b.players[me][0] - b.players[op][0])
        else:
            leaf.append(k)
    if leaf:
        X = np.stack([mod._quant(mod.encode(boards[k].players, boards[k].step, me,
                                            boards[k].carbon, full, None)) for k in leaf])
        SC = np.stack([mod._scalars(boards[k].players, boards[k].step, me) for k in leaf])
        _, _, v = yield (my_tag, X, SC)
        if value_scale == 'u':
            # VALUE-SCALE BRIDGE (ruling RULING_KATAGO_VALUE_SCALE_BRIDGE_
            # 20261119): the acting net is a trained checkpoint whose value
            # head outputs the plan's U scale; the deep plan's Q arithmetic
            # is the chassis' _terminal scale, so leaf values cross scales
            # exactly here.  Terminal boards above use mod._terminal on the
            # true margin and are never bridged.
            v = bridge.bridge(v)
        for k, vv in zip(leaf, v):
            vals[k] = float(vv)
    q = np.array([np.mean([vals[ci, mi] for mi in range(len(opp_resp))])
                  for ci in range(len(cands))], dtype=np.float64)
    p = np.exp((q - q.max()) / tau)
    p /= p.sum()
    # marginalize the candidate distribution onto units
    workers, _, bases, _ = mod.parse(players)
    wpos = {wid: w[0] for wid, w in workers.items() if w[3] == me}
    bpos = {bid: pos for bid, pos in bases[me]}
    tw = np.zeros((225, 5), np.float32)
    tb = np.zeros((225, 3), np.float32)
    for ci, cand in enumerate(cands):
        for uid, name in cand.items():
            if uid in wpos:
                tw[wpos[uid], mod.WIDX.get(name or 'NONE', 0)] += p[ci]
            elif uid in bpos:
                tb[bpos[uid], mod.BIDX.get(name or 'NONE', 0)] += p[ci]
    best = cands[int(np.argmax(q))]
    return best, (tw, tb), (x0, sc0)


# --------------------------------------------------------------------------
# Game loop as a coroutine -- vec_selfplay.py's play_game with the
# transport swapped; the RNG stream, the draw order and the record
# construction are unchanged.
# --------------------------------------------------------------------------
def play_game_g(mod, game_id, seed, deep_frac, hof_tags, smoke=False,
                max_steps=None, value_scale='terminal'):
    """Play one game (coroutine); returns a list of per-position records."""
    # Per-game seeds are seed_base + 1000003*gid and grow past numpy's
    # 2**32-1 limit once gid passes ~4294 (Gen-0 crash, 2026-11-19: all
    # 8 workers died starting gid 4295).  Wrap modulo 2**32 here -- the
    # single point where the seed enters any RNG (RandomState below and
    # np.random.seed inside layout.new_game_state) -- so low gids keep
    # their exact old seeds and the already-landed shards stay consistent.
    seed = int(seed) % (2 ** 32)
    rng = np.random.RandomState(seed)
    st = layout.new_game_state(mod, seed)
    last_step = mod.LAST_TURN if max_steps is None else min(mod.LAST_TURN, max_steps - 1)
    r = rng.rand()
    if r < 0.5:
        opp_tag, opp_seat = 'cur', None            # mirror: both seats are 'cur'
    elif r < 0.8 or not hof_tags:
        opp_tag, opp_seat = 'm3s2', game_id % 2
    else:
        opp_tag = hof_tags[rng.randint(len(hof_tags))]
        opp_seat = game_id % 2
    tags = {0: 'cur', 1: 'cur'} if opp_seat is None else {
        opp_seat: opp_tag, 1 - opp_seat: 'cur'}
    recorded = (0, 1) if opp_seat is None else (1 - opp_seat,)
    margins = {me: [] for me in recorded}
    records = {me: [] for me in recorded}
    prev = None
    cfg = mod.DEFAULT_CONFIG
    while st.step <= last_step:
        acts = [None, None]
        step_info = {}
        for me in (0, 1):
            tag = tags[me]
            deep = (not smoke and me in recorded and tag == 'cur'
                    and rng.rand() < deep_frac)
            if deep:
                act, (tw, tb), (x, sc) = yield from deep_move_g(
                    mod, 'cur', tags[1 - me], st, me, rng,
                    value_scale=value_scale)
                step_info[me] = (x, sc, tw, tb, 1)
            else:
                act, (x, sc, lw, lb) = yield from fast_action_g(
                    mod, tag, st, me, rng, prev)
                if me in recorded and tag == 'cur':
                    tw, tb = policy_targets_from_probs(mod, st.players, me, lw, lb)
                    step_info[me] = (x, sc, tw, tb, 0)
            acts[me] = act
        for me in recorded:
            x, sc, tw, tb, searched = step_info[me]
            wm, bm = masks_for(mod, st.players, me)
            records[me].append({
                'x': np.rint(np.clip(x, 0, 2.55) * 100).astype(np.uint8),
                'sc': sc, 'tw': tw.astype(np.float16), 'tb': tb.astype(np.float16),
                'wmask': wm, 'bmask': bm,
                'carbon': np.array(st.carbon, dtype=np.float16),
                'searched': np.uint8(searched)})
            margins[me].append(st.players[me][0] - st.players[1 - me][0])
        prev = st.players
        st = mod.next_state(st, acts, cfg)
    out = []
    for me in recorded:
        final = st.players[me][0] - st.players[1 - me][0]
        traj = margins[me] + [final]
        for t, rec in enumerate(records[me]):
            rec['margin'] = np.float32(final)
            rec['win'] = np.float32(1.0 if final > 0 else (0.5 if final == 0 else 0.0))
            rec['aux_delta'] = np.array(
                [traj[min(t + h, len(traj) - 1)] - traj[t] for h in AUX_HORIZONS],
                dtype=np.float32)
            out.append(rec)
    return out


def write_shard(path, rows):
    keys = ['x', 'sc', 'tw', 'tb', 'wmask', 'bmask', 'carbon', 'margin',
            'aux_delta', 'win', 'searched']
    arrs = {k: np.stack([r[k] for r in rows]) for k in keys}
    with open(path, 'wb') as f:
        np.savez_compressed(f, **arrs)


# --------------------------------------------------------------------------
# Backends: forward_batch(tag, X, SC) -> (lw, lb, v) numpy arrays for the
# WHOLE concatenated batch of one tag.  Chunking to --batch-cap rows is
# a memory guard inside the torch backend; results are concatenated
# back in request order.
# --------------------------------------------------------------------------
class NumpyLocalBackend:
    """Smoke-mode backend: the chassis module's numpy _Policy for every
    tag (identical semantics to vec_selfplay.py's NumpyLocalClient)."""

    def __init__(self, mod):
        self.mod = mod

    def forward_batch(self, tag, X, SC):
        lw = np.empty((len(X), 225, 5), np.float32)
        lb = np.empty((len(X), 225, 3), np.float32)
        for i in range(len(X)):
            lw[i], lb[i] = self.mod._POLICY.forward(X[i], self.mod._SID)
        # No critic without torch; _terminal of the current margin is the
        # documented smoke-mode leaf proxy (deep mode is disabled in smoke).
        v = np.array([self.mod._terminal(float(s[0]) * 3000.0) for s in SC],
                     dtype=np.float32)
        return lw, lb, v


class TorchBatchBackend:
    """Owns every net (as vec_selfplay.py's inference server did) and
    serves one batched forward per call."""

    def __init__(self, cfg):
        import torch
        import model as model_mod
        self.torch = torch
        torch.set_num_threads(cfg['threads'])
        self.device = cfg['device']
        self.batch_cap = int(cfg.get('batch_cap', 2048))
        self.nets = {}
        src_policy, src_value, src_snames, src_nb = layout.load_source_payloads(cfg['src'])
        self.nets['m3s2'] = model_mod.MainNet.from_source(src_policy, src_value, src_nb)
        self.nets['cur'], cur_snames = model_mod.load_main_npz(cfg['ckpt'])
        for i, p in enumerate(cfg['hof']):
            self.nets['hof%d' % i], _ = model_mod.load_main_npz(p)
        self.sids = {}
        for tag, net in self.nets.items():
            names = cur_snames if tag == 'cur' else src_snames
            self.sids[tag] = names.index(layout.SID_NAME) if layout.SID_NAME in names else len(names)
            net.eval().to(self.device)
        self.dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(cfg['dtype'])

    def _forward_chunk(self, tag, X, SC):
        torch = self.torch
        xt = torch.from_numpy(np.ascontiguousarray(X, np.float32)).to(self.device)
        st = torch.from_numpy(np.ascontiguousarray(SC, np.float32)).to(self.device)
        if self.dtype is not None:
            ctx = torch.autocast(device_type=xt.device.type, dtype=self.dtype)
        else:
            import contextlib
            ctx = contextlib.nullcontext()
        with torch.no_grad(), ctx:
            lw, lb, v, _ = self.nets[tag](xt, self.sids[tag], st)
        return (lw.float().cpu().numpy(), lb.float().cpu().numpy(),
                v.float().cpu().numpy())

    def forward_batch(self, tag, X, SC):
        if len(X) <= self.batch_cap:
            return self._forward_chunk(tag, X, SC)
        parts = [self._forward_chunk(tag, X[i:i + self.batch_cap],
                                     SC[i:i + self.batch_cap])
                 for i in range(0, len(X), self.batch_cap)]
        return tuple(np.concatenate([p[j] for p in parts]) for j in range(3))


# --------------------------------------------------------------------------
# Lockstep scheduler.  Games are coroutines yielding (tag, X, SC); the
# scheduler keeps up to `pool_size` games in flight (refilled from
# `next_game`, a callable returning (key, generator) or None), and in
# every round concatenates ALL outstanding requests per tag into one
# backend call.  Iteration order is the games' insertion order, so the
# assembly is deterministic; the scheduler never touches game RNGs.
# --------------------------------------------------------------------------
def run_pool(next_game, backend, pool_size, on_done=None):
    """Drive lockstep games to completion.

    next_game() -> (key, generator) or None when no games remain.
    on_done(key, records) is called in completion order if given.
    Returns {key: records}.
    """
    active = {}          # key -> generator (insertion ordered)
    pending = {}         # key -> (tag, X, SC) outstanding request
    results = {}

    def _advance(key, gen, value, first=False):
        try:
            req = next(gen) if first else gen.send(value)
        except StopIteration as stop:
            results[key] = stop.value
            if on_done is not None:
                on_done(key, stop.value)
            del active[key]
            pending.pop(key, None)
            return
        pending[key] = req

    def _fill():
        while len(active) < pool_size:
            item = next_game()
            if item is None:
                return
            key, gen = item
            active[key] = gen
            _advance(key, gen, None, first=True)

    _fill()
    while active:
        # one round: batch every outstanding request, grouped by tag
        by_tag = {}
        for key in active:
            if key in pending:
                tag, X, SC = pending[key]
                by_tag.setdefault(tag, []).append((key, X, SC))
        replies = {}
        for tag, reqs in by_tag.items():
            X = np.concatenate([r[1] for r in reqs])
            SC = np.concatenate([r[2] for r in reqs])
            lw, lb, v = backend.forward_batch(tag, X, SC)
            off = 0
            for key, Xk, _SCk in reqs:
                n = len(Xk)
                replies[key] = (lw[off:off + n], lb[off:off + n], v[off:off + n])
                off += n
        for key in list(active):
            if key in replies:
                _advance(key, active[key], replies[key])
        _fill()
    return results


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', default=None,
                    help='chassis source module (default: layout.MASTER3S2)')
    ap.add_argument('--ckpt', default=None, help='current-net checkpoint npz '
                    '(default: gen0_m3s2.npz next to the source payloads)')
    ap.add_argument('--hof', default='', help='comma-separated hall-of-fame npz paths')
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--deep-frac', type=float, default=0.25)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--pool-games', type=int, default=32,
                    help='games in flight in the lockstep pool')
    ap.add_argument('--batch-cap', type=int, default=2048,
                    help='max rows per single GPU forward chunk')
    ap.add_argument('--value-scale', choices=['terminal', 'u'], default='terminal',
                    help='scale of the --ckpt value head (see vec_selfplay.py)')
    ap.add_argument('--smoke', action='store_true',
                    help='numpy-only plumbing check: 2 short games, fast moves only')
    args = ap.parse_args()
    _load_deps()
    if args.src is None:
        args.src = layout.MASTER3S2
    os.makedirs(args.out, exist_ok=True)
    hof = [p for p in args.hof.split(',') if p]
    hof_tags = ['hof%d' % i for i in range(len(hof))]

    if args.smoke:
        mod = layout.load_module(args.src, 'sp_smoke2')
        backend = NumpyLocalBackend(mod)
        todo = iter([(gid, play_game_g(mod, gid, 500 + gid, 0.0, [],
                                       smoke=True, max_steps=48))
                     for gid in range(2)])
        results = run_pool(lambda: next(todo, None), backend, pool_size=2)
        rows = []
        for gid in range(2):
            rows.extend(results[gid])
        write_shard(os.path.join(args.out, 'shard_smoke.npz'), rows)
        print('smoke: %d rows, x %s, tw %s' % (len(rows), rows[0]['x'].shape,
                                               rows[0]['tw'].shape))
        return 0

    if args.ckpt is None:
        raise SystemExit('--ckpt is required outside --smoke')
    mod = layout.load_module(args.src, 'sp_src_pool')
    backend = TorchBatchBackend({'src': args.src, 'ckpt': args.ckpt, 'hof': hof,
                                 'device': args.device, 'dtype': args.dtype,
                                 'threads': args.threads,
                                 'batch_cap': args.batch_cap})
    state = {'next_gid': 0, 'buf': [], 'n_shards': 0}

    def next_game():
        gid = state['next_gid']
        if gid >= args.games:
            return None
        state['next_gid'] = gid + 1
        return gid, play_game_g(mod, gid, args.seed + 1000003 * gid,
                                args.deep_frac, hof_tags,
                                value_scale=args.value_scale)

    def on_done(gid, records):
        state['buf'].extend(records)
        if len(state['buf']) >= 1024:
            write_shard(os.path.join(
                args.out, 'shard_v2_%03d.npz' % state['n_shards']), state['buf'])
            state['n_shards'] += 1
            state['buf'] = []

    try:
        run_pool(next_game, backend, args.pool_games, on_done=on_done)
    except Exception as exc:
        # Failure must propagate, exactly as in vec_selfplay.py: never
        # report 'selfplay done' over a dead game.
        print('SELFPLAY2 FAILED: %r' % (exc,), file=sys.stderr)
        return 1
    if state['buf']:
        write_shard(os.path.join(
            args.out, 'shard_v2_%03d.npz' % state['n_shards']), state['buf'])
        state['n_shards'] += 1
    if state['n_shards'] == 0:
        print('SELFPLAY2 FAILED: no shards produced', file=sys.stderr)
        return 1
    with open(os.path.join(args.out, 'selfplay_config.json'), 'w') as f:
        json.dump({'games': args.games, 'seed': args.seed,
                   'deep_frac': args.deep_frac, 'ckpt': args.ckpt, 'hof': hof,
                   'value_scale': args.value_scale,
                   'deep': {'K': DEEP_K, 'D2': DEEP_D2, 'tau': DEEP_TAU},
                   'generator': 'vec_selfplay2',
                   'pool_games': args.pool_games,
                   'batch_cap': args.batch_cap}, f, indent=1)
    print('selfplay2 done: %d games -> %s' % (args.games, args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
