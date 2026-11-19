"""Plan stage 1: vectorized self-play data generation for the warm-start line.

Multiprocess CPU environment.  Game dynamics are NOT reimplemented here:
State / next_state / encode / _sample / _scalars / _terminal / _probs are
imported from the master3s2 module (--src), so self-play can never drift
from the chassis' own rules.  Worker processes own the game states; every
position evaluation is batched to ONE torch inference server process
(--device flag, BF16/FP16 autocast optional) that hosts the current net
('cur'), the frozen master3s2 net ('m3s2', built from the source payloads)
and any hall-of-fame checkpoints ('hof<i>').

Fixed, deterministic search configuration -- per owner ruling the training
side carries NO wall-clock budget and no step-time calibration at all:
  * playout-cap randomization per move: 75% of moves are 'fast' (a direct
    policy sample through the chassis' own _sample, targets = the net's
    per-unit probabilities), 25% are 'deep' (a fixed-plan search: fixed
    candidate set from the chassis' temperature table, 2 fixed opponent
    responses, fixed D2=6 rollout plies, critic leaf values);
  * opponent pool per game: 50% current self (mirror, both seats recorded),
    30% frozen master3s2, 20% a hall-of-fame checkpoint (falls back to
    'cur' when no hall-of-fame is configured);
  * deep targets: Q per candidate = mean leaf value over the 2 opponent
    responses; target distribution = softmax(Q / tau), marginalized onto
    the units that moved: worker cells get a 5-way distribution, base
    cells a 3-way distribution.

Value scale (Strategy ruling reports/RULING_KATAGO_VALUE_SCALE_BRIDGE_
20261119.md): the deep plan's Q arithmetic is the chassis' _terminal
scale, and terminal boards below are scored with mod._terminal directly
(never bridged).  When the acting checkpoint ('cur') is a trained one,
its value head outputs the plan's Score Utility U scale instead; then
--value-scale u applies the exact bridge B(u) = _terminal(U^-1(u))
(bridge.py) to the leaf values at exactly that consumption point.
Gen-0 / native checkpoints use the default --value-scale terminal and
pass through unbridged.  The choice is recorded in selfplay_config.json.

Self-play observation convention (documented divergence from the ladder):
positions are encoded with the TRUE carbon map and known = all ones,
exactly like the chassis' own _search rollouts (its `view()`), instead of
the belief-tracked estimate the ladder agent must use -- training sees the
same inputs the search evaluates.

Replay shards (.npz, one per worker flush) carry, per recorded position:
  x (54,15,15) uint8   feature planes, quantized exactly as the chassis
                       does (rint(x*100) clipped to [0,255]);
  sc (5,) float32      the critic scalars (_scalars);
  tw (225,5) float16   worker policy target;  tb (225,3) float16 base target;
  wmask/bmask (225,)   cells that hold an own worker / base (loss masks);
  margin float32       final cash margin from the recorded side's view;
  aux_delta (3,)       cash-margin change at +15/+40/+80 steps (clamped at
                       game end) -- the aux_short_delta target;
  carbon (225,) f16    the TRUE carbon map at this step (aux_fog target);
  win float32          1 / 0.5 / 0 from the final margin's sign;
  searched uint8       1 when the targets came from the deep plan.

--smoke runs a tiny numpy-only pass (no torch): the frozen chassis _Policy
drives every seat through a local backend, fast moves only, leaf values
unused.  It exists to prove the env/target plumbing on machines without
torch; it is not a data-generation mode.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge
import layout

DEEP_TAU = 1.0
DEEP_K = 24
DEEP_D2 = 6
# the chassis' own candidate temperature table (_S_GEN in master3s2.py)
GEN_TABLE = ((2.5, 8), (1.0, 3), (3.5, 8), (0.0, 6), (5.0, 8))
AUX_HORIZONS = (15, 40, 80)


# --------------------------------------------------------------------------
# Net backends: forward(tag, X, SC) -> (lw, lb, v) numpy arrays.
# --------------------------------------------------------------------------
class NumpyLocalClient:
    """Smoke-mode backend: the chassis module's numpy _Policy for every tag."""

    def __init__(self, mod):
        self.mod = mod

    def forward(self, tag, X, SC):
        lw = np.empty((len(X), 225, 5), np.float32)
        lb = np.empty((len(X), 225, 3), np.float32)
        for i in range(len(X)):
            lw[i], lb[i] = self.mod._POLICY.forward(X[i], self.mod._SID)
        # No critic without torch; _terminal of the current margin is the
        # documented smoke-mode leaf proxy (deep mode is disabled in smoke).
        v = np.array([self.mod._terminal(float(s[0]) * 3000.0) for s in SC],
                     dtype=np.float32)
        return lw, lb, v


def _server_main(req_q, resp_qs, cfg):
    """Inference server process: owns every net, serves batched forwards."""
    import torch
    import model as model_mod

    torch.set_num_threads(cfg['threads'])
    device = cfg['device']
    nets = {}
    src_policy, src_value, src_snames, src_nb = layout.load_source_payloads(cfg['src'])
    nets['m3s2'] = model_mod.MainNet.from_source(src_policy, src_value, src_nb)
    nets['cur'], cur_snames = model_mod.load_main_npz(cfg['ckpt'])
    for i, p in enumerate(cfg['hof']):
        nets['hof%d' % i], _ = model_mod.load_main_npz(p)
    sids = {}
    for tag, net in nets.items():
        names = cur_snames if tag == 'cur' else src_snames
        sids[tag] = names.index(layout.SID_NAME) if layout.SID_NAME in names else len(names)
        net.eval().to(device)
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(cfg['dtype'])
    while True:
        first = req_q.get()
        if first == 'STOP':
            return
        batch = [first]
        while len(batch) < 256:
            try:
                nxt = req_q.get_nowait()
            except Exception:
                break
            if nxt == 'STOP':
                for wid, rid, tag, X, SC in batch:
                    resp_qs[wid].put((rid, None))
                return
            batch.append(nxt)
        by_tag = {}
        for idx, (wid, rid, tag, X, SC) in enumerate(batch):
            by_tag.setdefault(tag, []).append(idx)
        results = [None] * len(batch)
        with torch.no_grad():
            for tag, idxs in by_tag.items():
                X = np.concatenate([batch[i][3] for i in idxs])
                SC = np.concatenate([batch[i][4] for i in idxs])
                xt = torch.from_numpy(X).to(device)
                st = torch.from_numpy(SC).to(device)
                if dtype is not None:
                    ctx = torch.autocast(device_type=xt.device.type, dtype=dtype)
                else:
                    import contextlib
                    ctx = contextlib.nullcontext()
                with ctx:
                    lw, lb, v, _ = nets[tag](xt, sids[tag], st)
                out = (lw.float().cpu().numpy(), lb.float().cpu().numpy(),
                       v.float().cpu().numpy())
                off = 0
                for i in idxs:
                    n = batch[i][3].shape[0]
                    results[i] = (out[0][off:off + n], out[1][off:off + n],
                                  out[2][off:off + n])
                    off += n
        for (wid, rid, tag, X, SC), res in zip(batch, results):
            resp_qs[wid].put((rid, res))


class ServerClient:
    """Worker-side handle to the inference server."""

    def __init__(self, wid, req_q, resp_q):
        self.wid = wid
        self.req_q = req_q
        self.resp_q = resp_q
        self.rid = 0

    def forward(self, tag, X, SC):
        self.rid += 1
        self.req_q.put((self.wid, self.rid, tag, np.ascontiguousarray(X, np.float32),
                        np.ascontiguousarray(SC, np.float32)))
        rid, res = self.resp_q.get()
        assert rid == self.rid
        if res is None:
            raise RuntimeError('inference server stopped')
        return res


# --------------------------------------------------------------------------
# Move generation
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


def fast_action(mod, client, tag, st, me, rng, prev):
    x = mod._quant(mod.encode(st.players, st.step, me, st.carbon,
                              [1.0] * mod.N, prev))
    sc = mod._scalars(st.players, st.step, me)
    lw, lb, _ = client.forward(tag, x[None], sc[None])
    act = sample(mod, st.players, me, lw[0], lb[0], rng, st.step, temp=1.0)
    return act, (x, sc, lw[0], lb[0])


def deep_move(mod, client, my_tag, opp_tag, st, me, rng, tau=DEEP_TAU,
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
    lw0, lb0, _ = client.forward(my_tag, x0[None], sc0[None])
    lwo, lbo, _ = client.forward(opp_tag, xo[None], mod._scalars(players, step, op)[None])
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
        lwm, lbm, _ = client.forward(my_tag, np.stack(Xm), np.stack(SCm))
        lwp, lbp, _ = client.forward(opp_tag, np.stack(Xo), np.stack(SCm))
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
        _, _, v = client.forward(my_tag, X, SC)
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
# Game loop (worker)
# --------------------------------------------------------------------------
def play_game(mod, client, game_id, seed, deep_frac, hof_tags, smoke=False,
              max_steps=None, value_scale='terminal'):
    """Play one game; return a list of per-position record dicts."""
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
                act, (tw, tb), (x, sc) = deep_move(
                    mod, client, 'cur', tags[1 - me], st, me, rng,
                    value_scale=value_scale)
                step_info[me] = (x, sc, tw, tb, 1)
            else:
                act, (x, sc, lw, lb) = fast_action(mod, client, tag, st, me, rng, prev)
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


def _worker_main(wid, game_ids, args, req_q, resp_q):
    mod = layout.load_module(args['src'], 'sp_src_%d' % wid)
    client = ServerClient(wid, req_q, resp_q)
    hof_tags = ['hof%d' % i for i in range(len(args['hof']))]
    buf, n_shards = [], 0
    for gid in game_ids:
        buf.extend(play_game(mod, client, gid, args['seed'] + 1000003 * gid,
                             args['deep_frac'], hof_tags,
                             value_scale=args.get('value_scale', 'terminal')))
        if len(buf) >= 1024:
            write_shard(os.path.join(args['out'], 'shard_w%d_%03d.npz' % (wid, n_shards)), buf)
            n_shards += 1
            buf = []
    if buf:
        write_shard(os.path.join(args['out'], 'shard_w%d_%03d.npz' % (wid, n_shards)), buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', default=layout.MASTER3S2)
    ap.add_argument('--ckpt', default=None, help='current-net checkpoint npz '
                    '(default: gen0_m3s2.npz next to the source payloads)')
    ap.add_argument('--hof', default='', help='comma-separated hall-of-fame npz paths')
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--deep-frac', type=float, default=0.25)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--value-scale', choices=['terminal', 'u'], default='terminal',
                    help='scale of the --ckpt value head: terminal = native '
                         'chassis _terminal scale (gen-0; leaf values pass '
                         'through unbridged); u = trained Score Utility scale '
                         '-- apply the exact bridge B(u) = _terminal(U^-1(u)) '
                         'to deep-plan leaf values (ruling '
                         'RULING_KATAGO_VALUE_SCALE_BRIDGE_20261119)')
    ap.add_argument('--smoke', action='store_true',
                    help='numpy-only plumbing check: 2 short games, fast moves only')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    hof = [p for p in args.hof.split(',') if p]

    if args.smoke:
        mod = layout.load_module(args.src, 'sp_smoke')
        client = NumpyLocalClient(mod)
        rows = []
        for gid in range(2):
            rows.extend(play_game(mod, client, gid, 500 + gid, 0.0, [],
                                  smoke=True, max_steps=48))
        write_shard(os.path.join(args.out, 'shard_smoke.npz'), rows)
        print('smoke: %d rows, x %s, tw %s' % (len(rows), rows[0]['x'].shape,
                                               rows[0]['tw'].shape))
        return 0

    if args.ckpt is None:
        raise SystemExit('--ckpt is required outside --smoke')
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    req_q = ctx.Queue()
    resp_qs = [ctx.Queue() for _ in range(args.workers)]
    cfg = {'src': args.src, 'ckpt': args.ckpt, 'hof': hof,
           'device': args.device, 'dtype': args.dtype, 'threads': args.threads}
    server = ctx.Process(target=_server_main, args=(req_q, resp_qs, cfg))
    server.start()
    argd = {'src': args.src, 'hof': hof, 'out': args.out, 'seed': args.seed,
            'deep_frac': args.deep_frac, 'value_scale': args.value_scale}
    procs = []
    worker_ids = []
    for wid in range(args.workers):
        ids = list(range(wid, args.games, args.workers))
        worker_ids.append(ids)
        p = ctx.Process(target=_worker_main, args=(wid, ids, argd, req_q, resp_qs[wid]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    req_q.put('STOP')
    server.join()
    # Worker failure must propagate: a worker that exited nonzero, or that
    # had games assigned but produced no shard, fails the whole run --
    # never report 'selfplay done' over dead workers.  (Stage-1 smoke,
    # 2026-11-19: all 8 workers died on a NaN and the driver still printed
    # 'selfplay done'.  run_loop.sh runs under set -e, so this exit code
    # is what stops the generation loop.)
    failed = []
    for wid, p in enumerate(procs):
        n_shards = len(glob.glob(os.path.join(args.out, 'shard_w%d_*.npz' % wid)))
        if p.exitcode != 0 or (worker_ids[wid] and n_shards == 0):
            failed.append('worker %d (exit %s, %d games, %d shards)'
                          % (wid, p.exitcode, len(worker_ids[wid]), n_shards))
    if server.exitcode != 0:
        failed.append('inference server (exit %s)' % (server.exitcode,))
    if failed:
        print('SELFPLAY FAILED: %s' % '; '.join(failed), file=sys.stderr)
        return 1
    with open(os.path.join(args.out, 'selfplay_config.json'), 'w') as f:
        json.dump({'games': args.games, 'seed': args.seed,
                   'deep_frac': args.deep_frac, 'ckpt': args.ckpt, 'hof': hof,
                   'value_scale': args.value_scale,
                   'deep': {'K': DEEP_K, 'D2': DEEP_D2, 'tau': DEEP_TAU}}, f, indent=1)
    print('selfplay done: %d games -> %s' % (args.games, args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
