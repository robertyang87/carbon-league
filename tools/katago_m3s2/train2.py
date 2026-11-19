"""train2: parallel-feed trainer, drop-in parallel of train.py (bypass piece).

EQUIVALENCE CONTRACT (hard constraint, gate_train2.py enforces it):
  * The batch sequence is bit-identical to train.py under the same seed:
    the index/RNG stream lives ONLY in the main process and is consumed in
    exactly train.py's order -- per epoch one rng.permutation(n), then,
    batch by batch in order, one rng.randint(4) per row in idx order.
    train.py interleaves those randint calls with training; train2
    pre-generates the whole epoch plan up front. That is sequence-identical
    because nothing else in train.py consumes the numpy RNG stream during
    training (torch has its own stream, seeded separately), and the
    permutation -> randint consumption order is unchanged.
  * Parallelism exists ONLY in materialization (index -> tensor): worker
    processes apply augment() with the pre-drawn g and stack/cast/scale
    exactly as train.py's inline code did. Workers draw NO random numbers.
  * Results return through a prefetch window and are consumed strictly in
    step order (reorder buffer), so worker completion order never leaks
    into the training order.
  * All recipe defaults and training math are verbatim from train.py:
    batch 512, AdamW lr 1.5e-4 cosine -> 3e-5, weight_decay 0.01,
    epochs 3, fast-policy-weight 0.0, loss weights 1.5 / 0.3 / 0.2 / 0.2,
    BF16 autocast, checkpoint/log formats unchanged.
  * torch.compile is opt-in (--compile, default OFF; the equivalence gate
    runs with it OFF) and wraps only the training forward handle; export
    still goes through the uncompiled module.

If the equivalence gate fails, this file is ABANDONED (no fix loop).
Source basis: tools/katago_m3s2/train.py md5 34898e7c0132b1834404bff01f0cde21
(read verbatim from repo main, 2026-11-20). Functions marked VERBATIM are
character-level ports of train.py; the only deliberate changes are in
feeding (plan generation, worker materialization, pinned async copy).

FIX (review item 2.7, 2026-11-20): OrderedFeeder no longer blocks
forever on a hard-killed worker. Every pooled .get() carries a timeout
(--feed-timeout, default 600 s -- see DEFAULT_FEED_TIMEOUT below) and a
timeout raises FeederTimeoutError naming the stuck step, the waited
time and per-worker liveness, so a dead worker surfaces as an explicit
failure instead of a silent hang. Nothing in the equivalence contract
changes: plan generation, the RNG stream and materialization are
untouched.
"""

import argparse
import glob
import math
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout

SIZE = 15
N = 225
NPL = 54
DR = (0, -1, 0, 1, 0)
DC = (0, 0, 1, 0, -1)


# --------------------------------------------------------------------------
# VERBATIM from train.py: V4 tables, augment, score_utility, storage, loader.
# --------------------------------------------------------------------------

def _v4_coords(r, c, idx):
    if idx == 0:
        return r, c
    if idx == 1:
        return SIZE - 1 - r, SIZE - 1 - c
    if idx == 2:
        return r, SIZE - 1 - c
    return SIZE - 1 - r, c


def build_v4_tables():
    """Identical construction to the fix2 chassis' V4 tables."""
    fwd_cell = np.zeros((4, N), dtype=np.intp)
    inv_cell = np.zeros((4, N), dtype=np.intp)
    fwd_act = np.zeros((4, 5), dtype=np.intp)
    inv_act = np.zeros((4, 5), dtype=np.intp)
    plane_perm = np.zeros((4, NPL), dtype=np.intp)
    for g in range(4):
        for i in range(N):
            r, c = divmod(i, SIZE)
            r2, c2 = _v4_coords(r, c, g)
            j = r2 * SIZE + c2
            fwd_cell[g, i] = j
            inv_cell[g, j] = i
        r0, c0 = _v4_coords(7, 7, g)
        ia = [0] * 5
        for a in range(5):
            r1, c1 = _v4_coords(7 + DR[a], 7 + DC[a], g)
            dr, dc = r1 - r0, c1 - c0
            for a2 in range(5):
                if DR[a2] == dr and DC[a2] == dc:
                    fwd_act[g, a] = a2
                    ia[a2] = a
                    break
        inv_act[g] = ia
        perm = np.arange(NPL, dtype=np.intp)
        for an in range(5):
            perm[25 + an] = 25 + ia[an]
            perm[30 + an] = 30 + ia[an]
        plane_perm[g] = perm
    return fwd_cell, inv_cell, fwd_act, inv_act, plane_perm


V4 = build_v4_tables()


def augment(row, g):
    """Apply V4 element g to one replay row (dict of numpy arrays)."""
    if g == 0:
        return row
    _, inv_cell, _, inv_act, plane_perm = V4
    out = dict(row)
    x = row['x'].reshape(NPL, N)
    out['x'] = x[:, inv_cell[g]][plane_perm[g]].reshape(NPL, SIZE, SIZE)
    out['tw'] = row['tw'][inv_cell[g]][:, inv_act[g]]
    out['tb'] = row['tb'][inv_cell[g]]
    out['wmask'] = row['wmask'][inv_cell[g]]
    out['bmask'] = row['bmask'][inv_cell[g]]
    out['carbon'] = row['carbon'][inv_cell[g]]
    return out


def score_utility(d):
    """U(d) = 0.6*tanh(d/600) + 0.4*clip(d/2500, -1, 1), elementwise."""
    return 0.6 * np.tanh(d / 600.0) + 0.4 * np.clip(d / 2500.0, -1.0, 1.0)


STORE_DTYPES = {
    'x': np.float16,
    'sc': np.float32,
    'tw': np.float16,
    'tb': np.float16,
    'wmask': np.uint8,
    'bmask': np.uint8,
    'carbon': np.float16,
    'margin': np.float32,
    'aux_delta': np.float32,
    'win': np.float32,
    'searched': np.uint8,
}


def _rss_gb():
    """Current RSS in GB from /proc (nan where unavailable)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float('nan')


class ReplayData:
    """The whole replay window as stacked arrays (verbatim semantics)."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.n = int(arrays['x'].shape[0])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: a[i] for k, a in self.arrays.items()}


def load_dataset(dirs):
    """Two-pass stacked load (verbatim from train.py).

    Row order is exactly the old loader's (dirs in order, shards sorted,
    rows in shard order).
    """
    per_dir = []
    total = 0
    keys = None
    row_shapes = None
    shard_dtypes = None
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, '*.npz')))
        counts = []
        for path in files:
            with np.load(path) as z:
                if keys is None:
                    keys = list(z.files)
                    row_shapes = {k: z[k].shape[1:] for k in keys}
                    shard_dtypes = {k: z[k].dtype for k in keys}
                counts.append(int(z['x'].shape[0]))
        per_dir.append((d, files, counts))
        total += sum(counts)
    if not total:
        raise SystemExit('no replay shards found in %s' % (dirs,))
    arrays = {}
    for k in keys:
        dt = STORE_DTYPES.get(k, shard_dtypes[k])
        arrays[k] = np.empty((total,) + row_shapes[k], dtype=dt)
    off = 0
    nfiles = 0
    for d, files, counts in per_dir:
        for path, n in zip(files, counts):
            with np.load(path) as z:
                for k in keys:
                    arrays[k][off:off + n] = z[k]
            off += n
            nfiles += 1
        print('load: dir %s done, %d/%d rows, rss %.1f GB'
              % (d, off, total, _rss_gb()), flush=True)
    print('loaded %d rows from %d shards' % (total, nfiles))
    return ReplayData(arrays)


# --------------------------------------------------------------------------
# Index/RNG plan: the equivalence-critical piece.
# --------------------------------------------------------------------------

def build_epoch_plan(rng, n_rows, batch):
    """One epoch of (idx, gs) in EXACTLY train.py's RNG consumption order.

    train.py per epoch: order = rng.permutation(n); then for each batch
    slice, for each row index in slice order, int(rng.randint(4)).
    This function performs the identical calls in the identical order and
    returns [(idx_array, gs_array), ...]. gs are the augment draws.
    """
    order = rng.permutation(n_rows)
    plan = []
    for s0 in range(0, len(order), batch):
        idx = order[s0:s0 + batch]
        gs = np.array([int(rng.randint(4)) for _ in idx], dtype=np.int64)
        plan.append((idx, gs))
    return plan


def generate_index_sequence(n_rows, batch, epochs, seed, max_steps=None):
    """Pure index/RNG sequence (no data, no torch) for the gate.

    Returns a list over steps of (idx_tuple, gs_tuple), consuming the RNG
    exactly as a full training run would, epoch after epoch.
    """
    rng = np.random.RandomState(seed)
    seq = []
    for _ in range(epochs):
        for idx, gs in build_epoch_plan(rng, n_rows, batch):
            seq.append((tuple(int(v) for v in idx),
                        tuple(int(v) for v in gs)))
            if max_steps is not None and len(seq) >= max_steps:
                return seq
    return seq


# --------------------------------------------------------------------------
# Materialization: the ONLY parallelized segment. No RNG anywhere below.
# --------------------------------------------------------------------------

def materialize_numpy(arrays, idx, gs, fast_policy_weight):
    """idx+gs -> the exact float32 numpy batch train.py built inline.

    Every stack/astype/scale below mirrors train.py's inline lines:
      x /100, sc plain, tw/tb/wm/bm plain, vt = score_utility(margin),
      aux_delta /1000, carbon /100, win plain,
      pw = searched + (1 - searched) * fast_policy_weight.
    Returns a dict of numpy arrays (torch conversion happens in main).
    """
    def row(i):
        return {k: a[i] for k, a in arrays.items()}

    batch = [augment(row(int(i)), int(g)) for i, g in zip(idx, gs)]
    x = np.stack([b['x'] for b in batch]).astype(np.float32) / 100.0
    sc = np.stack([b['sc'] for b in batch]).astype(np.float32)
    tw = np.stack([b['tw'] for b in batch]).astype(np.float32)
    tb = np.stack([b['tb'] for b in batch]).astype(np.float32)
    wm = np.stack([b['wmask'] for b in batch]).astype(np.float32)
    bm = np.stack([b['bmask'] for b in batch]).astype(np.float32)
    margin = np.stack([b['margin'] for b in batch]).astype(np.float32)
    vt = score_utility(margin).astype(np.float32)
    dt_ = np.stack([b['aux_delta'] for b in batch]).astype(np.float32) / 1000.0
    fog = np.stack([b['carbon'] for b in batch]).astype(np.float32) / 100.0
    win = np.stack([b['win'] for b in batch]).astype(np.float32)
    searched = np.stack([b['searched'] for b in batch]).astype(np.float32)
    pw = searched + (1 - searched) * fast_policy_weight
    return {'x': x, 'sc': sc, 'tw': tw, 'tb': tb, 'wm': wm, 'bm': bm,
            'vt': vt, 'dt': dt_, 'fog': fog, 'win': win,
            'pw': np.ascontiguousarray(pw, dtype=np.float32)}


# Worker side: dataset arrays arrive via fork inheritance (globals set in
# the parent before Pool creation); tasks carry only (step, idx, gs).
_W_ARRAYS = None
_W_FPW = 0.0


def _worker_init(arrays, fast_policy_weight):
    global _W_ARRAYS, _W_FPW
    _W_ARRAYS = arrays
    _W_FPW = fast_policy_weight
    # Workers must never contend for BLAS threads with each other/main.
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'


def _worker_task(task):
    step, idx, gs = task
    return step, materialize_numpy(_W_ARRAYS, idx, gs, _W_FPW)


DEFAULT_FEED_TIMEOUT = 600.0


class FeederTimeoutError(RuntimeError):
    """A pooled materialization step did not arrive within feed_timeout.

    Raised instead of blocking forever when a worker has been
    hard-killed (OOM killer / segfault): multiprocessing.Pool does not
    re-run a task whose worker died mid-flight, so without a timeout
    the consumer would hang silently (review item 2.7).
    """


class OrderedFeeder:
    """Prefetch window over a multiprocessing pool, strict in-order yield.

    Submits up to `prefetch` steps ahead; results are buffered by step
    number and yielded only in submission order. With workers <= 1 it
    degrades to inline materialization (same code path, no processes).
    """

    def __init__(self, arrays, fast_policy_weight, workers, prefetch,
                 feed_timeout=DEFAULT_FEED_TIMEOUT):
        if not (feed_timeout and float(feed_timeout) > 0):
            raise ValueError('feed_timeout must be > 0 seconds, got %r'
                             % (feed_timeout,))
        self.feed_timeout = float(feed_timeout)
        self.arrays = arrays
        self.fpw = fast_policy_weight
        self.workers = max(0, int(workers))
        self.prefetch = max(1, int(prefetch))
        self._pool = None
        if self.workers > 1:
            import multiprocessing as mp
            ctx = mp.get_context('fork')
            self._pool = ctx.Pool(processes=self.workers,
                                  initializer=_worker_init,
                                  initargs=(arrays, fast_policy_weight))

    def _worker_liveness(self):
        """Per-worker (pid, alive) snapshot for timeout diagnostics."""
        pool = self._pool
        if pool is None:
            return 'no pool (inline feeding)'
        workers = getattr(pool, '_pool', None)
        if not workers:
            return 'pool worker list unavailable'
        return ', '.join('pid=%s alive=%s' % (w.pid, w.is_alive())
                         for w in workers)

    def _get_step(self, pending, want):
        """pending[want].get() under the feed timeout; die naming the step."""
        ar = pending.pop(want)
        t0 = time.time()
        try:
            step, batch = ar.get(timeout=self.feed_timeout)
        except multiprocessing.TimeoutError:
            waited = time.time() - t0
            raise FeederTimeoutError(
                'OrderedFeeder stuck on step %d: no result after %.1fs '
                '(feed_timeout=%.1fs); a materialization worker was '
                'likely hard-killed (OOM/segfault) and the pool does '
                'not re-run its task. workers: %s'
                % (want, waited, self.feed_timeout,
                   self._worker_liveness()))
        return step, batch

    def run_epoch(self, plan):
        """Yield materialized batches in plan order for one epoch."""
        if self._pool is None:
            for idx, gs in plan:
                yield materialize_numpy(self.arrays, idx, gs, self.fpw)
            return
        pending = {}   # step -> AsyncResult
        results = {}   # step -> batch (arrived early, held for order)
        submitted = 0
        total = len(plan)
        for want in range(total):
            while submitted < total and submitted < want + self.prefetch:
                idx, gs = plan[submitted]
                pending[submitted] = self._pool.apply_async(
                    _worker_task, ((submitted, idx, gs),))
                submitted += 1
            if want in results:
                yield results.pop(want)
                continue
            step, batch = self._get_step(pending, want)
            assert step == want
            yield batch
            # Opportunistically drain anything already finished so the
            # reorder buffer stays small; order is still enforced above.
            for s in list(pending):
                if pending[s].ready():
                    st, bt = pending.pop(s).get(timeout=self.feed_timeout)
                    results[st] = bt

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None


def _to_device(np_batch, torch, device):
    """numpy batch -> pinned CPU tensors -> async copy to device.

    pin_memory is used only when the target is a CUDA device; on CPU the
    tensors are used directly (pinning would be a no-op copy). The copy
    itself uses non_blocking=True exactly as the brief requires.
    """
    pin = (device != 'cpu' and str(device).startswith('cuda'))
    out = {}
    for k, arr in np_batch.items():
        t = torch.from_numpy(arr)
        if pin:
            t = t.pin_memory()
        out[k] = t.to(device, non_blocking=True) if pin else t.to(device)
    return out


# --------------------------------------------------------------------------
# Training main: CLI and recipe VERBATIM from train.py, plus feed options.
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Plan stage 2: joint training on self-play replay shards.')
    ap.add_argument('--data', required=True,
                    help='comma-separated replay dirs (the last-3-generations window)')
    ap.add_argument('--init', required=True, help='init checkpoint npz (source layout)')
    ap.add_argument('--out', required=True, help='output checkpoint npz (source layout)')
    ap.add_argument('--out-pt', default=None, help='optional torch state_dict sidecar')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--lr-final', type=float, default=3e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--fast-policy-weight', type=float, default=0.0)
    ap.add_argument('--n-extra', type=int, default=0,
                    help='append this many zero-init blocks to a 6-block init')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    # --- new in train2: feeding only; defaults do not alter the recipe ---
    ap.add_argument('--feed-workers', type=int, default=8,
                    help='materialization worker processes (<=1 = inline)')
    ap.add_argument('--prefetch', type=int, default=4,
                    help='steps materialized ahead of consumption (in-order)')
    ap.add_argument('--feed-timeout', type=float,
                    default=DEFAULT_FEED_TIMEOUT,
                    help='max seconds to wait for one materialized step '
                         'before dying loudly, naming the step (worker '
                         'hard-death guard, review 2.7; default 600)')
    ap.add_argument('--compile', action='store_true',
                    help='torch.compile the training forward (default OFF; '
                         'keep OFF for equivalence-gate runs)')
    args = ap.parse_args()
    if not (args.feed_timeout and args.feed_timeout > 0):
        ap.error('--feed-timeout must be > 0 seconds, got %r'
                 % (args.feed_timeout,))

    import torch
    import torch.nn.functional as F
    import model as model_mod

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rows = load_dataset([d for d in args.data.split(',') if d])
    net, snames = model_mod.load_main_npz(args.init, n_extra=args.n_extra)
    net.to(args.device)
    sid = snames.index(layout.SID_NAME) if layout.SID_NAME in snames else len(snames)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(rows) / args.batch)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        t = min(1.0, step / max(1, total_steps))
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * t))

    # torch.compile wraps ONLY the training forward handle; `net` itself
    # (optimizer params, export) is untouched, so numerics/export paths
    # are the uncompiled ones whenever --compile is off (the default).
    train_net = torch.compile(net) if args.compile else net

    feeder = OrderedFeeder(rows.arrays, args.fast_policy_weight,
                           args.feed_workers, args.prefetch,
                           args.feed_timeout)
    step = 0
    t_train = time.time()
    try:
        for epoch in range(args.epochs):
            # RNG plan for the whole epoch, generated in the main process
            # in train.py's exact consumption order (see module docstring).
            plan = build_epoch_plan(rng, len(rows), args.batch)
            tot = {}
            nbatches = 0
            for np_batch in feeder.run_epoch(plan):
                b = _to_device(np_batch, torch, args.device)
                x, sc = b['x'], b['sc']
                tw, tb, wm, bm = b['tw'], b['tb'], b['wm'], b['bm']
                vt, dt_, fog, win, pw = (b['vt'], b['dt'], b['fog'],
                                         b['win'], b['pw'])
                for gp in opt.param_groups:
                    gp['lr'] = lr_at(step)
                with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                    lw, lb, v, aux = train_net(x, sid, sc, want_aux=True)
                    logw = F.log_softmax(lw.float(), dim=-1)
                    logb = F.log_softmax(lb.float(), dim=-1)
                    ce_w = -(tw * logw).sum(-1)                     # (n, 225)
                    ce_b = -(tb * logb).sum(-1)
                    lp_w = ((ce_w * wm).sum(1) / wm.sum(1).clamp(min=1) * pw).sum() / pw.sum().clamp(min=1)
                    lp_b = ((ce_b * bm).sum(1) / bm.sum(1).clamp(min=1) * pw).sum() / pw.sum().clamp(min=1)
                    l_v = F.mse_loss(v.float(), vt)
                    l_d = F.mse_loss(aux['delta'].float(), dt_)
                    l_f = F.mse_loss(aux['fog'].float(), fog.reshape(-1, SIZE, SIZE))
                    l_win = F.binary_cross_entropy_with_logits(aux['win'].float(), win)
                    loss = lp_w + lp_b + 1.5 * l_v + 0.3 * l_d + 0.2 * l_f + 0.2 * l_win
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                nbatches += 1
                for k, val in (('loss', loss), ('pw', lp_w), ('pb', lp_b), ('v', l_v),
                               ('delta', l_d), ('fog', l_f), ('win', l_win)):
                    tot[k] = tot.get(k, 0.0) + float(val)
                if step % 200 == 0:
                    el = time.time() - t_train
                    print('epoch %d step %d/%d: %.0fs elapsed, %.2f steps/s'
                          % (epoch, step, total_steps, el, step / max(el, 1e-9)),
                          flush=True)
            print('epoch %d: %s' % (epoch, {k: round(v / nbatches, 4) for k, v in tot.items()}))
    finally:
        feeder.close()

    out = net.export_source_dict(snames)
    with open(args.out, 'wb') as f:
        np.savez_compressed(f, **layout.to_f16(out))
    print('wrote %s' % args.out)
    if args.out_pt:
        torch.save(net.state_dict(), args.out_pt)
        print('wrote %s' % args.out_pt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
