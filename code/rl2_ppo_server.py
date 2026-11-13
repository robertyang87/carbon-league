"""rl2_ppo_server -- one inference process for all rollout workers of rl2_ppo.

Why: a rollout worker spent ~80% of its time in the NumPy convolutions of the trunk (measured: 20 of 26 ms per
position for the 128x8 net). Here the workers send the stacked inputs of all positions of one step to this process,
which runs the trunk of everything that is waiting as one batch on the trainer's device and sends back the per-token
and context vectors; the cheap autoregressive per-unit decode stays in the worker.

    server = start(ctx, adapter_name, device_str, n_workers)   # trainer side; server.initargs -> Pool initializer
    rl2_ppo_rollout._init_worker(*server.initargs)             # in each worker: installs the Client
    Client.encode(weights_path, stacked_inputs, teacher) -> tok, ctx

Models are addressed by their weights path (live/w_<it>.npz, snapshots); the server loads each once and keeps the
most recent RL2_SERVER_MODELS (default 12). Env: RL2_SERVER_MAXB (positions per forward, default 4096),
RL2_SERVER_AMP=1 (fp16 autocast on cuda; off by default so the recorded log-probs match the trainer's fp32 forward).
"""

import os
import queue
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _serve(adapter_name, device, req_q, resp_qs, stat_q, sid=0):
    import torch
    import rl2_ppo_adapters as AD
    dev = torch.device(device)
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("RL2_SERVER_THREADS", "4")))
    ad = AD.get_adapter(adapter_name)
    maxb = int(os.environ.get("RL2_SERVER_MAXB", "4096"))
    keep = int(os.environ.get("RL2_SERVER_MODELS", "12"))
    amp = os.environ.get("RL2_SERVER_AMP", "0") == "1"
    models = {}
    n_req = n_pos = n_fwd = 0; t_fwd = 0.0; t_last = time.time()
    # All workers write into ONE pipe whose buffer (64 KB) is smaller than a single request (~9 KB per position), so a
    # request only gets through while somebody is reading. Reading only between two forwards (get_nowait) let one or
    # two requests in per forward: with several models the batches fell to ~15 positions and the server, not the
    # workers, set the pace (L20, 2026-11-11: 257 games/min with 4 models). A reader thread keeps the pipe drained
    # during the forwards; the main loop takes everything that has arrived.
    use_reader = os.environ.get("RL2_SERVER_READER", "1") == "1"
    if use_reader:
        import collections
        import threading
        inbox = collections.deque(); cv = threading.Condition()

        def _reader():
            while True:
                it = req_q.get()
                with cv:
                    inbox.append(it); cv.notify()
                if it is None:
                    return
        threading.Thread(target=_reader, daemon=True).start()
    while True:
        if use_reader:
            with cv:
                while not inbox:
                    cv.wait()
                batch = []; n = 0
                while inbox and n < maxb:
                    it = inbox.popleft()
                    if it is None:
                        return
                    batch.append(it); n += len(it[3]["maps"])
        else:
            item = req_q.get()
            if item is None:
                return
            batch = [item]; n = len(item[3]["maps"])
            while n < maxb:
                try:
                    item = req_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    return
                batch.append(item); n += len(item[3]["maps"])
        groups = {}
        for it in batch:
            groups.setdefault((it[1], it[2]), []).append(it)
        for (key, teacher), its in groups.items():
            t0 = time.time()
            try:
                # Model loading and cache update are inside the per-group try
                # (REVIEW_FULL_20261113 P3): a missing/corrupt weights file
                # used to kill _serve before any response was queued, leaving
                # every worker of the group waiting for the full client
                # timeout. Now each rid of the group gets the error response
                # below, with the original exception type and message, and the
                # failed model is not cached (it is only inserted after a
                # successful load).
                m = models.pop(key, None)
                if m is None:
                    m = ad.make_model(AD.load_npz(key)).to(dev).eval()
                models[key] = m                               # most recently used last
                while len(models) > keep:
                    models.pop(next(iter(models)))
                d = {k: np.concatenate([it[3][k] for it in its]) for k in its[0][3]}
                tok, ctx = ad.server_encode(m, d, teacher, dev, amp)
                o = 0
                for it in its:
                    k = len(it[3]["maps"])
                    resp_qs[it[0]].put((it[4], tok[o:o + k], ctx[o:o + k])); o += k
            except Exception as e:                            # the workers must not hang on a server error
                for it in its:
                    resp_qs[it[0]].put((it[4], RuntimeError(f"rl2_ppo_server: {type(e).__name__}: {e}"), None))
            t_fwd += time.time() - t0; n_fwd += 1; n_req += len(its); n_pos += sum(len(it[3]["maps"]) for it in its)
        if time.time() - t_last > 30 and stat_q is not None:
            try:
                stat_q.put_nowait({"sid": sid, "req": n_req, "pos": n_pos, "fwd": n_fwd, "t_fwd": round(t_fwd, 2),
                                   "pos_per_fwd": round(n_pos / max(1, n_fwd), 1), "models": len(models)})
            except queue.Full:
                pass
            n_req = n_pos = n_fwd = 0; t_fwd = 0.0; t_last = time.time()


class Client:
    """submit() returns at once, collect() waits for that answer: a worker sends the requests of all models of one step
    first and only then waits, so they sit in the same server batch instead of one round trip after the other."""

    def __init__(self, wid, req_q, resp_q):
        self.wid, self.req_q, self.resp_q = wid, req_q, resp_q
        self.rid = 0; self.got = {}

    def submit(self, key, d, teacher):
        self.rid += 1
        self.req_q.put((self.wid, key, int(teacher), d, self.rid))
        return self.rid

    def collect(self, rid):
        while rid not in self.got:
            r = self.resp_q.get(timeout=float(os.environ.get("RL2_SERVER_TIMEOUT", "600")))
            self.got[r[0]] = r[1:]
        tok, ctx = self.got.pop(rid)
        if isinstance(tok, Exception):
            raise tok
        return tok, ctx

    def encode(self, key, d, teacher):
        return self.collect(self.submit(key, d, teacher))


class Server:
    def __init__(self, procs, stat_q, initargs, queues):
        self.procs, self.stat_q, self.initargs, self.queues = procs, stat_q, initargs, queues
        self.last = {}

    def stats(self):
        """latest 30-second window of every server process, summed (pos_per_fwd recomputed)."""
        while True:
            try:
                d = self.stat_q.get_nowait()
            except queue.Empty:
                break
            self.last[d.get("sid", 0)] = d
        if not self.last:
            return None
        out = {k: round(sum(d[k] for d in self.last.values()), 2) for k in ("req", "pos", "fwd", "t_fwd")}
        out["pos_per_fwd"] = round(out["pos"] / max(1, out["fwd"]), 1)
        out["models"] = max(d["models"] for d in self.last.values()); out["servers"] = len(self.procs)
        return out

    def stop(self):
        """terminate, do not wait: rollout workers killed by pool.terminate() can die holding the request queue's
        write lock, and then neither a sentinel nor the queue's feeder thread ever gets through (the trainer used to
        hang at interpreter exit in Queue._finalize_join)."""
        for q in self.queues:
            try:
                q.cancel_join_thread()
            except Exception:
                pass
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.join(timeout=5)


def start(ctx, adapter_name, device, n_workers):
    """RL2_SERVERS (default 1) inference processes on the same device, workers spread over them (worker id modulo):
    one process is limited by its Python loop (L20: ~40 ms per forward of 137 positions with the GPU far from full),
    several run that loop in parallel."""
    k = max(1, int(os.environ.get("RL2_SERVERS", "1")))
    req_qs = [ctx.Queue() for _ in range(k)]; stat_q = ctx.Queue(maxsize=4 * k)
    resp_qs = [ctx.Queue() for _ in range(n_workers)]
    ids = ctx.Queue()
    for i in range(n_workers):
        ids.put(i)
    procs = []
    for j in range(k):
        p = ctx.Process(target=_serve, args=(adapter_name, device, req_qs[j], resp_qs, stat_q, j), daemon=True)
        p.start(); procs.append(p)
    return Server(procs, stat_q, (ids, req_qs, resp_qs), req_qs + [stat_q, ids] + resp_qs)
