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


def _serve(adapter_name, device, req_q, resp_qs, stat_q):
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
    while True:
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
            m = models.pop(key, None)
            if m is None:
                m = ad.make_model(AD.load_npz(key)).to(dev).eval()
            models[key] = m                                   # most recently used last
            while len(models) > keep:
                models.pop(next(iter(models)))
            t0 = time.time()
            try:
                d = {k: np.concatenate([it[3][k] for it in its]) for k in its[0][3]}
                tok, ctx = ad.server_encode(m, d, teacher, dev, amp)
                o = 0
                for it in its:
                    k = len(it[3]["maps"])
                    resp_qs[it[0]].put((tok[o:o + k], ctx[o:o + k])); o += k
            except Exception as e:                            # the workers must not hang on a server error
                for it in its:
                    resp_qs[it[0]].put(RuntimeError(f"rl2_ppo_server: {type(e).__name__}: {e}"))
            t_fwd += time.time() - t0; n_fwd += 1; n_req += len(its); n_pos += sum(len(it[3]["maps"]) for it in its)
        if time.time() - t_last > 30 and stat_q is not None:
            try:
                stat_q.put_nowait({"req": n_req, "pos": n_pos, "fwd": n_fwd, "t_fwd": round(t_fwd, 2),
                                   "pos_per_fwd": round(n_pos / max(1, n_fwd), 1), "models": len(models)})
            except queue.Full:
                pass
            n_req = n_pos = n_fwd = 0; t_fwd = 0.0; t_last = time.time()


class Client:
    def __init__(self, wid, req_q, resp_q):
        self.wid, self.req_q, self.resp_q = wid, req_q, resp_q

    def encode(self, key, d, teacher):
        self.req_q.put((self.wid, key, int(teacher), d))
        r = self.resp_q.get(timeout=float(os.environ.get("RL2_SERVER_TIMEOUT", "600")))
        if isinstance(r, Exception):
            raise r
        return r


class Server:
    def __init__(self, proc, req_q, stat_q, initargs):
        self.proc, self.req_q, self.stat_q, self.initargs = proc, req_q, stat_q, initargs

    def stats(self):
        out = None
        while True:
            try:
                out = self.stat_q.get_nowait()
            except queue.Empty:
                return out

    def stop(self):
        self.req_q.put(None)
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()


def start(ctx, adapter_name, device, n_workers):
    req_q = ctx.Queue(); stat_q = ctx.Queue(maxsize=4)
    resp_qs = [ctx.Queue() for _ in range(n_workers)]
    ids = ctx.Queue()
    for i in range(n_workers):
        ids.put(i)
    p = ctx.Process(target=_serve, args=(adapter_name, device, req_q, resp_qs, stat_q), daemon=True)
    p.start()
    return Server(p, req_q, stat_q, (ids, req_q, resp_qs))
