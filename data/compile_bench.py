"""compile_bench DS W.npz -- training-step time and equivalence of torch.compile (default / reduce-overhead)
against eager, on real fixed-size batches (bs 512, drop-last). Equivalence: argmax agreement of the logits on legal
moves, max |dlogit|, and the max relative gradient difference over parameters (ignoring near-zero grads)."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import cnet2_train as T
from cnet2_model import CNet2
w = np.load(sys.argv[2]); cfg = json.loads(str(w["meta"]))["cfg"]
D, _, _ = T.load(sys.argv[1]); rng = np.random.default_rng(0)
bs = [T.to_dev(T.fetch(D, np.sort(rng.integers(0, len(D["y"]), 512))), True) for _ in range(14)]


def fresh():
    net = CNet2(**cfg).to(T.DEV); net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"})
    return net


def run(net, f, n_warm=4):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-6)
    for i, b in enumerate(bs):
        if i == n_warm:
            torch.cuda.synchronize(); t = time.time()
        lg, v, a = f(b); y = b["y"]; m = y >= 0
        loss = torch.nn.functional.cross_entropy(lg[m].float(), y[m])
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return 1000 * (time.time() - t) / (len(bs) - n_warm)


def grads(net, f, b):
    net.zero_grad(); lg, v, a = f(b); y = b["y"]; m = y >= 0
    torch.nn.functional.cross_entropy(lg[m].float(), y[m]).backward()
    return lg.detach(), {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}


ref = fresh(); print(f"eager: {run(ref, ref):.1f} ms/step", flush=True)
for mode in ("default", "reduce-overhead"):
    try:
        net = fresh(); t0 = time.time(); cf = torch.compile(net, mode=mode)
        ms = run(net, cf, n_warm=6)
        e = fresh(); c = fresh(); cc = torch.compile(c, mode=mode)
        l1, g1 = grads(e, e, bs[0]); l2, g2 = grads(c, cc, bs[0])
        ok = l1 > -1e8
        agree = float((l1.argmax(-1) == l2.argmax(-1)).float().mean())
        rel = max(float((g1[k] - g2[k]).abs().max() / g1[k].abs().max()) for k in g1 if g1[k].abs().max() > 1e-6)
        print(f"compile {mode}: {ms:.1f} ms/step (compile+warmup {time.time() - t0:.0f}s) | argmax agree {agree:.4f} "
              f"| max |dlogit| {float((l1 - l2)[ok].abs().max()):.2e} | max rel grad diff {rel:.2e}", flush=True)
    except Exception as ex:
        print(f"compile {mode}: FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)
