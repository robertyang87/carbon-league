"""compile_bench DS W.npz -- MEDIAN training-step time over 120 post-compile steps and equivalence of torch.compile (default / reduce-overhead)
against eager, on real fixed-size batches (bs 512, drop-last). Equivalence: argmax agreement of the logits on legal
moves, max |dlogit|, and the max relative gradient difference over parameters (ignoring near-zero grads)."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import cnet2_train as T
from cnet2_model import CNet2
if os.environ.get("BENCH_NO_TF32") == "1":       # strict fp32 everywhere, to tell precision from logic
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
EQ_ONLY = os.environ.get("BENCH_EQ_ONLY") == "1"
w = np.load(sys.argv[2]); cfg = json.loads(str(w["meta"]))["cfg"]
D, _, _ = T.load(sys.argv[1]); rng = np.random.default_rng(0)
bs = [T.to_dev(T.fetch(D, np.sort(rng.integers(0, len(D["y"]), 512))), True) for _ in range(14)]


def fresh():
    net = CNet2(**cfg).to(T.DEV); net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"})
    return net


STEPS = int(os.environ.get("BENCH_STEPS", "120"))


def run(net, f, n_warm=6):
    """median per-step time over STEPS steps AFTER warm-up/compile (fixed shape: the batches are all 512)."""
    opt = torch.optim.AdamW(net.parameters(), lr=1e-6); ts = []
    for i in range(n_warm + STEPS):
        b = bs[i % len(bs)]
        torch.cuda.synchronize(); t = time.time()
        lg, v, a = f(b); y = b["y"]; m = y >= 0
        loss = torch.nn.functional.cross_entropy(lg[m].float(), y[m])
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        if i >= n_warm:
            ts.append(1000 * (time.time() - t))
    return float(np.median(ts))


def grads(net, f, b):
    net.zero_grad(); lg, v, a = f(b); y = b["y"]; m = y >= 0
    torch.nn.functional.cross_entropy(lg[m].float(), y[m]).backward()
    return lg.detach(), {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}


ref = fresh()
if not EQ_ONLY:
    print(f"eager: {run(ref, ref):.1f} ms/step", flush=True)
for mode in ("default", "reduce-overhead"):
    try:
        net = fresh(); t0 = time.time(); cf = torch.compile(net, mode=mode)
        ms = float("nan") if EQ_ONLY else run(net, cf, n_warm=6)
        e = fresh(); c = fresh(); cc = torch.compile(c, mode=mode)
        l1, g1 = grads(e, e, bs[0]); l2, g2 = grads(c, cc, bs[0])
        ok = l1 > -1e8
        agree = float((l1.argmax(-1) == l2.argmax(-1)).float().mean())
        rr = sorted(((float((g1[k] - g2[k]).abs().max() / g1[k].abs().max()), k, float(g1[k].abs().max())) for k in g1
                     if g1[k].abs().max() > 1e-6), reverse=True)
        rel = rr[0][0]; worst = f"{rr[0][1]} (|g|max {rr[0][2]:.1e}); 2nd {rr[1][1]} {rr[1][0]:.1e}"
        print(f"compile {mode}: {ms:.1f} ms/step (compile+warmup {time.time() - t0:.0f}s) | argmax agree {agree:.4f} "
              f"| max |dlogit| {float((l1 - l2)[ok].abs().max()):.2e} | max rel grad diff {rel:.2e} at {worst}", flush=True)
    except Exception as ex:
        print(f"compile {mode}: FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)
