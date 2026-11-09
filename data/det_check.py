"""det_check DS W.npz -- is eager itself run-to-run reproducible on this GPU? Two eager backward passes on the same
batch (strict fp32, no TF32), with and without deterministic algorithms; then compiled (fullgraph, dynamic=False)
vs eager under the deterministic setting. Prints the max relative gradient difference (params with |g|max > 1e-6)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
import cnet2_train as T
from cnet2_model import CNet2
w = np.load(sys.argv[2]); cfg = json.loads(str(w["meta"]))["cfg"]
D, _, _ = T.load(sys.argv[1]); b = T.to_dev(T.fetch(D, np.sort(np.random.default_rng(0).integers(0, len(D["y"]), 512))), False)


def grads(f, net):
    net.zero_grad(); lg, v, a = f(b); y = b["y"]; m = y >= 0
    torch.nn.functional.cross_entropy(lg[m].float(), y[m]).backward()
    return {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}


def worst(g1, g2):
    """per parameter: ||dg|| / ||g|| (the reading that matters) and max|dg| / max|g| (single worst element)."""
    nr = sorted(((float((g1[k] - g2[k]).norm() / g1[k].norm()), k) for k in g1 if g1[k].norm() > 1e-8), reverse=True)
    mr = sorted(((float((g1[k] - g2[k]).abs().max() / g1[k].abs().max()), k) for k in g1 if g1[k].abs().max() > 1e-6), reverse=True)
    tot = float(sum((g1[k] - g2[k]).norm() ** 2 for k in g1) ** 0.5 / sum(g1[k].norm() ** 2 for k in g1) ** 0.5)
    return f"norm ratio: worst param {nr[0][0]:.1e} ({nr[0][1]}), all params {tot:.1e} | max-element ratio {mr[0][0]:.1e} ({mr[0][1]})"


net = CNet2(**cfg).to(T.DEV); net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"})
print("eager vs eager (default algos):      ", worst(grads(net, net), grads(net, net)), flush=True)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
print("eager vs eager (deterministic cudnn):", worst(grads(net, net), grads(net, net)), flush=True)
cf = torch.compile(net, fullgraph=True, dynamic=False)
ge = grads(net, net); gc = grads(cf, net)
print("compiled(fullgraph) vs eager (det):  ", worst(ge, gc), flush=True)
