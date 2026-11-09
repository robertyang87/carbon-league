"""compile_ab DS INIT.npz [STEPS] -- same-data A/B before a compiled training run: from the same initial weights,
the same batch order and the same augmentation seed, train STEPS steps eager and STEPS steps through
torch.compile(fullgraph, static), with the trainer's own losses/optimizer/schedule settings. Prints both loss curves
(every 25 steps), the max |loss_eager - loss_compiled|, and per-layer ||p_eager - p_compiled|| / ||p_eager - p_init||
after STEPS (how far the two runs drifted apart relative to how far training moved the weights).
Pass rule (2026-11-09): curves overlap and the weight drift ratio is ~1e-3 or below -> train dg6 compiled."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import cnet2_train as T
from cnet2_model import CNet2
ds, init = sys.argv[1], sys.argv[2]; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
w = np.load(init); cfg = json.loads(str(w["meta"]))["cfg"]
D, isv, _ = T.load(ds); tr = np.nonzero(~isv)[0]
order = np.random.default_rng(1234).permutation(tr)[: STEPS * 512]
bs, lr = 512, float(os.environ.get("BC_LR", "1.5e-4"))
if T.DEV.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = os.environ.get("BC_TF32", "1") == "1"


def train(compiled):
    torch.manual_seed(7)                                       # same augmentation draws in both runs
    net = CNet2(**cfg).to(T.DEV); net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"})
    p0 = {k: v.detach().clone() for k, v in net.named_parameters()}
    tnet = torch.compile(net, fullgraph=True, dynamic=False) if (compiled and T.DEV.type == "cuda") else net
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.05)
    curve = []; t0 = time.time()
    for i in range(STEPS):
        b = T.to_dev(T.fetch(D, order[i * bs:(i + 1) * bs]), True)
        lpi, lv, la, _ = T.losses(tnet, b); loss = lpi + 0.2 * lv + la
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(os.environ.get("BC_CLIP", "2.0")))
        opt.step(); sched.step(); curve.append(float(loss))
    return net, p0, curve, time.time() - t0


ne, p0, ce, te = train(False)
nc, _, cc, tc = train(True)
print("step  eager    compiled")
for i in range(0, STEPS, 25):
    print(f"{i:4d}  {ce[i]:.5f}  {cc[i]:.5f}")
print(f"max |loss diff| {max(abs(a - b) for a, b in zip(ce, cc)):.2e} | mean of last 50: eager {np.mean(ce[-50:]):.5f} compiled {np.mean(cc[-50:]):.5f}")
pe, pc = dict(ne.named_parameters()), dict(nc.named_parameters())
rr = sorted(((float((pe[k] - pc[k]).norm() / max((pe[k] - p0[k]).norm(), 1e-12)), k) for k in pe if (pe[k] - p0[k]).norm() > 1e-9), reverse=True)
tot = float(sum((pe[k] - pc[k]).norm() ** 2 for k in pe) ** 0.5 / sum((pe[k] - p0[k]).norm() ** 2 for k in pe) ** 0.5)
print(f"weight drift ||p_e - p_c|| / ||p_e - p_init||: all params {tot:.1e}, worst layer {rr[0][0]:.1e} ({rr[0][1]})")
print(f"wall: eager {te:.0f}s compiled (incl. compile) {tc:.0f}s")
