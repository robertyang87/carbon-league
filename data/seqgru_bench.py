"""seqgru_bench -- per-step training time (forward + backward + AdamW step) of the slot loop vs forward_seq,
on real batches, on whatever BC_DEVICE says. Also re-checks equivalence of the logits on that device."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import cnet2_train as T
from cnet2_model import CNet2
w = np.load(sys.argv[2]); cfg = json.loads(str(w["meta"]))["cfg"]
D, isv, meta = T.load(sys.argv[1])
rng = np.random.default_rng(0)
bs = [T.to_dev(T.fetch(D, np.sort(rng.integers(0, len(D["y"]), 512))), True) for _ in range(13)]
for name, seq in (("loop", False), ("seq", True)):
    net = CNet2(**cfg).to(T.DEV); net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"})
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5); f = net.forward_seq if seq else net
    for i, b in enumerate(bs):
        if i == 3:
            torch.cuda.synchronize() if T.DEV.type == "cuda" else None; t = time.time()
        lg, v, a = f(b); loss = Fn = torch.nn.functional.cross_entropy(lg.reshape(-1, 5), b["y"].clamp(min=0).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize() if T.DEV.type == "cuda" else None
    print(f"{name}: {1000 * (time.time() - t) / 10:.1f} ms/step", flush=True)
net.eval()
with torch.no_grad():
    l1, _, _ = net(bs[0]); l2, _, _ = net.forward_seq(bs[0])
ok = l1 > -1e8
print("device", T.DEV, "max |dlogit|", float((l1 - l2)[ok].abs().max()), "argmax agree", float((l1.argmax(-1) == l2.argmax(-1)).float().mean()))
