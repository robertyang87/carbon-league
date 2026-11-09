"""diag_open DS W.npz [TEACHER] -- per step range x unit type agreement on one teacher's validation rows.
Planters split by the teacher's label: NONE (stop: plants on a free cell / seizes) vs a move; for NONE also the net's
precision (when the net says NONE, does the teacher?). Question (2026-11-09): is the opening planter imitation weak?"""
import json, os, sys
import numpy as np, torch
os.environ.setdefault("BC_DEVICE", "cuda")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import cnet2_train as T
from cnet2_model import CNet2
D, isv, _ = T.load(sys.argv[1]); TCH = int(sys.argv[3]) if len(sys.argv) > 3 else 5
w = np.load(sys.argv[2]); net = CNet2(**json.loads(str(w["meta"]))["cfg"]).to(T.DEV)
net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"}); net.eval()
idx = np.nonzero(isv)[0]; idx = idx[np.asarray(D["teacher"])[idx] == TCH]
stp = np.asarray(D["step"])[idx]
buckets = [("0-50", 0, 50), ("51-100", 51, 100), ("101-240", 101, 240), ("241-300", 241, 300)]
acc = {b[0]: {k: [0, 0] for k in ("collector", "planter", "planter NONE (recall)", "planter move", "planter NONE (precision)", "base", "base recruit planter (recall)")} for b in buckets}
with torch.no_grad():
    for name, a, b in buckets:
        sel = idx[(stp >= a) & (stp <= b)]
        for o in range(0, len(sel), 1024):
            bt = T.fetch(D, sel[o:o + 1024]); lg, _, _ = net(T.to_dev({k: bt[k] for k in T.LOAD}, False))
            pred = lg.argmax(-1).cpu().numpy(); y = bt["y"]; S = y.shape[1]; pred = pred[:, :S]
            kind = bt["tok_kind"][:, :S]; m = (y >= 0) & bt["tok_mask"][:, :S].astype(bool)
            col = np.arange(S)[None, :] < 10
            def add(k, mask, hit): acc[name][k][0] += int((hit & mask).sum()); acc[name][k][1] += int(mask.sum())
            ok = pred == y
            add("collector", m & col & (kind == 0), ok); pl = m & col & (kind == 1)
            add("planter", pl, ok); add("planter NONE (recall)", pl & (y == 0), ok); add("planter move", pl & (y > 0), ok)
            add("planter NONE (precision)", pl & (pred == 0), y == 0)
            add("base", m & ~col, ok); add("base recruit planter (recall)", m & ~col & (y == 2), ok)
print(f"{os.path.basename(sys.argv[2])}  teacher slot {TCH} validation rows, agreement % (n)")
for name, _, _ in buckets:
    print(f"  steps {name:8s} " + "  ".join(f"{k} {100 * h / max(1, n):.1f} ({n})" for k, (h, n) in acc[name].items()))
