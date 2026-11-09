"""split_val DS W.npz [W.npz ...] -- validation agreement (all units, unweighted like the trainer's evaluate) split
into: old rows (teacher != SLOT) before step 240 / from step 240 on, and the new teacher's rows (teacher == SLOT).
Reading: old rows within 0.1 pt of the reference = no forgetting; old rows from 240 on dropping more than before 240
= the new endgame style leaking into the other style vectors; SLOT rows should climb epoch by epoch."""
import json, os, sys
import numpy as np, torch
os.environ.setdefault("BC_DEVICE", "cuda")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import cnet2_train as T
from cnet2_model import CNet2
SLOT = int(os.environ.get("SPLIT_SLOT", "2"))
D, isv, _ = T.load(sys.argv[1])
idx = np.nonzero(isv)[0]
tch = np.asarray(D["teacher"])[idx]; stp = np.asarray(D["step"])[idx]
groups = {"old <240": (tch != SLOT) & (stp < 240), "old >=240": (tch != SLOT) & (stp >= 240), f"slot{SLOT}": tch == SLOT}
print("rows:", {k: int(v.sum()) for k, v in groups.items()})
for wp in sys.argv[2:]:
    w = np.load(wp); cfg = json.loads(str(w["meta"]))["cfg"]
    net = CNet2(**cfg).to(T.DEV); net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != "meta"}); net.eval()
    res = {}
    for name, m in groups.items():
        sel = idx[m]; hit = tot = 0
        with torch.no_grad():
            for a in range(0, len(sel), 1024):
                b = T.fetch(D, sel[a:a + 1024]); lg, _, _ = net(T.to_dev({k: b[k] for k in T.LOAD}, False))
                pred = lg.argmax(-1).cpu().numpy(); y = b["y"]; ok = (y >= 0) & b["tok_mask"][:, :y.shape[1]].astype(bool)
                hit += int(((pred[:, :y.shape[1]] == y) & ok).sum()); tot += int(ok.sum())
        res[name] = round(100 * hit / max(1, tot), 2)
        if name == f"slot{SLOT}":     # base decisions are ~95% "none": recall on the recruits is what matters
            rec = {1: [0, 0], 2: [0, 0]}
            with torch.no_grad():
                for a in range(0, len(sel), 1024):
                    b = T.fetch(D, sel[a:a + 1024]); lg, _, _ = net(T.to_dev({k: b[k] for k in T.LOAD}, False))
                    pb = lg.argmax(-1).cpu().numpy()[:, 10:12]; yb = b["y"][:, 10:12]
                    for c in (1, 2):
                        m_ = yb == c; rec[c][0] += int((pb[m_] == c).sum()); rec[c][1] += int(m_.sum())
            res["base recall collector"] = f"{rec[1][0]}/{rec[1][1]}"; res["base recall planter"] = f"{rec[2][0]}/{rec[2][1]}"
    print(os.path.basename(wp), res, flush=True)
