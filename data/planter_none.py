"""planter_none W.npz [W.npz ...] -- on dataset v10's validation games: recall / precision of the planter NONE
class (= stay -> plant or seize) and overall planter accuracy, per weights file."""
import os, sys, json, numpy as np
os.environ.setdefault("BC_DEVICE", "mps")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import torch, cnet2_train as TR
from cnet2_model import CNet2
D, is_val, meta = TR.load('../data/cnet2_ds_v10')
va = np.where(is_val)[0]; rs = np.random.RandomState(0); va = np.sort(rs.choice(va, min(30000, len(va)), replace=False))
for wp in sys.argv[1:]:
    w = np.load(wp); cfg = json.loads(str(w['meta']))['cfg']
    net = CNet2(**cfg).to(TR.DEV); net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != 'meta'}); net.eval()
    tp = fn = fp = cor = tot = 0
    with torch.no_grad():
        for b0 in range(0, len(va), 1024):
            b = TR.to_dev(TR.fetch(D, va[b0:b0 + 1024]), False)
            lg, _, _ = net(b)
            pred = lg.argmax(-1)[:, :10].cpu().numpy(); y = b['y'][:, :10].cpu().numpy(); k = b['tok_kind'][:, :10].cpu().numpy()
            pl = (k == 1) & (y >= 0)
            tp += int(((pred == 0) & (y == 0) & pl).sum()); fn += int(((pred != 0) & (y == 0) & pl).sum())
            fp += int(((pred == 0) & (y != 0) & pl).sum()); cor += int(((pred == y) & pl).sum()); tot += int(pl.sum())
    print(f"{os.path.basename(wp)}: planter NONE recall {tp/max(1,tp+fn):.3f} precision {tp/max(1,tp+fp):.3f} | NONE share {(tp+fn)/max(1,tot):.3f} | planter acc {cor/max(1,tot):.3f} (n {tot})")
