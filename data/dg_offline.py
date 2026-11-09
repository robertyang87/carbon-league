"""dg_offline DAGGER_DIR W.npz [W.npz ...] -- on the held-out 10% of DAgger shards (crc % 10 == 0), agreement of each
net with the labeler's planter labels: on RELABELLED slots (uw == 3, where we originally disagreed) and on the rest."""
import glob, json, os, sys, zlib, numpy as np
os.environ.setdefault("BC_DEVICE", "mps")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import torch, cnet2_train as TR
from cnet2_model import CNet2
dg = sys.argv[1]
val = [s for s in sorted(glob.glob(os.path.join(dg, 'shards', '*.npz'))) if zlib.crc32(os.path.basename(s).encode()) % 10 == 0]
print('val shards', len(val))
for wp in sys.argv[2:]:
    w = np.load(wp); cfg = json.loads(str(w['meta']))['cfg']
    net = CNet2(**cfg).to(TR.DEV); net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != 'meta'}); net.eval()
    a_ch = n_ch = a_ok = n_ok = 0
    with torch.no_grad():
        for s in val:
            D = np.load(s); uw = D['uw'].astype(np.float32)[:, :10]
            b = TR.to_dev({k: D[k] for k in TR.LOAD}, False)
            lg, _, _ = net(b); pred = lg.argmax(-1)[:, :10].cpu().numpy(); y = D['y'][:, :10]
            ch = uw >= 2.5; ok = (uw > 0.5) & (uw < 2.5)
            a_ch += int(((pred == y) & ch).sum()); n_ch += int(ch.sum()); a_ok += int(((pred == y) & ok).sum()); n_ok += int(ok.sum())
    print(f"{os.path.basename(wp)}: relabelled planter decisions agree {100*a_ch/max(1,n_ch):.1f}% (n {n_ch}) | other planter decisions {100*a_ok/max(1,n_ok):.1f}% (n {n_ok})")
