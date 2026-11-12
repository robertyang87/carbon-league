"""cnet2_add_ta IN.npz OUT.npz TA [TNH] -- add TA cell-attention layers (cnet2_model.CellAttn) to a trained cnet2 net.
The new layers start as the identity (zero-initialised output projections), so OUT plays exactly like IN; training
(imitation, distillation or rl2_ppo) then decides what they learn. Prints the largest logit difference on random input.
  python3 cnet2_add_ta.py ../data/weights/cnet2_m3w3L.npz ../data/weights/cnet2_m3w3L_ta4.npz 4"""
import json
import sys

import numpy as np
import torch

from cnet2_model import CNet2

src, dst, ta = sys.argv[1], sys.argv[2], int(sys.argv[3]); tnh = int(sys.argv[4]) if len(sys.argv) > 4 else 4
w = np.load(src); cfg = dict(json.loads(str(w["meta"]))["cfg"])
if cfg.get("TA"):
    raise SystemExit(f"{src} already has TA={cfg['TA']}")
old = CNet2(**cfg); old.load_state_dict({k: torch.from_numpy(w[k].astype(np.float32)) for k in w.files if k != "meta"})
cfg2 = dict(cfg, TA=ta, TNH=tnh)
torch.manual_seed(0)
net = CNet2(**cfg2)
mk = net.load_state_dict(old.state_dict(), strict=False)
assert all(k.startswith("tattn.") for k in mk.missing_keys) and not mk.unexpected_keys, (mk.missing_keys[:4], mk.unexpected_keys[:4])
sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
np.savez(dst, **sd, meta=np.array(json.dumps({"cfg": cfg2, "from": src})))
x = torch.rand(4, old.stem.in_channels, 15, 15)
with torch.no_grad():
    h0 = torch.relu(old.stem_n(old.stem(__import__("cnet2_model").wrap(x)))); h1 = h0.clone()
    for i, b in enumerate(old.blocks):
        h0 = b(h0)
    for i, b in enumerate(net.blocks):
        h1 = b(h1)
        if i in net._ta:
            h1 = net.tattn[net._ta[i]](h1)
print(json.dumps({"params": [sum(p.numel() for p in old.parameters()), sum(p.numel() for p in net.parameters())],
                  "ta_after_blocks": sorted(net._ta), "trunk_max_diff": float((h0 - h1).abs().max())}))
