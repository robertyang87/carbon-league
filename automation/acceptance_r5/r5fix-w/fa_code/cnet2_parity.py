"""cnet2_parity -- check cnet2_np logits == torch logits (teacher-forced) on replay steps.
Usage: python cnet2_parity.py WEIGHTS.npz [replay.json.gz]"""
import gzip, json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("BC_DEVICE", "cpu")
import cnet2_feat as F
import cnet2_np as NP
from cnet2_model import CNet2
import cnet2_train as TR

def main(wp, rp):
    w = dict(np.load(wp)); meta = json.loads(str(w["meta"]))
    net = CNet2(**meta["cfg"]); net.load_state_dict({k: torch.from_numpy(v) for k, v in w.items() if k != "meta"}); net.eval()
    pol = NP.Policy(w)
    st = json.load(gzip.open(rp))["steps"]
    worst = 0.0; wv = 0.0
    for p in (0, 1):
        f = F.Feat(p)
        for t in range(0, 298):
            o0 = st[t][0]["observation"]
            obs = {"step": o0["step"], "players": o0["players"], "carbon": st[t][p]["observation"]["carbon"], "player": p}
            x = f.observe(obs)
            if t % 23:
                continue
            y = f.labels(st[t + 1][p].get("action") or {})
            pf = F.Prefix(x, f.aux)
            uf = np.zeros((F.OWN, F.UF), np.float32); mf = np.zeros((F.OWN, 5, F.MF), np.float32); lg = np.zeros((F.OWN, 5), bool)
            for s in range(F.OWN):
                if x["tok_mask"][s]:
                    uf[s], mf[s], lg[s] = pf.feats(s)
                    if not lg[s, y[s]]: y[s] = 0
                    pf.advance(s, int(y[s]))
            b = {k: x[k][None] for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move")}
            b.update(uf=uf[None].astype(np.float16), mf=mf[None].astype(np.float16), legal=lg[None], y=y[None],
                     carbon=np.zeros((1, F.N), np.uint8), value=np.zeros(1, np.float16), teacher=np.array([1]))
            for k in ("scal", "tok_cargo", "hist_cargo"):
                b[k] = b[k].astype(np.float16)
            tb = TR.to_dev(b, aug=False)
            with torch.no_grad():
                tl, tv, _ = net(tb)
            tl = tl[0].numpy(); tv = float(tv[0])
            f2 = F.Feat(p)  # numpy path uses a fresh prefix from the same x / aux
            lab, nl, nv = pol.act(x, f.aux, teacher=1, forced=y)
            m = x["tok_mask"][:F.OWN]
            ok = (nl > -1e8) & (tl > -1e8)
            d = np.abs(nl[m][ok[m]] - tl[m][ok[m]]).max() if ok[m].any() else 0
            worst = max(worst, d); wv = max(wv, abs(nv - tv))
    print(json.dumps({"max_logit_diff": float(worst), "max_value_diff": wv}))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "../data/ladder_top/replays/0b9ac0018dfaef15bc531c296332293c-1.json.gz"))
