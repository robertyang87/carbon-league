"""cnet2_tta_eval -- val accuracy with symmetry ensembling (teacher-forced), torch CPU/MPS.
Usage: python cnet2_tta_eval.py WEIGHTS.npz DATA_DIR [max_samples]"""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cnet2_train as TR
import cnet2_feat as F
from cnet2_model import CNet2


def main(wp, data, maxn=6000):
    w = dict(np.load(wp)); meta = json.loads(str(w["meta"]))
    net = CNet2(**meta["cfg"]).to(TR.DEV); net.load_state_dict({k: torch.from_numpy(v) for k, v in w.items() if k != "meta"}); net.eval()
    D, isv, _ = TR.load(data)
    va = np.nonzero(isv)[0]
    va = va[np.random.RandomState(0).permutation(len(va))[:int(maxn)]]
    ok = {k: np.zeros(3) for k in (1, 2, 4)}; n = np.zeros(3)
    for i in range(0, len(va), 512):
        b = TR.fetch(D, va[i:i + 512])
        outs = []
        for g in range(4):
            t = TR.to_dev(b, aug=False)
            if g:
                B = len(b["y"]); cm = TR.SYM_CELL[g].to(TR.DEV); mm = TR.SYM_MOVE[g].to(TR.DEV)
                t["maps"] = t["maps"].reshape(B, F.P, F.N)[:, :, cm].reshape(B, F.P, F.S, F.S)
                t["tok_pos"] = cm[t["tok_pos"]]
                hc = t["hist_cell"]; t["hist_cell"] = torch.where(hc >= 0, cm[hc.clamp(min=0)], hc)
                mm6 = torch.cat([mm, torch.tensor([5], device=TR.DEV)])
                t["hist_move"] = mm6[t["hist_move"]]
                t["mf"] = torch.cat([t["mf"][:, :10][:, :, mm], t["mf"][:, 10:]], 1)
                y = t["y"]; t["y"] = torch.cat([torch.where(y[:, :10] >= 0, mm[y[:, :10].clamp(min=0)], y[:, :10]), y[:, 10:]], 1)
            with torch.no_grad():
                lg, _, _ = net(t)
            lg = torch.log_softmax(lg.float(), -1)
            if g:
                lg = torch.cat([lg[:, :10][:, :, mm], lg[:, 10:]], 1)
            outs.append(lg)
        t0 = TR.to_dev(b, aug=False); y = t0["y"]; kind = t0["tok_kind"][:, :F.OWN]
        for k in (1, 2, 4):
            pred = torch.stack(outs[:k]).mean(0).argmax(-1)
            for c in range(3):
                m = (y >= 0) & (kind == c)
                ok[k][c] += ((pred == y) & m).sum().item()
                if k == 1:
                    n[c] += m.sum().item()
    for k in (1, 2, 4):
        print(json.dumps({"tta": k, "collector": round(ok[k][0] / n[0], 4), "planter": round(ok[k][1] / n[1], 4),
                          "base": round(ok[k][2] / n[2], 4), "all": round(ok[k].sum() / n.sum(), 4)}))


if __name__ == "__main__":
    main(*sys.argv[1:])
