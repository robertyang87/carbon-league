"""slot_val DS W.npz [W.npz ...] -- teacher validation games of DS (meta val), agreement of each net with the
teacher label PER STYLE SLOT (teacher id), split collector / planter / base / all units. Forgetting guard for DAgger
nets: DAgger only supervises planters on slots 5 / 7, every other head and slot lives on the teacher data alone."""
import json, os, sys, numpy as np
os.environ.setdefault("BC_DEVICE", "mps")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import torch, cnet2_train as TR
from cnet2_model import CNet2
D, is_val, meta = TR.load(sys.argv[1])
idx = np.nonzero(is_val)[0]
ex = os.environ.get("SV_EXCLUDE_NAME")          # e.g. 1C-Zero-v8-preview: drop rows of games that name played in
if ex:
    bad = np.array([ex in "".join(g.get("names", [])) for g in meta["games"]])
    idx = idx[~bad[np.asarray(D["gid"])[idx]]]
    print(f"excluding games with {ex}: {int(is_val.sum()) - len(idx)} val rows dropped", flush=True)
T = np.asarray(D["teacher"])[idx]
res = {}
for wp in sys.argv[2:]:
    w = np.load(wp); cfg = json.loads(str(w['meta']))['cfg']
    net = CNet2(**cfg).to(TR.DEV); net.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files if k != 'meta'}); net.eval()
    acc = {}
    with torch.no_grad():
        for a in range(0, len(idx), 1024):
            ii = idx[a:a + 1024]; b = TR.fetch(D, ii)
            lg, _, _ = net(TR.to_dev({k: b[k] for k in TR.LOAD}, False)); pred = lg.argmax(-1).cpu().numpy()[:, :12]
            y = b["y"]; n_own = y.shape[1]; ok = (y >= 0) & b["tok_mask"][:, :n_own].astype(bool)
            kind = np.where(np.arange(y.shape[1])[None] >= 10, 2, (b["tok_kind"][:, :n_own] == 1).astype(int))   # 0 col 1 pla 2 base
            for t in np.unique(b["teacher"]):
                r = (b["teacher"] == t)[:, None] & ok
                A = acc.setdefault(int(t), np.zeros((4, 2)))
                for k in range(3):
                    m = r & (kind == k); A[k] += [((pred == y) & m).sum(), m.sum()]
                A[3] += [((pred == y) & r).sum(), r.sum()]
    res[os.path.basename(wp)] = {t: [round(float(100 * v[0] / max(1, v[1])), 2) for v in A] for t, A in sorted(acc.items())}
    print(os.path.basename(wp), "slot: [collector, planter, base, all] %")
    for t, v in res[os.path.basename(wp)].items():
        print(f"  {t}: {v}  (n units {int(acc[t][3][1])})")
names = list(res)
if len(names) == 2:
    a, b = names; bad = [(t, round(res[b][t][3] - res[a][t][3], 2)) for t in res[a] if res[b][t][3] < res[a][t][3] - 1.0]
    heads = [(t, ["collector", "planter", "base"][k], round(res[b][t][k] - res[a][t][k], 2)) for t in res[a] for k in range(3)
             if res[b][t][k] < res[a][t][k] - 1.0]
    print(json.dumps({"compare": f"{b} vs {a}", "slots_down_over_1pt_all_units": bad, "GATE": "FAIL" if bad else "PASS",
                      "info_heads_down_over_1pt": heads}, ensure_ascii=False))
