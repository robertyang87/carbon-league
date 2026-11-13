"""rl2_ppo_adapters -- policy adapters for the rl2_ppo PPO framework.

An adapter couples one network family to the generic rollout/trainer code.

Actor side (NumPy, runs inside CPU rollout workers, 1 BLAS thread):
    actor = adapter.make_actor(weights)          # weights: dict name -> np.ndarray
    st = actor.new_game(side)                    # per-game state (trackers, history)
    outs = actor.act(states, obss, rng, temp, greedy)
        batched over concurrent games; returns per game (env_actions, rec)
        rec: dict of NumPy arrays for that decision (adapter specific) plus
             rec["logp"] (float, SUM of per-unit log-probs of the joint
             action under softmax(logits/temp) with legal masks) and
             rec["value"] (float, actor-side value estimate).

Trainer side (torch, MPS/CPU):
    model = adapter.make_model(weights)          # nn.Module
    adapter.export(model) -> weights dict (NumPy, actor-loadable)
    buf = adapter.build_buffer(trajs)            # trajs: list of list-of-recs
    batch = buf.batch(idx, device)               # idx: decision indices
    out = adapter.evaluate(model, batch, temp)   # dict with
        lp  [U, A]  per-unit log-probs over a padded action set (illegal = -1e9-ish,
                    already log-softmaxed over legal entries; temp applied),
        act [U]     taken action per unit, seg [U] decision index in batch (0..B-1),
        legal [U, A] bool, ukey [U] (long) stable unit key in [0, buf.n_ukeys) so
                    per-unit quantities (reference log-probs) can be cached,
        value [B]   value prediction per decision.
    For an autoregressive policy `evaluate` must teacher-force the stored
    joint action so lp[u] is the conditional distribution given earlier units.

Joint log-prob of a decision = segment-sum over its units of lp[u, act[u]].
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nnfeat as F  # noqa: E402

S, N = 15, 225


def _rc(i):
    return i // S, i % S


# EGO_IDX[c, k]: flat source cell so that ego(G, c).reshape(C,225)[:, k] == G[:, EGO_IDX[c, k]]
_r = np.arange(N) // S
_c = np.arange(N) % S
EGO_IDX = np.stack([((_r + (ci // S) - 7) % S) * S + (_c + (ci % S) - 7) % S for ci in range(N)]).astype(np.int64)


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def save_npz(weights, path, meta):
    sd = {k: np.asarray(v, np.float32) for k, v in weights.items() if k != "meta"}
    m = {}
    if "meta" in weights:                      # keep model meta (e.g. cnet2 cfg), add ours
        try:
            m = json.loads(str(weights["meta"]))
        except Exception:
            m = {}
    m.update(meta)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **sd, meta=np.array(json.dumps(m)))
    os.replace(tmp, path)


# =====================================================================
# Small egocentric CNN (nn_train_bc.Net / nn_agent), independent per unit
# =====================================================================
class SmallNetActor:
    """Workers sample their move from softmax(logits[0:5]/T) in a fixed order
    (obs order); a move onto a cell already claimed by an earlier own unit is
    masked (sequential legal masking -> the stored masks make the joint
    log-prob exactly recomputable). Bases use the nn_agent rule recruit gate
    (not a policy sample, excluded from the log-prob): the BC head almost
    never recruits on its own."""

    def __init__(self, weights):
        import nn_agent as A
        self.A = A
        self.pol = A.Policy(weights)
        self.vw = np.asarray(weights["v.weight"], np.float32)
        self.vb = np.asarray(weights["v.bias"], np.float32)

    def new_game(self, side):
        return {"tr": F.Tracker(side), "side": side}

    def _forward(self, x, s):
        p = self.pol.sd; wm = self.pol.wm; conv = self.A.conv
        x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        h = np.maximum(conv(x, wm["stem.weight"], p["stem.bias"]), 0)
        for i in range(self.pol.nb):
            t = np.maximum(conv(h, wm[f"blocks.{i}.0.weight"], p[f"blocks.{i}.0.bias"]), 0)
            h = np.maximum(h + conv(t, wm[f"blocks.{i}.1.weight"], p[f"blocks.{i}.1.bias"]), 0)
        patch = h[:, 6:9, 6:9, :].transpose(0, 3, 1, 2).reshape(h.shape[0], -1)
        g = h.mean(axis=(1, 2))
        z = np.maximum(np.concatenate([patch, g, s], axis=1) @ p["fc.weight"].T + p["fc.bias"], 0)
        return z @ p["pi.weight"].T + p["pi.bias"], np.tanh(z @ self.vw.T + self.vb)[:, 0]

    def act(self, states, obss, rng, temp, greedy):
        per, xs, ss = [], [], []
        for st, obs in zip(states, obss):
            tr = st["tr"]
            tr.update(obs)
            G, scal, info = tr.global_planes(obs)
            me = obs["player"]
            mine = obs["players"][me]
            units = []
            for wid, w in mine[2].items():
                k = 0 if w[2] == "COLLECTOR" else 1
                x, s = F.unit_inputs(G, scal, w[0], k, w[1])
                units.append((wid, k, w[0], w[1])); xs.append(x); ss.append(s)
            for bid, b in mine[1].items():
                x, s = F.unit_inputs(G, scal, b, 2, 0.0)
                units.append((bid, 2, b, 0.0)); xs.append(x); ss.append(s)
            per.append((G, scal, info, units))
        if xs:
            lg_all, v_all = self._forward(np.stack(xs).astype(np.float32), np.stack(ss).astype(np.float32))
        outs = []
        off = 0
        for (G, scal, info, units), obs in zip(per, obss):
            n = len(units)
            lg = lg_all[off:off + n] if n else np.zeros((0, 8), np.float32)
            v = v_all[off:off + n] if n else np.zeros(0, np.float32)
            off += n
            me = obs["player"]
            mine = obs["players"][me]
            acts, claimed = {}, set()
            act_arr = np.zeros(n, np.int8)
            mask = np.zeros((n, 8), np.uint8)
            pol = np.zeros(n, np.uint8)
            logp = 0.0
            lpu = np.zeros(n, np.float32)
            for i, (uid, k, pos, cargo) in enumerate(units):
                if k == 2:
                    continue
                tg = [F.nb(pos, F.MOVES[a]) for a in range(5)]
                m = np.array([t not in claimed for t in tg], bool)
                if not m.any():
                    m[:] = True
                z = np.where(m, lg[i, 0:5] / temp, -1e9)
                z = z - z.max(); pr = np.exp(z); pr /= pr.sum()
                a = int(np.argmax(pr)) if greedy else int(rng.choice(5, p=pr))
                lpu[i] = float(np.log(max(pr[a], 1e-12)))
                logp += float(lpu[i])
                claimed.add(tg[a])
                if a:
                    acts[uid] = F.MOVES[a]
                act_arr[i] = a; mask[i, 0:5] = m; pol[i] = 1
            cash = info["cash"]; nworkers = len(mine[2])
            n_pl = sum(1 for w in mine[2].values() if w[2] != "COLLECTOR")
            for i, (uid, k, pos, cargo) in enumerate(units):
                if k != 2:
                    continue
                a = self.A.recruit_choice(lg[i, 5:8], cash, nworkers, 10, n_pl, info["tree_price"],
                                          obs["step"], pos in claimed)
                act_arr[i] = 5 + a
                if a:
                    acts[uid] = F.BASE_ACTS[a]
                    cash -= 30; nworkers += 1; claimed.add(pos); n_pl += (a == 2)
            rec = {"G": G.astype(np.float32), "scal": scal.astype(np.float32),
                   "center": np.array([u[2] for u in units], np.int16),
                   "kind": np.array([u[1] for u in units], np.int8),
                   "cargo": np.array([u[3] for u in units], np.float32),
                   "act": act_arr, "mask": mask, "pol": pol, "lpu": lpu,
                   "logp": logp, "value": float(v.mean()) if n else 0.0}
            outs.append((acts, rec))
        return outs


class SmallNetBuffer:
    def __init__(self, trajs):
        recs = [r for t in trajs for r in t]
        self.n = len(recs)
        self.G = np.stack([r["G"] for r in recs])                    # [D,14,225] f32
        self.scal = np.stack([r["scal"] for r in recs])
        nu = np.array([len(r["kind"]) for r in recs], np.int64)
        self.uoff = np.concatenate([[0], np.cumsum(nu)])
        cat = lambda k: np.concatenate([r[k] for r in recs]) if recs else np.zeros(0)
        self.center = cat("center").astype(np.int64)
        self.kind = cat("kind").astype(np.int64)
        self.cargo = cat("cargo").astype(np.float32)
        self.act = cat("act").astype(np.int64)
        self.mask = np.concatenate([r["mask"] for r in recs]).astype(bool)
        self.pol = cat("pol").astype(bool)
        self.n_ukeys = len(self.act)
        self.lp0u = cat("lpu").astype(np.float32)        # behaviour per-unit log-prob, indexed by ukey

    def nbytes(self):
        return sum(getattr(self, k).nbytes for k in ("G", "scal", "center", "kind", "cargo", "act", "mask", "pol"))

    def batch(self, idx, device):
        import torch
        idx = np.asarray(idx)
        us = [np.arange(self.uoff[d], self.uoff[d + 1]) for d in idx]
        seg = np.concatenate([np.full(len(u), j) for j, u in enumerate(us)]).astype(np.int64)
        u = np.concatenate(us)
        keep = self.pol[u]                         # only policy-sampled units enter the loss
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        return {"G": t(self.G[idx].astype(np.float32)), "scal": t(self.scal[idx]), "B": len(idx),
                "seg": t(seg), "center": t(self.center[u]), "kind": t(self.kind[u]),
                "cargo": t(self.cargo[u]), "act": t(self.act[u]), "mask": t(self.mask[u]),
                "keep": t(keep), "ukey": t(u[keep])}


class SmallNetAdapter:
    name = "small"
    n_act = 5

    def make_actor(self, weights):
        return SmallNetActor(weights)

    def make_model(self, weights):
        import torch
        import nn_train_bc as TB
        net = TB.Net()
        net.load_state_dict({k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in weights.items() if k != "meta"})
        return net

    def export(self, model):
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}

    def build_buffer(self, trajs):
        return SmallNetBuffer(trajs)

    def evaluate(self, model, b, temp):
        import torch
        dev = b["G"].device
        ego = torch.from_numpy(EGO_IDX).to(dev)
        # all units of the batch (bases included for the value), per-unit egocentric inputs
        gi = ego[b["center"]]                                         # [U,225]
        Gu = b["G"][b["seg"]]                                         # [U,14,225]
        x = torch.gather(Gu, 2, gi[:, None, :].expand(-1, Gu.shape[1], -1)).reshape(-1, Gu.shape[1], S, S)
        s = b["scal"][b["seg"]].clone()
        s[:, 15] = (b["kind"] == 1).float()
        s[:, 16] = torch.clamp(b["cargo"], max=400.0) / 200.0
        s[:, 17] = (b["kind"] == 2).float()
        logits, v = model(x, s)
        B = b["B"]
        vsum = torch.zeros(B, device=dev).index_add_(0, b["seg"], v)
        cnt = torch.zeros(B, device=dev).index_add_(0, b["seg"], torch.ones_like(v))
        value = vsum / cnt.clamp(min=1)
        k = b["keep"]
        z = logits[k, 0:5] / temp
        z = torch.where(b["mask"][k, 0:5], z, torch.full_like(z, -1e9))
        lp = torch.log_softmax(z, dim=1)
        return {"lp": lp, "act": b["act"][k], "seg": b["seg"][k], "value": value,
                "legal": b["mask"][k, 0:5], "ukey": b["ukey"]}


def get_adapter(name):
    if name == "small":
        return SmallNetAdapter()
    if name == "cnet2":
        import rl2_ppo_cnet2
        return rl2_ppo_cnet2.Cnet2Adapter()
    raise ValueError(name)
