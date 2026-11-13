"""rl2_ppo_cnet2 -- rl2_ppo adapter for the cnet2 policy (cnet2_feat / cnet2_np /
cnet2_model, read-only imports; those files are owned by the BC line).

Actor: cnet2_feat.Feat(seat).observe(obs) -> x, aux; cnet2_np.Policy.encode;
autoregressive GRU decode over own slots 0..11 (workers 0-9, bases 10-11)
with cnet2_feat.Prefix -- the same loop as cnet2_np.Policy.act, but sampling
from softmax(logits / T) and recording, per slot, the prefix-dependent inputs
(uf, mf, legal) so the torch model can teacher-force the exact joint action.
No rule layer: legality is cnet2_feat.Prefix's mask (recruits need cash >= 30
and a free slot; worker moves are always legal, collisions are learned).

Trainer: cnet2_model.CNet2 forward(b) (teacher-forced with b["y"]) -> masked
logits [B,12,5], value [B] (tanh; BC target tanh(margin/3000), so use
--rscale 3.33e-4 for a calibrated value start).

Env: CNET2_TEACHER (teacher-embedding id used at play; default 1 = 1C-Zero-v9).
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cnet2_feat as F  # noqa: E402

NEG = np.float32(-1e9)
TEACHER = int(os.environ.get("CNET2_TEACHER", "1"))
# CNET2_TEMP_KIND="c,p,b": sampling temperature per unit kind (collector, planter, base) for the LEARNER only --
# it replaces the trainer's scalar --temp when the learner actor samples and when the learner recomputes the
# log-probs (evaluate below; same env in both). R04: it must NOT leak into snapshot / opponent actors, which
# always play at the temperature they are handed (RL2_SNAP_TEMP); Cnet2Actor.temp_kind marks the learner.
_TK = os.environ.get("CNET2_TEMP_KIND")
TEMP_KIND = [float(v) for v in _TK.split(",")] if _TK else None
# CNET2_TEACHER_LATE / CNET2_LATE_STEP: style row from that step on (Grove1's pack plays row 2, then row 7 from step 240);
# used by the actor (learner and snapshots; "#T" opponents keep their one row) and by the learner's forward alike.
_TL = os.environ.get("CNET2_TEACHER_LATE")
TEACHER_LATE = int(_TL) if _TL else None
LATE_STEP = int(os.environ.get("CNET2_LATE_STEP", "240"))
XKEYS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo", "hist_cell", "hist_cargo", "hist_move")


class Cnet2Actor:
    def __init__(self, weights):
        import cnet2_np as NP
        self.pol = NP.Policy(weights)
        self.remote = None; self.key = None; self.teacher = TEACHER; self.sched = True
        # R04: False = snapshot/opponent actor: _decode uses the temperature it is passed.
        # The rollout worker flags the task's primary (learner) actor True; only it may use TEMP_KIND.
        self.temp_kind = False

    def _t(self, x):
        if self.sched and TEACHER_LATE is not None and int(round(float(x["scal"][1]) * 300)) >= LATE_STEP:
            return TEACHER_LATE
        return self.teacher

    def new_game(self, side):
        return {"f": F.Feat(side), "side": side}

    def _decode(self, x, aux, rng, temp, greedy, enc=None):
        P = self.pol; w = P.w
        tok, ctx = P.encode(x, self._t(x)) if enc is None else enc
        value = math.tanh(float(ctx @ w["value.weightT"][:, 0] + w["value.bias"][0]))
        h = P.lin(ctx, "h0")
        prev = 8
        pf = F.Prefix(x, aux)
        y = np.full(F.OWN, -1, np.int8)
        UF = np.zeros((F.OWN, F.UF), np.float16)
        MF = np.zeros((F.OWN, 5, F.MF), np.float16)
        LG = np.zeros((F.OWN, 5), bool)
        logp = 0.0
        lpu = np.zeros(F.OWN, np.float32)
        for s in range(F.OWN):
            if not x["tok_mask"][s]:
                continue
            uf, mf, legal = pf.feats(s)
            UF[s] = uf; MF[s] = mf; LG[s] = legal
            inp = P.lin(np.concatenate([tok[s], ctx, uf, mf.reshape(-1)]), "din") + w["aemb.weight"][prev]
            h = P.gru(inp, h)
            base = P.lin(h, "whead") if s < 10 else np.concatenate([P.lin(h, "bhead"), np.zeros(2, np.float32)])
            lg = base + P.lin(mf.astype(np.float32), "V") @ P.lin(h, "U")
            # R04: TEMP_KIND applies to the learner instance only; any other actor uses the passed temp.
            ts = temp if (TEMP_KIND is None or not getattr(self, "temp_kind", False)) else TEMP_KIND[2 if s >= 10 else int(x["tok_kind"][s])]
            z = np.where(legal, lg / ts, -np.inf)
            z = z - z.max(); p = np.exp(z); p /= p.sum()
            a = int(np.argmax(p)) if greedy else int(rng.choice(5, p=p))
            lpu[s] = float(np.log(max(p[a], 1e-12)))
            logp += float(lpu[s])
            y[s] = a
            pf.advance(s, a)
            prev = a + (0 if s < 10 else 5)
        return y, UF, MF, LG, logp, value, lpu

    @staticmethod
    def pack(recs):
        return PackedTraj(recs)

    def prepare(self, states, obss):
        """features of all positions; with an inference server the trunk request is SENT here (not awaited)."""
        xs = [st["f"].observe(obs) for st, obs in zip(states, obss)]
        h = None
        rem = getattr(self, "remote", None)
        if rem is not None and xs:
            ts = [self._t(x) for x in xs]
            h = []
            for t in sorted(set(ts)):          # lockstep games share the step, so this is one request in practice
                ii = [i for i, v in enumerate(ts) if v == t]
                h.append((ii, rem.submit(self.key, {k: np.stack([np.asarray(xs[i][k]) for i in ii]) for k in XKEYS}, t)))
        return xs, h

    def finish(self, states, prep, rng, temp, greedy):
        xs, h = prep
        outs = []
        encs = [None] * len(xs)
        if h is not None:
            for ii, rid in h:
                tok, ctx = self.remote.collect(rid)
                for j, i in enumerate(ii):
                    encs[i] = (tok[j], ctx[j])
        for st, x, enc in zip(states, xs, encs):
            f = st["f"]
            y, UF, MF, LG, logp, value, lpu = self._decode(x, f.aux, rng, temp, greedy, enc)
            acts = f.to_actions(y)
            rec = {k: np.asarray(x[k]) for k in XKEYS}
            rec.update(uf=UF, mf=MF, legal=LG, y=y, logp=logp, value=value, lpu=lpu)
            outs.append((acts, rec))
        return outs

    def act(self, states, obss, rng, temp, greedy):
        """remote (rl2_ppo_server.Client, set by the rollout worker) -> the trunk of all positions runs as one batched
        request on the inference server; the per-unit decode stays here. Without it: NumPy trunk per position."""
        return self.finish(states, self.prepare(states, obss), rng, temp, greedy)


_DT = {"maps": np.uint8, "scal": np.float16, "tok_pos": np.int16, "tok_kind": np.int8, "tok_mask": bool,
       "tok_cargo": np.float16, "hist_cell": np.int16, "hist_cargo": np.float16, "hist_move": np.int8,
       "uf": np.float16, "mf": np.float16, "legal": bool, "y": np.int8}


class PackedTraj:
    """one recorded trajectory as stacked compact arrays (built in the rollout worker: ~8x less to pickle and to hold
    than a list of per-step dicts, and the trainer only concatenates)."""
    __slots__ = ("d", "n")

    def __init__(self, recs):
        self.n = len(recs)
        self.d = {k: np.stack([np.asarray(r[k]).astype(dt) for r in recs]) for k, dt in _DT.items()}
        self.d["lpu"] = np.stack([r["lpu"] for r in recs]).astype(np.float32)
        self.d["logp"] = np.array([r["logp"] for r in recs], np.float32)

    def __len__(self):
        return self.n


class Cnet2Buffer:
    def __init__(self, trajs):
        if trajs and isinstance(trajs[0], PackedTraj):
            self.n = sum(t.n for t in trajs)
            self.d = {k: np.concatenate([t.d[k] for t in trajs]) for k in _DT}
            self.lp0u = np.concatenate([t.d["lpu"] for t in trajs]).reshape(-1)
            self.lp0 = np.concatenate([t.d["logp"] for t in trajs])
            self.n_ukeys = self.n * F.OWN
            self.act = np.zeros(int((self.d["y"] >= 0).sum()))
            return
        recs = [r for t in trajs for r in t]
        self.n = len(recs)
        self.d = {k: np.stack([np.asarray(r[k]).astype(dt) for r in recs]) for k, dt in _DT.items()}
        self.n_ukeys = self.n * F.OWN
        self.lp0 = np.array([r["logp"] for r in recs], np.float32)
        self.lp0u = np.concatenate([r["lpu"] for r in recs]).astype(np.float32)   # ukey = d*OWN + s
        self.act = np.zeros(int((self.d["y"] >= 0).sum()))      # for unit counting in logs

    def nbytes(self):
        return sum(v.nbytes for v in self.d.values())

    def batch(self, idx, device):
        import torch
        idx = np.asarray(idx)
        t = {}
        for k, v in self.d.items():
            a = v[idx]
            if a.dtype == np.float16:
                a = a.astype(np.float32)
            t[k] = torch.from_numpy(np.ascontiguousarray(a))
        B = len(idx)
        t["maps"] = (t["maps"].float() / 255.0).reshape(B, F.P, F.S, F.S)
        for k in ("tok_pos", "tok_kind", "hist_cell", "hist_move", "y"):
            t[k] = t[k].long()
        t["teacher"] = torch.full((B,), TEACHER, dtype=torch.long)
        if TEACHER_LATE is not None:
            late = torch.round(t["scal"][:, 1].float() * 300) >= LATE_STEP
            t["teacher"] = torch.where(late, torch.full_like(t["teacher"], TEACHER_LATE), t["teacher"])
        t["ukey"] = torch.from_numpy((idx[:, None] * F.OWN + np.arange(F.OWN)[None, :]).astype(np.int64))
        out = {k: v.to(device) for k, v in t.items()}
        out["B"] = B
        return out


def lg_dev(model):
    return next(model.parameters()).device.type


class Cnet2Adapter:
    name = "cnet2"
    n_act = 5

    def make_actor(self, weights):
        return Cnet2Actor(weights)

    def make_model(self, weights):
        import torch
        from cnet2_model import CNet2, migrate_state_dict
        meta = weights.get("meta")
        cfg = json.loads(str(meta))["cfg"] if meta is not None else {}
        net = CNet2(**cfg)
        # F01 (final acceptance): pre-heads weights (98-era npz) lack the 6
        # Stage 0 head keys; a bare strict load made every old PPO init /
        # resume through this adapter fail. Route through the shared
        # migration -- it tolerates ONLY the head keys and raises on any
        # other missing/unexpected key, so new-format weights load exactly
        # as before.
        migrate_state_dict(net, {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in weights.items() if k != "meta"},
                           source="rl2 Cnet2Adapter.make_model")
        net._rl2_cfg = cfg
        return net

    def server_encode(self, model, d, teacher, device, amp=False):
        """trunk of a batch of positions for the inference server: d = stacked Feat.observe outputs (XKEYS, NumPy)
        -> tok [B,OWN,D], ctx [B,D] float32 NumPy (what cnet2_np.Policy.encode returns per position)."""
        import torch
        B = len(d["maps"])
        t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in d.items()}
        b = {"maps": (t["maps"].float() / 255.0).reshape(B, F.P, F.N).reshape(B, F.P, F.S, F.S),
             "scal": t["scal"].float(), "tok_cargo": t["tok_cargo"].float(), "hist_cargo": t["hist_cargo"].float(),
             "tok_mask": t["tok_mask"].bool(), "teacher": torch.full((B,), int(teacher), dtype=torch.long, device=device)}
        for k in ("tok_pos", "tok_kind", "hist_cell", "hist_move"):
            b[k] = t[k].long()
        with torch.no_grad(), torch.autocast(device.type, enabled=bool(amp) and device.type == "cuda"):
            tok, ctx, _ = model.encode(b)
        return tok[:, :F.OWN].float().cpu().numpy(), ctx.float().cpu().numpy()

    def export(self, model):
        sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
        sd["meta"] = np.array(json.dumps({"cfg": getattr(model, "_rl2_cfg", model.cfg)}))
        return sd

    def build_buffer(self, trajs):
        return Cnet2Buffer(trajs)

    def evaluate(self, model, b, temp):
        import torch
        amp = bool(getattr(model, "_amp", False)) and lg_dev(model) == "cuda"
        with torch.autocast("cuda", enabled=amp):
            lg, v, _ = model(b)                               # [B,12,5] masked (NEG), [B]
        lg = lg.float(); v = v.float()
        keep = b["tok_mask"][:, :F.OWN] & (b["y"] >= 0)
        legal = b["legal"][keep]
        if TEMP_KIND is None:
            z = lg[keep] / temp
        else:
            tk = torch.tensor(TEMP_KIND, device=lg.device, dtype=lg.dtype)
            kind = b["tok_kind"][:, :F.OWN].clamp(max=2)          # 0 collector, 1 planter, 2 own base
            z = lg[keep] / tk[kind[keep]][:, None]
        z = torch.where(legal, z, torch.full_like(z, -1e9))
        lp = torch.log_softmax(z, dim=1)
        B = b["B"]
        seg = torch.arange(B, device=lg.device)[:, None].expand(-1, F.OWN)[keep]
        return {"lp": lp, "act": b["y"][keep], "seg": seg, "value": v, "legal": legal, "ukey": b["ukey"][keep]}
