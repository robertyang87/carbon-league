"""rl2_ppo_train -- PPO (league self-play) trainer for Carbon policies.

Rollouts: persistent spawn-pool of CPU workers (rl2_ppo_rollout.play_batch),
each playing --lockstep games at once with a batched NumPy policy forward.
Learner: torch on MPS (or CPU). Policy adapters: rl2_ppo_adapters
("small" = nn_train_bc.Net, "cnet2" = rl2_ppo_cnet2 adapter).

Objective per decision (one game step = the joint action of all our units):
  logp = sum_u log pi(a_u | s, a_<u)      (temperature-consistent, legal-masked)
  L = -min(r A, clip(r) A) + vf * (V - G)^2 - ent * H + kl * KL(pi_ref || pi)
  r = exp(logp - logp_behaviour); H, KL summed over units (teacher-forced
  conditionals for autoregressive policies); pi_ref = frozen BC init.
Reward per step: rscale * d(margin)  (or terminal-only margin, or --reward worth: d(worth margin))  + win bonus
  + anneal * (c_dep * delivered_cargo - c_lost * cargo_lost) * rscale,
  anneal = max(0, 1 - it / shape_iters). GAE(gamma, lambda) on the value head.
League: bot files + frozen learner snapshots; opponent sampling
  p ~ prior * (1 - winrate_ema + eps)^pfsp  (weights toward the strongest).
Eval every --eval-every its (async, greedy, paired seeds >= 980000) ->
  OUT/eval.jsonl; train log -> OUT/train.jsonl; league table -> OUT/league.json;
  checkpoints OUT/ckpt_<it>.npz (+ ckpt_last.npz, state_last.pt for --resume).

Example (small net smoke):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 \
  ../.venv/bin/python rl2_ppo_train.py --adapter small \
     --init ../data/weights/shadow_1czero_v1.npz --out ../data/rl2/small_smoke --iters 3
"""

import argparse
import pickle
import zlib
import json
import math
import multiprocessing as mp
import os
import random
import resource
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BC_W", "64")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rl2_ppo_adapters as AD  # noqa: E402

DEFAULT_POOL = [["exp/opp_1cz_v5.py", 1.0], ["exp/cand_c1z_v1.py", 1.0], ["exp/submission_nn_v9.py", 0.5]]


def parse():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--adapter", default="small")
    a("--init", required=True, help="BC weights npz (also the KL reference unless --ref)")
    a("--ref", default=None)
    a("--out", required=True)
    a("--iters", type=int, default=1000000)
    a("--minutes", type=float, default=1e9)
    a("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    a("--games", type=int, default=32, help="games per iteration")
    a("--lockstep", type=int, default=4, help="concurrent games per worker task")
    a("--overlap", type=int, default=1, help="1: collect iteration k+1 while training on k (1-iter stale behaviour policy)")
    a("--device", default="mps")
    a("--temp", type=float, default=1.0)
    a("--lr", type=float, default=1e-5)
    a("--vlr", type=float, default=3e-4)
    a("--epochs", type=int, default=2)
    a("--mb", type=int, default=256, help="decisions per minibatch")
    a("--clip", type=float, default=0.2)
    a("--ratio", default="joint", choices=["joint", "unit"],
      help="joint: one PPO ratio per decision (sum of unit log-probs); unit: per-unit ratios x decision advantage")
    a("--vf", type=float, default=0.5)
    a("--ent", type=float, default=0.001)
    a("--kl", type=float, default=0.1, help="KL(pi_ref||pi) coefficient (per decision, summed over units)")
    a("--kl-target", type=float, default=0.0, help=">0: adapt --kl to keep mean KL near target")
    a("--gclip", type=float, default=1.0)
    a("--gamma", type=float, default=0.995)
    a("--lam", type=float, default=0.95)
    a("--rscale", type=float, default=1 / 5000.0)
    a("--reward", default="dense", choices=["dense", "terminal", "worth"],
      help="dense: per-step change of the cash margin; worth: per-step change of the worth margin (cash + cargo on board + "
           "remaining tree value; sums to the same final margin, credit arrives at pick-up / planting time)")
    a("--win-bonus", type=float, default=0.1)
    a("--c-dep", type=float, default=0.0)
    a("--c-lost", type=float, default=0.0)
    a("--shape-iters", type=int, default=200)
    a("--sep-value", type=int, default=1, help="1: separate value network (copy of init)")
    a("--vwarm", type=int, default=2, help="iterations training only the value function")
    a("--pool", default=json.dumps(DEFAULT_POOL), help="json [[bot_file, prior], ...]")
    a("--snap-every", type=int, default=10)
    a("--snap-prior", type=float, default=0.5)
    a("--max-snaps", type=int, default=6)
    a("--pfsp", type=float, default=1.0)
    a("--ckpt-every", type=int, default=10)
    a("--eval-every", type=int, default=10)
    a("--eval-pairs", type=int, default=8, help="paired seeds per eval opponent")
    a("--eval-opps", default="exp/opp_1cz_v5.py,exp/cand_c1z_v1.py")
    a("--eval-seed0", type=int, default=980000)
    a("--seed0", type=int, default=3000000, help="training game seeds start here")
    a("--amp", type=int, default=0,
      help="1: fp16 autocast for the learner's forward/backward on cuda (log-probs and losses stay fp32); watch dlogp")
    a("--server", type=int, default=0,
      help="1: one inference process on --device runs the trunk for all rollout workers (rl2_ppo_server); the workers "
           "keep only features, the per-unit decode and the simulator")
    a("--self-prior", type=float, default=0.0,
      help=">0: add opponent SELF (current weights on both sides, both sides recorded) to the league with this prior")
    a("--resume", default=None, help="state_last.pt to resume from")
    a("--remote", default=None, help="split mode: consume cloud actor shards (rl2_actor.py) from this local dir")
    a("--dec-frac", type=float, default=1.0, help="fraction of decisions used in the PPO epochs (values: all)")
    a("--max-stale", type=int, default=2, help="split mode: drop games whose behaviour weights are older than it-K")
    return p.parse_args()


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def children_rss_mb():
    try:
        import subprocess
        out = subprocess.run(["ps", "-o", "rss=", "--ppid", str(os.getpid())] if sys.platform != "darwin"
                             else ["ps", "-A", "-o", "rss=,ppid="], capture_output=True, text=True).stdout
        tot = 0
        for line in out.strip().splitlines():
            parts = line.split()
            if sys.platform == "darwin":
                if len(parts) == 2 and int(parts[1]) == os.getpid():
                    tot += int(parts[0])
            else:
                tot += int(parts[0])
        return tot / 1e3
    except Exception:
        return -1


class League:
    def __init__(self, path, pool, snap_prior, pfsp):
        self.path = path
        self.pfsp = pfsp
        self.snap_prior = snap_prior
        self.tab = {}
        if os.path.exists(path):
            self.tab = json.load(open(path))
        for f, pr in pool:
            self.tab.setdefault(f, {"prior": pr, "n": 0, "wr": 0.5, "margin": 0.0})
            self.tab[f]["prior"] = pr

    def add_snap(self, spec, max_snaps):
        self.tab[spec] = {"prior": self.snap_prior, "n": 0, "wr": 0.5, "margin": 0.0, "snap": True}
        snaps = [k for k, v in self.tab.items() if v.get("snap") and v["prior"] > 0]
        for k in snaps[:-max_snaps]:
            self.tab[k]["prior"] = 0.0          # retired (row kept for the record)

    def sample(self, rng):
        keys = [k for k, v in self.tab.items() if v["prior"] > 0]
        w = np.array([self.tab[k]["prior"] * (1.0 - self.tab[k]["wr"] + 0.05) ** self.pfsp for k in keys])
        return keys[rng.choice(len(keys), p=w / w.sum())]

    def update(self, opp, margin):
        e = self.tab[opp]
        e["n"] += 1
        a = max(0.05, 1.0 / e["n"])
        e["wr"] = (1 - a) * e["wr"] + a * (1.0 if margin > 0 else 0.0)
        e["margin"] = (1 - a) * e["margin"] + a * margin

    def save(self):
        json.dump(self.tab, open(self.path + ".tmp", "w"), indent=1)
        os.replace(self.path + ".tmp", self.path)


def gae(rew, val, gamma, lam):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val


def game_rewards(g, args, it):
    dm = g["dmargin"].astype(np.float64)
    if args.reward == "terminal":
        r = np.zeros_like(dm); r[-1] = dm.sum()
    elif args.reward == "worth":
        r = g["dworth"].astype(np.float64)
    else:
        r = dm.copy()
    ann = max(0.0, 1.0 - it / max(1, args.shape_iters))
    r = r + ann * (args.c_dep * g["dep"] - args.c_lost * g["lost"])
    r = r * args.rscale
    r[-1] += args.win_bonus * (1 if g["margin"] > 0 else (-1 if g["margin"] < 0 else 0))
    return r.astype(np.float32)


def eval_tasks(args, adapter, wkey, it):
    tasks = []
    for opp in args.eval_opps.split(","):
        for j in range(args.eval_pairs):
            for side in (0, 1):
                tasks.append({"adapter": adapter, "wkey": wkey, "games": [(opp, args.eval_seed0 + j, side)],
                              "temp": 1.0, "greedy": True, "record": False, "it": it})
    return tasks


def summarize_eval(it, results):
    by = {}
    for r in results:
        for g in r["games"]:
            by.setdefault(g["opp"], []).append(g["margin"])
    rows = []
    for opp, m in by.items():
        m = np.array(m)
        rows.append({"it": it, "opp": opp, "n": len(m), "wins": int((m > 0).sum()),
                     "mean": round(float(m.mean()), 1),
                     "se": round(float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 1 else 0.0, 1),
                     "t": time.time()})
    return rows


def main():
    args = parse()
    args.out = os.path.abspath(args.out)
    import torch
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    live = os.path.join(args.out, "live"); os.makedirs(live, exist_ok=True)
    dev = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("LEARNER_THREADS", "8")))
    ad = AD.get_adapter(args.adapter)
    w0 = AD.load_npz(args.init)
    wref = AD.load_npz(args.ref) if args.ref else w0
    net = ad.make_model(w0).to(dev)
    ref = ad.make_model(wref).to(dev).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    vnet = ad.make_model(w0).to(dev) if args.sep_value else None
    use_amp = bool(args.amp) and dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    for m_ in (net, ref, vnet):
        if m_ is not None:
            m_._amp = use_amp
    sc_p = torch.amp.GradScaler("cuda", enabled=use_amp); sc_v = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    vopt = torch.optim.Adam(vnet.parameters(), lr=args.vlr) if vnet is not None else None
    # shared model: value-warmup steps touch only the value head (trunk updates would move the policy)
    vhead = [q for n, q in net.named_parameters() if n.startswith(("v.", "value."))]
    hopt = torch.optim.Adam(vhead, lr=args.vlr) if (vnet is None and vhead) else None
    it0 = 0
    kl_coef = args.kl
    if args.resume:
        stt = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(stt["net"]); opt.load_state_dict(stt["opt"])
        if vnet is not None and stt.get("vnet"):
            vnet.load_state_dict(stt["vnet"]); vopt.load_state_dict(stt["vopt"])
        it0 = stt["it"]; kl_coef = stt.get("kl_coef", kl_coef)
    league = League(os.path.join(args.out, "league.json"), json.loads(args.pool), args.snap_prior, args.pfsp)
    if args.self_prior > 0:
        league.tab.setdefault("SELF", {"prior": args.self_prior, "n": 0, "wr": 0.5, "margin": 0.0})
        league.tab["SELF"]["prior"] = args.self_prior
    league.save()                         # split mode: the cloud actor needs the table before iteration 1
    for k, v in league.tab.items():      # resumed snapshots keep their files
        if v.get("snap") and not os.path.exists(k.split(":", 2)[2]):
            v["prior"] = 0.0
    tlog = open(os.path.join(args.out, "train.jsonl"), "a")
    elog = open(os.path.join(args.out, "eval.jsonl"), "a")
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    rng = np.random.default_rng(int(time.time()) % 100000 + it0)
    seed = args.seed0 + it0 * args.games

    cur_it = [it0]

    class _Remote:
        """split mode: newest cloud games (behaviour weights >= it - max_stale), at least args.games of them"""
        def get(self):
            while True:
                fs = sorted(f for f in os.listdir(args.remote) if f.endswith(".pkl.z"))
                keep = []
                for f in fs:
                    if int(f[2:8]) < cur_it[0] - args.max_stale:
                        os.remove(os.path.join(args.remote, f))
                    else:
                        keep.append(f)
                n = sum(int(f.rsplit("_", 1)[1].split(".")[0]) for f in keep)
                if n >= args.games:
                    games = []
                    for f in reversed(keep):
                        if len(games) >= args.games:
                            break
                        fp = os.path.join(args.remote, f)
                        games.extend(pickle.loads(zlib.decompress(open(fp, "rb").read())))
                        os.remove(fp)
                    return [{"games": games, "t_pol": 0.0, "t_opp": 0.0}]
                time.sleep(3)

    def publish(it):
        cur_it[0] = it
        wk = os.path.join(live, f"w_{it}.npz")
        AD.save_npz(ad.export(net), wk, {"it": it})
        olds = sorted([f for f in os.listdir(live) if f.startswith("w_")], key=lambda s: int(s[2:-4]))
        for f in olds[:-4]:
            if f != os.path.basename(wk):
                try:
                    os.remove(os.path.join(live, f))
                except OSError:
                    pass
        return wk

    ctx = mp.get_context("spawn")
    server = None
    if args.server and not args.remote:
        import rl2_ppo_server as SV
        import rl2_ppo_rollout as _RO
        server = SV.start(ctx, args.adapter, str(dev), args.workers)
        pool = ctx.Pool(args.workers, initializer=_RO._init_worker, initargs=server.initargs)
    else:
        pool = ctx.Pool(1 if args.remote else args.workers)

    def submit_rollouts(wk):
        nonlocal seed
        if args.remote:
            return _Remote(), time.time()
        specs = []
        for gi in range(args.games):
            specs.append((league.sample(rng), seed + gi // 2, gi % 2))   # same seed both sides
        seed += args.games
        tasks = []
        for i in range(0, len(specs), args.lockstep):
            tasks.append({"adapter": args.adapter, "wkey": wk, "games": specs[i:i + args.lockstep],
                          "temp": args.temp, "greedy": False, "record": True,
                          "rng_seed": int(rng.integers(1 << 30))})
        return pool.map_async(__import__("rl2_ppo_rollout").play_batch, tasks), time.time()

    import rl2_ppo_rollout as RO
    pending_eval = []
    wk = publish(it0)
    if it0 == 0 and args.eval_every > 0:
        pending_eval.append((0, [pool.apply_async(RO.play_batch, (t,)) for t in eval_tasks(args, args.adapter, wk, 0)]))
    nxt = submit_rollouts(wk)
    t_start = time.time()
    it = it0
    while it < it0 + args.iters and time.time() - t_start < 60 * args.minutes:
        it += 1
        res_async, t_sub = nxt
        res = res_async.get()
        t_roll = time.time() - t_sub
        if args.overlap:
            nxt = submit_rollouts(wk)            # behaviour policy = current weights (1-iter stale for next update)
        games = [g for r in res for g in r["games"]]
        trajs = [g["recs"] for g in games]
        ndec = sum(len(t) for t in trajs)
        t_pol = sum(r["t_pol"] for r in res); t_opp = sum(r["t_opp"] for r in res)
        # ---------------- values, GAE
        t1 = time.time()
        buf = ad.build_buffer(trajs)
        offs = np.cumsum([0] + [len(t) for t in trajs])
        LP0 = np.array([r["logp"] for t in trajs for r in t], np.float32)
        LP0U = torch.from_numpy(buf.lp0u).to(dev)
        vmodel = vnet if vnet is not None else net
        vmodel.eval()
        V = np.zeros(buf.n, np.float32)
        REFLP = None
        # --dec-frac < 1: values for every decision (GAE needs them), but the reference pass and the PPO epochs only
        # touch a random subset -> more games per GPU-second (decisions within a game are strongly correlated)
        SUB = (np.sort(np.random.choice(buf.n, max(1, int(buf.n * args.dec_frac)), replace=False))
               if args.dec_frac < 1 else np.arange(buf.n))
        with torch.no_grad():
            for b0 in range(0, buf.n, args.mb):
                idx = np.arange(b0, min(buf.n, b0 + args.mb))
                bt = buf.batch(idx, dev)
                V[idx] = ad.evaluate(vmodel, bt, args.temp)["value"].float().cpu().numpy()
                if args.dec_frac >= 1:
                    orf = ad.evaluate(ref, bt, args.temp)        # reference policy cached once per iteration
                    if REFLP is None:
                        REFLP = torch.zeros((buf.n_ukeys, orf["lp"].shape[1]), device=dev)
                    REFLP[orf["ukey"]] = orf["lp"].float()
            if args.dec_frac < 1:
                for b0 in range(0, len(SUB), args.mb):
                    bt = buf.batch(SUB[b0:b0 + args.mb], dev)
                    orf = ad.evaluate(ref, bt, args.temp)
                    if REFLP is None:
                        REFLP = torch.zeros((buf.n_ukeys, orf["lp"].shape[1]), device=dev)
                    REFLP[orf["ukey"]] = orf["lp"].float()
        ADV = np.zeros(buf.n, np.float32); RET = np.zeros(buf.n, np.float32)
        for gi, g in enumerate(games):
            a_, r_ = gae(game_rewards(g, args, it), V[offs[gi]:offs[gi + 1]], args.gamma, args.lam)
            ADV[offs[gi]:offs[gi + 1]] = a_; RET[offs[gi]:offs[gi + 1]] = r_
        ADVn = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
        # ---------------- PPO epochs
        net.train()
        if vnet is not None:
            vnet.train()
        stats = []
        warm = (it - it0) <= args.vwarm and not args.resume
        tq = lambda a: torch.from_numpy(a).to(dev)
        for ep in range(args.epochs):
            perm = np.random.permutation(SUB)
            for b0 in range(0, len(perm), args.mb):
                idx = np.sort(perm[b0:b0 + args.mb])
                bt = buf.batch(idx, dev)
                o = ad.evaluate(net, bt, args.temp)
                B = len(idx)
                lpu = o["lp"].gather(1, o["act"][:, None]).squeeze(1)
                lp = torch.zeros(B, device=dev).index_add_(0, o["seg"], lpu)
                pu = o["lp"].exp()
                legal = o["legal"]
                lpz = torch.where(legal, o["lp"], torch.zeros_like(o["lp"]))
                ent = torch.zeros(B, device=dev).index_add_(0, o["seg"], -(pu * lpz).sum(1))
                with torch.no_grad():
                    rlp = REFLP[o["ukey"]]
                    lr_ = torch.where(legal, rlp, torch.zeros_like(rlp))
                    pr_ = torch.where(legal, rlp.exp(), torch.zeros_like(rlp))
                klu = (pr_ * (lr_ - lpz)).sum(1)
                kl = torch.zeros(B, device=dev).index_add_(0, o["seg"], klu)
                ratio = torch.exp(torch.clamp(lp - tq(LP0[idx]), -20, 20))
                A_ = tq(ADVn[idx])
                if args.ratio == "joint":
                    pl = -torch.min(ratio * A_, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * A_).mean()
                else:
                    ru = torch.exp(torch.clamp(lpu - LP0U[o["ukey"]], -20, 20))
                    Au = A_[o["seg"]]
                    pl = -torch.min(ru * Au, torch.clamp(ru, 1 - args.clip, 1 + args.clip) * Au).mean()
                    ratio = ru
                ret = tq(RET[idx])
                if vnet is not None:
                    vv = ad.evaluate(vnet, bt, args.temp)["value"]
                    vl = ((vv - ret.clamp(-1, 1)) ** 2).mean()
                    vopt.zero_grad(); sc_v.scale(vl).backward(); sc_v.unscale_(vopt)
                    torch.nn.utils.clip_grad_norm_(vnet.parameters(), args.gclip); sc_v.step(vopt); sc_v.update()
                    loss = pl - args.ent * ent.mean() + kl_coef * kl.mean()
                else:
                    vl = ((o["value"] - ret.clamp(-1, 1)) ** 2).mean()
                    loss = pl - args.ent * ent.mean() + kl_coef * kl.mean() + args.vf * vl
                    if warm:
                        loss = args.vf * vl
                if warm and hopt is not None:
                    hopt.zero_grad(); (args.vf * vl).backward()
                    torch.nn.utils.clip_grad_norm_(vhead, args.gclip); hopt.step()
                    gn = 0.0
                elif not (warm and vnet is not None):
                    opt.zero_grad(); sc_p.scale(loss).backward(); sc_p.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.gclip).item()
                else:
                    gn = 0.0
                if not warm:
                    sc_p.step(opt); sc_p.update()
                with torch.no_grad():
                    cf = ((ratio - 1).abs() > args.clip).float().mean().item()
                stats.append([pl.item(), vl.item(), ent.mean().item(), kl.mean().item(), cf, gn,
                              (lp - tq(LP0[idx])).mean().item()])
        st = np.mean(stats, axis=0)
        if args.kl_target > 0:
            kl_coef *= 1.5 if st[3] > 1.5 * args.kl_target else (1 / 1.5 if st[3] < args.kl_target / 1.5 else 1.0)
            kl_coef = float(np.clip(kl_coef, 1e-4, 10.0))
        t_upd = time.time() - t1
        wk = publish(it)
        if not args.overlap:
            nxt = submit_rollouts(wk)
        for g in games:
            league.update(g["opp"], g["margin"])
        by = {}
        for g in games:
            by.setdefault(g["opp"], []).append(g["margin"])
        srv = server.stats() if server is not None else None
        rec = {"it": it, "t": round(time.time()), "games": len(games), "srv": srv, "dec": ndec, "units": int(len(buf.act)),
               "roll_s": round(t_roll, 1), "upd_s": round(t_upd, 1),
               "games_per_min": round(60 * len(games) / max(t_roll, 1e-6), 1),
               "dec_per_s": round(ndec / max(t_roll, 1e-6), 1),
               "pol_s": round(t_pol, 1), "opp_s": round(t_opp, 1),
               "pl": round(st[0], 4), "vl": round(st[1], 4), "ent": round(st[2], 3), "kl": round(st[3], 4),
               "kl_coef": round(kl_coef, 4), "clipfrac": round(st[4], 3), "gnorm": round(st[5], 3),
               "dlogp": round(st[6], 4), "warm": bool(warm),
               "adv_std": round(float(ADV.std()), 4), "v_mean": round(float(V.mean()), 3),
               "margin": {k: [round(float(np.mean(v))), len(v), int(sum(x > 0 for x in v))] for k, v in by.items()},
               "winrate": round(float(np.mean([g["margin"] > 0 for g in games])), 3),
               "buf_mb": round(buf.nbytes() / 1e6, 1), "rss_mb": round(rss_mb()), "workers_rss_mb": round(children_rss_mb())}
        tlog.write(json.dumps(rec) + "\n"); tlog.flush()
        print(json.dumps(rec), flush=True)
        del buf, trajs, games, res
        if it % args.snap_every == 0:
            sp = os.path.join(args.out, f"snap_{it}.npz")
            AD.save_npz(ad.export(net), sp, {"it": it})
            league.add_snap(f"SNAP:{args.adapter}:{sp}", args.max_snaps)
        league.save()
        if it % args.ckpt_every == 0 or it == it0 + args.iters:
            AD.save_npz(ad.export(net), os.path.join(args.out, f"ckpt_{it}.npz"), {"it": it})
        AD.save_npz(ad.export(net), os.path.join(args.out, "ckpt_last.npz"), {"it": it})
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "vnet": vnet.state_dict() if vnet is not None else None,
                    "vopt": vopt.state_dict() if vopt is not None else None,
                    "it": it, "kl_coef": kl_coef}, os.path.join(args.out, "state_last.pt"))
        if args.eval_every > 0 and it % args.eval_every == 0:
            ek = os.path.join(args.out, f"eval_w_{it}.npz")
            AD.save_npz(ad.export(net), ek, {"it": it})
            pending_eval.append((it, [pool.apply_async(RO.play_batch, (t,)) for t in eval_tasks(args, args.adapter, ek, it)]))
        still = []
        for eit, lst in pending_eval:
            if all(x.ready() for x in lst):
                for row in summarize_eval(eit, [x.get() for x in lst]):
                    elog.write(json.dumps(row) + "\n"); elog.flush()
                    print("EVAL", json.dumps(row), flush=True)
                ek = os.path.join(args.out, f"eval_w_{eit}.npz")
                if os.path.exists(ek):
                    os.remove(ek)
            else:
                still.append((eit, lst))
        pending_eval = still
    # drain evals before exit
    for eit, lst in pending_eval:
        for row in summarize_eval(eit, [x.get() for x in lst]):
            elog.write(json.dumps(row) + "\n"); elog.flush()
            print("EVAL", json.dumps(row), flush=True)
        ek = os.path.join(args.out, f"eval_w_{eit}.npz")
        if os.path.exists(ek):
            os.remove(ek)
    pool.terminate()
    if server is not None:
        server.stop()


if __name__ == "__main__":
    main()
