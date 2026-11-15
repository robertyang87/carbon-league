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
import hashlib
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
from rl2_checkpoint import require_finite, atomic_json, commit_checkpoint  # noqa: E402

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
    # R03: --lr/--vlr/--kl default to None = "not given on the CLI". main() resolves the built-in
    # defaults (1e-5 / 3e-4 / 0.1); on --resume an explicitly given value wins over the checkpoint,
    # an unspecified one inherits it. (Previously the checkpoint silently overrode fresh CLI values.)
    a("--lr", type=float, default=None)
    a("--vlr", type=float, default=None)
    a("--epochs", type=int, default=2)
    a("--mb", type=int, default=256, help="decisions per minibatch")
    a("--clip", type=float, default=0.2)
    a("--ratio", default="joint", choices=["joint", "unit"],
      help="joint: one PPO ratio per decision (sum of unit log-probs); unit: per-unit ratios x decision advantage")
    a("--vf", type=float, default=0.5)
    a("--ent", type=float, default=0.001)
    a("--kl", type=float, default=None, help="KL(pi_ref||pi) coefficient (per decision, summed over units)")
    a("--kl-target", type=float, default=0.0, help=">0: adapt --kl to keep mean KL near target")
    a("--gclip", type=float, default=1.0)
    a("--gamma", type=float, default=0.995)
    a("--lam", type=float, default=0.95)
    a("--rscale", type=float, default=1 / 5000.0)
    a("--reward", default="dense", choices=["dense", "terminal", "worth", "u"],
      help="dense: per-step change of the cash margin; worth: per-step change of the worth margin (cash + cargo on board + "
           "remaining tree value; sums to the same final margin, credit arrives at pick-up / planting time)")
    a("--value-head", default="tanh", choices=["tanh", "utility"],
      help="critic readout. tanh: the legacy value head (default, unchanged). utility: the Stage 0 utility head "
           "(E[U]) mapped to 2*(U-0.5) -- the Terminal-U critic; requires --reward u and the cnet2 adapter, and the "
           "effective critic weights must carry the utility head (see --utility-source)")
    a("--utility-source", default="auto", choices=["auto", "checkpoint", "init"],
      help="utility critic source: auto uses init for a fresh run, or a checkpoint declaring U/utility training; "
           "checkpoint explicitly accepts a legacy checkpoint's critic provenance; init explicitly resets the "
           "complete separate critic and its optimizer after resume (never just a head on a different trunk)")
    a("--win-bonus", type=float, default=0.1)
    a("--c-dep", type=float, default=0.0)
    a("--c-lost", type=float, default=0.0)
    a("--shape-iters", type=int, default=200)
    a("--sep-value", type=int, default=1, help="1: separate value network (copy of init)")
    a("--vwarm", type=int, default=2, help="iterations training only the value function")
    a("--resume-vwarm", type=int, default=0,
      help="critic-only warmup batches after resume, before formal iterations; policy/optimizer and formal it stay frozen")
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
    a("--pre-mb", type=int, default=4096, help="batch of the no-grad passes before the PPO epochs (values, reference)")
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


# Terminal-U: U = 0.5*W + 0.5*sigmoid(margin/1000); W = 1 win / 0 loss /
# 0.5 draw. The draw value is forced, not a knob (strategy-group ruling):
# U(-M) = 1 - U(M) holds for every terminal state iff the two seats' draw
# W are complementary, whose symmetric solution is 0.5 -- then U(draw, 0)
# = 0.5 and the seats sum to 1 (zero-sum consistent). This is the same U
# as the teacher terminal (eval_v53 _terminal_u) and the utility-head
# training labels: one definition across gate/teacher/value/PPO.
DRAW_W = 0.5


def terminal_u(margin):
    w = 1.0 if margin > 0 else (DRAW_W if margin == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-margin / 1000.0))


def u_mode_check(args):
    """Terminal-U launch contract (spec SS15). --reward u pays 2*(U-0.5) at
    the terminal step only, so the return equals the objective exactly iff
    gamma == 1 and rscale == 1: with the defaults (gamma 0.995, rscale
    1/5000) a 299-step game discounts the terminal reward to 0.995**298 =
    0.22 of its value at the first decision and scales it by 2e-4 -- the
    critic would then be fitting something that is not U. Returns a list of
    problems (empty = consistent); main() prints them loudly and records
    them in effective_params.json. Not a refusal: a staged switch may pass
    through intermediate values deliberately."""
    probs = []
    if args.reward == "u":
        if abs(args.gamma - 1.0) > 1e-12:
            probs.append(f"gamma={args.gamma} (Terminal-U needs 1.0: the terminal reward reaches step 1 "
                         f"scaled by {args.gamma ** 298:.3f})")
        if abs(args.rscale - 1.0) > 1e-12:
            probs.append(f"rscale={args.rscale} (Terminal-U needs 1.0: the reward is already in [-1, 1])")
        if args.c_dep or args.c_lost:
            probs.append("c_dep/c_lost are ignored in u mode (no shaping terms)")
    if args.value_head == "utility" and args.reward != "u":
        probs.append("--value-head utility without --reward u: the utility head is an E[U] readout")
    return probs


def game_rewards(g, args, it):
    dm = g["dmargin"].astype(np.float64)
    if args.reward == "u":
        # Terminal-U: non-terminal steps pay 0; terminal pays 2*(U-0.5) in
        # [-1, 1]. No shaping terms and no win bonus in this mode (U already
        # carries the win term). Launch with --rscale 1.0 --gamma 1.0.
        r = np.zeros_like(dm)
        r[-1] = 2.0 * (terminal_u(float(g["margin"])) - 0.5)
        return (r * args.rscale).astype(np.float32)
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
        u = np.array([terminal_u(float(x)) for x in m])
        rows.append({"it": it, "opp": opp, "n": len(m), "wins": int((m > 0).sum()),
                     "mean_u": round(float(u.mean()), 4),
                     "se_u": round(float(u.std(ddof=1) / math.sqrt(len(u))) if len(u) > 1 else 0.0, 4),
                     "mean": round(float(m.mean()), 1),
                     "se": round(float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 1 else 0.0, 1),
                     "t": time.time()})
    return rows


def _resume_sd(model, sd, opt, opt_sd, tag):
    """F01 (final acceptance): resume one model+optimizer pair from a
    checkpoint. A NEW-format checkpoint (same key set) takes the plain
    strict path. A PRE-HEADS checkpoint (only the 6 Stage 0 head keys
    absent, nothing unexpected) resumes through the shared migration:
    weights via cnet2_model.migrate_state_dict, optimizer via
    cnet2_train.migrate_optimizer_checked (the id->name mapping is proven
    from the checkpoint's own key order, moments post-verified). Anything
    else falls back to the strict load so the real mismatch raises."""
    missing = set(model.state_dict()) - set(sd)
    extra = set(sd) - set(model.state_dict())
    if not missing and not extra:
        model.load_state_dict(sd); opt.load_state_dict(opt_sd)
        return "strict"
    from cnet2_model import NEW_HEAD_KEYS
    if extra or not missing <= set(NEW_HEAD_KEYS):
        model.load_state_dict(sd)          # raises with the real mismatch
        opt.load_state_dict(opt_sd)
        return "strict"
    from cnet2_model import migrate_state_dict
    from cnet2_train import migrate_optimizer_checked
    migrate_state_dict(model, sd, source=f"rl2 resume {tag}")
    rep = migrate_optimizer_checked(opt, opt_sd, sd, model, source=f"rl2 resume {tag}")
    print(f"[resume] {tag}: pre-heads checkpoint migrated; carried "
          f"{len(rep['carried'])} params, cold-started {len(rep['cold'])} head params, "
          f"steps preserved {rep['steps_preserved']}", flush=True)
    return "migrated"


def training_contract(args):
    """Persist the target/readout actually used, not merely the head key set."""
    return {k: getattr(args, k) for k in ("reward", "value_head", "gamma", "rscale", "sep_value")}


def utility_source_plan(args, init_weights, checkpoint):
    """Resolve the effective critic before loading it. Key presence establishes
    shape/availability only, never calibration. Legacy checkpoints need an
    explicit source declaration; a conflicting init is never silently used as
    evidence about the critic subsequently restored from a checkpoint.
    """
    source = args.utility_source
    if args.value_head != "utility":
        if source != "auto":
            raise SystemExit("--utility-source is only valid with --value-head utility")
        return None
    if args.adapter != "cnet2":
        raise SystemExit("--value-head utility needs --adapter cnet2")
    if checkpoint is None:
        if source == "checkpoint":
            raise SystemExit("--utility-source checkpoint needs --resume")
        source = "init"
    elif source == "auto":
        declared = checkpoint.get("training", {})
        fields = ("reward", "value_head", "gamma", "rscale", "sep_value")
        if any(declared.get(k) != getattr(args, k) for k in fields):
            raise SystemExit(
                "utility resume: checkpoint training target/readout is unknown or differs from this run; "
                "use --utility-source checkpoint to explicitly accept its critic, or --utility-source init "
                "to reset a separate critic; consider --resume-vwarm for a target switch")
        source = "checkpoint"
    if source == "init" and checkpoint is not None and not args.sep_value:
        raise SystemExit("--utility-source init after resume requires --sep-value 1: resetting a shared "
                         "critic's trunk would overwrite the restored policy")
    tag = "vnet" if args.sep_value else "net"
    sd = init_weights if source == "init" else checkpoint.get(tag)
    if not isinstance(sd, dict) or not all(k in sd for k in ("utility.weight", "utility.bias")):
        label = "init weights" if source == "init" else f"checkpoint {tag} weights"
        raise SystemExit(f"--value-head utility: {label} carry no utility head (weight and bias required); "
                         "refusing to start from a random critic")
    h = hashlib.sha256()
    arrays = []
    for k in ("utility.weight", "utility.bias"):
        v = sd[k]
        v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        if not np.isfinite(v).all():
            raise SystemExit(f"utility source {source}: non-finite {k}")
        arrays.append(v)
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode())
        h.update(v.tobytes())
    if arrays[0].ndim != 2 or arrays[0].shape[0] != 1 or arrays[1].shape != (1,):
        raise SystemExit(f"utility source {source}: invalid weight/bias shapes")
    return {"source": source, "checkpoint_model": tag if source == "checkpoint" else None,
            "path": os.path.abspath(args.resume if source == "checkpoint" else args.init),
            "initial_head_sha256": h.hexdigest(),
            "declaration": ("explicit-checkpoint" if args.utility_source == "checkpoint" else
                            "matching-checkpoint-contract" if source == "checkpoint" else "init-file"),
            "previous_training": checkpoint.get("training") if checkpoint is not None else None,
            "reset_critic_optimizer": source == "init" and checkpoint is not None,
            "calibration_verified": False}


def apply_utility_source(plan, ad, init_weights, net, vnet, vopt):
    """Called AFTER checkpoint restore; an explicit init reset copies the whole
    separate critic and clears its stale moments. Actor weights remain intact.
    """
    if plan is None:
        return
    if plan["reset_critic_optimizer"]:
        fresh = ad.make_model(init_weights)
        vnet.load_state_dict(fresh.state_dict())
        vopt.state.clear()
    for model in (net, vnet):
        if model is not None:
            model._value_head = "utility"
    # A head hash alone cannot identify the estimator: the same head on a
    # different trunk is a different critic. Hash the complete effective
    # initial critic AFTER all restore/reset operations.
    h = hashlib.sha256()
    model = vnet if vnet is not None else net
    for k, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().numpy()
        h.update(k.encode()); h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
        h.update(value.tobytes())
    plan["initial_critic_sha256"] = h.hexdigest()


def main():
    args = parse()
    args.out = os.path.abspath(args.out)
    # R03: resolve built-in defaults for the None-sentinel hyper-parameters (see parse()).
    lr0 = 1e-5 if args.lr is None else args.lr
    vlr0 = 3e-4 if args.vlr is None else args.vlr
    kl0 = 0.1 if args.kl is None else args.kl
    import torch
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    progress = {"it": None, "phase": "initialization"}
    def finite(value, label):
        try:
            require_finite(value, label)
        except FloatingPointError as exc:
            atomic_json(dict(progress, error=str(exc)), os.path.join(args.out, "numerical_failure.json"))
            raise
    def clip_gradients(parameters, label):
        parameters = list(parameters)
        finite([p.grad for p in parameters], label + ".gradients")
        try:
            return torch.nn.utils.clip_grad_norm_(parameters, args.gclip, error_if_nonfinite=True)
        except RuntimeError as exc:
            if "non-finite" not in str(exc):
                raise
            message = f"non-finite {label}.gradient_norm: {exc}"
            atomic_json(dict(progress, error=message), os.path.join(args.out, "numerical_failure.json"))
            raise FloatingPointError(message) from exc
    live = os.path.join(args.out, "live"); os.makedirs(live, exist_ok=True)
    dev = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("LEARNER_THREADS", "8")))
    ad = AD.get_adapter(args.adapter)
    w0 = AD.load_npz(args.init)
    stt = torch.load(args.resume, map_location="cpu") if args.resume else None
    if args.resume_vwarm < 0 or args.vwarm < 0:
        raise SystemExit("value warmup iteration counts must be non-negative")
    if args.resume_vwarm and not args.resume:
        raise SystemExit("--resume-vwarm needs --resume (use --vwarm for a fresh run)")
    utility_plan = utility_source_plan(args, w0, stt)
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
    opt = torch.optim.Adam(net.parameters(), lr=lr0)
    vopt = torch.optim.Adam(vnet.parameters(), lr=vlr0) if vnet is not None else None
    # shared model: value-warmup steps touch only the value head (trunk updates would move the policy)
    vhead = [q for n, q in net.named_parameters()
             if n.startswith(("utility.",) if args.value_head == "utility" else ("v.", "value."))]
    hopt = torch.optim.Adam(vhead, lr=vlr0) if (vnet is None and vhead) else None
    it0 = 0
    kl_coef = kl0
    if args.resume:
        _resume_sd(net, stt["net"], opt, stt["opt"], "net")
        if vnet is not None and stt.get("vnet"):
            _resume_sd(vnet, stt["vnet"], vopt, stt["vopt"], "vnet")
        it0 = stt["it"]
        # R03: load_state_dict restored the checkpoint's optimizer LR and the old code took the
        # checkpoint's kl_coef unconditionally, so a fresh --lr/--vlr/--kl on the command line was
        # silently ignored. Explicitly given values (not None) now win; unspecified ones inherit.
        kl_coef = args.kl if args.kl is not None else stt.get("kl_coef", kl_coef)
        if args.lr is not None:
            for _pg in opt.param_groups:
                _pg["lr"] = args.lr
        if vnet is not None and args.vlr is not None:
            for _pg in vopt.param_groups:
                _pg["lr"] = args.vlr
    apply_utility_source(utility_plan, ad, w0, net, vnet, vopt)
    for name, model in (("actor", net), ("critic", vnet), ("reference", ref)):
        if model is not None:
            finite(model.state_dict(), name)
    finite(opt.state_dict(), "actor_optimizer")
    if vopt is not None:
        finite(vopt.state_dict(), "critic_optimizer")
    if utility_plan is not None:
        h = hashlib.sha256()
        with open(utility_plan["path"], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        utility_plan["source_file_sha256"] = h.hexdigest()
    warm_iters = args.resume_vwarm if args.resume else args.vwarm
    warm_done = 0
    prior_warm = (stt or {}).get("resume_warmup") or {}
    if (args.resume and not (utility_plan or {}).get("reset_critic_optimizer")
            and prior_warm.get("at_it") == it0
            and prior_warm.get("requested") == args.resume_vwarm):
        warm_done = int(prior_warm["completed"])
        if not 0 <= warm_done <= warm_iters:
            raise SystemExit("invalid checkpoint resume_warmup progress")
        if hopt is not None and stt.get("hopt") is not None:
            hopt.load_state_dict(stt["hopt"])
            if args.vlr is not None:
                for group in hopt.param_groups:
                    group["lr"] = args.vlr
            finite(hopt.state_dict(), "warmup_optimizer")
    # R03: record the hyper-parameters that actually took effect (post-resume), so any run can be
    # audited against what was claimed on the command line (config.json keeps the raw claims).
    _eff = {"lr": float(opt.param_groups[0]["lr"]),
            "vlr": float(vopt.param_groups[0]["lr"]) if vopt is not None else
                   (float(hopt.param_groups[0]["lr"]) if hopt is not None else None),
            "kl_coef": float(kl_coef), "resumed_from": args.resume, "it0": it0,
            "reward": args.reward, "gamma": args.gamma, "rscale": args.rscale,
            "value_head": args.value_head, "u_mode_problems": u_mode_check(args),
            "utility_provenance": utility_plan, "critic_warmup_iters": warm_iters}
    _eff["resume_warmup_completed"] = warm_done
    _eff["resume_warmup_counts_as_iteration"] = False
    for _p in _eff["u_mode_problems"]:
        print(f"[u-mode] WARNING: {_p}", flush=True)
    json.dump(_eff, open(os.path.join(args.out, "effective_params.json"), "w"), indent=1)
    league = League(os.path.join(args.out, "league.json"), json.loads(args.pool), args.snap_prior, args.pfsp)
    if args.self_prior > 0:
        league.tab.setdefault("SELF", {"prior": args.self_prior, "n": 0, "wr": 0.5, "margin": 0.0})
        league.tab["SELF"]["prior"] = args.self_prior
    league.save()                         # split mode: the cloud actor needs the table before iteration 1
    for k, v in league.tab.items():      # resumed snapshots keep their files
        if v.get("snap") and not os.path.exists(k.split(":", 2)[2]):
            v["prior"] = 0.0
    tlog = open(os.path.join(args.out, "train.jsonl"), "a")
    wlog = open(os.path.join(args.out, "warmup.jsonl"), "a")
    elog = open(os.path.join(args.out, "eval.jsonl"), "a")
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    rng = np.random.default_rng(int(time.time()) % 100000 + it0)
    # Warm trajectories have their own reproducible seed range and do not
    # consume seeds belonging to formal iterations.
    seed = args.seed0 + it0 * args.games
    formal_seed = seed
    if args.resume and warm_done < warm_iters:
        seed = args.seed0 + 1000000000 + (it0 * max(1, warm_iters) + warm_done) * args.games

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
    if not args.resume and it0 == 0 and args.eval_every > 0:
        pending_eval.append((0, [pool.apply_async(RO.play_batch, (t,)) for t in eval_tasks(args, args.adapter, wk, 0)]))
    nxt = submit_rollouts(wk)
    t_start = time.time()
    it = it0
    while (it < it0 + args.iters or (args.resume and warm_done < warm_iters)) and time.time() - t_start < 60 * args.minutes:
        resume_warm = bool(args.resume and warm_done < warm_iters)
        if not resume_warm:
            it += 1
        warm = resume_warm or (not args.resume and it <= args.vwarm)
        progress.update(it=it, phase="warmup" if resume_warm else "train")
        res_async, t_sub = nxt
        res = res_async.get()
        t_roll = time.time() - t_sub
        if args.overlap and not resume_warm:
            nxt = submit_rollouts(wk)            # behaviour policy = current weights (1-iter stale for next update)
        games = [g for r in res for g in r["games"]]
        trajs = [g["recs"] for g in games]
        ndec = sum(len(t) for t in trajs)
        t_pol = sum(r["t_pol"] for r in res); t_opp = sum(r["t_opp"] for r in res)
        # ---------------- values, GAE
        t1 = time.time()
        buf = ad.build_buffer(trajs)
        t_buf = time.time() - t1
        offs = np.cumsum([0] + [len(t) for t in trajs])
        LP0 = buf.lp0 if hasattr(buf, "lp0") else np.array([r["logp"] for t in trajs for r in t], np.float32)
        for g in games:                      # the buffer holds the data now; drop the per-game copies
            g["recs"] = None
        trajs = None
        pmb = max(args.mb, args.pre_mb)      # the no-grad passes (values, reference policy) take larger batches
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
            for b0 in range(0, buf.n, pmb):
                idx = np.arange(b0, min(buf.n, b0 + pmb))
                bt = buf.batch(idx, dev)
                V[idx] = ad.evaluate(vmodel, bt, args.temp)["value"].float().cpu().numpy()
                if args.dec_frac >= 1:
                    orf = ad.evaluate(ref, bt, args.temp)        # reference policy cached once per iteration
                    if REFLP is None:
                        REFLP = torch.zeros((buf.n_ukeys, orf["lp"].shape[1]), device=dev)
                    REFLP[orf["ukey"]] = orf["lp"].float()
            if args.dec_frac < 1:
                for b0 in range(0, len(SUB), pmb):
                    bt = buf.batch(SUB[b0:b0 + pmb], dev)
                    orf = ad.evaluate(ref, bt, args.temp)
                    if REFLP is None:
                        REFLP = torch.zeros((buf.n_ukeys, orf["lp"].shape[1]), device=dev)
                    REFLP[orf["ukey"]] = orf["lp"].float()
        t_pre = time.time() - t1 - t_buf
        ADV = np.zeros(buf.n, np.float32); RET = np.zeros(buf.n, np.float32)
        for gi, g in enumerate(games):
            a_, r_ = gae(game_rewards(g, args, it), V[offs[gi]:offs[gi + 1]], args.gamma, args.lam)
            ADV[offs[gi]:offs[gi + 1]] = a_; RET[offs[gi]:offs[gi + 1]] = r_
        ADVn = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
        finite({"values": V, "advantages": ADV, "returns": RET,
                "normalized_advantages": ADVn, "behavior_logp": LP0}, "rollout")
        # ---------------- PPO epochs
        net.train()
        if vnet is not None:
            vnet.train()
        stats = []
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
                    finite(vl, "critic_loss")
                    vopt.zero_grad(); sc_v.scale(vl).backward(); sc_v.unscale_(vopt)
                    clip_gradients(vnet.parameters(), "critic")
                    sc_v.step(vopt); sc_v.update()
                    loss = pl - args.ent * ent.mean() + kl_coef * kl.mean()
                else:
                    vl = ((o["value"] - ret.clamp(-1, 1)) ** 2).mean()
                    loss = pl - args.ent * ent.mean() + kl_coef * kl.mean() + args.vf * vl
                    if warm:
                        loss = args.vf * vl
                finite({"loss": loss, "policy": pl, "value": vl, "kl": kl, "entropy": ent}, "losses")
                if warm and hopt is not None:
                    hopt.zero_grad(); (args.vf * vl).backward()
                    clip_gradients(vhead, "warmup"); hopt.step()
                    gn = 0.0
                elif not (warm and vnet is not None):
                    opt.zero_grad(); sc_p.scale(loss).backward(); sc_p.unscale_(opt)
                    gn = clip_gradients(net.parameters(), "actor").item()
                else:
                    gn = 0.0
                if not warm:
                    sc_p.step(opt); sc_p.update()
                with torch.no_grad():
                    cf = ((ratio - 1).abs() > args.clip).float().mean().item()
                stats.append([pl.item(), vl.item(), ent.mean().item(), kl.mean().item(), cf, gn,
                              (lp - tq(LP0[idx])).mean().item()])
        st = np.mean(stats, axis=0)
        finite(st, "update_statistics")
        for name, model in (("actor", net), ("critic", vnet)):
            if model is not None:
                finite(model.state_dict(), name)
        finite(opt.state_dict(), "actor_optimizer")
        if vopt is not None:
            finite(vopt.state_dict(), "critic_optimizer")
        if args.kl_target > 0 and not warm:
            kl_coef *= 1.5 if st[3] > 1.5 * args.kl_target else (1 / 1.5 if st[3] < args.kl_target / 1.5 else 1.0)
            kl_coef = float(np.clip(kl_coef, 1e-4, 10.0))
        t_upd = time.time() - t1
        if not resume_warm:
            for g in games:
                league.update(g["opp"], g["margin"])
        by = {}
        for g in games:
            by.setdefault(g["opp"], []).append(g["margin"])
        srv = server.stats() if server is not None else None
        rec = {"it": it, "t": round(time.time()), "games": len(games), "srv": srv, "dec": ndec, "units": int(len(buf.act)),
               "roll_s": round(t_roll, 1), "upd_s": round(t_upd, 1), "buf_s": round(t_buf, 1), "pre_s": round(t_pre, 1),
               "games_per_min": round(60 * len(games) / max(t_roll, 1e-6), 1),
               "dec_per_s": round(ndec / max(t_roll, 1e-6), 1),
               "pol_s": round(t_pol, 1), "opp_s": round(t_opp, 1),
               "pl": round(st[0], 4), "vl": round(st[1], 4), "ent": round(st[2], 3), "kl": round(st[3], 4),
               "kl_coef": round(kl_coef, 4), "clipfrac": round(st[4], 3), "gnorm": round(st[5], 3),
               "dlogp": round(st[6], 4), "warm": bool(warm),
               "adv_std": round(float(ADV.std()), 4), "v_mean": round(float(V.mean()), 3),
               "margin": {k: [round(float(np.mean(v))), len(v), int(sum(x > 0 for x in v))] for k, v in by.items()},
               "winrate": round(float(np.mean([g["margin"] > 0 for g in games])), 3),
               "planted": {k: [round(float(np.mean([g.get("planted", 0) for g in games if g["opp"] == k])), 1),
                               round(float(np.mean([g.get("planted_op", 0) for g in games if g["opp"] == k])), 1)]
                           for k in by},
               "buf_mb": round(buf.nbytes() / 1e6, 1), "rss_mb": round(rss_mb()), "workers_rss_mb": round(children_rss_mb())}
        if resume_warm:
            warm_done += 1
            rec["warmup_batch"] = warm_done
            rec["phase"] = "warmup"
        del buf, trajs, games, res
        if not resume_warm and it % args.snap_every == 0:
            sp = os.path.join(args.out, f"snap_{it}.npz")
            AD.save_npz(ad.export(net), sp, {"it": it})
            league.add_snap(f"SNAP:{args.adapter}:{sp}", args.max_snaps)
        league.save()
        if not resume_warm and (it % args.ckpt_every == 0 or it == it0 + args.iters):
            AD.save_npz(ad.export(net), os.path.join(args.out, f"ckpt_{it}.npz"), {"it": it})
        AD.save_npz(ad.export(net), os.path.join(args.out, "ckpt_last.npz"), {"it": it})
        state = {"net": net.state_dict(), "opt": opt.state_dict(),
                    "vnet": vnet.state_dict() if vnet is not None else None,
                    "vopt": vopt.state_dict() if vopt is not None else None,
                    "it": it, "kl_coef": kl_coef, "training": training_contract(args),
                    "utility_provenance": utility_plan,
                    "hopt": hopt.state_dict() if hopt is not None else None,
                    "resume_warmup": {"at_it": it0, "requested": warm_iters, "completed": warm_done}
                    if resume_warm else None}
        rec["checkpoint"] = commit_checkpoint(args.out, state, os.path.join(args.out, "ckpt_last.npz"), progress["phase"])
        log = wlog if resume_warm else tlog
        log.write(json.dumps(rec, allow_nan=False) + "\n"); log.flush(); os.fsync(log.fileno())
        print(json.dumps(rec, allow_nan=False), flush=True)
        wk = publish(it)
        if resume_warm and warm_done == warm_iters:
            seed = formal_seed
        if resume_warm or not args.overlap:
            nxt = submit_rollouts(wk)
        if not resume_warm and args.eval_every > 0 and it % args.eval_every == 0:
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
    tlog.close(); wlog.close(); elog.close()
    if server is not None:
        server.stop()


if __name__ == "__main__":
    main()
