"""PPO Terminal-U wiring checks (spec SS15 item 2). Runs from a repo checkout:
  python3 automation/acceptance_s1/test_ppo_u_mode.py
No fixtures: a tiny random CNet2 is built in memory; nothing is trained.

  1. reward: --reward u pays 2*(U-0.5) at the terminal step only, 0 elsewhere,
     antisymmetric between the seats, bounded by [-1, 1], no win bonus/shaping.
  2. return: with gamma=1 the GAE return of every step equals the terminal
     reward (lambda=1, zero baseline); with the default gamma=0.995 the first
     decision of a 299-step game sees 0.995**298 of it -- the launch contract
     u_mode_check() reports.
  3. critic readout: --value-head utility makes the cnet2 adapter return
     2*u-1 from the Stage 0 utility head (same log-probs as the tanh path,
     value in [-1, 1], gradient reaches utility.* and not value.*).
  4. u_mode_check flags gamma/rscale/shaping/value-head mismatches and is
     silent for the canonical launch.
  5. --value-head utility refuses init weights without a utility head.
Exit 0 iff everything passes.
"""
import math
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE = os.path.join(REPO, "code")
sys.path.insert(0, CODE)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rl2_ppo_train as PT  # noqa: E402
import rl2_ppo_adapters as AD  # noqa: E402
from cnet2_model import CNet2  # noqa: E402
from eval_v53 import CFG  # noqa: E402
from fastsim import FastCarbon  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)


def args(**kw):
    d = dict(reward="u", gamma=1.0, rscale=1.0, win_bonus=0.1, c_dep=0.0, c_lost=0.0, shape_iters=200,
             value_head="tanh")
    d.update(kw)
    return types.SimpleNamespace(**d)


def game(margin, T=299):
    rng = np.random.default_rng(1)
    return {"dmargin": rng.normal(0, 50, T).astype(np.float32), "dworth": rng.normal(0, 50, T).astype(np.float32),
            "dep": rng.random(T).astype(np.float32), "lost": rng.random(T).astype(np.float32), "margin": float(margin)}


def main():
    # 1. reward
    for m in (-13320.0, -500.0, -1.0, 0.0, 1.0, 500.0, 9000.0):
        r = PT.game_rewards(game(m), args(c_dep=0.3, c_lost=0.7), it=0)
        u = 0.5 * (1.0 if m > 0 else (0.5 if m == 0 else 0.0)) + 0.5 / (1.0 + math.exp(-m / 1000.0))
        r2 = PT.game_rewards(game(-m), args(), it=0)
        check(f"reward u, margin {m:+.0f}: terminal-only, = 2*(U-0.5), antisymmetric, no shaping/bonus",
              np.all(r[:-1] == 0) and abs(float(r[-1]) - 2 * (u - 0.5)) < 1e-6 and abs(float(r[-1] + r2[-1])) < 1e-6
              and -1.0 <= float(r[-1]) <= 1.0, f"r_T={float(r[-1]):+.4f} U={u:.4f}")
    # 2. return
    r = PT.game_rewards(game(800.0), args(), it=0)
    z = np.zeros(len(r), np.float32)
    _, ret1 = PT.gae(r, z, 1.0, 1.0)
    _, ret995 = PT.gae(r, z, 0.995, 1.0)
    check("gamma=1: every step's return equals the terminal reward", np.allclose(ret1, r[-1]), f"ret={float(ret1[0]):+.4f}")
    check("gamma=0.995: the first decision sees 0.995**298 of it",
          abs(float(ret995[0]) / float(r[-1]) - 0.995 ** 298) < 1e-4, f"share={float(ret995[0]) / float(r[-1]):.4f}")
    # 3. critic readout
    torch.manual_seed(0)
    cfg = dict(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)
    net = CNet2(**cfg).eval()
    ad = AD.get_adapter("cnet2")
    w = ad.export(net)
    actor = ad.make_actor(w)
    sim = FastCarbon(CFG).reset(seed=5)
    sts = [actor.new_game(0), actor.new_game(1)]
    rng = np.random.default_rng(0)
    recs = []
    for _ in range(12):
        outs = actor.act(sts, [sim.obs(0), sim.obs(1)], rng, 1.0, False)
        recs += [outs[0][1], outs[1][1]]
        sim.step(outs[0][0], outs[1][0])
    buf = ad.build_buffer([recs])
    b = buf.batch(np.arange(buf.n), torch.device("cpu"))
    o_t = ad.evaluate(net, b, 1.0)
    net._value_head = "utility"
    o_u = ad.evaluate(net, b, 1.0)
    with torch.no_grad():
        _, v_t, _, u, _, _ = net.forward_heads(b)
    vu = o_u["value"].detach()
    check("utility readout: value == 2*u-1 from forward_heads, within [-1, 1]",
          torch.allclose(vu, 2 * u - 1, atol=1e-6) and float(vu.abs().max()) <= 1.0,
          f"n={buf.n} range=[{float(vu.min()):+.3f}, {float(vu.max()):+.3f}]")
    check("utility readout: log-probs identical to the tanh path", torch.allclose(o_u["lp"], o_t["lp"], atol=1e-6))
    check("tanh path unchanged: value is the legacy tanh head", torch.allclose(o_t["value"], v_t, atol=1e-6)
          and not torch.allclose(o_t["value"], o_u["value"], atol=1e-4))
    net.zero_grad()
    ((o_u["value"] - 0.3) ** 2).mean().backward()
    gu = net.utility.weight.grad; gv = net.value.weight.grad
    check("utility readout: critic gradient reaches utility.*, not value.*",
          gu is not None and float(gu.abs().sum()) > 0 and (gv is None or float(gv.abs().sum()) == 0))
    # 4. launch contract
    check("u_mode_check: canonical launch is silent", PT.u_mode_check(args(value_head="utility")) == [])
    p = PT.u_mode_check(args(gamma=0.995, rscale=1 / 5000.0, c_dep=0.1))
    check("u_mode_check: default gamma / rscale / shaping are all flagged", len(p) == 3, "; ".join(x.split(" (")[0] for x in p))
    check("u_mode_check: utility head without --reward u is flagged",
          len(PT.u_mode_check(args(reward="dense", value_head="utility", gamma=0.995))) == 1)
    check("u_mode_check: legacy modes are not touched", PT.u_mode_check(args(reward="dense", gamma=0.995, rscale=2e-4)) == [])
    # 5. refusal without a utility head
    with tempfile.TemporaryDirectory() as td:
        w_no = {k: v for k, v in w.items() if not k.startswith(("utility.", "wdl.", "margin."))}
        wp = os.path.join(td, "nohead.npz")
        np.savez(wp, **w_no)
        cp = subprocess.run([sys.executable, os.path.join(CODE, "rl2_ppo_train.py"), "--adapter", "cnet2", "--init", wp,
                             "--out", os.path.join(td, "out"), "--device", "cpu", "--reward", "u", "--gamma", "1.0",
                             "--rscale", "1.0", "--value-head", "utility", "--iters", "0", "--workers", "1"],
                            capture_output=True, text=True, timeout=300, cwd=CODE)
        check("--value-head utility refuses init weights without a utility head",
              cp.returncode != 0 and "carry no utility head" in (cp.stderr + cp.stdout), f"rc={cp.returncode}")
    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
