"""Regression fixtures for the resume-warmup claim guard in rl2_ppo_train.

An unfinished resume warmup recorded in a checkpoint must never be silently
skipped: a relaunch either claims it with the identical --resume-vwarm, or
abandons it explicitly with --abandon-resume-vwarm (the abandonment stays on
the record in effective_params and in the next checkpoint state). Any other
relaunch against an unfinished (or inconsistent) record is refused before
training starts, and nothing is published.

Builds tiny models/checkpoints in a temporary directory; no production data.
Run: python test_ppo_warmup_claim.py  (from code/, or from an acceptance tree)
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import types

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "rl2_ppo_train.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))
sys.path.insert(0, CODE)

import numpy as np
import torch
import rl2_ppo_train as PT
import rl2_ppo_adapters as AD
from cnet2_model import CNet2

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def args(**kw):
    d = dict(adapter="cnet2", reward="u", value_head="utility", gamma=1., rscale=1.,
             sep_value=1, utility_source="auto", init="init.npz", resume="state.pt")
    d.update(kw)
    return types.SimpleNamespace(**d)


def trainer_cmd(init_path, resume, out, iters, extra=()):
    cmd = [sys.executable, os.path.join(CODE, "rl2_ppo_train.py"), "--adapter", "cnet2",
           "--init", init_path, "--resume", resume, "--out", out, "--device", "cpu",
           "--reward", "u", "--gamma", "1", "--rscale", "1", "--value-head", "utility",
           "--sep-value", "1", "--workers", "1", "--games", "2",
           "--pool", "[]", "--self-prior", "1", "--eval-every", "0", "--iters", str(iters),
           "--epochs", "1", "--mb", "512", "--pre-mb", "1024", "--overlap", "0"]
    cmd.extend(extra)
    return cmd


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=CODE)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    ad = AD.get_adapter("cnet2")
    net = CNet2(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)
    opt = torch.optim.Adam(net.parameters(), lr=2e-5)
    sum(p.square().sum() for p in net.parameters()).backward()
    opt.step(); opt.zero_grad()
    with torch.no_grad():
        net.utility.bias.fill_(-2.)
    base_sd = copy.deepcopy(net.state_dict())
    base_opt = copy.deepcopy(opt.state_dict())
    legacy = dict(net=base_sd, opt=base_opt, vnet=copy.deepcopy(base_sd),
                  vopt=copy.deepcopy(base_opt), it=7, kl_coef=.15)
    declared = dict(legacy, training=PT.training_contract(args()))
    w = ad.export(net)
    nohead = {k: v for k, v in w.items() if not k.startswith(("utility.", "wdl.", "margin."))}

    with tempfile.TemporaryDirectory() as td:
        init_path = os.path.join(td, "legacy_init.npz")
        np.savez(init_path, **nohead)
        state_path = os.path.join(td, "state.pt")
        torch.save(copy.deepcopy(declared), state_path)

        # Base run: complete a one-batch warm stage, then re-mark it as batch
        # 1 of 2 -- the interrupted-stage record the guard exists for.
        warm_out = os.path.join(td, "warm_run")
        cp = run(trainer_cmd(init_path, state_path, warm_out, 0, ("--resume-vwarm", "1")))
        check("base warm run completes", cp.returncode == 0)
        if cp.returncode:
            print(cp.stdout[-2000:] + cp.stderr[-2000:])
            return 1
        partial = torch.load(os.path.join(warm_out, "state_last.pt"),
                             map_location="cpu", weights_only=False)
        assert partial["resume_warmup"] == {"at_it": 7, "requested": 1, "completed": 1}, \
            partial["resume_warmup"]
        partial["resume_warmup"]["requested"] = 2
        partial_path = os.path.join(td, "partial_warm.pt")
        torch.save(partial, partial_path)

        # A. Flag omitted against an unfinished record: refuse, publish nothing.
        out_a = os.path.join(td, "refuse_omitted")
        cp = run(trainer_cmd(init_path, partial_path, out_a, 1))
        check("omitted flag against unfinished warmup is refused",
              cp.returncode != 0 and "unfinished resume warmup" in cp.stderr)
        check("refused launch publishes no checkpoint",
              not os.path.exists(os.path.join(out_a, "state_last.pt")))

        # B. Mismatched flag (1 != recorded 2): refuse as well.
        out_b = os.path.join(td, "refuse_mismatch")
        cp = run(trainer_cmd(init_path, partial_path, out_b, 0, ("--resume-vwarm", "1")))
        check("mismatched flag against unfinished warmup is refused",
              cp.returncode != 0 and "unfinished resume warmup" in cp.stderr)

        # F. Inconsistent record (completed > requested): refuse without the flag.
        corrupt = copy.deepcopy(partial)
        corrupt["resume_warmup"]["completed"] = 3
        corrupt_path = os.path.join(td, "corrupt_warm.pt")
        torch.save(corrupt, corrupt_path)
        out_f = os.path.join(td, "refuse_corrupt")
        cp = run(trainer_cmd(init_path, corrupt_path, out_f, 0))
        check("inconsistent warmup record is refused, not erased",
              cp.returncode != 0 and "unfinished resume warmup" in cp.stderr)

        # C. Explicit abandonment: formal training proceeds, and the discard
        # is on the record in effective_params and in the saved state.
        out_c = os.path.join(td, "abandoned")
        cp = run(trainer_cmd(init_path, partial_path, out_c, 1,
                             ("--abandon-resume-vwarm",)))
        check("abandon flag lets formal training proceed", cp.returncode == 0)
        if cp.returncode:
            print(cp.stdout[-2000:] + cp.stderr[-2000:])
        else:
            eff = json.load(open(os.path.join(out_c, "effective_params.json")))
            st_c = torch.load(os.path.join(out_c, "state_last.pt"),
                              map_location="cpu", weights_only=False)
            check("abandonment recorded in effective_params",
                  eff.get("resume_warmup_abandoned") == {"at_it": 7, "requested": 2, "completed": 1})
            check("abandonment recorded in checkpoint state",
                  st_c["resume_warmup"] == {"at_it": 7, "requested": 2, "completed": 1,
                                            "abandoned": True})
            check("abandoned run trains formally, no warmup rows",
                  st_c["it"] == 8
                  and not open(os.path.join(out_c, "warmup.jsonl")).read()
                  and len(open(os.path.join(out_c, "train.jsonl")).readlines()) == 1)

            # D. The abandoned record is inert history for the next launch.
            out_d = os.path.join(td, "after_abandon")
            cp = run(trainer_cmd(init_path, os.path.join(out_c, "state_last.pt"),
                                 out_d, 1))
            check("launch after abandonment needs no flag", cp.returncode == 0)
            if cp.returncode == 0:
                st_d = torch.load(os.path.join(out_d, "state_last.pt"),
                                  map_location="cpu", weights_only=False)
                check("post-abandon formal state carries no pending record",
                      st_d["resume_warmup"] is None and st_d["it"] == 9)

        # E. Identical flag claims the record: only the remaining batch runs.
        out_e = os.path.join(td, "claimed")
        cp = run(trainer_cmd(init_path, partial_path, out_e, 0, ("--resume-vwarm", "2")))
        check("identical flag claims the unfinished warmup", cp.returncode == 0)
        if cp.returncode == 0:
            st_e = torch.load(os.path.join(out_e, "state_last.pt"),
                              map_location="cpu", weights_only=False)
            warm_rows = [json.loads(x) for x in open(os.path.join(out_e, "warmup.jsonl"))]
            check("claim runs only the remaining batch",
                  len(warm_rows) == 1 and warm_rows[0]["warmup_batch"] == 2
                  and st_e["resume_warmup"]["completed"] == 2 and st_e["it"] == 7)

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
