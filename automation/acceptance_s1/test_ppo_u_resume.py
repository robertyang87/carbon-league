"""Utility resume source and actual critic-only warmup regression tests.

Builds tiny models/checkpoints in a temporary directory; no production data.
Run: python automation/acceptance_s1/test_ppo_u_resume.py
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
CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../code"))
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


def reject(fn, message):
    try:
        fn()
    except SystemExit as exc:
        return message in str(exc)
    return False


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def args(**kw):
    d = dict(adapter="cnet2", reward="u", value_head="utility", gamma=1., rscale=1.,
             sep_value=1, utility_source="auto", init="init.npz", resume="state.pt")
    d.update(kw)
    return types.SimpleNamespace(**d)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    ad = AD.get_adapter("cnet2")
    net = CNet2(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)
    opt = torch.optim.Adam(net.parameters(), lr=2e-5)
    # Nonzero moments/steps make accidental optimizer resets observable.
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
    w_init = {k: v.copy() for k, v in w.items()}
    w_init["utility.bias"][:] = 2.
    w_init["stem.weight"] += .25  # resetting only the head must fail the full-critic check
    nohead = {k: v for k, v in w.items() if not k.startswith(("utility.", "wdl.", "margin."))}

    plan = PT.utility_source_plan(args(), nohead, declared)
    check("init without heads + matching U checkpoint accepted", plan["source"] == "checkpoint")
    check("legacy checkpoint cannot silently override calibrated init",
          reject(lambda: PT.utility_source_plan(args(), w_init, legacy), "unknown or differs"))
    manual = PT.utility_source_plan(args(utility_source="checkpoint"), nohead, legacy)
    check("legacy source declaration accepts checkpoint and records uncertainty",
          manual["declaration"] == "explicit-checkpoint" and not manual["calibration_verified"])
    for field, value in (("reward", "worth"), ("value_head", "tanh"), ("gamma", .995),
                         ("rscale", .5), ("sep_value", 0)):
        other = copy.deepcopy(declared); other["training"][field] = value
        check("automatic resume rejects changed " + field,
              reject(lambda: PT.utility_source_plan(args(), w_init, other), "unknown or differs"))
    missing = copy.deepcopy(legacy)
    for k in ("utility.weight", "utility.bias"):
        del missing["vnet"][k]
    check("missing effective critic head refused even when init carries heads",
          reject(lambda: PT.utility_source_plan(args(utility_source="checkpoint"), w_init, missing), "carry no"))
    check("both sources missing heads refused",
          reject(lambda: PT.utility_source_plan(args(utility_source="checkpoint"), nohead, missing), "carry no"))
    bad = copy.deepcopy(legacy); bad["vnet"]["utility.bias"][0] = float("nan")
    check("non-finite checkpoint utility refused",
          reject(lambda: PT.utility_source_plan(args(utility_source="checkpoint"), nohead, bad), "non-finite"))
    check("explicit init reset cannot overwrite shared policy",
          reject(lambda: PT.utility_source_plan(args(utility_source="init", sep_value=0), w_init, legacy),
                 "requires --sep-value 1"))
    check("utility source flag cannot silently affect legacy tanh mode",
          reject(lambda: PT.utility_source_plan(args(value_head="tanh", utility_source="init"), w_init, legacy),
                 "only valid"))

    actor, critic = ad.make_model(w_init), ad.make_model(w_init)
    ao = torch.optim.Adam(actor.parameters(), lr=1e-5)
    vo = torch.optim.Adam(critic.parameters(), lr=1e-5)
    PT._resume_sd(actor, legacy["net"], ao, copy.deepcopy(legacy["opt"]), "test net")
    PT._resume_sd(critic, legacy["vnet"], vo, copy.deepcopy(legacy["vopt"]), "test vnet")
    PT.apply_utility_source(manual, ad, w_init, actor, critic, vo)
    check("checkpoint source preserves parameters, moments, lr and steps exactly",
          same(actor.state_dict(), base_sd) and same(critic.state_dict(), base_sd)
          and same(ao.state_dict(), base_opt) and same(vo.state_dict(), base_opt))
    reset = PT.utility_source_plan(args(utility_source="init"), w_init, legacy)
    PT.apply_utility_source(reset, ad, w_init, actor, critic, vo)
    check("explicit init resets complete critic and moments while preserving actor",
          same(critic.state_dict(), ad.make_model(w_init).state_dict()) and not vo.state
          and same(actor.state_dict(), base_sd) and same(ao.state_dict(), base_opt))

    # Exercise main, rollout, GAE, backward and checkpoint publication together.
    with tempfile.TemporaryDirectory() as td:
        init_path = os.path.join(td, "legacy_init.npz"); np.savez(init_path, **nohead)
        for separate in (1, 0):
            state = copy.deepcopy(declared)
            state["training"]["sep_value"] = separate
            if not separate:
                state["vnet"] = state["vopt"] = None
            state_path = os.path.join(td, f"state_{separate}.pt")
            torch.save(state, state_path)
            out = os.path.join(td, f"run_{separate}")
            cmd = [sys.executable, os.path.join(CODE, "rl2_ppo_train.py"), "--adapter", "cnet2",
                   "--init", init_path, "--resume", state_path, "--out", out, "--device", "cpu",
                   "--reward", "u", "--gamma", "1", "--rscale", "1", "--value-head", "utility",
                   "--sep-value", str(separate), "--resume-vwarm", "1", "--workers", "1", "--games", "2",
                   "--pool", "[]", "--self-prior", "1", "--eval-every", "0", "--iters", "0",
                   "--epochs", "1", "--mb", "512", "--pre-mb", "1024", "--overlap", "0"]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=CODE)
            check(f"actual main resume/warmup, separate={separate}", cp.returncode == 0)
            if cp.returncode:
                print(cp.stdout[-2000:] + cp.stderr[-2000:]); continue
            got = torch.load(os.path.join(out, "state_last.pt"), map_location="cpu", weights_only=False)
            effective = json.load(open(os.path.join(out, "effective_params.json")))
            check(f"warmup does not consume formal iteration or budget, separate={separate}",
                  got["it"] == 7 and not open(os.path.join(out, "train.jsonl")).read()
                  and json.loads(open(os.path.join(out, "warmup.jsonl")).readline())["it"] == 7)
            policy_keys = [k for k in base_sd if separate or not k.startswith("utility.")]
            check(f"warmup preserves policy and policy optimizer, separate={separate}",
                  all(torch.equal(base_sd[k], got["net"][k]) for k in policy_keys)
                  and same(base_opt, got["opt"]))
            got_critic = got["vnet"] if separate else got["net"]
            check(f"warmup actually trains utility, separate={separate}",
                  not torch.equal(base_sd["utility.weight"], got_critic["utility.weight"]))
            check(f"effective/checkpoint provenance and contract persist, separate={separate}",
                  effective["utility_provenance"]["source"] == "checkpoint"
                  and len(effective["utility_provenance"]["initial_critic_sha256"]) == 64
                  and len(effective["utility_provenance"]["source_file_sha256"]) == 64
                  and effective["critic_warmup_iters"] == 1
                  and got["training"] == PT.training_contract(args(sep_value=separate)))
            check(f"warmup does not publish a formal snapshot, separate={separate}",
                  not os.path.exists(os.path.join(out, "ckpt_7.npz"))
                  and not os.path.exists(os.path.join(out, "snap_7.npz")))
            if separate:
                # Continue a completed warm stage: do not warm twice, and
                # --iters is still the number of FORMAL PPO updates.
                resumed = os.path.join(td, 'formal_after_warm')
                formal_cmd = cmd.copy()
                for key, value in (("--resume", os.path.join(out, "state_last.pt")),
                                   ("--out", resumed), ("--iters", "1")):
                    formal_cmd[formal_cmd.index(key)+1] = value
                follow = subprocess.run(formal_cmd, capture_output=True, text=True, timeout=180, cwd=CODE)
                check("completed warm stage resumes into exactly one formal iteration", follow.returncode == 0)
                if follow.returncode:
                    print(follow.stdout[-2000:]+follow.stderr[-2000:])
                else:
                    formal_state = torch.load(os.path.join(resumed, 'state_last.pt'), weights_only=False)
                    check("formal iteration budget and actor update preserved after warm",
                          formal_state['it'] == 8 and formal_state['resume_warmup'] is None
                          and not same(formal_state['net'], got['net'])
                          and not open(os.path.join(resumed, 'warmup.jsonl')).read()
                          and len(open(os.path.join(resumed, 'train.jsonl')).readlines()) == 1)
                # Fixture for an interrupted two-batch warm stage, saved
                # after batch one. Only the remaining critic batch may run.
                partial = copy.deepcopy(got)
                partial['resume_warmup']['requested'] = 2
                partial_path = os.path.join(td, 'partial_warm.pt'); torch.save(partial, partial_path)
                partial_out = os.path.join(td, 'partial_continued')
                partial_cmd = cmd.copy()
                for key, value in (("--resume", partial_path), ("--out", partial_out), ("--resume-vwarm", "2")):
                    partial_cmd[partial_cmd.index(key)+1] = value
                continued = subprocess.run(partial_cmd, capture_output=True, text=True, timeout=180, cwd=CODE)
                check("interrupted critic warm stage can resume", continued.returncode == 0)
                if continued.returncode:
                    print(continued.stdout[-2000:]+continued.stderr[-2000:])
                else:
                    final = torch.load(os.path.join(partial_out, 'state_last.pt'), weights_only=False)
                    warm_rows = [json.loads(x) for x in open(os.path.join(partial_out, 'warmup.jsonl'))]
                    check("warm continuation runs only the remaining batch, preserves it/actor/optimizer",
                          final['it'] == 7 and final['resume_warmup']['completed'] == 2
                          and len(warm_rows) == 1 and warm_rows[0]['warmup_batch'] == 2
                          and same(final['net'], partial['net']) and same(final['opt'], partial['opt']))
        for bad_field in ('backbone', 'optimizer'):
            bad_state = copy.deepcopy(declared)
            if bad_field == 'backbone':
                bad_state['vnet']['stem.weight'].flatten()[0] = float('nan')
            else:
                first = next(iter(bad_state['vopt']['state'].values()))
                first['exp_avg'].flatten()[0] = float('nan')
            bad_path = os.path.join(td, 'bad_'+bad_field+'.pt'); torch.save(bad_state, bad_path)
            bad_out = os.path.join(td, 'bad_'+bad_field); os.makedirs(bad_out)
            last = os.path.join(bad_out, 'state_last.pt'); torch.save(declared, last)
            healthy = open(last, 'rb').read()
            bad_cmd = cmd.copy()
            for key, value in (("--resume", bad_path), ("--out", bad_out), ("--sep-value", "1")):
                bad_cmd[bad_cmd.index(key)+1] = value
            failed = subprocess.run(bad_cmd, capture_output=True, text=True, timeout=180, cwd=CODE)
            check('non-finite '+bad_field+' fails before publishing and preserves recovery point',
                  failed.returncode != 0 and 'non-finite' in failed.stderr
                  and os.path.exists(os.path.join(bad_out, 'numerical_failure.json'))
                  and open(last, 'rb').read() == healthy
                  and not os.path.exists(os.path.join(bad_out, 'ckpt_last.npz')))
    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
