"""Regression fixture for the PPO run environment record.

effective_params.json now carries an "env" block: the CNET2_*/RL2_*
variables (plus BC_W / LEARNER_THREADS) the run actually played under.
The adapter and rollout read these at import; an offline evaluation
run under a bare environment once produced a false crash reading, and
before this record a run's receipt could not show its environment at
all. Unrelated variables must NOT be swept into the record.

Run: python test_ppo_env_record.py  (from code/, or an acceptance tree)
"""
import json
import os
import subprocess
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "rl2_ppo_train.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))
sys.path.insert(0, CODE)

import torch
import rl2_ppo_adapters as AD
from cnet2_model import CNet2

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def main():
    torch.set_num_threads(1)
    ad = AD.get_adapter("cnet2")
    net = CNet2(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)
    with tempfile.TemporaryDirectory() as td:
        init = os.path.join(td, "init.npz")
        AD.save_npz(ad.export(net), init, {})
        out = os.path.join(td, "run")
        env = dict(os.environ)
        env.update({"CNET2_TEACHER": "2", "RL2_SNAP_TEMP": "0.5",
                    "CNET2_TEMP_KIND": "0.15,0.3,0.3",
                    "UNRELATED_FIXTURE_VAR": "must-not-be-recorded"})
        cmd = [sys.executable, os.path.join(CODE, "rl2_ppo_train.py"), "--adapter", "cnet2",
               "--init", init, "--out", out, "--device", "cpu",
               "--reward", "u", "--gamma", "1", "--rscale", "1", "--value-head", "utility",
               "--sep-value", "1", "--workers", "1", "--games", "2",
               "--pool", "[]", "--self-prior", "1", "--eval-every", "0", "--iters", "1",
               "--epochs", "1", "--mb", "512", "--pre-mb", "1024", "--overlap", "0"]
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=CODE, env=env)
        eff_path = os.path.join(out, "effective_params.json")
        check("tiny run produced effective_params.json", os.path.exists(eff_path))
        if os.path.exists(eff_path):
            rec = json.load(open(eff_path)).get("env", {})
            check("operative CNET2_*/RL2_* values are recorded verbatim",
                  rec.get("CNET2_TEACHER") == "2" and rec.get("RL2_SNAP_TEMP") == "0.5"
                  and rec.get("CNET2_TEMP_KIND") == "0.15,0.3,0.3")
            check("unrelated variables are not swept in",
                  "UNRELATED_FIXTURE_VAR" not in rec)
        else:
            print(cp.stdout[-800:], cp.stderr[-800:])
    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
