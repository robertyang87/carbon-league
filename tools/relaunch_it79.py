#!/usr/bin/env python3
"""Relaunch snipe_n1 from it79 after the 01:14 OOM crash.
Rebuilds the exact it79-swap argv as a list (no shell quoting) and
starts the trainer detached, appending to the run log.
Run once from the shell: /root/carbon/.venv/bin/python relaunch_it79.py"""
import os, subprocess, sys

POOL = ('[["SNAP:cnet2:/root/carbon/data/weights/cnet2_m3s2g1L.npz#T2", 0.5], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_m6r1L.npz#T2", 0.5], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_m3s2r1L.npz#T2", 1.0], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_m7bsr1L.npz#T2", 0.5], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj260wr1_fromstudio.npz#T2", 0.5], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj260r1.npz#T2", 0.5], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj260g1.npz#T2", 0.25], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_m7bsv2r1.npz#T2", 0.5], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_m7bsv2g1.npz#T2", 0.25], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_caragor1_ep19.npz#T2", 0.5], '
        '["SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj320r1.npz#T2", 0.5]]')

argv = ["/root/carbon/.venv/bin/python", "rl2_ppo_train.py",
        "--adapter", "cnet2", "--device", "cuda", "--server", "1",
        "--reward", "u", "--vwarm", "3", "--self-prior", "0.5",
        "--snap-prior", "1", "--workers", "30", "--lockstep", "16",
        "--dec-frac", "0.25", "--epochs", "1", "--mb", "768",
        "--pre-mb", "4096", "--rscale", "1.0", "--gamma", "1.0",
        "--ckpt-every", "1", "--snap-every", "10", "--max-snaps", "2",
        "--eval-every", "5", "--eval-pairs", "32",
        "--eval-opps", "exp/opp_master2r.py,exp/opp_m3echo.py",
        "--lr", "1e-5", "--kl", "0.15",
        "--init", "/root/carbon/data/weights/cnet2_m3n1L.npz",
        "--games", "4000", "--out", "/root/carbon/data/rl3/snipe_n1",
        "--resume", "/root/carbon/data/rl3/snipe_n1/state_last.pt",
        "--pool", POOL,
        "--value-head", "utility", "--utility-source", "checkpoint",
        "--resume-vwarm", "1"]

env = dict(os.environ)
env.update({"CNET2_TEACHER": "2", "RL2_SNAP_TEMP": "0.5",
            "CNET2_TEMP_KIND": "0.15,0.3,0.3", "RL2_SERVERS": "1",
            "RL2_SERVER_MAXB": "1024",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

logf = open("/root/carbon/data/rl3/snipe_n1.log", "ab")
p = subprocess.Popen(argv, cwd="/root/carbon/code", env=env,
                     stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True)
print("RELAUNCHED pid=", p.pid, flush=True)
