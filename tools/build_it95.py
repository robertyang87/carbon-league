import hashlib
import os
import subprocess
import sys

CONF = '{"SEARCH": False, "ROLL": "self", "GUARD": True, "GUARD_NARROW": True, "MODE": "sweep", "MIX": False, "TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240, "BUDGET": 0.7, "HARD": 0.85, "ABORT_DL": 0.95, "START_DL": None}'

assert hashlib.md5(CONF.encode()).hexdigest() == "4ce785bf6bccc34b3adbd74aedd25dc0", "CONF literal md5 mismatch"

env = dict(os.environ)
env["CNET2_NO_ROLL"] = "1"

r = subprocess.run(
    [
        "/home/arm/junyang/carbon/.venv/bin/python",
        "cnet2_search_pack.py",
        "/home/arm/junyang/carbon-repo/code/exp/it95.py",
        CONF,
        "/home/arm/junyang/carbon-repo/data/weights/ckpt_95.npz",
    ],
    cwd="/home/arm/junyang/a4_pack_stage",
    env=env,
)
sys.exit(r.returncode)
