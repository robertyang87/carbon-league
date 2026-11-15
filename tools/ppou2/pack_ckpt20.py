#!/usr/bin/env python3
"""Pack ppou2_m ckpt_20 with the base pack's exact CONF; verify hashes."""
import hashlib
import json
import os
import subprocess
import sys

CODE = "/root/carbon/probe_m2/code"
EXP = os.path.join(CODE, "exp")
CKPT = "/root/carbon/data/rl3/ppou2_m/ckpt_20.npz"
OUT = os.path.join(EXP, "ppou2_m_it20.py")

base_src = open(os.path.join(EXP, "m3s2r1_base.py")).read()
i = base_src.find('CONF = {')
j = base_src.find('}', i)
conf = base_src[i + len('CONF = '):j + 1]
expect = ('{"SEARCH": False, "ROLL": "self", "GUARD": True, '
          '"GUARD_NARROW": True, "MODE": "sweep", "MIX": False, '
          '"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240, '
          '"BUDGET": 0.7, "HARD": 0.85, "ABORT_DL": 0.95, "START_DL": None}')
assert conf == expect, f"CONF mismatch:\n{conf}\nvs\n{expect}"

env = dict(os.environ)
env.pop("CNET2_ENS", None)
p = subprocess.run(
    ["/root/carbon/.venv/bin/python",
     "/root/carbon/code/packs/cnet2_search_pack.py", OUT, conf, CKPT],
    capture_output=True, text=True, cwd=EXP, env=env)
print(p.stdout.strip())
if p.returncode != 0:
    print("PACK FAILED rc=", p.returncode)
    print(p.stderr[-2000:])
    sys.exit(1)

data = open(OUT, "rb").read()
rec = {"out": OUT, "bytes": len(data),
       "md5": hashlib.md5(data).hexdigest(),
       "ah": json.loads(p.stdout.strip().splitlines()[-1]).get("out")}
print(json.dumps({"pack_md5": rec["md5"], "bytes": rec["bytes"]}))

base = [json.loads(l) for l in
        open("/root/carbon/probe_m2/data/bigeval_live/ppou2_base_m2_a.jsonl")
        if l.strip()]
print("baseline ah:", base[0]["ah"], "hh:", base[0]["hh"])
