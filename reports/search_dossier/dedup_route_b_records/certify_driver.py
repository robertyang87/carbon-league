#!/usr/bin/env python3
"""Certify both packs under identical conditions (relative gate, section 5.4).

Opponent: switch/ref_echo4_int8.py (baseline Echo4 pack, md5 819250bf) --
Master2's file is not present in the repo tree or on this VM; the section 5.4
criterion is relative (not worse than the current pack) under identical
conditions, which this pairing measures directly. Declared in the evidence.
Runs sequentially, --mem-limit-mib 512 (rusage hard gate), seed0 993000,
2 games. Quota: this VM's cgroup namespace exposes no cpu.max/memory.max
(/sys/fs/cgroup is empty even inside a systemd scope -- measured), so the
certify quota self-proof records nulls report-only and --require-quota is
NOT passed; the 150%/512M venue caliber absolute timing belongs to the C1
venue (S2). Attempt 1 (systemd scope + --require-quota) failed at harness
import because the checkout lacked env_src/; its records are kept under
certify_records/attempt1_incomplete_checkout/.
Resumable: skips a pack whose cert JSON already exists.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.path.join(HERE, "..", "certify_checkout")
OPP = os.path.join(HERE, "..", "ref_echo4_int8.py")
OUTDIR = os.path.join(HERE, "certify_records")
os.makedirs(OUTDIR, exist_ok=True)
LOG = os.path.join(HERE, "certify_driver.log")


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


JOBS = [
    ("orig", os.path.join(HERE, "..", "echo4_search_int8.py")),
    ("dedup", os.path.join(HERE, "echo4_search_int8_dedup.py")),
]

for name, pack in JOBS:
    out_json = os.path.join(OUTDIR, "cert_%s.json" % name)
    out_txt = os.path.join(OUTDIR, "cert_%s.stdout.txt" % name)
    if os.path.exists(out_json):
        log("skip %s (cert exists)" % name)
        continue
    cmd = [
        sys.executable, os.path.join(CHECKOUT, "certify_pack.py"),
        os.path.abspath(pack), "--opp", os.path.abspath(OPP),
        "--seed0", "993000", "--games", "2", "--steps-per-game", "299",
        "--mem-limit-mib", "512", "--out", out_json,
    ]
    log("run cert %s" % name)
    t0 = time.time()
    with open(out_txt, "w") as f:
        rc = subprocess.run(cmd, cwd=CHECKOUT, stdout=f, stderr=subprocess.STDOUT).returncode
    log("done cert %s rc=%d wall=%.0fs cert_exists=%s" % (name, rc, time.time() - t0, os.path.exists(out_json)))

log("CERTIFY-DRIVER-DONE")
