#!/usr/bin/env python3
"""One-click execution of the PPO-U value-side diagnostic (A + B),
per the SIGNED PREREG_VALUE. Run on L20 (gpu-l20-6) by the L20 line:

    cd /root/ppo_attrib/value-diag && \
        /root/carbon/.venv/bin/python run_all.py

Pipeline: (0) copy the five surviving trainer states from the run
dir into value-diag/state/ (read-only source; training dir zero
writes) and verify every artifact md5 (six ckpt npz vs the Stage 1
receipt, five states vs the prep record, roots.npz, rootset32.json);
(1) realized_u.py; (2) extract_vhat.py (ckpt_82 -> ARTIFACT_MISSING
marker, the other five extracted); (3) ab_tables.py -> out/results.json.
All outputs land in /root/ppo_attrib/value-diag/out/.
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CKPT_MD5, ROOTS_MD5, ROOTSET_MD5, RUN_DIR, STATE_DIR,  # noqa: E402
                    STATE_MD5, STATE_SRC, VD, WORK, verify_md5)

HERE = os.path.dirname(os.path.abspath(__file__))


def step(name, script):
    print(f"=== {name} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       check=False)
    if r.returncode != 0:
        raise SystemExit(f"FAILED at {name} (rc={r.returncode})")


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    print("=== step 0: state copies + md5 verification ===", flush=True)
    for k, src_name in STATE_SRC.items():
        dst = os.path.join(STATE_DIR, f"state_it{k}.pt")
        src = os.path.join(RUN_DIR, src_name)
        if not (os.path.exists(dst)
                and __import__("common").md5_file(dst) == STATE_MD5[k]):
            shutil.copyfile(src, dst)
        verify_md5(dst, STATE_MD5[k], f"state_it{k}")
        print(f"state_it{k}.pt md5 OK ({src_name})")
    for k, expect in CKPT_MD5.items():
        verify_md5(os.path.join(WORK, "state", f"ckpt_{k}.npz"),
                   expect, f"ckpt_{k}")
    print("six ckpt npz md5 OK (82 verified as an artifact; its vhat "
          "state is the documented gap)")
    verify_md5(os.path.join(WORK, "roots.npz"), ROOTS_MD5, "roots.npz")
    verify_md5(os.path.join(WORK, "rootset32.json"), ROOTSET_MD5,
               "rootset32.json")
    print("roots.npz / rootset32.json md5 OK")
    step("step 1: realized U", "realized_u.py")
    step("step 2: vhat extraction", "extract_vhat.py")
    step("step 3: A/B tables + verdicts", "ab_tables.py")
    print("=== DONE ===")
    print("outputs:", os.path.join(VD, "out"))


if __name__ == "__main__":
    main()
