"""Launch one S3 enumeration lane in its own process group (Mac scope equivalent).

Usage: launch_lane.py <lane_id> <games A-B> <out_dir> <shard_id>
The lane process becomes its own process-group leader (setsid), so the
watchdog can police/kill the whole lane by PGID. BLAS pinned to 1 before
numpy import (the script also pins it, this is belt-and-braces).
stdout/stderr go to <out_dir>/lane<lane_id>.log.
"""
import os
import sys

lane, games, out_dir, shard = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"
os.setsid()
log = open(os.path.join(out_dir, f"lane{lane}.log"), "ab", buffering=0)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
os.execv("/Users/arm/junyang/carbon/.venv/bin/python",
         ["python",
          "/Users/arm/junyang/carbon/data/s3_enum_studio/s3_freeze/enumerate_roots.py",
          "--games", games, "--out", out_dir, "--shard-id", shard])
