"""W05 repro: published generation's meta.json is corrupted IN PLACE;
an identical republish then goes through exit_merge's reuse verification.
Question 1: does merge notice? (expected baseline: no -- meta.json is the
one product file never re-hashed on the reuse path.)
Question 2: does the trainer's pointer-side verify catch it downstream?
Also records the clean-republish behaviour that must be preserved."""
import hashlib
import io
import json
import os
import sys
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "run")
sys.path.insert(0, RUN)
WORK = os.path.join(ROOT, "work")
SHARDS = os.path.join(WORK, "shards_w05")

import exit_merge  # noqa: E402
from cnet2_exit_train import _verify_exit_generation, load_exit, resolve_exit_arg  # noqa: E402


def merge(out):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_merge.main(out, [SHARDS])
    lines = [l for l in buf.getvalue().splitlines() if l.startswith("{")]
    return json.loads(lines[-1])


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


res = {}
out = os.path.join(WORK, "ds_w05")
r1 = merge(out)
gen = r1["generation"]
gen_dir = os.path.join(out, "generations", gen)
meta_p = os.path.join(gen_dir, "meta.json")
man_p = os.path.join(gen_dir, "manifest.json")
res["gen"] = gen
res["meta_sha_at_publish"] = sha(meta_p)
man1 = open(man_p, "rb").read()
ino1 = os.stat(gen_dir).st_ino

# ---- clean republish (behaviour to preserve) ----
r2 = merge(out)
res["clean_republish"] = {"reused": r2.get("reused"),
                          "manifest_bytes_identical": open(man_p, "rb").read() == man1,
                          "gen_dir_same_inode": os.stat(gen_dir).st_ino == ino1}

# ---- corrupt meta.json in place (valid JSON, wrong content) ----
meta = json.load(open(meta_p))
meta["total"] = int(meta["total"]) + 1
meta["games"][0]["n"] = int(meta["games"][0]["n"]) + 1
with open(meta_p, "w") as fh:
    json.dump(meta, fh)
res["meta_sha_after_corruption"] = sha(meta_p)
res["meta_corrupted"] = res["meta_sha_after_corruption"] != res["meta_sha_at_publish"]

# ---- identical republish against the corrupted published generation ----
try:
    r3 = merge(out)
    res["corrupt_republish"] = {"outcome": "PUBLISHED", "reused": r3.get("reused"),
                                "generation": r3["generation"]}
except SystemExit as e:
    res["corrupt_republish"] = {"outcome": "DIED", "message": str(e)[:220]}

# ---- trainer backstop: pointer-side full verify + load path ----
ptr = json.load(open(os.path.join(out, "current.json")))
try:
    _verify_exit_generation(gen_dir, ptr)
    res["trainer_verify"] = "PASS"
except SystemExit as e:
    res["trainer_verify"] = "REFUSED: " + str(e)[:220]
buf = io.StringIO()
try:
    with redirect_stdout(buf):
        d = resolve_exit_arg(out)
        load_exit(d)
    res["trainer_load_path"] = "LOADED"
except SystemExit as e:
    res["trainer_load_path"] = "REFUSED: " + str(e)[:220]

print(json.dumps(res, indent=1))
