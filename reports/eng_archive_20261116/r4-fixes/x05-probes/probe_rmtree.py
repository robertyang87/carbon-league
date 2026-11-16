"""probe_rmtree -- X05 deletion half.

Build one generation with the current main cnet2_build, then make the
published generation INCOMPLETE under the same id (delete one declared field
file) and plant a sentinel file inside the generation directory. Re-run the
identical build. On the reviewed bytes `_gen_complete` returns False, the
publish path does shutil.rmtree(gen_dir) and renames a fresh staging dir into
its place: the whole old directory -- sentinel included, and including files
that were perfectly intact -- is destroyed instead of being quarantined for
inspection.

Run:  python3 probe_rmtree.py
Exit: 0 = defect reproduced; 2 = not reproduced (expected after a fix that
quarantine-renames incomplete generations instead of deleting them).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x05_common import first_build, prepare, run_build, sha256_file

MISSING_FIELD = "step"     # the declared field file we delete
BYSTANDER_FIELD = "value"  # an intact field: proves the WHOLE dir is replaced
SENTINEL = "X05_SENTINEL.txt"


def main():
    pr, workdir, vendor, corpus = prepare("rmtree")
    out_dir = os.path.join(workdir, "out")
    gen_id, gen_dir, man = first_build(pr, vendor, out_dir, corpus)

    missing_path = os.path.join(gen_dir, MISSING_FIELD + ".npy")
    bystander_path = os.path.join(gen_dir, BYSTANDER_FIELD + ".npy")
    pr.check("fresh generation is sound: step.npy sha256 == manifest sha256",
             sha256_file(missing_path) == man["fields"][MISSING_FIELD]["sha256"])
    gen_dir_ino = os.stat(gen_dir).st_ino
    bystander_ino = os.stat(bystander_path).st_ino
    declared_step_sha = man["fields"][MISSING_FIELD]["sha256"]

    # Make the same-id generation incomplete + plant forensic evidence.
    os.remove(missing_path)
    sentinel_path = os.path.join(gen_dir, SENTINEL)
    with open(sentinel_path, "w") as fh:
        fh.write("old generation dir marker -- a quarantine would keep me\n")
    pr.check("setup: declared field file missing, sentinel planted",
             not os.path.exists(missing_path) and os.path.exists(sentinel_path),
             f"gen_dir_ino={gen_dir_ino} bystander_ino={bystander_ino}")

    rc, rep, _so, se = run_build(vendor, out_dir, corpus)
    pr.check("re-run with identical inputs exits 0", rc == 0,
             f"rc={rc} stderr_tail={se[-200:]}")
    pr.check("re-run reports the same generation id",
             bool(rep) and rep.get("generation") == gen_id,
             f"reported={rep.get('generation') if rep else None}")
    ptr = json.load(open(os.path.join(out_dir, "current.json")))
    pr.check("current.json points at the rebuilt generation",
             ptr["generation"] == gen_id)
    pr.check("missing field file was rebuilt and matches the manifest again",
             os.path.exists(missing_path)
             and sha256_file(missing_path) == declared_step_sha)
    pr.check("DELETED, not quarantined: sentinel file is gone",
             not os.path.exists(sentinel_path))
    pr.check("whole directory replaced: generation dir inode changed",
             os.stat(gen_dir).st_ino != gen_dir_ino,
             f"ino {gen_dir_ino} -> {os.stat(gen_dir).st_ino}")
    pr.check("whole directory replaced: intact bystander field got a new inode too",
             os.stat(bystander_path).st_ino != bystander_ino,
             f"ino {bystander_ino} -> {os.stat(bystander_path).st_ino}")

    return pr.finish(workdir)


if __name__ == "__main__":
    sys.exit(main())
