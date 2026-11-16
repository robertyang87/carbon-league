"""probe_adopt -- X05 adoption half.

Build one generation with the current main cnet2_build, corrupt ONE field
file in place inside the published generation directory (manifest.json left
untouched, file size unchanged), then re-run the identical build. On the
reviewed bytes the publisher's `_gen_complete` sees manifest + files present
and ADOPTS the corrupted generation wholesale: the bad bytes survive the
re-run, the generation directory is not rebuilt (inodes unchanged),
current.json keeps pointing at it, and the flat mirror is refreshed FROM the
corrupted generation -- while the module's own consumer-side verifier
(--resolve, verify=True) rejects the very same bytes.

Run:  python3 probe_adopt.py
Exit: 0 = defect reproduced; 2 = not reproduced (expected after a fix).
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x05_common import (first_build, md5_file, prepare, run_build, sha256_file)

FIELD = "value"  # small float16 field; one flipped data byte stays loadable


def main():
    pr, workdir, vendor, corpus = prepare("adopt")
    out_dir = os.path.join(workdir, "out")
    gen_id, gen_dir, man = first_build(pr, vendor, out_dir, corpus)

    field_path = os.path.join(gen_dir, FIELD + ".npy")
    man_path = os.path.join(gen_dir, "manifest.json")
    healthy_md5 = md5_file(field_path)
    declared_sha = man["fields"][FIELD]["sha256"]
    pr.check("fresh generation is sound: field sha256 == manifest sha256",
             sha256_file(field_path) == declared_sha,
             f"md5={healthy_md5}")
    gen_dir_ino = os.stat(gen_dir).st_ino
    field_ino = os.stat(field_path).st_ino
    manifest_md5_before = md5_file(man_path)

    # In-place tamper: flip one data byte (past the .npy header), size kept,
    # manifest untouched -- the "published, then changed under the same id"
    # case (bit rot / partial overwrite / drifted writer), NOT an input change.
    data = bytearray(open(field_path, "rb").read())
    off = 200 if len(data) > 208 else len(data) - 1
    data[off] ^= 0xFF
    with open(field_path, "wb") as fh:
        fh.write(bytes(data))
    bad_md5 = md5_file(field_path)
    pr.check("tamper landed: field md5 changed, size unchanged",
             bad_md5 != healthy_md5 and os.path.getsize(field_path) > 0,
             f"healthy_md5={healthy_md5} bad_md5={bad_md5} offset={off}")
    pr.check("tampered bytes now contradict the manifest's own sha256",
             sha256_file(field_path) != declared_sha,
             f"actual_sha256={sha256_file(field_path)} declared={declared_sha}")
    pr.check("manifest.json untouched by the tamper",
             md5_file(man_path) == manifest_md5_before)

    rc, rep, _so, se = run_build(vendor, out_dir, corpus)
    pr.check("re-run with identical inputs exits 0", rc == 0,
             f"rc={rc} stderr_tail={se[-200:]}")
    pr.check("re-run reports the same generation id",
             bool(rep) and rep.get("generation") == gen_id,
             f"reported={rep.get('generation') if rep else None}")
    ptr = json.load(open(os.path.join(out_dir, "current.json")))
    pr.check("current.json still points at the tampered generation",
             ptr["generation"] == gen_id)
    pr.check("ADOPTED, not rebuilt: generation dir inode unchanged",
             os.stat(gen_dir).st_ino == gen_dir_ino,
             f"ino {gen_dir_ino} -> {os.stat(gen_dir).st_ino}")
    pr.check("ADOPTED, not rebuilt: corrupted field inode unchanged",
             os.stat(field_path).st_ino == field_ino)
    pr.check("bad bytes survive the re-run (publisher never compared hashes)",
             md5_file(field_path) == bad_md5,
             f"post-run md5={md5_file(field_path)} bad_md5={bad_md5}")
    pr.check("manifest.json still not rewritten (no re-seal happened)",
             md5_file(man_path) == manifest_md5_before)
    flat_field = os.path.join(out_dir, FIELD + ".npy")
    pr.check("flat mirror now carries the BAD bytes (refreshed from the adopted gen)",
             md5_file(flat_field) == bad_md5,
             f"flat md5={md5_file(flat_field)}")

    # Contrast: the consumer-side verifier in the same module rejects these
    # exact bytes -- the publish path is the one that never looks.
    p = subprocess.run([sys.executable, os.path.join(vendor, "cnet2_build.py"),
                        "--resolve", out_dir],
                       capture_output=True, text=True, timeout=300)
    pr.check("contrast: --resolve (verify=True) rejects the adopted generation",
             p.returncode != 0 and "hash mismatch" in p.stderr,
             f"rc={p.returncode}")

    return pr.finish(workdir)


if __name__ == "__main__":
    sys.exit(main())
