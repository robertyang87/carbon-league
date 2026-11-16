"""X04 replay-half proof against the FIXED driver: validate via
load_roots, swap the replay file's content at the same path, then
run_duel with the validated blobs. The measurement must consume
the VALIDATED content: u_parent must equal the established true
value 0.16894850197364647 for this root, despite the path now
holding a different game's bytes. Run from the stage2 dir."""
import shutil
import sys

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402

CANON = "/home/hatch/workspace/heads-v2-diff/dgx.npz"
TRUE_U = 0.16894850197364647
specs, sha, blobs = D.load_roots("../fixtures/manifest_swap.json", 200, 299)
shutil.copy("/home/hatch/workspace/carbon-league/replays/replays/"
            "85a5134446fc00fea1b44b3f85a3a1a0-1.json.gz", specs[0]["path"])
print(f"path swapped after validation; md5_of(path) now "
      f"{D.md5_of(specs[0]['path'])[:16]}... (validated: "
      f"{specs[0]['replay_md5'][:16]}...)")
res = D.run_duel(CANON, CANON, specs, sha, replay_blobs=blobs)
u = res["rows"][0]["u_parent"]
print(f"u_parent = {u!r}; true = {TRUE_U!r}")
assert u == TRUE_U, "consumed content differs from validated content"
print("RESULT: BOUND -- post-validation swap is inert; the measured "
      "scene came from the validated bytes")
