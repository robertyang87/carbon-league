"""X04 replay-half repro (pre-fix driver): validate a manifest via
load_roots, then REPLACE the replay file's content at the same path
(a different game's bytes), then run_duel. Pre-fix, extract_scene
re-reads by path, so the measurement consumes the swapped content
while the rows/header keep the pre-swap roots_sha256 and game id.
True u_parent for this root (healthy, established): 0.16894850197364647.
Run from the stage dir."""
import json
import shutil
import sys

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402

CANON = "/home/hatch/workspace/heads-v2-diff/dgx.npz"
TRUE_U = 0.16894850197364647
specs, sha = D.load_roots("../fixtures/manifest_swap.json", 200, 299)
print(f"validated: roots_sha256={sha[:16]}... replay_md5="
      f"{specs[0]['replay_md5'][:16]}...")
# swap: 85a5 content over the 8573-named copy (same path, same basename)
shutil.copy("/home/hatch/workspace/carbon-league/replays/replays/"
            "85a5134446fc00fea1b44b3f85a3a1a0-1.json.gz",
            specs[0]["path"])
print(f"swapped file at path; md5_of(path) now {D.md5_of(specs[0]['path'])[:16]}...")
res = D.run_duel(CANON, CANON, specs, sha)
row = res["rows"][0]
print(f"result: game={row['game']} roots_sha256={res['roots_sha256'][:16]}... "
      f"u_parent={row['u_parent']!r}")
print(f"true u_parent for the validated content: {TRUE_U!r}")
print("RESULT:", "CONSUMED SWAPPED CONTENT under the old roots identity"
      if row["u_parent"] != TRUE_U else "consumed validated content")
