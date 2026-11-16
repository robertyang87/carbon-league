"""Root-manifest hard validation: every malformed manifest must die
with SystemExit naming the cause (no continuations are run).
Run from the stage dir."""
import json
import sys

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402

GOOD = json.load(open("../fixtures/roots.json"))


def expect_die(manifest, tag, lo=200, hi=299):
    p = f"../evidence/f0_{tag}.json"
    json.dump(manifest, open(p, "w"))
    try:
        D.load_roots(p, lo, hi)
    except SystemExit as e:
        print(f"  die[{tag}]: {e}")
        return
    raise AssertionError(f"manifest {tag} was accepted, expected die")


m = json.loads(json.dumps(GOOD))
m[0]["step"] = 150
expect_die(m, "window")                     # step outside 200..299
m = json.loads(json.dumps(GOOD))
m[0]["step"] = 299                          # step == first terminal (299)
expect_die(m, "padding")
m = json.loads(json.dumps(GOOD))
m.append(dict(m[0]))
expect_die(m, "duplicate")                  # duplicate (game, step, me)
m = json.loads(json.dumps(GOOD))
m[0]["game"] = "deadbeefcafe-1"
expect_die(m, "gameid")                     # game id != replay file id
m = json.loads(json.dumps(GOOD))
m[0]["me"] = 2
expect_die(m, "seat")                       # me not in (0, 1)
specs, sha = D.load_roots("../fixtures/roots.json", 200, 299)
assert len(specs) == 4 and len(sha) == 64
print(f"F0 PASS: 5 malformed manifests all die by name; good manifest "
      f"loads 4 roots, roots_sha256={sha[:16]}...")
