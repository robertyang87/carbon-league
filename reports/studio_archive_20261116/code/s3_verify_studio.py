"""S3 Studio on-machine md5 verification (frozen set + script + pack).

Rules per protocol SS10.2 / real_env.py:
- whole-file md5 for script, 7 frozen proto files, vendor feat/np/fastsim, pack
- cnet2_search: md5 of file minus first 3 header comment lines (tail -n +4)
"""
import hashlib
import sys

BASE = "/Users/arm/junyang/carbon/data/s3_enum_studio"


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_tail4(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        lines = f.readlines()
    for ln in lines[3:]:
        h.update(ln)
    return h.hexdigest()


EXPECT = [
    ("script", f"{BASE}/s3_freeze/enumerate_roots.py",
     "1218d573e81f6f9b614a61597968ea45", "file"),
    ("s1_verify3", f"{BASE}/proto/s1_verify3.py",
     "17310cc1dfc2f32519bef6521f38048b", "file"),
    ("real_env", f"{BASE}/proto/real_env.py",
     "f9bfacb72e112859459a57aceebbbf46", "file"),
    ("comparator", f"{BASE}/proto/comparator.py",
     "53e5c1a1dd6ccc1a1f3343a47fc9b589", "file"),
    ("smoke_real_s1", f"{BASE}/proto/smoke_real_s1.py",
     "2425f34066c1b2b56208bdd4a2b7ac10", "file"),
    ("telemetry", f"{BASE}/proto/telemetry.py",
     "91a7121f02bc848045a5dcae9554ec0e", "file"),
    ("leaf_adapters", f"{BASE}/proto/leaf_adapters.py",
     "a5852c73f7187bb3e179428a81a6fba9", "file"),
    ("pack_bridge", f"{BASE}/proto/pack_bridge.py",
     "ad522153d32b7cbc5008ef0b59638c0f", "file"),
    ("cnet2_feat", f"{BASE}/proto/vendor/cnet2_feat.py",
     "2a45b08b6f97508541e62867e408748e", "file"),
    ("cnet2_np", f"{BASE}/proto/vendor/cnet2_np.py",
     "d8c15ad4fdd8ee079363cf215d52daf3", "file"),
    ("fastsim", f"{BASE}/proto/vendor/fastsim.py",
     "aefe6c83826efae4de83bef8c643c80b", "file"),
    ("cnet2_search", f"{BASE}/proto/vendor/cnet2_search.py",
     "94f7fb6438bb7577b141e388f068e122", "tail4"),
    ("echo4_pack", "/tmp/echo4_int8.py",
     "819250bfc82632786063dabd46a616bd", "file"),
]

ok = True
for name, path, want, rule in EXPECT:
    got = md5_tail4(path) if rule == "tail4" else md5_file(path)
    if got != want:
        ok = False
    print(f"{'OK ' if got == want else 'FAIL'} {name}: {got} (want {want})")
print("VERIFY_ALL_OK" if ok else "VERIFY_FAILED")
sys.exit(0 if ok else 1)
