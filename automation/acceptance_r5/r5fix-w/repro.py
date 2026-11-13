"""Weight-chain P1 reproductions (owner re-review, post-73ebd4af).

P1-A: generation-owned fields (w) must be read from the verified
      generation, never from the flat root.
P1-B: set_uw must pin+verify the generation, refuse a torn publish
      (BUILD_FAILED.json), read features from the generation directory,
      and re-check the binding at registration time.

Run: <venv python> repro.py [workdir]
Exit 0 iff every scenario shows the FIXED behaviour.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import fixture  # noqa: E402

PY = sys.executable
BASE = os.path.join(HERE, "data")
RESULTS = []


def note(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'GREEN' if ok else 'RED  '}] {name}: {detail}", flush=True)


def run_load(ds):
    r = subprocess.run([PY, os.path.join(HERE, "child_load.py"), ds],
                       capture_output=True, text=True,
                       env={**os.environ, "BC_DEVICE": "cpu"})
    for line in r.stdout.splitlines():
        if line.startswith("VERDICT "):
            return json.loads(line[8:])
    return {"ok": False, "error": "no verdict", "stdout": r.stdout[-400:],
            "stderr": r.stderr[-400:]}


def run_set_uw(ds, *args):
    r = subprocess.run([PY, os.path.join(HERE, "code", "set_uw.py"), ds,
                        *[str(a) for a in args]],
                       capture_output=True, text=True)
    return r


def main():
    shutil.rmtree(BASE, ignore_errors=True)
    os.makedirs(BASE)

    # ---------- P1-A1: sealed w=[2,2,1,1], root w tampered to 123 ----------
    ds = fixture.build_ds(os.path.join(BASE, "dsA1"), with_w=True)
    np.save(os.path.join(ds, "w.npy"), np.full(4, 123, dtype=np.float16))
    v = run_load(ds)
    note("P1-A1 root-tampered w ignored",
         v.get("ok") and v.get("w") == [2.0, 2.0, 1.0, 1.0],
         f"load w={v.get('w')} (sealed [2,2,1,1], root tampered to 123) "
         f"mode={v.get('mode')} err={v.get('systemexit') or v.get('error')}")

    # ---------- P1-A2: sealed w, root w deleted ----------
    ds = fixture.build_ds(os.path.join(BASE, "dsA2"), with_w=True)
    os.remove(os.path.join(ds, "w.npy"))
    v = run_load(ds)
    note("P1-A2 sealed w not silently dropped",
         v.get("ok") and v.get("w") == [2.0, 2.0, 1.0, 1.0],
         f"load w={v.get('w')} (root w.npy deleted) mode={v.get('mode')} "
         f"err={v.get('systemexit') or v.get('error')}")

    # ---------- P1-A3 (regression): derived uw via set_uw still verified --
    ds = fixture.build_ds(os.path.join(BASE, "dsA3"), with_w=False)
    r = run_set_uw(ds, 0, 3, 1)
    exp = fixture.expected_uw_from_gen(ds)
    v = run_load(ds)
    got = v.get("uw_row0_head")
    note("P1-A3 derived uw consumed verified",
         r.returncode == 0 and v.get("ok") and v.get("mode") == "verified"
         and got == exp[0, :10].astype(float).tolist(),
         f"set_uw rc={r.returncode}, load mode={v.get('mode')}, "
         f"uw row0[:10]={got}, expected={exp[0, :10].tolist()}")
    # tamper the derived file -> sha mismatch must refuse
    uw_p = os.path.join(ds, "uw.npy")
    arr = np.load(uw_p); arr[0, 0] = 9.0; np.save(uw_p, arr)
    v = run_load(ds)
    note("P1-A3b tampered derived uw refused",
         not v.get("ok") and "sha256" in (v.get("systemexit") or ""),
         f"load ok={v.get('ok')} err={v.get('systemexit') or v.get('error')}")

    # ---------- P1-B1: torn publish -> set_uw must refuse ----------
    ds = fixture.build_ds(os.path.join(BASE, "dsB1"), with_w=False)
    torn = fixture.field_arrays(kind_torn=True)
    np.save(os.path.join(ds, "tok_kind.npy"), torn["tok_kind"])  # root = new gen
    with open(os.path.join(ds, "BUILD_FAILED.json"), "w") as fh:
        json.dump({"failed_stage": "publish", "generation": "gen-BBBB"}, fh)
    side_before = os.path.exists(os.path.join(ds, "derived.json"))
    r = run_set_uw(ds, 0, 3, 1)
    side_after = os.path.exists(os.path.join(ds, "derived.json"))
    w_written = os.path.exists(os.path.join(ds, "w.npy"))
    note("P1-B1 torn publish: set_uw refuses, no sidecar",
         r.returncode != 0 and side_after == side_before and not w_written,
         f"rc={r.returncode}, sidecar {side_before}->{side_after}, "
         f"w written={w_written}, out={(r.stdout + r.stderr).strip()[-160:]}")

    # ---------- P1-B2: normal set_uw reads GENERATION features ----------
    ds = fixture.build_ds(os.path.join(BASE, "dsB2"), with_w=False)
    # root mirror deliberately poisoned (no BUILD_FAILED): if set_uw read
    # the root it would compute the torn weights; the generation is truth.
    torn = fixture.field_arrays(kind_torn=True)
    np.save(os.path.join(ds, "tok_kind.npy"), torn["tok_kind"])
    r = run_set_uw(ds, 0, 3, 1)
    exp = fixture.expected_uw_from_gen(ds)
    uw_file = (np.load(os.path.join(ds, "uw.npy"))
               if os.path.exists(os.path.join(ds, "uw.npy")) else None)
    v = run_load(ds)
    note("P1-B2 set_uw weights from generation features",
         r.returncode == 0 and uw_file is not None
         and np.array_equal(uw_file, exp)
         and v.get("ok") and v.get("mode") == "verified"
         and v.get("uw_row0_head") == exp[0, :10].astype(float).tolist(),
         f"rc={r.returncode}, uw==hand-computed-from-generation: "
         f"{uw_file is not None and np.array_equal(uw_file, exp)}, "
         f"load mode={v.get('mode')}")

    # ---------- P1-B3: registration race -- pointer switched ----------
    sys.path.insert(0, os.path.join(HERE, "code"))
    import cnet2_build as B
    # second generation gen-BBBB with identical shape, pointer still AAAA
    genB = os.path.join(ds, "generations", "gen-BBBB")
    os.makedirs(genB)
    A = fixture.field_arrays(seed=7)
    import hashlib
    fields = {}
    for k, a in A.items():
        p = os.path.join(genB, k + ".npy"); np.save(p, a)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        fields[k] = {"file": k + ".npy", "rows": fixture.N, "sha256": h}
    with open(os.path.join(genB, "meta.json"), "w") as fh:
        json.dump({"games": [{"val": False, "key": "g0"}], "total": fixture.N}, fh)
    with open(os.path.join(genB, "manifest.json"), "w") as fh:
        json.dump({"generation": "gen-BBBB", "total": fixture.N,
                   "fields": fields}, fh)
    side_path = os.path.join(ds, "derived.json")
    side_bytes_before = open(side_path, "rb").read()
    # pointer switches to BBBB (a publish lands mid-flight)
    with open(os.path.join(ds, "current.json"), "w") as fh:
        json.dump({"generation": "gen-BBBB", "dir": "generations/gen-BBBB",
                   "total": fixture.N}, fh)
    raised = None
    try:
        B.register_derived(ds, [os.path.join(ds, "uw.npy")],
                           params={"tool": "race-test"},
                           expected_base="gen-AAAA")
    except Exception as e:  # noqa: BLE001
        raised = f"{type(e).__name__}: {e}"[:160]
    side_unchanged = open(side_path, "rb").read() == side_bytes_before
    note("P1-B3 register with stale expected_base refused",
         raised is not None and "TypeError" not in raised
         and "gen-BBBB" in raised and side_unchanged,
         f"raised={raised}, sidecar unchanged={side_unchanged}")
    # compat: without expected_base the call still works (binds current)
    try:
        B.register_derived(ds, [os.path.join(ds, "uw.npy")],
                           params={"tool": "race-test"})
        compat = True
    except Exception as e:  # noqa: BLE001
        compat = f"{type(e).__name__}: {e}"[:120]
    note("P1-B3b register_derived without expected_base still works",
         compat is True, f"result={compat}")

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n==== {len(RESULTS) - len(bad)}/{len(RESULTS)} GREEN"
          + (f"; RED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
