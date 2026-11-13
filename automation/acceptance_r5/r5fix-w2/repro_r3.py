"""Weight-chain round-3 reproductions (owner review bound to cda64b42).

P1: a generation switch DURING load() must not let the loader consume
    generation A's features with generation B's derived weights under a
    "verified" label -- identity is judged against the manifest load()
    pinned at resolve time; a pointer that moved mid-load refuses.
P2: set_uw must treat "pointer present but sealed generation damaged"
    as a hard refusal, not as a legacy dataset -- the legacy fallback is
    only for a dataset with NO generation pointer at all.

Run: <venv python> repro_r3.py
Exit 0 iff every scenario shows the FIXED behaviour.
"""
import hashlib
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
BASE = os.path.join(HERE, "data_r3")
RESULTS = []


def note(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'GREEN' if ok else 'RED  '}] {name}: {detail}", flush=True)


def run_child(script, ds):
    r = subprocess.run([PY, os.path.join(HERE, script), ds],
                       capture_output=True, text=True,
                       env={**os.environ, "BC_DEVICE": "cpu"})
    for line in r.stdout.splitlines():
        if line.startswith("VERDICT "):
            return json.loads(line[8:])
    return {"ok": False, "error": "no verdict",
            "stdout": r.stdout[-400:], "stderr": r.stderr[-400:]}


def run_set_uw(ds, *args):
    return subprocess.run([PY, os.path.join(HERE, "code", "set_uw.py"), ds,
                           *[str(a) for a in args]],
                          capture_output=True, text=True)


def main():
    shutil.rmtree(BASE, ignore_errors=True)
    os.makedirs(BASE)

    # ---------- P1: generation switch mid-load ----------
    ds = fixture.build_ds(os.path.join(BASE, "dsP1"), with_w=False)
    a_maps = os.path.join(ds, "generations", "gen-AAAA", "maps.npy")
    a_sha = hashlib.sha256(np.load(a_maps).tobytes()).hexdigest()
    v = run_child("child_load_switch.py", ds)
    err = v.get("systemexit") or v.get("error") or ""
    note("P1 mid-load switch refused (pinned A, pointer moved to B)",
         (not v.get("ok")) and "gen-AAAA" in err and "gen-BBBB" in err,
         f"load ok={v.get('ok')} mode={v.get('mode')} "
         f"check_gen={v.get('check_gen')} "
         f"uw[:4]={None if v.get('uw_row0_head') is None else v['uw_row0_head'][:4]} "
         f"(A truth 0.0, B weights 2.0) "
         f"features==A: {v.get('maps_sha') == a_sha} err={err[:200]}")

    # control: same dataset WITHOUT the switch loads verified normally
    # (set_uw for A first, so a derived layer exists to validate).
    ds = fixture.build_ds(os.path.join(BASE, "dsP1ctl"), with_w=False)
    r = run_set_uw(ds, 0, 3, 1)
    v = run_child("child_load.py", ds)
    exp = fixture.expected_uw_from_gen(ds)
    note("P1 control: no switch -> verified load, A's own weights",
         r.returncode == 0 and v.get("ok") and v.get("mode") == "verified"
         and v.get("check_gen") == "gen-AAAA"
         and v.get("uw_row0_head") == exp[0, :10].astype(float).tolist(),
         f"set_uw rc={r.returncode}, load ok={v.get('ok')} "
         f"mode={v.get('mode')} gen={v.get('check_gen')}")

    # ---------- P2a: pointer present, sealed maps.npy missing ----------
    ds = fixture.build_ds(os.path.join(BASE, "dsP2"), with_w=False)
    os.remove(os.path.join(ds, "generations", "gen-AAAA", "maps.npy"))
    r = run_set_uw(ds, 0, 3, 1)
    out = (r.stdout + r.stderr)
    wrote = [f for f in ("w.npy", "uw.npy", "derived.json")
             if os.path.exists(os.path.join(ds, f))]
    note("P2a damaged generation: set_uw refuses, names maps.npy, no writes",
         r.returncode != 0 and not wrote and "maps.npy" in out
         and "no generation pointer" not in out,
         f"rc={r.returncode}, written={wrote}, "
         f"out tail={out.strip()[-200:]}")

    # ---------- P2b: true legacy (no pointer) still works ----------
    ds = os.path.join(BASE, "legacy")
    os.makedirs(ds)
    A = fixture.field_arrays()
    for k, a in A.items():
        np.save(os.path.join(ds, k + ".npy"), a)
    with open(os.path.join(ds, "meta.json"), "w") as fh:
        json.dump({"games": [{"val": False, "key": "g0"}],
                   "total": fixture.N}, fh)
    r = run_set_uw(ds, 0, 3, 1)
    wrote = all(os.path.exists(os.path.join(ds, f))
                for f in ("w.npy", "uw.npy"))
    note("P2b true legacy dataset: set_uw completes with legacy warning",
         r.returncode == 0 and wrote
         and "no generation pointer" in (r.stdout + r.stderr),
         f"rc={r.returncode}, w/uw written={wrote}, "
         f"out tail={(r.stdout + r.stderr).strip()[-160:]}")

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n==== {len(RESULTS) - len(bad)}/{len(RESULTS)} GREEN"
          + (f"; RED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
