"""rng_stream_probe -- batch-2 validation frame B (design S6.3):
direct, falsifiable comparison of the teacher's random streams
between the per-root replay path (pinned form: _eval_root falls back
to G.extract_scene) and the snapshot path (_game_root_scenes scenes
injected into the same _eval_root).

A recording wrapper around np.random.default_rng is installed in
this (dedicated) process: every Generator the teacher constructs is
recorded as a stream -- its seed argument verbatim, the number of
draw calls, and a rolling record of (method, return-value digest)
whose sha256 is the stream hash. Per root, the pinned path and the
snapshot path must produce identical stream lists (same seeds, same
call counts, same hashes): the belief stream R2
(default_rng([seed_base + m, 0xC0])) and the opponent stream R3
(default_rng(seed_base + m)) are thereby compared at the level of the
drawn values themselves, not only through their downstream effect on
q/Q (frame A's group-level evidence).

B3 negative control (--negative-control): the first root is re-run
on the pinned path with seed_base + 1; EVERY stream hash must
change. If any hash survives the seed shift, the recorder is not
hooked into the stream it claims to measure and the probe's green is
void -- fix the probe, not the annotator.

Validation-only: this probe never writes annotation output; it only
reads replays/weights and prints/writes its JSON report.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_REAL_DEFAULT_RNG = np.random.default_rng
_STREAMS = []


def _digest(out):
    if isinstance(out, np.ndarray):
        return ["arr", out.dtype.str, list(out.shape),
                hashlib.sha256(out.tobytes()).hexdigest()]
    return ["scalar", type(out).__name__, repr(out)]


class _RecGen:
    """Forwarding proxy over a numpy Generator that records every
    method call's return-value digest, in call order."""

    def __init__(self, gen, rec):
        object.__setattr__(self, "_gen", gen)
        object.__setattr__(self, "_rec", rec)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_gen"), name)
        if not callable(attr):
            return attr
        rec = object.__getattribute__(self, "_rec")

        def wrap(*a, **kw):
            out = attr(*a, **kw)
            rec.append((name, _digest(out)))
            return out
        return wrap


def _recording_default_rng(seed=None, *a, **kw):
    gen = _REAL_DEFAULT_RNG(seed, *a, **kw)
    rec = {"seed": repr(seed), "calls": []}
    _STREAMS.append(rec)
    return _RecGen(gen, rec["calls"])


def _stream_summary(rec):
    canon = json.dumps({"seed": rec["seed"], "calls": rec["calls"]},
                       sort_keys=True).encode()
    n_elems = 0
    for _name, dig in rec["calls"]:
        if dig[0] == "arr":
            n = 1
            for s in dig[2]:
                n *= s
            n_elems += n
        else:
            n_elems += 1
    return {"seed": rec["seed"], "n_calls": len(rec["calls"]),
            "n_elems": n_elems,
            "hash": hashlib.sha256(canon).hexdigest()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="k0_annotate_queue",
                    help="annotator module to take _eval_root / "
                         "_game_root_scenes from (its directory must "
                         "be the cwd or on sys.path)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--steps", default="200,230,260,290",
                    help="comma list of root steps to probe (both "
                         "seats each)")
    ap.add_argument("--belief", default="sample")
    ap.add_argument("--teacher-style", type=int, default=2)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--seed-teacher", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--negative-control", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    mod = __import__(a.module)
    steps = {int(s) for s in a.steps.split(",") if s.strip()}
    ns = argparse.Namespace(
        weights=a.weights, replays=a.replays, games=a.games,
        window_lo=0, window_hi=299, belief=a.belief,
        teacher_style=a.teacher_style, K=a.K, M=a.M, H=a.H,
        seed_teacher=a.seed_teacher, train_temp=1.0, behavior_temp=1.0,
        seed=a.seed, pi0=1.0, pi1=1.0, isolation_file=[])
    plan = mod.build_plan(ns)
    cfg = plan["config"]
    mod._init_worker(plan["weights_bytes"])
    np.random.default_rng = _recording_default_rng

    def run_eval(spec, cfg_):
        del _STREAMS[:]
        mod._eval_root(spec, cfg_)
        return [_stream_summary(r) for r in _STREAMS]

    roots_out = []
    first_root = None
    for pg in plan["per_game"]:
        want = [(s, seat) for (s, seat) in pg["roots"] if s in steps]
        if not want:
            continue
        trimmed = dict(pg)
        trimmed["roots"] = want
        scenes = mod._game_root_scenes(trimmed)
        for (step, seat) in want:
            spec_pin = {"game": pg["game"], "path": pg["path"],
                        "step": step, "me": seat, "ft": pg["ft"]}
            spec_snap = dict(spec_pin)
            spec_snap["scene"] = scenes[(step, seat)]
            s_pin = run_eval(spec_pin, cfg)
            s_snap = run_eval(spec_snap, cfg)
            roots_out.append({
                "game": pg["game"], "step": step, "seat": seat,
                "streams_pin": s_pin, "streams_snap": s_snap,
                "equal": s_pin == s_snap})
            if first_root is None:
                first_root = (pg, spec_pin, s_pin)
    b3 = None
    if a.negative_control:
        if first_root is None:
            raise SystemExit("negative control: no roots probed")
        pg, spec_pin, s_pin = first_root
        cfg_off = dict(cfg)
        cfg_off["seed_teacher"] = cfg["seed_teacher"] + 1
        s_off = run_eval(spec_pin, cfg_off)
        changed = [x["hash"] != y["hash"]
                   for x, y in zip(s_pin, s_off)]
        b3 = {"root": [pg["game"], spec_pin["step"], spec_pin["me"]],
              "n_streams": len(s_pin), "n_streams_offset": len(s_off),
              "all_hashes_changed": bool(len(s_pin) == len(s_off)
                                         and all(changed)),
              "changed": changed,
              "streams_offset": s_off}
    summary = {"module": a.module,
               "n_roots": len(roots_out),
               "all_equal": all(r["equal"] for r in roots_out),
               "roots": roots_out,
               "negative_control": b3}
    print(json.dumps({"n_roots": summary["n_roots"],
                      "all_equal": summary["all_equal"],
                      "negative_control": (
                          None if b3 is None else {
                              "n_streams": b3["n_streams"],
                              "all_hashes_changed":
                                  b3["all_hashes_changed"]})},
                     indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(summary, fh, indent=1)
    ok = summary["all_equal"] and (b3 is None
                                   or b3["all_hashes_changed"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
