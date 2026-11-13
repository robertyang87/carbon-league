"""certify_pack -- OUTER platform resource certification for a packed submission (N04).

Why this exists: reports/live/pack_submit_runbook.md requires a full platform
resource certification (init, per-step 1s target / 3s hard limit, slow-step
count and overage pool, whole-match wall, peak memory, p95). The older
code/cnet2_search_timing.py is only a health + function-timing sub-test: it
times the in-process agent() call and exits nonzero solely on internal agent
errors, so a stub that sleeps 3.05s and returns a legal empty action passes
it with exit 0. This runner is the certification the runbook means.

Method / measurement scope (koujing):
  * The pack is loaded in a CHILD process (one per game), platform-style:
    exec of the file bytes with no __file__ semantics relied upon.
  * init_wall_s: parent wall clock from child spawn until the child reports
    READY, i.e. interpreter start + pack exec (compile / import / top-level
    load) + harness setup ONLY. The platform contract counts nothing else
    in the 20s init budget: anything the agent loads lazily inside agent()
    is charged to the first step, never to init. Gate: <= --init-limit-s
    (default 20). There is NO warmup call (F04 fix): the first agent() call
    IS step 0 of the measured game -- timed under the normal per-step gates
    and advanced into the game like every other step.
  * step_wall_ms: per-step wall clock measured by the PARENT around the
    request/response round trip to the child (so it includes IPC and the
    top-level agent() call), minus the child's reported previous sim-advance
    time, clamped below by the child's own wall around the top-level
    agent() call (child_agent_ms, also reported). Gates: every step
    <= --step-hard-ms (default 3000); PER GAME (F04 fix): slow steps
    (> --step-soft-ms, default 1000) <= --slow-max (default 3) and overage
    pool sum(max(0, step - soft)) <= --overage-limit-s (default 5) -- the
    platform grants these budgets per game, so counts are never summed
    across games.
  * game_wall_s: parent wall clock from child spawn until the child's final
    SUMMARY is received, i.e. it covers the LAST sim advance as well (F04
    fix: previously truncated at the last step reply, before the final
    advance completed). Gate per game: <= --game-limit-s (default 1500).
  * peak_rss_mib: child's own resource.getrusage(RUSAGE_SELF).ru_maxrss at
    game end (Linux: KiB, converted to MiB) -- the peak RSS of the pack
    process itself. The parent's RUSAGE_CHILDREN ru_maxrss (max over all
    waited-for children) is reported alongside as a cross-check. Reported
    always; gated only when --mem-limit-mib is given (on the platform the
    binding 512MiB limit is the cgroup's; pass --mem-limit-mib 512 to make
    it a hard local gate too).
  * a_errors: unified internal-error total with the C03 rule (the "errors"
    key plus every numeric *_errors category in the search agent stats),
    plus any exception raised by a top-level agent() call. Gate: == 0.
  * Explicit opponent / dependency load failure is FATAL (F04 fix): if
    --opp names a file, a missing file or any exception while loading it
    aborts the certification with a nonzero exit -- the harness never
    silently substitutes an empty-action opponent.
  * The certificate carries an identity block: candidate pack md5/bytes,
    opponent md5/bytes (or null), the harness identity (this file's md5
    plus the fastsim / eval_v53 module md5s the child actually imported)
    and the full parameter set, so a PASS is bound to exact artifacts.
  * p95 / p50 / mean / max of step_wall_ms are reported; p95 itself is a
    reported metric, not a gate (the binding per-step rules are the hard
    limit, slow count and overage pool above).

Usage:
  python3 certify_pack.py PACK.py [--opp OPP.py] [--seed0 996000]
      [--games 1] [--steps-per-game 299] [gate overrides] [--out CERT.json]

--games / --steps-per-game only shrink the workload for short runs; the
gate thresholds are never scaled. Exit code: 0 iff every gated check
passes; the full per-item PASS/FAIL summary is printed as JSON either way.
"""
import argparse
import json
import os
import resource
import select
import subprocess
import sys
import time

PREFIX = "CERT1 "


def _md5_bytes(b):
    import hashlib
    return hashlib.md5(b).hexdigest()


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


# --------------------------------------------------------------------------
# Child: loads one pack, plays one game on parent request.
# Protocol: JSON lines on stdout prefixed with PREFIX; commands are JSON
# lines on stdin: {"cmd": "step", "i": k} / {"cmd": "end"}.
# --------------------------------------------------------------------------

def _send(obj):
    sys.stdout.write(PREFIX + json.dumps(obj) + "\n")
    sys.stdout.flush()


def child_main(argv):
    pack, seed, steps, opp = argv[0], int(argv[1]), int(argv[2]), argv[3]
    g = {"__name__": "__main__", "__builtins__": __builtins__}
    t0 = time.perf_counter()
    with open(pack) as f:
        src = f.read()
    exec(compile(src, "<submission>", "exec"), g)
    exec_ms = (time.perf_counter() - t0) * 1000.0

    code_dir = os.path.dirname(os.path.abspath(__file__))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    def setup_fail(stage, exc):
        _send({"type": "setup_error", "stage": stage,
               "error": f"{type(exc).__name__}: {exc}"})
        return 3

    # Harness + dependencies. Any failure here is a certification failure,
    # never a silent downgrade (F04: an explicit opponent that fails to
    # load must not be replaced by an empty-action opponent, and a missing
    # simulator must not turn the run into a synthetic-obs pass).
    try:
        import eval_v53
        from fastsim import FastCarbon
        cfg = eval_v53.CFG
    except Exception as e:
        return setup_fail("harness_import", e)
    harness_mods = {}
    for _m in (eval_v53, FastCarbon.__module__ and sys.modules.get("fastsim")):
        try:
            if _m is not None and getattr(_m, "__file__", None):
                with open(_m.__file__, "rb") as _f:
                    harness_mods[os.path.basename(_m.__file__)] = _md5_bytes(_f.read())
        except Exception:
            pass
    opp_fn = None
    opp_id = None
    if opp:
        if not os.path.exists(opp):
            return setup_fail("opponent_missing",
                              FileNotFoundError(f"opponent file not found: {opp}"))
        try:
            opp_fn = eval_v53.load_agent_fn(opp)
        except Exception as e:
            return setup_fail("opponent_load", e)
        with open(opp, "rb") as _f:
            _ob = _f.read()
        opp_id = {"md5": _md5_bytes(_ob), "bytes": len(_ob)}
    try:
        sim = FastCarbon(cfg).reset(seed=seed)
    except Exception as e:
        return setup_fail("sim_reset", e)
    side = seed % 2
    harness = True

    call_errors = 0

    def call_agent(obs):
        nonlocal call_errors
        try:
            return g["agent"](obs, cfg), None
        except Exception as e:  # agent failure == internal error for the gate
            call_errors += 1
            return {}, f"{type(e).__name__}: {e}"

    # NO warmup (F04): the first agent() call is step 0 of the measured
    # game. init_wall therefore covers only spawn + exec + harness setup;
    # lazy construction inside agent() is charged to step 0 like the
    # platform charges it to the first move.
    _send({"type": "ready", "exec_ms": round(exec_ms, 1), "harness": harness,
           "harness_mods": harness_mods, "opp_loaded": opp_fn is not None,
           "opp_id": opp_id})

    prev_advance_ms = 0.0
    final_advance_ms = 0.0
    n_steps_done = 0
    for line in sys.stdin:
        try:
            cmd = json.loads(line)
        except Exception:
            continue
        if cmd.get("cmd") == "step":
            i = cmd["i"]
            o = [sim.obs(0), sim.obs(1)]
            obs = o[side]
            t2 = time.perf_counter()
            a, err = call_agent(obs)
            child_ms = (time.perf_counter() - t2) * 1000.0
            _send({"type": "step", "i": i, "child_ms": round(child_ms, 1),
                   "advance_ms": round(prev_advance_ms, 1), "error": err})
            n_steps_done += 1
            # advance AFTER replying; timed into advance_ms. The final
            # advance runs before the "end" command is read, so the
            # parent-side game wall (spawn -> SUMMARY) covers it.
            t3 = time.perf_counter()
            try:
                b = opp_fn(o[1 - side], cfg) if opp_fn else {}
                sim.step(*((a, b) if side == 0 else (b, a)))
            except Exception as e:
                call_errors += 1
                _send({"type": "advance_error",
                       "error": f"{type(e).__name__}: {e}"})
            prev_advance_ms = (time.perf_counter() - t3) * 1000.0
            final_advance_ms = prev_advance_ms
        elif cmd.get("cmd") == "end":
            break

    err_by_cat = {}
    ag = g.get("_AG")
    st = getattr(ag, "stats", None) if ag is not None else None
    if isinstance(st, dict):
        for k, v in st.items():
            if (k == "errors" or k.endswith("_errors")) and isinstance(v, (int, float)):
                err_by_cat[k] = err_by_cat.get(k, 0) + (v.item() if hasattr(v, "item") else v)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    _send({"type": "summary", "a_errors": sum(err_by_cat.values()) + call_errors,
           "err_by_cat": err_by_cat, "call_errors": call_errors,
           "peak_rss_mib": round(peak_mib, 1),
           "final_advance_ms": round(final_advance_ms, 1),
           "steps_done": n_steps_done,
           "first_call_ms": None})
    return 0


# --------------------------------------------------------------------------
# Parent: spawns one child per game, times everything, applies the gates.
# --------------------------------------------------------------------------

class LineReader:
    def __init__(self, proc):
        self.fd = proc.stdout.fileno()
        self.buf = b""

    def readline(self, deadline):
        while b"\n" not in self.buf:
            if time.time() > deadline:
                return None
            r, _, _ = select.select([self.fd], [], [], max(0.0, deadline - time.time()))
            if not r:
                return None
            chunk = os.read(self.fd, 65536)
            if not chunk:
                return None  # EOF
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode("utf-8", errors="replace")


def run_game(args, seed):
    cmd = [sys.executable, os.path.abspath(__file__), "--child",
           os.path.abspath(args.pack), str(seed), str(args.steps_per_game),
           os.path.abspath(args.opp) if args.opp else ""]
    hard_deadline = time.time() + args.init_limit_s + args.game_limit_s + 120
    t_spawn = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    rd = LineReader(proc)
    game = {"seed": seed, "fatal": None, "steps_ms": [], "child_ms": []}
    ready = None
    while True:
        line = rd.readline(hard_deadline)
        if line is None:
            game["fatal"] = "no READY from child (EOF/timeout)"
            break
        if line.startswith(PREFIX):
            msg = json.loads(line[len(PREFIX):])
            if msg.get("type") == "setup_error":
                game["fatal"] = f"setup_error[{msg.get('stage')}]: {msg.get('error')}"
                break
            if msg.get("type") == "ready":
                ready = msg
                break
    if ready is None:
        proc.kill()
        proc.wait()
        game["rc"] = proc.returncode
        return game
    game["init_wall_s"] = time.perf_counter() - t_spawn
    game["exec_ms"] = ready["exec_ms"]
    game["harness"] = ready["harness"]
    game["harness_mods"] = ready.get("harness_mods")
    game["opp_loaded"] = ready.get("opp_loaded", False)
    game["opp_id"] = ready.get("opp_id")
    if args.opp and not ready.get("opp_loaded"):
        game["fatal"] = "explicit opponent requested but child did not load it"

    advance_errors = 0
    for i in range(args.steps_per_game):
        proc.stdin.write((json.dumps({"cmd": "step", "i": i}) + "\n").encode())
        proc.stdin.flush()
        t_step = time.perf_counter()
        msg = None
        while True:
            line = rd.readline(hard_deadline)
            if line is None:
                game["fatal"] = f"no step reply at step {i} (EOF/timeout)"
                break
            if line.startswith(PREFIX):
                m = json.loads(line[len(PREFIX):])
                if m.get("type") == "advance_error":
                    advance_errors += 1
                    continue
                if m.get("type") == "step":
                    msg = m
                    break
        if msg is None:
            break
        raw_ms = (time.perf_counter() - t_step) * 1000.0
        step_wall = max(msg["child_ms"], raw_ms - msg["advance_ms"])
        game["steps_ms"].append(round(step_wall, 1))
        game["child_ms"].append(msg["child_ms"])
        if i == 0:
            # No warmup: the first agent() call IS step 0; its latency is
            # reported for visibility but it is gated as a normal step.
            game["first_call_ms"] = msg["child_ms"]
        if msg.get("error"):
            game.setdefault("step_errors", []).append({"i": i, "error": msg["error"]})
    summary = None
    if game["fatal"] is None:
        proc.stdin.write(b'{"cmd": "end"}\n')
        proc.stdin.flush()
        while True:
            line = rd.readline(hard_deadline)
            if line is None:
                break
            if line.startswith(PREFIX):
                m = json.loads(line[len(PREFIX):])
                if m.get("type") == "advance_error":
                    advance_errors += 1
                    continue
                if m.get("type") == "summary":
                    summary = m
                    break
    # Game wall is stamped only now: the child finished the final sim
    # advance before answering "end", so spawn -> SUMMARY covers the
    # complete match task including the last advance (F04 fix).
    game["game_wall_s"] = time.perf_counter() - t_spawn
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
        proc.wait()
    game["rc"] = proc.returncode
    if summary:
        game["final_advance_ms"] = summary.get("final_advance_ms")
        game["a_errors"] = summary["a_errors"] + advance_errors
        game["err_by_cat"] = summary["err_by_cat"]
        game["call_errors"] = summary["call_errors"] + advance_errors
        game["peak_rss_mib"] = summary["peak_rss_mib"]
    else:
        game["a_errors"] = None
        if game["fatal"] is None:
            game["fatal"] = "no SUMMARY from child"
    return game


def main():
    ap = argparse.ArgumentParser(description="Outer platform resource certification for a packed submission (N04).")
    ap.add_argument("pack")
    ap.add_argument("--opp", default="")
    ap.add_argument("--seed0", type=int, default=996000)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--steps-per-game", type=int, default=299)
    ap.add_argument("--init-limit-s", type=float, default=20.0)
    ap.add_argument("--step-soft-ms", type=float, default=1000.0)
    ap.add_argument("--step-hard-ms", type=float, default=3000.0)
    ap.add_argument("--slow-max", type=int, default=3)
    ap.add_argument("--overage-limit-s", type=float, default=5.0)
    ap.add_argument("--game-limit-s", type=float, default=1500.0)
    ap.add_argument("--mem-limit-mib", type=float, default=None,
                    help="hard peak-RSS gate in MiB (default: report only; platform cgroup is 512MiB)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # An explicitly named opponent that does not exist fails the whole
    # certification before any game starts (F04: no silent fallback).
    if args.opp and not os.path.exists(args.opp):
        print(json.dumps({"tool": "certify_pack", "overall": "FAIL",
                          "fatal": f"explicit opponent not found: {args.opp}"}, indent=1))
        sys.exit(2)
    if not os.path.exists(args.pack):
        print(json.dumps({"tool": "certify_pack", "overall": "FAIL",
                          "fatal": f"pack not found: {args.pack}"}, indent=1))
        sys.exit(2)

    games = [run_game(args, args.seed0 + k) for k in range(args.games)]
    all_steps = [ms for g in games for ms in g["steps_ms"]]
    # Slow-step budget and overage pool are PER GAME on the platform
    # (F04 fix): judge each game on its own counts, never the cross-game
    # sum. Aggregate totals below are report-only.
    per_game_slow = [sum(1 for ms in g["steps_ms"] if ms > args.step_soft_ms) for g in games]
    per_game_over = [sum(max(0.0, ms - args.step_soft_ms) for ms in g["steps_ms"]) / 1000.0
                     for g in games]
    slow = [ms for ms in all_steps if ms > args.step_soft_ms]
    overage_s = sum(max(0.0, ms - args.step_soft_ms) for ms in all_steps) / 1000.0
    max_slow = max(per_game_slow, default=0)
    max_over = max(per_game_over, default=0.0)
    max_init = max((g.get("init_wall_s", float("inf")) for g in games), default=float("inf"))
    max_game = max((g.get("game_wall_s", float("inf")) for g in games), default=float("inf"))
    peak_rss = max((g.get("peak_rss_mib") or 0.0 for g in games), default=0.0)
    tot_errors = sum(g["a_errors"] for g in games if g.get("a_errors") is not None)
    fatals = [g["fatal"] for g in games if g.get("fatal")]
    bad_rc = [g["rc"] for g in games if g.get("rc") != 0]
    parent_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0

    checks = {
        "init_le_limit": {"value_s": round(max_init, 3), "limit_s": args.init_limit_s,
                          "pass": max_init <= args.init_limit_s},
        "step_hard_le_limit": {"value_ms": max(all_steps) if all_steps else None,
                               "limit_ms": args.step_hard_ms,
                               "pass": bool(all_steps) and max(all_steps) <= args.step_hard_ms},
        "slow_count_le_limit": {"value": max_slow, "limit": args.slow_max,
                                "per_game": per_game_slow, "scope": "per game",
                                "pass": all(v <= args.slow_max for v in per_game_slow) and bool(games)},
        "overage_pool_le_limit": {"value_s": round(max_over, 3), "limit_s": args.overage_limit_s,
                                  "per_game_s": [round(v, 3) for v in per_game_over], "scope": "per game",
                                  "pass": all(v <= args.overage_limit_s for v in per_game_over) and bool(games)},
        "game_wall_le_limit": {"value_s": round(max_game, 3), "limit_s": args.game_limit_s,
                               "pass": max_game <= args.game_limit_s},
        "internal_errors_zero": {"value": tot_errors, "limit": 0,
                                 "pass": tot_errors == 0 and not fatals},
        "child_exit_clean": {"value": bad_rc, "limit": [],
                             "pass": not bad_rc and not fatals},
        "peak_rss": {"value_mib": round(peak_rss, 1), "limit_mib": args.mem_limit_mib,
                     "pass": (peak_rss <= args.mem_limit_mib) if args.mem_limit_mib else None,
                     "note": "child RUSAGE_SELF ru_maxrss; report-only unless --mem-limit-mib given"},
        "step_p95_report": {"value_ms": round(_pct(all_steps, 95), 1) if all_steps else None,
                            "pass": None, "note": "reported metric, not a gate"},
    }
    overall = all(c["pass"] for c in checks.values() if c["pass"] is not None)
    with open(args.pack, "rb") as _f:
        _pb = _f.read()
    _opp_id = None
    if args.opp:
        with open(args.opp, "rb") as _f:
            _ob = _f.read()
        _opp_id = {"path": args.opp, "md5": _md5_bytes(_ob), "bytes": len(_ob)}
    result = {
        "tool": "certify_pack", "pack": args.pack, "opp": args.opp or None,
        "identity": {
            "pack": {"md5": _md5_bytes(_pb), "bytes": len(_pb)},
            "opp": _opp_id,
            "harness": {"certify_pack_md5": _md5_bytes(open(os.path.abspath(__file__), "rb").read()),
                        "modules": games[0].get("harness_mods") if games else None},
            "params": {"seed0": args.seed0, "games": args.games,
                       "steps_per_game": args.steps_per_game,
                       "init_limit_s": args.init_limit_s, "step_soft_ms": args.step_soft_ms,
                       "step_hard_ms": args.step_hard_ms, "slow_max": args.slow_max,
                       "overage_limit_s": args.overage_limit_s,
                       "game_limit_s": args.game_limit_s, "mem_limit_mib": args.mem_limit_mib},
        },
        "games": [{k: v for k, v in g.items() if k not in ("steps_ms", "child_ms")} | {
            "steps": len(g["steps_ms"]),
            "step_wall_ms": {"mean": round(sum(g["steps_ms"]) / len(g["steps_ms"]), 1) if g["steps_ms"] else None,
                             "p50": round(_pct(g["steps_ms"], 50), 1) if g["steps_ms"] else None,
                             "p95": round(_pct(g["steps_ms"], 95), 1) if g["steps_ms"] else None,
                             "max": max(g["steps_ms"]) if g["steps_ms"] else None}} for g in games],
        "aggregate": {
            "steps": len(all_steps),
            "step_wall_ms": {"mean": round(sum(all_steps) / len(all_steps), 1) if all_steps else None,
                             "p50": round(_pct(all_steps, 50), 1) if all_steps else None,
                             "p95": round(_pct(all_steps, 95), 1) if all_steps else None,
                             "p99": round(_pct(all_steps, 99), 1) if all_steps else None,
                             "max": max(all_steps) if all_steps else None},
            "slow_steps_over_soft": len(slow), "overage_pool_s": round(overage_s, 3),
            "max_init_wall_s": round(max_init, 3), "max_game_wall_s": round(max_game, 3),
            "peak_rss_mib_child_self": round(peak_rss, 1),
            "peak_rss_mib_parent_rusage_children": round(parent_peak, 1),
            "a_errors": tot_errors, "fatals": fatals,
        },
        "checks": checks,
        "overall": "PASS" if overall else "FAIL",
    }
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        sys.exit(child_main(sys.argv[2:]))
    main()
