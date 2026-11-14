"""Fixtures for rl2_ppo_eval --temp: field pass-through AND behavior.

History, three rounds:
1. Codex re-review item 2: main() built every task with
   "temp": max(a.temp, 1.0), silently raising any requested temp in
   (0, 1) to 1.0. Fix: pass the requested temperature through.
2. Grok re-review (regression from the round-1 fix): the verbatim
   pass-through sent temp=0.0 for the DEFAULT invocation (0 = greedy),
   but greedy does not bypass the temperature in the adapters --
   rl2_ppo_cnet2's act computes z = lg / ts BEFORE the argmax (the
   small adapter divides likewise), so ts=0.0 poisons every logit with
   inf/NaN and the policy collapses to argmax over NaNs (all-STAY).
   Correct contract: temp < 0 is refused; temp == 0 (greedy) sends
   1.0 with greedy=True; temp > 0 passes through verbatim.
3. Owner re-review V04/V10: play_batch built the evaluated net with
   role="learner" unconditionally, so with CNET2_TEMP_KIND inherited
   from a trainer env the adapter replaced the CLI temperature with
   the learner's per-kind temperatures for SAMPLING (measured: env
   0.7,0.7,0.7 makes --temp 0.2 and 1.0 produce identical actions,
   logp and margins). Fix: eval declares primary_role="eval" (the
   trainer's tasks keep the "learner" default, R04 by design) and the
   summary row records temp_cli / temp_eff / greedy / temp_kind_env.
   V10: this fixture previously played against exp/opp_1cz_v5.py,
   which is not in the tree (nor is exp/cand_c1z_v1.py) -- the
   behavior section now plays SELF games (play_batch-native, no
   external files) and gains an inherited-env section that re-executes
   this file in a child process with CNET2_TEMP_KIND set, the shape
   the round-2 version could not see (it popped the variable).

The behavior sections are the load-bearing ones: they play real games
through rl2_ppo_rollout.play_batch with tasks captured from main()
itself and assert the greedy default plays exactly like explicit
T=1.0 greedy (action-for-action) and is not all-STAY -- that the
poisoned construction (temp=0.0, greedy=True) IS all-STAY, proving
the test can see the failure it guards against -- and, under an
inherited CNET2_TEMP_KIND, that two CLI temperatures now play
DIFFERENTLY (V04) while greedy stays exactly the no-env greedy.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = HERE if os.path.exists(os.path.join(HERE, "rl2_ppo_rollout.py")) else os.path.normpath(os.path.join(HERE, "..", "..", "code"))
sys.path.insert(0, CODE)
os.environ["RL2_PACK"] = "0"          # keep per-step recs unpacked for action extraction
_CHILD = "--child" in sys.argv
if not _CHILD:
    # TEMP_KIND governs the learner when set (by design); the plain
    # sections test the no-env path. The env section sets it explicitly
    # in a child process instead of popping it here.
    os.environ.pop("CNET2_TEMP_KIND", None)

spec = importlib.util.spec_from_file_location("rl2_ppo_eval_under_test", os.path.join(CODE, "rl2_ppo_eval.py"))
EV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EV)

CAPTURED = []
FAKE_GAMES = []   # games the fake pool "plays"; row_case fills it


class _FakePool:
    def __init__(self, workers):
        self.workers = workers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, tasks, chunksize=1):
        CAPTURED.extend(tasks)
        return [{"games": list(FAKE_GAMES)} for _ in tasks]


class _FakeCtx:
    def Pool(self, workers):
        return _FakePool(workers)


EV.mp.get_context = lambda name: _FakeCtx()

LOG = "/tmp/eval_temp_fixture/eval.jsonl"


def run(extra_argv, w="/tmp/nonexistent_ckpt.npz", opps="exp/opp_a.py", adapter="small"):
    CAPTURED.clear()
    argv = ["rl2_ppo_eval.py", "--adapter", adapter, "--w", w,
            "--opps", opps, "--pairs", "1",
            "--log", LOG] + extra_argv
    old = sys.argv
    sys.argv = argv
    try:
        EV.main()
    finally:
        sys.argv = old
    assert CAPTURED, "no tasks captured"
    return list(CAPTURED)


def field_cases():
    cases = [
        (["--temp", "0.2"], 0.2, False),
        (["--temp", "0.5"], 0.5, False),
        ([], 1.0, True),                  # default 0 = greedy -> sends 1.0
        (["--temp", "0"], 1.0, True),
        (["--temp", "1.5"], 1.5, False),
    ]
    for argv, want_temp, want_greedy in cases:
        tasks = run(argv)
        for t in tasks:
            assert t["temp"] == want_temp, f"{argv}: task temp {t['temp']} != {want_temp}"
            assert t["greedy"] == want_greedy, f"{argv}: greedy {t['greedy']} != {want_greedy}"
            # V04: eval must declare itself -- the evaluated net is not
            # the trainer's learner and must not inherit TEMP_KIND rule.
            assert t.get("primary_role") == "eval", f"{argv}: primary_role {t.get('primary_role')!r} != 'eval'"
        print(f"ok field {argv or ['(default)']}: temp={want_temp} greedy={want_greedy} role=eval x{len(tasks)} tasks")
    # temp < 0 is refused (argparse error -> SystemExit), no tasks built
    CAPTURED.clear()
    old = sys.argv
    sys.argv = ["rl2_ppo_eval.py", "--w", "/tmp/x.npz", "--temp", "-1",
                "--log", LOG]
    try:
        EV.main()
        raise AssertionError("--temp -1 was not refused")
    except SystemExit as e:
        assert e.code != 0, f"refusal exit code {e.code}"
    finally:
        sys.argv = old
    assert not CAPTURED, "tasks were built despite the refusal"
    print("ok field ['--temp', '-1']: refused, no tasks built")


def row_case():
    """The summary row records the effective temperature (V04)."""
    FAKE_GAMES.clear()
    FAKE_GAMES.append({"opp": "SELF", "seed": 980000, "margin": 3.0})
    try:
        before = 0
        if os.path.exists(LOG):
            with open(LOG) as f:
                before = sum(1 for _ in f)
        run([], opps="SELF")
        run(["--temp", "0.2"], opps="SELF")
        with open(LOG) as f:
            rows = [json.loads(l) for l in list(f)[before:]]
    finally:
        FAKE_GAMES.clear()
    assert len(rows) == 2, f"expected 2 summary rows, got {len(rows)}"
    r0, r1 = rows
    assert (r0["temp_cli"], r0["temp_eff"], r0["greedy"], r0["temp_kind_env"]) == (0.0, 1.0, True, False), r0
    assert (r1["temp_cli"], r1["temp_eff"], r1["greedy"], r1["temp_kind_env"]) == (0.2, 0.2, False, False), r1
    print("ok row: temp_cli/temp_eff/greedy/temp_kind_env recorded (default + --temp 0.2)")


def _actions(res):
    import numpy as np
    ys = []
    for g in res["games"]:
        for rec in g.get("recs", []):
            y = np.asarray(rec["y"]).ravel()   # cnet2 rec dict; y<0 marks padding, 0 = STAY
            ys.append(y[y >= 0])
    if not ys:
        return None
    return np.concatenate(ys)


def _logps(res):
    import numpy as np
    lp = []
    for g in res["games"]:
        for rec in g.get("recs", []):
            lp.append(float(rec["logp"]))
    return np.array(lp, dtype=np.float64)


def _play_all(tasks):
    """Play captured eval tasks in-process (record on), reseeding the
    global RNGs per task the way production's fresh pool processes do."""
    import random as _random
    import numpy as np
    import rl2_ppo_rollout as RO
    acts, logps, margins = [], [], []
    for t in tasks:
        _random.seed(20261114)
        np.random.seed(20261114)
        t = dict(t)
        t["record"] = True
        r = RO.play_batch(t)
        a = _actions(r)
        if a is not None:
            acts.append(a)
        logps.append(_logps(r))
        margins += [g["margin"] for g in r["games"]]
    import numpy as _np
    return (_np.concatenate(acts) if acts else None,
            _np.concatenate(logps) if logps else None, margins)


def _sha(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def _tiny_weights():
    import numpy as np
    import torch
    from cnet2_model import CNet2
    import rl2_ppo_adapters as AD
    torch.manual_seed(0)
    cfg = dict(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)
    net = CNet2(**cfg).eval()
    w = AD.get_adapter("cnet2").export(net)
    td = tempfile.mkdtemp(prefix="eval_temp_behavior_")
    wp = os.path.join(td, "tiny.npz")
    np.savez(wp, **w)
    return wp


def behavior_cases():
    import numpy as np
    wp = _tiny_weights()
    # SELF is play_batch-native: no external opponent files (V10 --
    # exp/opp_1cz_v5.py and exp/cand_c1z_v1.py are both absent from
    # the tree, so the old reference died with FileNotFoundError).
    opp = "SELF"
    ts_default = run([], w=wp, opps=opp, adapter="cnet2")
    assert all(t["temp"] == 1.0 and t["greedy"] is True for t in ts_default)
    # The explicit counterpart is the canonical T=1 GREEDY task, built by
    # hand: the CLI cannot express it (--temp 1.0 labels sampling mode,
    # greedy=False -- correct as labeled), so the review's "greedy must
    # equal T=1 greedy" is tested against this canonical dict.
    ts_explicit = [dict(t, temp=1.0, greedy=True) for t in ts_default]

    a_default, _, m_default = _play_all(ts_default)
    a_explicit, _, m_explicit = _play_all(ts_explicit)
    # The tiny random-init net dies young (~10 steps), so the pair yields
    # ~70 decisions; the divide-by-zero failure poisons decision #1, so
    # the count suffices -- what carries the assertion is the contrast
    # with the poison control below, over the identical task.
    assert a_default is not None and len(a_default) >= 40, f"too few decisions: {None if a_default is None else len(a_default)}"
    assert np.array_equal(a_default, a_explicit), "default greedy play != explicit T=1.0 greedy play"
    assert m_default == m_explicit
    n_nonstay = int((a_default != 0).sum())
    assert n_nonstay > 0, f"greedy default is all-STAY ({len(a_default)} decisions)"
    print(f"ok behavior: default greedy == T=1.0 greedy over {len(a_default)} decisions, "
          f"non-STAY={n_nonstay}, margins={m_default}")

    # The poisoned construction the round-1 fix produced: temp=0.0 with
    # greedy=True must be visibly degenerate -- this proves the fixture
    # can detect the failure mode it now guards against.
    ts_poison = [dict(t, temp=0.0) for t in ts_default]
    a_poison, _, _ = _play_all(ts_poison)
    assert a_poison is not None and int((a_poison != 0).sum()) == 0, \
        "poison control did not collapse to all-STAY; the behavior test cannot see the failure"
    print(f"ok behavior: poison control (temp=0.0) collapses to all-STAY over {len(a_poison)} decisions, as diagnosed")
    return wp, a_default, m_default


def _child_main(wp):
    """Child mode: CNET2_TEMP_KIND is set in OUR environment (inherited
    from the parent test). Play greedy / temp 0.2 / temp 1.0 tasks
    captured from main() and print hashes + margins as JSON."""
    import numpy as np
    out = {}
    for name, argv in (("greedy", []), ("t02", ["--temp", "0.2"]), ("t10", ["--temp", "1.0"])):
        tasks = run(argv, w=wp, opps="SELF", adapter="cnet2")
        a, lp, m = _play_all(tasks)
        out[name] = {"a_sha": _sha(a), "lp_sha": _sha(np.round(lp, 6)),
                     "n": int(len(a)), "margins": m}
    print("CHILDJSON " + json.dumps(out))


def behavior_env_cases(wp, a_ref, m_ref):
    """V04: with CNET2_TEMP_KIND inherited, two CLI temperatures must
    play differently (pre-fix they were identical), and greedy must
    stay exactly the no-env greedy the parent just played."""
    ref = {"a_sha": _sha(a_ref), "margins": m_ref}
    for tk in ("0.7,0.7,0.7", "0.15,0.3,0.3"):
        env = dict(os.environ)
        env["CNET2_TEMP_KIND"] = tk
        env["RL2_PACK"] = "0"
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", wp],
                           env=env, capture_output=True, text=True, timeout=1200)
        line = [l for l in p.stdout.splitlines() if l.startswith("CHILDJSON ")]
        assert line, f"child (TEMP_KIND={tk}) produced no result: rc={p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
        out = json.loads(line[0][len("CHILDJSON "):])
        assert out["t02"]["a_sha"] != out["t10"]["a_sha"] or out["t02"]["lp_sha"] != out["t10"]["lp_sha"], \
            f"TEMP_KIND={tk}: --temp 0.2 and 1.0 still play identically -- CLI temperature overridden (V04)"
        assert out["greedy"]["a_sha"] == ref["a_sha"] and out["greedy"]["margins"] == ref["margins"], \
            f"TEMP_KIND={tk}: greedy under inherited env != no-env greedy"
        print(f"ok behavior-env TEMP_KIND={tk}: temp 0.2 != temp 1.0 (a {out['t02']['a_sha']} vs {out['t10']['a_sha']}), "
              f"greedy == no-env greedy over {out['greedy']['n']} decisions")


def main():
    if _CHILD:
        _child_main(sys.argv[sys.argv.index("--child") + 1])
        return
    field_cases()
    row_case()
    wp, a_default, m_default = behavior_cases()
    behavior_env_cases(wp, a_default, m_default)
    print("PASS test_eval_temp")


if __name__ == "__main__":
    main()
