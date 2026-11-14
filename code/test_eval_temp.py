"""Fixtures for rl2_ppo_eval --temp: field pass-through AND behavior.

History, two rounds:
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

The behavior section is the load-bearing one: it plays real games
through rl2_ppo_rollout.play_batch with tasks captured from main()
itself and asserts the greedy default plays exactly like explicit
T=1.0 greedy (action-for-action) and is not all-STAY -- and that the
poisoned construction (temp=0.0, greedy=True) IS all-STAY, proving
the test can see the failure it guards against.
"""
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = HERE if os.path.exists(os.path.join(HERE, "rl2_ppo_rollout.py")) else os.path.normpath(os.path.join(HERE, "..", "..", "code"))
sys.path.insert(0, CODE)
os.environ["RL2_PACK"] = "0"          # keep per-step recs unpacked for action extraction
os.environ.pop("CNET2_TEMP_KIND", None)  # TEMP_KIND governs the learner when set (by design); test the plain path

spec = importlib.util.spec_from_file_location("rl2_ppo_eval_under_test", os.path.join(CODE, "rl2_ppo_eval.py"))
EV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EV)

CAPTURED = []


class _FakePool:
    def __init__(self, workers):
        self.workers = workers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, tasks, chunksize=1):
        CAPTURED.extend(tasks)
        return [{"games": []} for _ in tasks]


class _FakeCtx:
    def Pool(self, workers):
        return _FakePool(workers)


EV.mp.get_context = lambda name: _FakeCtx()


def run(extra_argv, w="/tmp/nonexistent_ckpt.npz", opps="exp/opp_a.py", adapter="small"):
    CAPTURED.clear()
    argv = ["rl2_ppo_eval.py", "--adapter", adapter, "--w", w,
            "--opps", opps, "--pairs", "1",
            "--log", "/tmp/eval_temp_fixture/eval.jsonl"] + extra_argv
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
        print(f"ok field {argv or ['(default)']}: temp={want_temp} greedy={want_greedy} x{len(tasks)} tasks")
    # temp < 0 is refused (argparse error -> SystemExit), no tasks built
    CAPTURED.clear()
    old = sys.argv
    sys.argv = ["rl2_ppo_eval.py", "--w", "/tmp/x.npz", "--temp", "-1",
                "--log", "/tmp/eval_temp_fixture/eval.jsonl"]
    try:
        EV.main()
        raise AssertionError("--temp -1 was not refused")
    except SystemExit as e:
        assert e.code != 0, f"refusal exit code {e.code}"
    finally:
        sys.argv = old
    assert not CAPTURED, "tasks were built despite the refusal"
    print("ok field ['--temp', '-1']: refused, no tasks built")


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


def behavior_cases():
    import numpy as np
    import torch
    from cnet2_model import CNet2
    import rl2_ppo_adapters as AD
    import rl2_ppo_rollout as RO

    torch.manual_seed(0)
    cfg = dict(C=16, NB=1, D=16, NH=2, L=1, E=8, G=4)
    net = CNet2(**cfg).eval()
    ad = AD.get_adapter("cnet2")
    w = ad.export(net)
    td = tempfile.mkdtemp(prefix="eval_temp_behavior_")
    wp = os.path.join(td, "tiny.npz")
    np.savez(wp, **w)

    opp = "exp/opp_1cz_v5.py"  # eval's own default opponent; self-contained (embedded weights)
    ts_default = run([], w=wp, opps=opp, adapter="cnet2")
    assert all(t["temp"] == 1.0 and t["greedy"] is True for t in ts_default)
    # The explicit counterpart is the canonical T=1 GREEDY task, built by
    # hand: the CLI cannot express it (--temp 1.0 labels sampling mode,
    # greedy=False -- correct as labeled), so the review's "greedy must
    # equal T=1 greedy" is tested against this canonical dict.
    ts_explicit = [dict(t, temp=1.0, greedy=True) for t in ts_default]

    def play(tasks):
        # eval emits one task per (pair, seat); play them all and merge.
        # Production runs each task in a fresh pool process; replaying
        # several tasks in THIS process must reseed the global RNGs per
        # task or the opponent/sim streams carry over between plays.
        import random as _random
        acts, margins = [], []
        for t in tasks:
            _random.seed(20261114)
            np.random.seed(20261114)
            t = dict(t)
            t["record"] = True
            r = RO.play_batch(t)
            a = _actions(r)
            if a is not None:
                acts.append(a)
            margins += [g["margin"] for g in r["games"]]
        import numpy as _np
        return (_np.concatenate(acts) if acts else None), margins

    a_default, m_default = play(ts_default)
    a_explicit, m_explicit = play(ts_explicit)
    r_default = {"games": [{"margin": m_default[0]}]}
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
    a_poison, _ = play(ts_poison)
    assert a_poison is not None and int((a_poison != 0).sum()) == 0, \
        "poison control did not collapse to all-STAY; the behavior test cannot see the failure"
    print(f"ok behavior: poison control (temp=0.0) collapses to all-STAY over {len(a_poison)} decisions, as diagnosed")


def main():
    field_cases()
    behavior_cases()
    print("PASS test_eval_temp")


if __name__ == "__main__":
    main()
