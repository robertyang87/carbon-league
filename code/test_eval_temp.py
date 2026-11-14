"""Fixture: rl2_ppo_eval --temp must reach the play tasks verbatim.

Codex re-review (item 2): main() built every task with
"temp": max(a.temp, 1.0), so any requested temperature in (0, 1) was
silently raised to 1.0 (greedy, temp <= 0, and temp >= 1 were the only
requests that behaved as labelled). The trainer builds the same task
dicts with "temp": args.temp verbatim (rl2_ppo_train.py), and the
adapter applies softmax(logits/temp) with greedy overriding -- there
is no second clamp downstream. This fixture stubs the process pool so
main() runs in-process and captures the exact task dicts it builds.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = HERE if os.path.exists(os.path.join(HERE, "rl2_ppo_rollout.py")) else os.path.normpath(os.path.join(HERE, "..", "..", "code"))
sys.path.insert(0, CODE)

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


def run(extra_argv):
    CAPTURED.clear()
    argv = ["rl2_ppo_eval.py", "--w", "/tmp/nonexistent_ckpt.npz",
            "--opps", "exp/opp_a.py", "--pairs", "2",
            "--log", "/tmp/eval_temp_fixture/eval.jsonl"] + extra_argv
    old = sys.argv
    sys.argv = argv
    try:
        EV.main()
    finally:
        sys.argv = old
    assert CAPTURED, "no tasks captured"
    return CAPTURED


def main():
    cases = [
        (["--temp", "0.2"], 0.2, False),
        (["--temp", "0.5"], 0.5, False),
        ([], 0.0, True),                 # default: greedy
        (["--temp", "0"], 0.0, True),
        (["--temp", "1.5"], 1.5, False),
        (["--temp", "-1"], -1.0, True),  # <= 0 means greedy; value passes through
    ]
    for argv, want_temp, want_greedy in cases:
        tasks = run(argv)
        for t in tasks:
            assert t["temp"] == want_temp, f"{argv}: task temp {t['temp']} != {want_temp}"
            assert t["greedy"] == want_greedy, f"{argv}: greedy {t['greedy']} != {want_greedy}"
        print(f"ok {argv or ['(default)']}: temp={want_temp} greedy={want_greedy} x{len(tasks)} tasks")
    print("PASS test_eval_temp")


if __name__ == "__main__":
    main()
