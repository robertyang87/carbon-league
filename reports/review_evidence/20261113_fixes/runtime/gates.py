"""Minimal R05 checks: 2 steps/game, at most 3 games; all outputs stay outside the repo."""
import os
from pathlib import Path
import subprocess
import sys
from common import setup, emit


def worker(name, seed, n):
    _, out = setup()
    os.environ["EVAL_STEPS"] = "2"
    os.environ["EVO_WORKERS"] = "1"
    import bigeval
    # Relocate result storage only; eval_v53 retains the real repository code path.
    bigeval.HERE = str(out / "exam_output" / "code")
    Path(bigeval.HERE).mkdir(parents=True, exist_ok=True)
    bigeval.main(str(out / (name + ".py")), str(out / "idle.py"), int(seed), int(n), name)


def main():
    _, out = setup()
    for name in ("valid", "invalid", "conditional_error", "idle"):
        if not (out / (name + ".py")).is_file():
            raise RuntimeError("Run prepare.py first")
    rows = []
    for name, n, seed in (("valid", 2, 970500), ("invalid", 2, 970500), ("conditional_error", 3, 970501)):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", name, str(seed), str(n)]
        result = subprocess.run(command, capture_output=True, text=True)
        (out / (name + "_gate.out")).write_text(result.stdout + result.stderr)
        rows.append({"test": "gate", "kind": name, "exit_code": result.returncode,
                     "output": result.stdout + result.stderr})
    emit("gates.jsonl", rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        main()
