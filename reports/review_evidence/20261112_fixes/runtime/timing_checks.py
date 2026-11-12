"""Optional R11 verification: one 299-step game per valid/invalid pure-network pack."""
import subprocess
import sys
from common import setup, emit


def main():
    repo, out = setup()
    rows = []
    for name in ("valid", "invalid"):
        if not (out / (name + ".py")).is_file() or not (out / "idle.py").is_file():
            raise RuntimeError("Run prepare.py first")
        result = subprocess.run(
            [sys.executable, str(repo / "code" / "cnet2_search_timing.py"),
             str(out / (name + ".py")), str(out / "idle.py"), "970500", "1"],
            capture_output=True, text=True,
        )
        (out / (name + "_timing.out")).write_text(result.stdout + result.stderr)
        rows.append({"kind": name, "exit_code": result.returncode, "output": result.stdout + result.stderr})
    emit("timing_checks.jsonl", rows)


if __name__ == "__main__":
    main()
