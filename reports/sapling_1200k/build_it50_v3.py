"""Package our two frozen, independently trained nets into one legal-prefix policy."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "data/weights/sapling_1200k"
IT50 = ROOT / "code/packs/ppou2_m_it50.py"
STUDENT = WEIGHTS / "cash_v3_agent.py"
OUTPUT = WEIGHTS / "hybrid_it50_v3.py"
INPUT_SHA256 = {
    STUDENT: "4cd4c1fcf40f05217ecba2a59095ba40286c5f96017d2b15285c942a3b0f5348",
    IT50: "57e2557665636a1bbc92ae46a4c33ee9dac609efcffefa1c30df1233746c2211",
}


def main():
    for path, expected in INPUT_SHA256.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed source: {path}")
    old = STUDENT.read_text().split("\n_AG = None\n")
    if len(old) != 2:
        raise ValueError("missing unique student loader boundary")
    expert = {}
    for statement in ast.parse(IT50.read_text()).body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in ("_Q1", "_M1", "_B1"):
                expert[name] = ast.literal_eval(statement.value)
    if set(expert) != {"_Q1", "_M1", "_B1"}:
        raise ValueError("incomplete expert weight payload")
    src = old[0] + "\n".join(f"{k} = {expert[k]!r}" for k in sorted(expert)) + "\n"
    src += Path(__file__).with_name("it50_v3_runtime.py").read_text()
    data = src.encode()
    if len(data) > 10_000_000:
        raise ValueError("pack size exceeds deployment format")
    OUTPUT.write_bytes(data)
    print(OUTPUT, len(data), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
