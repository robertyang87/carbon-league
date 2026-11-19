"""Package the frozen V5 actor with the existing production BankruptcyGuard."""
import hashlib
from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
import cnet2_search_pack as packer


def main():
    directory = ROOT / "data/weights/sapling_1200k"
    weights = directory / "soft_it50_v5.npz"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != "d2ddac41d70d2df22b582adfc5b5d28677e286d4cf5b061374828c223a7a49e2":
        raise ValueError("unexpected V5 checkpoint")
    out = directory / "soft_it50_v5_guard_agent.py"
    conf = dict(SEARCH=False, ROLL="self", GUARD=True, GUARD_NARROW=True, GUARD_CASH=40.0, LOWW=2, TEACHER=2)
    with tempfile.TemporaryDirectory(dir=directory) as work:
        tmp = Path(work) / "pack.py"
        os.environ["CNET2_QMODE"] = "int8"
        os.environ["CNET2_NO_ROLL"] = "1"
        packer.main(str(tmp), repr(conf), str(weights))
        header, marker, tail = tmp.read_text().partition("\n_W = None\n")
        if not marker or "def agent(" not in tail:
            raise ValueError("unexpected native package template")
        runtime = HERE.joinpath("v5_guard_runtime.py").read_text()
        result = (header + "\n" + runtime).encode()
        if len(result) > 10_000_000:
            raise ValueError("pack exceeds deployment size")
        tmp.write_bytes(result)
        tmp.replace(out)
    print(hashlib.sha256(result).hexdigest(), len(result))


if __name__ == "__main__":
    main()
