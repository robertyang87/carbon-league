"""Build a separate endgame-only candidate from the byte-frozen hybrid parent."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "data/weights/sapling_1200k/hybrid_it50_v3.py"
OUTPUT = PARENT.with_name("hybrid_it50_v3_endhome.py")
PARENT_SHA = "b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca"


def build(out):
    data = PARENT.read_bytes()
    if hashlib.sha256(data).hexdigest() != PARENT_SHA or out.resolve() == PARENT.resolve():
        raise ValueError("wrong parent or output aliases parent")
    src = data.decode()
    patches = (("def _mixed_actions(x, aux):", "def _return_decode(x, aux, changes=None):"),
               ("        labels[slot] = action\n",
                "        if changes is not None:\n"
                "            action = _return_decision(F, x, prefix, slot, action, combined, legal, changes)\n"
                "        labels[slot] = action\n"))
    for old, new in patches:
        if src.count(old) != 1:
            raise ValueError(f"decoder site count changed: {old!r}")
        src = src.replace(old, new)
    src += "\n" + Path(__file__).with_name("it50_endhome_runtime.py").read_text()
    compile(src, str(out), "exec")
    payload = src.encode()
    if len(payload) > 10_000_000:
        raise ValueError("exceeds arena package size")
    if out.exists() and out.read_bytes() != payload:
        raise FileExistsError("refusing to replace different candidate")
    if not out.exists():
        with out.open("xb") as handle:
            handle.write(payload)
    print(len(payload), hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUTPUT)
    build(p.parse_args().out)
