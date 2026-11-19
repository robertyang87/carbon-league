"""Build the freight candidate on the exact public hybrid actor and both weight blobs."""
import hashlib
from pathlib import Path

from build_it50_endhome import PARENT, PARENT_SHA

HERE = Path(__file__).resolve().parent
OUTPUT = PARENT.with_name("freight_it50_v3.py")


def build():
    raw = PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise RuntimeError("frozen hybrid parent mismatch")
    src = raw.decode()
    patches = [
        ("def _mixed_actions(x, aux):", "def _freight_encode(x):"),
        ("    prefix = F.Prefix(x, aux)", "    return states\n\n\ndef _freight_decode(x, aux, encoded, changes=None):\n    F = sys.modules['cnet2_feat']\n    states = [list(state) for state in encoded]\n    prefix = F.Prefix(x, aux)"),
        ("        labels[slot] = action", "        if changes is not None:\n            action = _freight_decide(F, x, prefix, slot, action, combined, legal, changes)\n        labels[slot] = action"),
    ]
    for old, new in patches:
        if src.count(old) != 1:
            raise ValueError(old)
        src = src.replace(old, new)
    # Reuse the paired candidate/parent collision, hostile-base, recruitment and
    # threatened-cargo consistency check already exercised by the endhome fixtures.
    prior = (HERE / "it50_endhome_runtime.py").read_text()
    joint = prior[prior.index("def _return_destinations"):prior.index("def _mixed_actions")]
    src += "\n" + joint + "\n" + (HERE / "freight_runtime.py").read_text()
    compile(src, str(OUTPUT), "exec")
    payload = src.encode()
    if len(payload) > 10_000_000:
        raise ValueError("pack size")
    if OUTPUT.exists() and OUTPUT.read_bytes() != payload:
        raise FileExistsError(OUTPUT)
    if not OUTPUT.exists():
        with OUTPUT.open("xb") as stream:
            stream.write(payload)
    print(len(payload), hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    build()
