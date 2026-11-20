"""Regenerate a complete standalone search-only derivative of the published V1 parent."""
import argparse
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "cold_start/candidate.py"
PARENT_SHA = "7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
REPLACEMENTS = (
    ("return (self.KA @ M.reshape(36, -1)).reshape(256, n, U.shape[2]) + b", "return (self.KA @ M.reshape(36, -1)).reshape(256, n, U.shape[2]).add_(b)"),
    ("_torch.relu(self._conv(xb[self.winb], self.stem[0], self.stem[1], n) + self.emb[sid])", "self._conv(_torch.index_select(xb, 0, self.winb), self.stem[0], self.stem[1], n).add_(self.emb[sid]).relu_()"),
    ("_torch.relu(self._conv(h[self.win], U1, b1, n))", "self._conv(_torch.index_select(h, 0, self.win), U1, b1, n).relu_()"),
    ("t = t + (t[self.board].mean(0) @ gwT + gb)", "t.add_((_torch.index_select(t, 0, self.board).mean(0) @ gwT).add_(gb))"),
    ("h = _torch.relu(h + self._conv(t[self.win], U2, b2, n))", "h.add_(self._conv(_torch.index_select(t, 0, self.win), U2, b2, n)).relu_()"),
    ("hf = h[self.board]", "hf = _torch.index_select(h, 0, self.board)"),
    ("(hf @ self.hw[0] + self.hw[1])", "(hf @ self.hw[0]).add_(self.hw[1])"),
    ("(hf @ self.hb[0] + self.hb[1])", "(hf @ self.hb[0]).add_(self.hb[1])"),
    ("_torch.relu(hf @ self.v1[0] + self.v1[1])", "(hf @ self.v1[0]).add_(self.v1[1]).relu_()"),
    ("_torch.relu(z @ self.v2[0].T + self.v2[1])", "(z @ self.v2[0].T).add_(self.v2[1]).relu_()"),
)


def build():
    source = PARENT.read_bytes()
    if hashlib.sha256(source).hexdigest() != PARENT_SHA:
        raise ValueError("Parent source changed")
    start = source.index(b"class _WinoNet:")
    end = source.index(b"_S_WINO = True", start)
    body = source[start:end]
    for original, replacement in REPLACEMENTS:
        key = original.encode()
        if body.count(key) != 1:
            raise ValueError(original)
        body = body.replace(key, replacement.encode())
    return source[:start] + body + source[end:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = build()
    args.output.write_bytes(raw)
    print(hashlib.sha256(raw).hexdigest(), len(raw))
