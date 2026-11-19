"""Build self-contained NumPy competitors with identical policy tensors and decoder."""
import base64
import numpy as np
from common import HERE, PROTOCOL, sha, save

runtime = (HERE / "runtime.py").read_text()
parent = (HERE.parent / "candidate.py").read_text()
before = '''    enc = []
    for policy in _NETS:
        tokens, context = policy.encode(x, _HYBRID["style"])'''
after = '''    enc = []
    student_x = _carbon_view(x, _FEAT, F, _CARBON_WEIGHTS)
    for index, policy in enumerate(_NETS):
        tokens, context = policy.encode(student_x if index == 1 else x, _HYBRID["style"])'''
assert parent.count(before) == 1
parts = np.load(HERE / "weights.npz")
rows = {}
for name, keys in (("candidate","wbva"), ("constant","c")):
    raw = b"".join(np.ascontiguousarray(parts[k], dtype="<f4").tobytes() for k in keys)
    shapes = [tuple(parts[k].shape) for k in keys]
    blob = base64.b64encode(raw).decode("ascii")
    payload = runtime + f'\n_CARBON_BLOB = {blob!r}\n_CARBON_SHAPES = {shapes!r}\n'
    payload += '''_CARBON_VALUES = np.frombuffer(base64.b64decode(_CARBON_BLOB), dtype="<f4")
_CARBON_WEIGHTS = [a.reshape(shape) for a, shape in zip(
    np.split(_CARBON_VALUES, np.cumsum([np.prod(shape) for shape in _CARBON_SHAPES], dtype=int)[:-1]),
    _CARBON_SHAPES)]
'''
    out = (parent.replace(before, after) + "\n" + payload).encode()
    assert len(out) <= 10_000_000
    (HERE/(name+".py")).write_bytes(out)
    rows[name] = dict(sha256=sha(out),bytes=len(out),float_parameters=len(raw)//4,fp32_sha256=sha(raw))
save(HERE/"packages.json", dict(parent=PROTOCOL["incumbent"],parent_bytes=len(parent.encode()),packages=rows))
