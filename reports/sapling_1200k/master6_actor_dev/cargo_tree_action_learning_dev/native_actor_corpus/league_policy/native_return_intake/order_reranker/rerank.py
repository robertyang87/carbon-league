"""Build a deployable NumPy package that changes only the final two-order probabilities."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "natural_original/candidate_sampled.py"
SOURCE_SHA = "851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee"


def replace_once(value, old, new):
    if value.count(old) != 1:
        raise ValueError("Unexpected serving code surrounding: " + old)
    return value.replace(old, new, 1)


def export_pack(weights, path):
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Incorrect natural parent")
    parent = raw.decode()
    source = replace_once(parent, "        probs = np.exp(logits - logits.max())",
        """        rows = _order_rows(_order_observation, _FEAT.own_uid[slot], (base_cmd, alt_cmd), logits)
        refined = logits.copy()
        refined[0] += _order_logit_delta(rows, _order_head)
        probs = np.exp(refined - refined.max())""")
    source = replace_once(source, "    global _obs_key, _behavior_rng\n",
        "    global _obs_key, _behavior_rng, _order_observation\n    _order_observation = observation\n")
    definition = (HERE / "features.py").read_text()
    arrays = json.dumps(weights, allow_nan=False, separators=(",", ":"), sort_keys=True)
    runtime = definition + "\n_order_head = {k: np.array(v, dtype=np.float64) for k, v in " + arrays + ".items()}\n"
    source = replace_once(source, "_behavior_rng = None\n", runtime + "\n_behavior_rng = None\n")
    ast.parse(source)
    data = source.encode()
    if len(data) > 10_000_000:
        raise ValueError("Single-file size limit exceeded")
    path = Path(path)
    if path.exists() and path.read_bytes() != data:
        raise ValueError("Existing candidate mismatch")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
