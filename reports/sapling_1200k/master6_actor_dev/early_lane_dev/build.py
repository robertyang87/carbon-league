"""Build the fixed early-lane experimental agent from exactly the scored model and policy."""

import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
PIN = {
    "protocol.json": "6d1bec75712e110ebf089ccf8db31050109fd6ee667c52fa3d1d7df723c9272f",
    "weights.json": "b530ea1ece47c846a363cafbfd558ec6a1a457b72087e11c330ae9a209b39a5d",
    "forecast.py": "64137e5a7186355e5b7d2c71255729108eea8b45652d294ecc42ae8909e8759b",
    "chooser.py": "c11d23c9064512e9a171e5ab8d06b0740070da71ff393a69e7551bbfc60ac4cf",
}

RUNTIME = """
from collections import deque as _early_deque
_EARLY_BASE = agent
_EARLY_HISTORY = _early_deque(maxlen=10)
_EARLY_TRACE = None
_EARLY_DEFAULTS = dict(size=15, episodeSteps=300, treeLifespan=50, workerLimit=10,
    initialCollectRate=0.3, collectDecreaseRate=0., recCollectorCost=30., recPlanterCost=30.,
    recCollectorIncreaseCost=0., recPlanterIncreaseCost=0., moveCost=0.)

def agent(observation, configuration=None):
    global _EARLY_TRACE
    seat, step = observation["player"], observation["step"]
    if (step == 0 or not _EARLY_HISTORY or _EARLY_HISTORY[-1]["player"] != seat
            or _EARLY_HISTORY[-1]["step"] + 1 != step):
        _EARLY_HISTORY.clear()
    # Every turn is first decoded through the genuine, frozen two-network parent.
    base = _EARLY_BASE(observation, configuration)
    cfg = _EARLY_DEFAULTS if configuration is None else {
        key: configuration.get(key, value) for key, value in _EARLY_DEFAULTS.items()
    }
    predictions = {}
    if step + 5 <= cfg["episodeSteps"] - 1:
        for uid in _EARLY_FILTER.sources_at(observation):
            predictions[uid] = {
                h: _EARLY_PREDICT.predict(_EARLY_WEIGHTS, observation, _EARLY_HISTORY, uid, h, cfg)
                for h in (3, 5)
            }
    decision = _EARLY_CHOOSER.choose(observation, base, cfg, predictions)
    _EARLY_TRACE = {"step": step, "seat": seat, "v1_commands": dict(base), "decision": decision}
    # The forecast consumes enemy worker positions in these prior frames only. Copy records
    # to detach history from a caller that mutates its observation between invocations.
    past_players = [None, None]
    past_players[1-seat] = (None, None, {uid: tuple(w) for uid,w in observation["players"][1-seat][2].items()})
    _EARLY_HISTORY.append({"player": seat, "step": step, "players": past_players})
    return dict(decision["commands"])
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def publish(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(data)
        name = f.name
    try:
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("different existing frozen artifact at " + str(path))
    finally:
        os.unlink(name)


def build(destination=HERE):
    source = {name: (HERE / name).read_bytes() for name in PIN}
    if any(sha(source[name]) != digest for name, digest in PIN.items()):
        raise ValueError("registered build source changed")
    protocol, weights = (json.loads(source[name]) for name in ("protocol.json", "weights.json"))
    parent = (ROOT / protocol["parent_pack"]).read_bytes()
    if sha(parent) != BASE or protocol["parent_sha256"] != BASE:
        raise ValueError("parent no longer matches training")
    if weights["protocol_sha256"] != PIN["protocol.json"] or weights["forecast_sha256"] != PIN["forecast.py"]:
        raise ValueError("weights/forecast registration mismatch")
    text = (HERE / "train.py").read_text()
    defs = {
        n.name: ast.get_source_segment(text, n)
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name in {"distance", "sources_at"}
    }
    filtering = defs["distance"] + "\n\n" + defs["sources_at"]
    gate_text = []
    for symbol, code in (
        ("_EARLY_PREDICT", source["forecast.py"].decode()),
        ("_EARLY_CHOOSER", source["chooser.py"].decode()),
        ("_EARLY_FILTER", filtering),
    ):
        gate_text.extend(
            [f"{symbol} = types.ModuleType({symbol.lower()!r})", f"exec({code!r}, {symbol}.__dict__)"]
        )
    gate_text.append(f"_EARLY_WEIGHTS = {weights!r}")
    data = parent + b"\n" + ("\n".join(gate_text) + "\n" + RUNTIME).encode()
    compile(data, "candidate.py", "exec")
    if len(data) > 10_000_000:
        raise ValueError("single-file competition limit exceeded")
    manifest = {
        "pack_sha256": sha(data),
        "parent_sha256": BASE,
        "parent_pack_sha256": BASE,
        "bytes": len(data),
        "inputs_sha256": PIN,
        "public_filter_sha256": sha(filtering.encode()),
        "build_sha256": sha(Path(__file__).read_bytes()),
        "training_source": protocol["parent_pack"],
    }
    destination = Path(destination)
    publish(destination / "candidate.py", data)
    publish(
        destination / "release_manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
