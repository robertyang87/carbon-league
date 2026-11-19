"""Rebuild the frozen dual actor plus shared public freight model and policy as one importable file."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
ROOT = HERE.parents[3]
PARENT = ROOT / "data/weights/sapling_1200k/master6_actor_hybrid.py"
INPUTS = {
    PARENT: "416cc31b8da7d38cc97f66903fbe2797d7eadda0ee175f105227057afe5d3874",
    DEV / "freight_horizon/model.json": "376fdde5bba0ea816c597f9d2aa88d20354edec7e4b641466ad8879400d353ff",
    DEV / "freight_horizon/predictor.py": "764075bd986a33379500cde2ae43294bd4822b449c20db2b2377a03f881e3d69",
    DEV / "freight_response/policy.py": "f48da3f1255ea4ab1b869fde202fe3c9504b3fac2bda80ad94bbf5fbfbe1a00c",
}
def sha(data):
    return hashlib.sha256(data).hexdigest()


def replay_decoder(parent):
    start = parent.index("def _mixed_actions(x, aux):")
    end = parent.index("\n\ndef agent(", start)
    decoder = parent[start:end]
    score_start = decoder.index("        expert, student =")
    score_end = decoder.index("        labels[slot] = action", score_start)
    # The underlying original decoder is taken verbatim, minus telemetry. Only one
    # slot can be forced. Both networks choose each later token on the changed prefix.
    decoder = (decoder[:score_start] +
               "        if slot == forced_slot:\n"
               "            if not legal[forced_label]:\n"
               "                return None\n"
               "            action = forced_label\n" + decoder[score_end:])
    return decoder.replace("def _mixed_actions(x, aux):", "def _forced_decode(x, aux, forced_slot, forced_label):", 1)


BOOT = """
def _freight_module(name):
    module = types.ModuleType('freight_' + name)
    module.__file__ = '<embedded ' + name + '>'
    exec(compile(_FREIGHT_SOURCES[name], module.__file__, 'exec'), module.__dict__)
    return module

_freight_predictor = _freight_module('predictor')
_freight_policy = _freight_module('policy')
_freight_runtime = _freight_module('runtime')
_model = json.loads(zlib.decompress(base64.b64decode(_FREIGHT_MODEL)))
if tuple(_model['features']) != _freight_predictor.NAMES:
    raise ValueError('freight model inputs disagree')
_model_wrapper = _freight_predictor.FreightPredictor.__new__(_freight_predictor.FreightPredictor)
_model_wrapper.model = _model
_FREIGHT = _freight_runtime.FreightOverlay(_model_wrapper, _freight_policy)
_original_agent = agent
_original_mixed = _mixed_actions
_freight_root = None

def _mixed_actions(x, aux):
    global _freight_root
    _freight_root = (x, aux)
    return _original_mixed(x, aux)

def _freight_reselect(unit, action):
    slots = list(_FEAT.own_uid)
    if unit not in slots or slots.index(unit) >= 10:
        return None
    index = slots.index(unit)
    forced = int(_FEAT.labels({unit: action})[index])
    labels = _forced_decode(*_freight_root, index, forced)
    return None if labels is None else _FEAT.to_actions(labels)

def agent(observation, configuration=None):
    base = _original_agent(observation, configuration)
    return _FREIGHT.consider(observation, configuration or {}, base, _freight_reselect)
"""


def build():
    for path, expected in INPUTS.items():
        if sha(path.read_bytes()) != expected:
            raise ValueError(f"Frozen source changed: {path.name}")
    parent = PARENT.read_text()
    sources = {name: path.read_text() for name, path in (
        ("predictor", DEV/"freight_horizon/predictor.py"), ("policy", DEV/"freight_response/policy.py"),
        ("runtime", HERE/"runtime.py"))}
    blob = base64.b64encode(zlib.compress((DEV/"freight_horizon/model.json").read_bytes(), 9)).decode()
    code = (parent + "\n_FREIGHT_SOURCES = " + repr(sources) + "\n_FREIGHT_MODEL = " + repr(blob) +
            "\n\n" + replay_decoder(parent) + BOOT).encode()
    compile(code, "freight_deployment/agent.py", "exec")
    if len(code) > 10_000_000:
        raise ValueError("Package exceeds 10,000,000 bytes")
    (HERE/"agent.py").write_bytes(code)
    manifest = {
        "agent": "agent.py", "agent_sha256": sha(code), "bytes": len(code),
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in
                          [*INPUTS, Path(__file__).resolve(), HERE/"runtime.py", HERE/"PROTOCOL.md"]},
        "max_interventions_per_episode": 1, "source_of_tactical_proposals": "freight_response/policy.py",
        "actor": "Frozen Master6 75/25 two-network logits and unchanged joint prefixes on fallback.",
    }
    (HERE/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(manifest["agent_sha256"], manifest["bytes"])
    return manifest


if __name__ == "__main__":
    build()
