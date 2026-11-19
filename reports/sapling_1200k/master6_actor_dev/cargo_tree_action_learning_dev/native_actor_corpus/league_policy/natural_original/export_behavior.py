"""Export the registered two-order behavior policy from the signed Sapling greedy pack.

From any working directory: python path/to/export_behavior.py --arm candidate
Use --out to independently reproduce the package in a scratch directory.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
from tempfile import TemporaryDirectory

FOLDER = Path("reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy")
PARENT_SHA = "8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"
SAMPLER_SHA = "2df318842cddfaf830cf69e5b630e9f19fd1e564926be82702af415d26df05e8"
REGISTER_SHA = "bf2df87e797f64308c3001186ffa890f9e6bb82e970bef421a9c777b20fd7dbc"
STEPS = tuple(range(12, 277, 24))


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / FOLDER / "fit.json").is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_once(content, needle):
    if content.count(needle) != 1:
        raise ValueError("Unrecognized frozen source: " + needle[:65])
    return content.split(needle, 1)


def sources(root, arm):
    league = root / FOLDER
    base = league.parents[1]
    registered = league / "collection_registration.json"
    if sha(registered) != REGISTER_SHA:
        raise ValueError("Wrong training protocol")
    reg = json.loads(registered.read_bytes())
    sampler = base / "onpolicy_native/training_policy.py"
    parent = base / "candidate.py"
    if sha(sampler) != SAMPLER_SHA or sha(parent) != PARENT_SHA or tuple(reg["sample_steps"]) != STEPS:
        raise ValueError("Wrong reference paths/weights/schedule")
    actor = parent if arm == "parent" else league/"candidate.py"
    weights = actor.with_suffix(".npz")
    if arm == "candidate":
        fit = json.loads((league / "fit.json").read_bytes())
        if (sha(actor) != fit["export"]["package_sha256"] or sha(weights) != fit["export"]["weights_sha256"]
                or not fit["physical_gate"] or fit["export"]["quantized_changed"] <= 0):
            raise ValueError("Missing changed, signed whole-policy optimizer export")
    original = parent.read_text()
    native = actor.read_text()
    registered_sampler = sampler.read_text()
    marker = "def _return_decode("
    # The .npz is provenance. Actual deployed weights come from the signed .py itself.
    native_before, native_decode = split_once(native, marker)
    _, original_decode = split_once(original, marker)
    if native_decode != original_decode:
        raise ValueError("Candidate changed the known decoder/safety route")
    _, sampler_tail = split_once(registered_sampler, marker)
    training_instrument = native_before + marker + sampler_tail
    if arm == "parent" and training_instrument != registered_sampler:
        raise ValueError("Parent instrument differs from the training collector")
    decoder, training_wrapper = split_once(training_instrument,
        '"""Appended only to the training build. The exported greedy pack has none of these hooks."""')
    sampler_prefix, _ = split_once(training_wrapper, '        stream = int(os.environ["SAPLING_NATIVE_STREAM"])')
    _, sampling = split_once(sampler_prefix, "_original_mixed = _mixed_actions")
    sampling = ("_original_mixed = _mixed_actions" + sampling).replace("last_training_event = None\n", "")
    sampling = sampling.replace("    global last_training_event\n", "")
    source = decoder + "\nimport os\n_SAMPLE_STEPS = " + repr(STEPS) + "\n" + sampling + """        return (baseline, alternate)[int(_behavior_rng.random() >= probs[0])]
    return baseline


_behavior_rng = None

def agent(observation, configuration=None):
    global _obs_key, _behavior_rng
    _obs_key = int(observation["step"]), int(observation["player"])
    if _behavior_rng is None or _obs_key[0] == 0:
        _behavior_rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))
    return _normal_agent(observation, configuration)
"""
    ast.parse(source)
    if "SAPLING_NATIVE_STREAM" in source or "SAPLING_NATIVE_TRACE" in source:
        raise ValueError("Training instrumentation leaked into the deployment")
    return source, training_instrument, actor, weights


@contextmanager
def reference_package(root, arm):
    """The training instrument is created only for offline verification, then discarded."""
    _, reference, _, _ = sources(root, arm)
    with TemporaryDirectory(prefix="sapling-training-reference-") as temporary:
        reference_path = Path(temporary) / "training_reference.py"
        reference_path.write_text(reference)
        yield reference_path


def build(root, arm, out):
    source, _, actor, weights = sources(root, arm)
    raw = source.encode()
    if len(raw) > 10_000_000:
        raise ValueError("Single-file size budget exceeded")
    out.mkdir(parents=True, exist_ok=True)
    dest = out / (arm + "_sampled.py")
    if dest.exists() and dest.read_bytes() != raw:
        raise ValueError("Different deployment already at output path")
    dest.write_bytes(raw)
    return dict(arm=arm, source=str(actor.relative_to(root)), source_sha256=sha(actor),
                training_weights=str(weights.relative_to(root)), training_weights_sha256=sha(weights),
                training_sampler_sha256=SAMPLER_SHA, registration_sha256=REGISTER_SHA,
                schedule=list(STEPS), sample_file=dest.name, sha256=sha(dest), bytes=len(raw))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=("candidate", "parent"), required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.arm, args.out), indent=2))
