"""Reproduce the trained policy's actual conditional support using the byte-exact validation package."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NATURAL = HERE.parents[1] / "natural_original"
ROOT = next(p for p in HERE.parents if (p/"code/env_src").is_dir())
PARENT = "851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee"
ARMS = ("sampled_parent", "sampled_child")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_verifier(plan_file):
    """Instantiate a private copy of the original checker with the two registered releases."""
    plan_file = Path(plan_file)
    plan = json.loads(plan_file.read_bytes())
    if plan["arms"]!=list(ARMS):
        raise ValueError("Unexpected validation arm identities")
    inputs = plan["files"]
    packages = {}
    for arm in ARMS:
        card = inputs[arm]
        path = ROOT / card["path"]
        if path.stat().st_size!=card["bytes"] or digest(path)!=card["sha256"]:
            raise ValueError(f"Incorrect selected artifact for {arm}")
        packages[arm] = (path, card["sha256"])
    if packages[ARMS[0]][1]!=PARENT or packages[ARMS[1]][1]==PARENT:
        raise ValueError("Need the 851c control and a distinct successfully trained validation release")
    implementation = NATURAL/"audit_menu.py"
    entries = [e for e in inputs.values() if e.get("path")==str(implementation.relative_to(ROOT))]
    if not entries or any(e["sha256"]!=digest(implementation) for e in entries):
        raise ValueError("Original conditional-action auditor not bound by the validation protocol")
    sys.path.insert(0, str(NATURAL))
    spec = importlib.util.spec_from_file_location("validation_actual_menu_instance", implementation)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Normally this checker binds a historical package per key. Here its local
    # instance binds both releases from the selected validation matrix instead.
    module.PACKS = packages
    protocol_sha = digest(plan_file)

    def verify(directory, arm):
        folder = Path(directory)
        registration = json.loads((folder/"registration.json").read_bytes())
        card=inputs[arm]
        if (registration["matrix_sha256"]!=protocol_sha or registration["arm"]!=arm or
            registration["files"]["candidate"]!=card or registration["files"]["candidate_weights"]!=inputs[arm+"_weights"]):
            raise ValueError("Unregistered role, package, editable weight or matrix")
        result = module.verify_menu(folder, arm, persist=False)
        if result["deployment_sha256"]!=card["sha256"]:
            raise ValueError("Action support was replayed using the wrong release")
        return result

    return verify

def verifier(plan_file):
    """Only the network and editable tensors selected by the signed original-return update can occupy the child arm."""
    entries=json.loads(Path(plan_file).read_bytes())["files"]
    fit_path=HERE.parent/"fit.json"
    items=[e for e in entries.values() if e.get("path")==str(fit_path.relative_to(ROOT))]
    if not items or any(e["sha256"]!=digest(fit_path) for e in items):
        raise ValueError("Training selection receipt is missing or unsigned")
    fit=json.loads(fit_path.read_bytes())
    if (fit.get("status")!="training_mechanism_pass" or
            fit.get("selected_package_sha256")!=entries["sampled_child"]["sha256"] or
            fit.get("selected_weights_sha256")!=entries["sampled_child_weights"]["sha256"]):
        raise ValueError("Validation child does not match the selected full-game return update")
    return release_verifier(plan_file)
