"""Freeze an admitted terminal-trained actor and evaluate both it and its parent on the same new original-examiner map."""
import json
import os
from pathlib import Path
import subprocess
import sys

import train


def main():
    report = train.HERE / "results.json"
    result = json.loads(report.read_bytes())
    if result.get("status") != "complete" or not result["admitted_to_original_development"]:
        raise ValueError("proxy train and holdout gates did not admit a candidate")
    pack = train.ACTORS / "terminal_blend_agent.py"
    expected = result["chosen_pack_sha256"]
    if train.prior.digest(pack) != expected or pack.read_bytes() != train.packed(train.ARMS[result["chosen"]]):
        raise ValueError("terminal-trained actor cannot be reconstructed")
    identity = train.HERE / "chosen_identity.json"
    manifest = {"pack_sha256": expected, "parent_sha256": train.PARENT_SHA,
                "train_report_sha256": train.prior.digest(report),
                "train_source_sha256": result["recipe"]["source_sha256"],
                "chosen": result["chosen"], "student_role_phase_weights": train.ARMS[result["chosen"]],
                "parent_input_weights_manifest_sha256": train.prior.digest(train.BASE/"delivery_parent_dev/build_manifest.json")}
    if identity.exists() and json.loads(identity.read_bytes()) != manifest:
        raise ValueError("different candidate was previously frozen")
    train.prior.shared.atomic_json(identity, manifest)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for label, actor, pack_sha, weights in (
            ("candidate", pack, expected, identity),
            ("parent", train.PARENT, train.PARENT_SHA, train.BASE/"delivery_parent_dev/build_manifest.json")):
        output = train.HERE / ("original_"+label)
        subprocess.run([sys.executable, "-u", str(train.BASE/"tree_pack_arena.py"),
                        "--weights",str(weights),"--pack",str(actor),"--pack-sha256",pack_sha,
                        "--run",str(output),"--maps","621453920","--cpus","2","3"],check=True,env=env)
        official = output/"official_replay.json"
        if not official.exists():
            subprocess.run([sys.executable,str(train.BASE/"official_game_replay.py"),str(output),
                            "--out",str(official)],check=True,env=env)


if __name__=="__main__":
    main()
