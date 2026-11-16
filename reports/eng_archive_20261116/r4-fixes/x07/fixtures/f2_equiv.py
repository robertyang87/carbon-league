"""Acceptance 2: the driver's parent-arm terminal U equals a direct
stage1_teacher_gate.continue_game reference run value-for-value.
The reference uses the GATE's own extract_scene + continue_game with
the gate's greedy-arm call shape (first_labels=None, guided off,
cont_seed = seed_cont + step, teacher style 2). The fixture (not the
driver) imports the gate. Driver values come from f1's run1 artifact.
Run from the stage dir, after f1."""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402
import stage1_teacher_gate as G  # noqa: E402  (fixture-only reference)

PARENT = "/home/hatch/workspace/heads-v2-diff/dgx.npz"
f1 = json.load(open("../evidence/f1_run1.json"))
rows = {(r["game"], r["step"]): r for r in f1["rows"]}
specs_all, _, _blobs = D.load_roots("../fixtures/roots.json", 200, 299)
specs = [s for s in specs_all if s["step"] in (285, 260)]

pol = G.CachePolicy(dict(np.load(PARENT)))
refs = {}
for spec in specs:
    scene = G.extract_scene(spec)
    u_ref = G.continue_game(pol, scene, None, 995400 + spec["step"],
                            teacher=2)
    key = (spec["game"], spec["step"])
    refs[f"{key[0]}@{key[1]}"] = u_ref
    got = rows[key]["u_parent"]
    assert got == u_ref, f"{key}: driver {got!r} != gate {u_ref!r}"
    print(f"root {key}: driver u_parent == gate continue_game == {u_ref!r}")
json.dump(refs, open("../evidence/f2_refs.json", "w"), indent=1)
print("F2 PASS: parent-arm U bitwise equal to direct gate "
      "continue_game on all sampled roots")
