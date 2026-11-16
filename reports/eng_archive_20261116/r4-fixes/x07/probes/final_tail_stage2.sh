#!/bin/bash
cd ~/workspace/r4-fixes/x07/stage2
CANON=/home/hatch/workspace/heads-v2-diff/dgx.npz
python3 k0_duel.py --student ../fixtures/student_perturbed.npz --parent $CANON --roots ../fixtures/roots_late.json --out ../evidence/cli_student_final.json
echo "== workers=2 smoke =="
python3 k0_duel.py --student $CANON --parent $CANON --roots ../fixtures/roots_x07.json --out ../evidence/cli_workers2_final.json --workers 2
echo "workers2 exit=$?"
echo "== corrupt A must die =="
python3 k0_duel.py --student ../fixtures/corrupt_nan.npz --parent ../fixtures/corrupt_nan.npz --roots ../fixtures/roots_x07.json --out ../evidence/reproA_final.json
echo "corruptA exit=$?"
echo "== X08 units =="
python3 - <<'PYEOF'
import sys; sys.path.insert(0, ".")
import k0_duel as D
try:
    D.load_roots("../fixtures/manifest_x08.json", 200, 299); print("X08 conflict: ACCEPTED (BUG)")
except SystemExit as e: print("X08 conflict die:", str(e)[:80], "...")
specs, sha, blobs = D.load_roots("../fixtures/manifest_same_content.json", 200, 299)
print(f"X08 same-content two paths: ACCEPTED ({len(specs)} roots)")
PYEOF
echo "== X04 bind probe =="
python3 ../probes/x04_bind_probe.py
echo "== X04 replay swap bound =="
python3 ../probes/replay_swap_probe_fixed.py
echo "== X07 inference gate =="
python3 ../probes/inference_gate_probe.py
echo "== X07 feature NaN gate =="
python3 ../probes/feat_nan_probe_fixed.py
echo "FINAL_TAIL_STAGE2_DONE"
