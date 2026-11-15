#!/usr/bin/env python3
"""MECHANISM validation only (prep-time; the ONLY computation run
before sign-off execution). Per available checkpoint: take the
first 8 roots, run the HEAD_MAP extraction path, and verify
MECHANISM properties only:
  - vhat shape == (8,)
  - all values finite and inside [0,1]
  - values not all identical (non-constant across the 8 roots)
  - policy logits obtainable from the SAME forward pass
    (shape (8,12,5), finite)
Prints PASS/FAIL booleans + the tensor names read. NO cross-root
aggregate (no means, no errors, no correlations) is computed or
printed. For ckpt_82 it verifies the documented artifact gap
(state slot absent -> ARTIFACT_MISSING), which is itself the
mechanism conclusion for that checkpoint.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CKPTS, READOUT, STATE_DIR, build_critic,  # noqa: E402
                    forward_vhat, load_roots)

IDX = np.arange(8)


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = load_roots()
    all_ok = True
    for k in CKPTS:
        readout, wk, bk = READOUT[k]
        spath = os.path.join(STATE_DIR, f"state_it{k}.pt")
        if not os.path.exists(spath):
            print(f"ckpt_{k}: ARTIFACT_MISSING (no state slot; readout "
                  f"would be {readout} via vnet {wk})")
            if k != 82:
                all_ok = False
            continue
        model, info = build_critic(k, device)
        vhat, logits = forward_vhat(model, z, readout, IDX, device)
        checks = {
            "shape_(8,)": tuple(vhat.shape) == (8,),
            "finite": bool(np.isfinite(vhat).all()),
            "in_[0,1]": bool(vhat.min() >= 0.0 and vhat.max() <= 1.0),
            "non_constant": bool(vhat.max() > vhat.min()),
            "logits_same_forward": bool(
                tuple(logits.shape) == (8, 12, 5)
                and np.isfinite(logits).all()),
            "net==ckpt_identity": info["net_vs_ckpt_diffs"] == 0,
            "embedded_it": info["embedded_it"] == k,
        }
        ok = all(checks.values())
        all_ok = all_ok and ok
        print(f"ckpt_{k}: {'PASS' if ok else 'FAIL'} readout={readout} "
              f"tensors=vnet[{wk},{bk}] checks={checks}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print("MECH_CHECK:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
