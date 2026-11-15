#!/usr/bin/env python3
"""vhat extraction per checkpoint (PREREG_VALUE test A/B input).

For each checkpoint k in {74,78,79,82,85,95}: build k's critic (the
trainer state's vnet, via common.build_critic -- which also verifies
the state's net == ckpt_<k>.npz and embedded it == k), forward the
6,336 Stage 1 roots through the repo model path, read the era head
per HEAD_MAP, save per-root vhat + the tensor it was read from.

ckpt_82: the end-of-it82 trainer state was never preserved (rolling
state overwritten by it83+; freeze/backups exist at 74/78/79/85 and
previous=95, none at 82). Extraction writes an ARTIFACT_MISSING
marker for 82 and continues with the other five; any OTHER missing
state is a hard error.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CKPTS, OUT, STATE_DIR, build_critic, forward_vhat,  # noqa: E402
                    load_roots)


def main():
    os.makedirs(OUT, exist_ok=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = load_roots()
    idx_all = np.arange(6336)
    for k in CKPTS:
        spath = os.path.join(STATE_DIR, f"state_it{k}.pt")
        if not os.path.exists(spath):
            if k != 82:
                raise SystemExit(f"state missing for ckpt {k}: {spath}")
            marker = {
                "checkpoint": 82, "status": "ARTIFACT_MISSING",
                "missing": "end-of-it82 trainer state .pt (holder of the vnet critic)",
                "evidence": ("run-dir state inventory: backups at 74/78/79, "
                             "freeze at 85, state_previous=it95, state_last=it96; "
                             "rolling prev1 froze at it81 (== pre_r19 backup); "
                             "no state file embeds it=82"),
                "readout_if_present": "vnet utility head (r19 provenance "
                                      "hashes verified against the it81 backup)",
            }
            with open(os.path.join(OUT, "vhat_82.json"), "w") as f:
                json.dump(marker, f, indent=1)
            print("ckpt_82: ARTIFACT_MISSING (marker written)")
            continue
        model, info = build_critic(k, device)
        vhat, _logits = forward_vhat(model, z, info["readout"], idx_all, device)
        assert vhat.shape == (6336,), vhat.shape
        assert np.isfinite(vhat).all(), f"ckpt {k}: non-finite vhat"
        assert vhat.min() >= 0.0 and vhat.max() <= 1.0, \
            f"ckpt {k}: vhat outside [0,1] -- wrong head?"
        np.savez(os.path.join(OUT, f"vhat_{k}.npz"),
                 vhat=vhat.astype(np.float64))
        with open(os.path.join(OUT, f"vhat_{k}.json"), "w") as f:
            json.dump(info, f, indent=1, default=str)
        print(f"ckpt_{k}: vhat extracted readout={info['readout']} "
              f"head={info['head_tensors']} state_md5={info['state_md5']}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
