"""nn_widen -- function-preserving widening of a BC/RL policy (Net2Net style).

python3 nn_widen.py IN.npz OUT.npz NEW_W [init_scale]

Old channels are copied; new channels get small random *incoming* weights and zero *outgoing*
weights (into old channels and into fc), so the widened net computes exactly the same logits/value
at init while the new channels can still learn (their outgoing gradients are non-zero).
Layout follows nn_train_bc.Net: stem, blocks.{i}.{0,1}, fc (input = 9*W patch [c-major] + W mean + scalars), pi, v.
"""
import json
import sys

import numpy as np


def main(src, dst, new_w, scale=0.01):
    rng = np.random.default_rng(0)
    w = dict(np.load(src))
    meta = json.loads(str(w.pop("meta"))) if "meta" in w else {}
    old = w["stem.weight"].shape[0]
    assert new_w > old
    out = {}
    for k, v in w.items():
        if k == "stem.weight":                                  # [W, C, 3, 3]
            n = rng.normal(0, scale, (new_w,) + v.shape[1:]).astype(np.float32); n[:old] = v; out[k] = n
        elif k == "stem.bias":
            n = np.zeros(new_w, np.float32); n[:old] = v; out[k] = n
        elif k.startswith("blocks.") and k.endswith(".weight"):  # [W, W, 3, 3]
            n = rng.normal(0, scale, (new_w, new_w) + v.shape[2:]).astype(np.float32)
            n[:old, old:] = 0.0                                   # new inputs never feed old outputs
            n[:old, :old] = v
            out[k] = n
        elif k.startswith("blocks.") and k.endswith(".bias"):
            n = np.zeros(new_w, np.float32); n[:old] = v; out[k] = n
        elif k == "fc.weight":                                  # [HID, 9W + W + S]
            hid, fin = v.shape
            n_s = fin - 10 * old
            n = np.zeros((hid, 10 * new_w + n_s), np.float32)
            n[:, : 9 * old] = v[:, : 9 * old]                    # patch of old channels (c-major: c*9+k)
            n[:, 9 * new_w: 9 * new_w + old] = v[:, 9 * old: 10 * old]   # global mean of old channels
            n[:, 10 * new_w:] = v[:, 10 * old:]                  # scalars
            out[k] = n
        else:
            out[k] = v
    meta.update({"W": new_w, "widened_from": src, "old_W": old})
    np.savez_compressed(dst, **out, meta=np.array(json.dumps(meta)))
    print(json.dumps({"old_W": old, "new_W": new_w, "params": int(sum(x.size for x in out.values()))}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.01)
