"""Child runner: import the workdir cnet2_train and load() one dataset,
printing a single JSON verdict line."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
os.environ.setdefault("BC_DEVICE", "cpu")

import numpy as np  # noqa: E402

ds = sys.argv[1]
try:
    import cnet2_train as T
    D, is_val, meta = T.load(ds)
    out = {"ok": True,
           "w": (np.asarray(D["w"]).astype(float).tolist() if "w" in D else None),
           "uw_row0_head": (np.asarray(D["uw"])[0, :10].astype(float).tolist()
                            if "uw" in D else None),
           "mode": (meta.get("derived_layer_check") or {}).get("mode")}
except SystemExit as e:
    out = {"ok": False, "systemexit": str(e)[:300]}
except Exception as e:  # noqa: BLE001
    out = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
print("VERDICT " + json.dumps(out))
