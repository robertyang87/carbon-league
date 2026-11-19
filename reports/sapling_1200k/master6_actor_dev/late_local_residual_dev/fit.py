"""Fit three map-weighted treatment contrasts to verified training arms only."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from runtime import FEATURES

HERE = Path(__file__).resolve().parent
NATIVE = HERE / "native"
TARGETS = ("margin", "principal", "own_delivery")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def freeze(path, payload):
    raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"frozen evidence differs: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(raw)
    temp.replace(path)


def checked(split):
    protocol = read(HERE / "protocol.json")
    if digest(HERE / "runtime.py") != protocol["proposal_sha256"]:
        raise ValueError("runtime differs from frozen selection")
    authority = read(NATIVE / f"{split}_labels.json")
    assert authority["source_protocol_sha256"] == digest(HERE / "protocol.json")
    assert authority["runner_sha256"] == digest(NATIVE / "run.py")
    assert authority["counter_sha256"] == digest(HERE / "evidence" / "account.py")
    expected = {(r["seed"], r["seat"], r["map"]): r for r in protocol["source_groups"] if r["split"] == split}
    actual = authority["cases"]
    assert len(actual) == len(expected) == 8
    assert {(r["seed"], r["seat"], r["map"]) for r in actual} == set(expected)
    for r in actual:
        assert r["split"] == split
        assert r["selection_sha256"] == digest(NATIVE / f'selection_{r["seed"]}_{r["seat"]}.json')
        if r["status"] == "paired_full_original":
            for k in TARGETS:
                assert np.isclose(r["arms"]["alt"][k] - r["arms"]["base"][k], r["delta"][k])
            assert all(np.isfinite(r["proposal"]["features"])) and len(r["proposal"]["features"]) == len(FEATURES)
    return actual


def ridge(rows):
    x = np.asarray([r["proposal"]["features"] for r in rows], dtype=np.float64)
    y = np.asarray([[r["delta"][t] for t in TARGETS] for r in rows], dtype=np.float64)
    counts = Counter(r["map"] for r in rows)
    weights = np.asarray([1. / (len(counts) * counts[r["map"]]) for r in rows])
    coeff = np.linalg.solve((x.T * weights) @ x + 0.5 * np.eye(len(FEATURES)),
                            (x.T * weights) @ np.clip(y, -400., 400.))
    if not np.isfinite(coeff).all():
        raise ValueError("nonfinite coefficients")
    return {name: coeff[:, k].tolist() for k, name in enumerate(TARGETS)}


def main():
    rows = checked("train")
    assert all(r["status"] in {"no_eligible_root", "paired_full_original"} for r in rows)
    rows = [r for r in rows if r["status"] == "paired_full_original"]
    count_maps = len({r["map"] for r in rows})
    model = {"enabled": len(rows) >= 4 and count_maps >= 3, "features": list(FEATURES),
             "heads": ridge(rows) if rows else {k: [0.] * len(FEATURES) for k in TARGETS},
             "labels_sha256": digest(NATIVE / "train_labels.json"), "protocol_sha256": digest(HERE / "protocol.json"),
             "training_pairs": len(rows), "training_maps": count_maps}
    freeze(HERE / "weights.json", model)
    loo = []
    if count_maps > 1:
        for r in rows:
            coefs = ridge([a for a in rows if a["map"] != r["map"]])
            pred = {k: float(np.dot(r["proposal"]["features"], coefs[k])) for k in TARGETS}
            loo.append({"seed": r["seed"], "map": r["map"], "seat": r["seat"], "t": r["t"],
                        "predicted": pred, "actual": {k: r["delta"][k] for k in TARGETS}})
    freeze(HERE / "fit_evidence.json", {"weights_sha256": digest(HERE / "weights.json"), "fit_sha256": digest(Path(__file__)),
                                      "leave_one_training_map_out": loo,
                                      "raw_abs_label_exceeds_400": {k: sum(abs(r["delta"][k]) > 400 for r in rows) for k in TARGETS}})
    print("frozen", digest(HERE / "weights.json"), model["enabled"], len(rows), count_maps)


if __name__ == "__main__":
    main()
