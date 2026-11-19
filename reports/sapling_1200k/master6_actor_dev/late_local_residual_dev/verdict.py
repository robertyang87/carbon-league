"""Evaluate the frozen development-root admission rule, counting every registered seat."""
from collections import Counter
import gzip
import math

from fit import HERE, NATIVE, checked, digest, freeze, read
from runtime import predict

METRICS = ("margin", "principal", "own_delivery", "new_direct", "others_delivery", "enemy_delivery")


def average(rows, field):
    return math.fsum(r["delta"][field] for r in rows) / len(rows)


def main():
    model = read(HERE / "weights.json")
    assert model["labels_sha256"] == digest(NATIVE / "train_labels.json")
    assert digest(HERE / "weights.json") == read(NATIVE / "held_labels.json")["weights"]
    cases = checked("held")
    train = checked("train")
    assert not {r["map"] for r in cases} & {r["map"] for r in train}
    actual = []
    changed = []
    for r in cases:
        if "proposal" in r and model["enabled"]:
            predictions = predict(r["proposal"], model)
            takes = predictions["margin"] > 10 and predictions["principal"] >= -5 and predictions["own_delivery"] >= -5
            assert takes == (r["status"] == "paired_full_original")
        else:
            predictions = {}
            assert r["status"] != "paired_full_original"
        if r["status"] == "paired_full_original":
            changed.append(r)
        else:
            assert r["status"] in ("no_eligible_root", "frozen_model_kept_parent")
            assert all(v == 0 for v in r.get("delta", {}).values())
        actual.append({"map": r["map"], "seed": r["seed"], "seat": r["seat"], "t": r["t"], "status": r["status"],
                       "action": r.get("proposal", {}), "predicted": predictions,
                       "delta": {k: r.get("delta", {}).get(k, 0.0) for k in METRICS}})
    maps = sorted({r["map"] for r in actual})
    by_map = {}
    for m in maps:
        subset = [r for r in actual if r["map"] == m]
        assert {r["seat"] for r in subset} == {0, 1} and len(subset) == 2
        by_map[m] = {k: average(subset, k) for k in METRICS}
    mean = {k: math.fsum(s[k] for s in by_map.values()) / len(maps) for k in METRICS}
    seats = {str(s): {k: average([r for r in actual if r["seat"] == s], k) for k in METRICS} for s in (0, 1)}
    package = read(HERE / "package.json")
    assert package["candidate_sha256"] == digest(HERE / "candidate.py")
    assert package["weights_sha256"] == digest(HERE / "weights.json")
    import json
    with gzip.open(HERE / "package_parity.json.gz") as f:
        serving = json.load(f)
    assert serving["source"] == package and serving["serving"]["disabled_public_agent_comparisons"] >= 2
    with gzip.open(HERE / "mechanism_fixture.json.gz") as f:
        fixture = json.load(f)
    assert fixture["serving"]["decision_reasons"]["accepted"] > 0
    checks = {"trained_sufficiently": bool(model["enabled"]), "parent_and_local_edits_verified": True,
              "two_changed_held_maps": len({r["map"] for r in changed}) >= 2,
              "mean_official_margin_increases": mean["margin"] > 0,
              "neither_seat_worse": min(s["margin"] for s in seats.values()) >= 0,
              "principal_non_decreasing": mean["principal"] >= 0, "own_delivery_increases": mean["own_delivery"] > 0}
    out = {"decision": "ADVANCE_TO_FRESH_PILOT" if all(checks.values()) else "HOLD",
           "checks": checks, "by_map": by_map, "by_seat": seats, "all_8_seats_map_mean": mean,
           "root_status_counts": dict(Counter(r["status"] for r in actual)), "root_evidence": actual,
           "identities": {str(p.relative_to(HERE)): digest(p) for p in (HERE/"protocol.json", HERE/"weights.json",
                         HERE/"package.json", HERE/"package_parity.json.gz", HERE/"mechanism_fixture.json.gz",
                         NATIVE/"train_labels.json", NATIVE/"held_labels.json")}}
    freeze(HERE / "verdict.json", out)
    print(out["decision"], out["root_status_counts"], mean, checks)


if __name__ == "__main__":
    main()
