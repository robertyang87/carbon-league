"""Bind a development registration to the original-based training and physical-action receipts."""
import argparse
import gzip
import json
import matrix as m

def bind(classification, decision=None):
    train = m.TRAIN
    fit = m.read(train / "fit.json")
    search = json.loads(gzip.decompress((train / "search_probe/full_search.json.gz").read_bytes()))
    names = ("protocol.json", "register.py", "prepare.py", "original_policy.py", "runtime.py", "fit.py",
             "fit.json", "held.json", "physical.py", "physical.json", "held_orders.json.gz", "source_identity.json",
             "search_probe/full_search.json.gz")
    inputs = {f"training_{i:02d}": m.record(train / name) for i, name in enumerate(names)}
    package = m.record(m.PACKAGE)
    weights = m.record(train / "weights.npz")
    m.require(package["sha256"] == fit["candidate_sha256"] == search["candidate_sha256"] and weights["sha256"] == fit["head_sha256"],
              "Weight, candidate or searched package mismatch")
    if classification == "qualified_development":
        m.require(fit["observational_gate"] and search["dispatch_eligible"], "Qualified registration requires both gates")
    elif not decision:
        raise ValueError("Record the pre-game diagnostic decision before freezing this status")
    reg = dict(registration_class=classification, eligible_for_development=True, original_sha256=m.ORIGINAL_SHA,
               observational_gate=fit["observational_gate"], mechanism_gate=search["dispatch_eligible"], diagnostic_decision=decision,
               package=package, weights=weights, inputs=inputs)
    dest = m.HERE / "release.json"
    if dest.exists():
        m.require(m.read(dest) == reg, "Different existing development release")
    else:
        m.play.live.save(dest,reg)
    return m.sha(dest)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("classification", choices=("qualified_development", "diagnostic_development"))
    p.add_argument("--diagnostic-decision")
    a=p.parse_args()
    print(bind(a.classification,a.diagnostic_decision))
