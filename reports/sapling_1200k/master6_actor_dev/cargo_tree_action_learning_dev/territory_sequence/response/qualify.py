"""Read frozen final actions and recompute the preregistered release gate; never plays games."""
from collections import defaultdict
import math
import run as runner


def verify():
    R = runner.R
    d = runner.SEQUENCE
    plan = runner.preflight()
    fit, q = R.load(d/"fit.json"), R.load(d/"qualification.json")
    sources = R.load(d/"sources.json")
    prohibited = {c["terrain"] for c in plan["cases"]}
    assert all(c["terrain"] not in prohibited for c in sources)
    for name, digest in fit["data_sha256"].items():
        assert R.sha(d/name) == digest
    # Separate encodings and source archives may share a physical terrain: compare all actual row identities.
    for name in ("pairs.json.gz", "sequence.json.gz"):
        assert all(row["map"] not in prohibited for row in R.load(d/name))
    assert fit["protocol"] == R.sha(d/"protocol.json")
    assert fit["parent"] == q["parent"] == plan["parent_sha256"] == R.sha(runner.PARENT)
    assert fit["parent_weights"] == R.sha(d.parent/"candidate.npz")
    for arm in ("candidate", "control"):
        out = fit["outputs"][arm]
        assert out["package"] == q[arm] == R.sha(d/(arm+".py"))
        assert out["weights"] == R.sha(d/(arm+".npz")) and sum(out["quantized_parameters_changed"].values()) > 0
        assert (d/(arm+".py")).stat().st_size == out["bytes"] <= 10_000_000
    assert R.sha(d/"held_actions.json.gz") == q["held_actions"] and R.sha(d/"held_native_pairs.json.gz") == q["held_pairs"]
    rows = R.load(d/"held_actions.json.gz")
    by_map = defaultdict(list)
    for r in rows:
        assert r["map"] not in prohibited and math.isfinite(r["w"]) and r["w"]>0
        by_map[r["map"]].append(r)
    mean = {}
    for arm in ("candidate", "control", "parent"):
        per_map = [sum(r[arm+"_nll"]*r["w"] for r in group) / sum(r["w"] for r in group) for group in by_map.values()]
        mean[arm] = sum(per_map)/len(per_map)
        assert math.isclose(mean[arm],q["metrics"]["all"][arm]["nll"],rel_tol=1e-9)
    changed = [r for r in rows if r["parent_cmd"]!=r["candidate_cmd"]]
    n_maps = len({r["map"] for r in changed})
    assert (q["changed_rows"],q["changed_maps"]) == (len(changed), n_maps)
    passed = mean["candidate"] < min(mean["control"],mean["parent"]) and len(changed)>=4 and n_maps>=2
    assert q["plain_entry_crosschecks"] > 0 and q["eligible_for_response"] is passed
    record = dict(protocol=runner.REGISTERED, runner=R.sha(runner.HERE/"run.py"), checker=R.sha(runner.HERE/"qualify.py"),
                  receipt_file="qualification.json", receipt=R.sha(d/"qualification.json"), fit=R.sha(d/"fit.json"), fit_protocol=fit["protocol"],
                  student=q["candidate"], control=q["control"], parent=q["parent"], weights=fit["outputs"]["candidate"]["weights"],
                  nll=mean, actual_focal_changes=len(changed), affected_terrains=n_maps, eligible=passed,
                  application="Only a passed receipt allows the paired original continuations.")
    R.write(runner.HERE/("qualified.json" if passed else "blocked.json"), record)
    print(record)


if __name__=="__main__":
    verify()
