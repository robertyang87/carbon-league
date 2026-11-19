"""Issue a native-match ticket after verifying changed public commands for both seats."""
import collections
import numpy as np
from match import ACTOR, FIT, HERE, native, plan

def main():
    fit = native.read(FIT / "fit.json")
    assert fit["contract"] == plan["training"] and fit["parent"] == plan["parent"]
    digest = native.sha(FIT / "candidate.py")
    assert digest == "8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"
    weights = FIT / "candidate.npz"
    parent = ACTOR / "deployed_mix_policy_gradient_dev/full_episode/selected.npz"
    with np.load(weights,allow_pickle=False) as new, np.load(parent,allow_pickle=False) as old:
        assert set(new.files)==set(old.files)
        changed = []
        for k in new.files:
            if new[k].dtype.kind in "fc":
                assert np.isfinite(new[k]).all()
            if not np.array_equal(new[k],old[k]):
                changed.append(k)
        assert set(changed)=={"whead.weight","whead.bias"}
    ev = FIT / "normal_entry_actions.json.gz"
    byseat = collections.defaultdict(set)
    for row in native.read(ev):
        if row["split"] == "test" and row["observed"]["candidate"]["issued"] != row["observed"]["parent"]["issued"]:
            byseat[row["seat"]].add((row["raw"],row["t"]))
    count = {str(s):len(byseat[s]) for s in (0,1)}
    assert min(count.values()) > 0
    native.save(HERE / "binding.json", dict(candidate=digest,candidate_weights=native.sha(weights),parent=plan["parent"],original=plan["original"],protocol=native.REG,changed_turns_by_seat=count,evidence=native.sha(ev),evidence_file=ev.name,fit=native.sha(FIT/"fit.json"),runner=native.sha(HERE/"match.py")))

if __name__ == "__main__":
    main()
