"""Replay the candidate's public observations through both packages without calling the examiner."""
from match import CAND, HERE, PARENT, native


def compare(raw):
    parent = native.module(PARENT, "comparison_parent")
    candidate = native.module(CAND, "comparison_candidate")
    seat = raw["candidate_seat"]
    sim = native.exam.FastCarbon(native.exam.CFG).reset(seed=raw["seed"])
    changed = []
    for t, pair in enumerate(raw["actions"]):
        obs = sim.obs(seat)
        actual = candidate.agent(obs, sim.config)
        baseline = parent.agent(obs, sim.config)
        assert actual == pair[seat], (t, seat)
        if actual != baseline:
            keys = sorted(set(actual) | set(baseline))
            changed.append(dict(step=t, units={k:[baseline.get(k), actual.get(k)] for k in keys
                                              if actual.get(k) != baseline.get(k)}))
        sim.step(*pair)
    return dict(seat=seat, turns=len(raw["actions"]), changed=changed)


if __name__ == "__main__":
    rows = [compare(native.read(path)) for path in sorted((HERE / "games/candidate").glob("raw_*.json.gz"))]
    assert len(rows) == 2
    native.save(HERE / "changed.json", rows)
