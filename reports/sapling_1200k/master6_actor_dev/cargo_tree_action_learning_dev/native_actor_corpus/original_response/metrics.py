"""Replay settled action tapes to add physical-seat worker roles and three inventory checkpoints."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from settle import K, R, REPO, WIDTH, digest


def inventory(obs):
    teams = []
    for cash, _, workers, trees in obs.players:
        roles = Counter(w[2] for w in workers.values())
        assert set(roles) <= {"COLLECTOR", "PLANTER"}
        teams.append(dict(cash=float(cash), trees=len(trees), collectors=roles["COLLECTOR"], planters=roles["PLANTER"]))
    assert len(teams) == 2
    return dict(observation=int(obs.step), physical_seats=teams)


def summarize(freeze, directory):
    assert __debug__
    plan = R.load(freeze)
    settled = directory / "settlement.json"
    summary = R.load(settled)
    assert summary["release_file_sha"] == R.sha(freeze) and summary["freeze_canonical_sha"] == digest(plan)
    runs, traces = [], []
    for case, check in zip(plan["cases"], summary["cases"], strict=True):
        seed, seat, t0 = case["seed"], case["seat"], case["t0"]
        assert (seed, seat, t0) == (check["seed"], check["seat"], check["t0"])
        source = REPO / case["source"]
        assert R.sha(source) == case["source_sha256"] == check["source_sha"]
        raw = R.load(source)
        folder = directory / "native" / f"{seed}_{seat}"
        for arm in case["arm_order"]:
            verified = check["arms"][arm]
            path = folder / (arm + ".json.gz")
            assert R.sha(path) == verified["receipt_file_sha"]
            tape = R.load(path)["tape"]
            assert digest(tape) == verified["tape_sha"] and len(tape) + t0 == verified["verified_joint_steps"]
            env = K.checked_env(raw)
            for joint in raw["actions"][:t0]:
                env.step(joint)
            root = inventory(env.states[0].observation)
            states = [root]
            for joint in tape:
                assert not env.done
                env.step(joint)
                states.append(inventory(env.states[0].observation))
            scores, cash, status = K.score(env)
            assert (scores, cash, status) == (verified["official_scores"], verified["physical_cash"], verified["status"])
            width = min(len(tape), WIDTH)
            after, end = states[width], states[-1]
            assert [s["observation"] for s in states] == list(range(t0, t0 + len(tape) + 1))
            for field, state, key in (("cash", root, "root_cash"), ("cash", after, "window_end_cash"),
                                      ("trees", after, "window_end_trees"), ("trees", end, "terminal_trees")):
                actual = [p[field] for p in state["physical_seats"]]
                if key == "root_cash":
                    assert all(abs(a-b) <= .011 for a,b in zip(actual, verified[key], strict=True))
                else:
                    assert actual == verified[key]
            counts = [[[team["collectors"], team["planters"]] for team in st["physical_seats"]] for st in states]
            traces.append(dict(seed=seed, seat=seat, arm=arm, first_observation=t0, by_observation=counts))
            runs.append(dict(seed=seed, seat=seat, arm=arm, receipt_sha256=R.sha(path), controlled_steps=width,
                             root=root, window_end=after, terminal=end, official_scores=scores, status=status,
                             own_margin=verified["margin"], changed_window_proposal_steps=len(verified["window_proposal_differences"])))
    assert len(runs) == 4
    series = directory / "verified_worker_roles.json.gz"
    R.write(series, dict(layout="by_observation[step-first_observation][physical_seat] = [collectors, planters]", traces=traces))
    result = dict(status="all_four_official_terminals_verified", settlement_sha256=R.sha(settled), runs=runs,
                  role_trace=series.name, role_trace_sha256=R.sha(series),
                  summed_candidate_margin_delta=summary["summed_candidate_margin_delta"])
    R.write(directory / "verified_inventories.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.freeze, args.out)
