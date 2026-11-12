"""One real inference call before/after R12; requires the baseline Git object."""
import os
import subprocess
import types
from common import setup, fixture_weights, emit


def main():
    repo, _ = setup()
    import cnet2_search as current
    import eval_v53
    from fastsim import FastCarbon

    ref = os.environ.get("CARBON_REVIEW_BEFORE", "5f338eb")
    source = subprocess.check_output(["git", "-C", str(repo), "show", ref + ":code/cnet2_search.py"], text=True)
    before = types.ModuleType("review_search_before")
    before.__file__ = str(repo / "code" / "cnet2_search.py")
    exec(compile(source, ref + ":code/cnet2_search.py", "exec"), before.__dict__)
    weights = fixture_weights()
    obs = FastCarbon(eval_v53.CFG).reset(seed=970500).obs(0)
    conf = {"TEACHER": 1, "SEARCH": False, "ROLL": "self", "GUARD": True,
            "OPEN_T": 3.0, "OPEN_STEPS": 100, "OPEN_SEED": 8}
    rows = []
    for label, module in (("before", before), ("after", current)):
        agent = module.make_search_agent(conf, weights=weights)
        action = agent(obs, eval_v53.CFG)
        rows.append({"version": label, "actions": action, "stats": agent.stats})
    emit("sampling_before_after.jsonl", rows)


if __name__ == "__main__":
    main()
