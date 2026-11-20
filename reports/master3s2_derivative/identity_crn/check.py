"""Check the draw contract and sixteen public native decisions on eight old seat streams."""
import copy
import gzip
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
spec = importlib.util.spec_from_file_location("registered_parent_replay", FAMILY / "return_rank/development/root_responses.py")
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
np = trace.np


def contracts(parent, child):
    ps = [
        [200, {"b0": 4, "b1": 24}, {"w0": [61, 12., "COLLECTOR"], "w1": [92, 0., "PLANTER"], "w2": [125, 29., "COLLECTOR"]}, {}],
        [200, {"e0": 50}, {"a": [81, 18., "COLLECTOR"]}, {}],
    ]
    rng = np.random.RandomState(7121)
    lw = rng.normal(size=(225, 5)).astype(np.float32)
    lb = rng.normal(size=(225, 3)).astype(np.float32)
    changed_after_death = 0
    for seed in range(512):
        expected = parent._sample(ps, 0, lw, lb, np.random.RandomState(seed), 111)
        draw = child._identity_uniforms(seed, ps[0])
        assert child._sample(ps, 0, lw, lb, None, 111, choose=draw) == expected
        # These root workers' five destinations do not overlap. A death changes no occupancy here.
        reduced = copy.deepcopy(ps)
        del reduced[0][2]["w1"]
        def order(ordering):
            return (ordering.get("w0", "NONE"), ordering.get("w2", "NONE"))
        separate = child._sample(reduced, 0, lw, lb, None, 111, choose=draw)
        assert order(separate) == order(expected)
        shifted = parent._sample(reduced, 0, lw, lb, np.random.RandomState(seed), 111)
        changed_after_death += order(shifted) != order(expected)
        alternative = {"a-new": [15, 0., "PLANTER"], **reduced[0][2]}
        alternative = [reduced[0][0], reduced[0][1], alternative, {}]
        draw2 = child._identity_uniforms(seed, ps[0])
        u = child._sample([alternative, ps[1]], 0, lw, lb, None, 111, choose=draw2)
        assert order(u) == order(expected)
    return dict(seeds=512, same_roster_exact=512, reordered_with_birth_exact=512,
                removed_middle_survivor_pair_preserved=512, parent_pair_changed_after_removal=changed_after_death)


def sampled_call(model, history, obs, cfg, seed):
    tracker, previous = history
    model._ST = dict(tr=copy.deepcopy(tracker), prev=copy.deepcopy(previous), rng=np.random.RandomState(seed))
    model._S_COST[:] = [.001, .002, .001, .002]
    model._S_STATS.pop("last", None)
    model._S_REC = []
    order, clock = trace.base.measured(model, obs, cfg)
    vals = model._S_REC
    model._S_REC = None
    if not vals:
        return dict(order=order,clock=clock,menu=None,selected=None)
    _, _, menu, _, _, selected = vals[0]
    assert order == menu[selected]
    return dict(order=order, clock=clock, selected=selected, menu=trace.base.digest(menu))


def run():
    import torch
    torch.set_num_threads(1)
    os.environ["CARBON_DEBUG"] = "1"
    parent, child = [trace.base.load("identity_crn_"+label, path) for label, path in (
        ("parent", FAMILY/"cold_start/candidate.py"), ("candidate", HERE/"candidate.py"))]
    for model in (parent, child):
        model._net()
        assert model._TEMP == 1 and model._SEED is None and model._S_ABORT == .58 and model._SEARCH_OK
    result = dict(contracts=contracts(parent, child), roots=[], completed=False)
    plan = json.loads((FAMILY/"return_rank/development/root_protocol.json").read_bytes())
    public_plan = json.loads((HERE/"protocol.json").read_bytes())
    assert sha256((HERE/"candidate.py").read_bytes()).hexdigest() == public_plan["candidate"]
    cfg = trace.base.econ.config()
    cfg.pop("randomSeed", None)
    for ticket in plan["roots"]:
        i = ticket["ordinal"]
        if i not in public_plan["probe_ordinals"]:
            continue
        seat, t = ticket["seat"], ticket["step"]
        frames, commands = trace.tape(ticket)
        history = trace.historical_root(ticket,seat,parent,cfg,frames,commands)
        obs = json.loads(frames[t])["observations"][seat]
        sequence = ("parent","child","child","parent") if i%2==0 else ("child","parent","parent","child")
        calls = [(arm, sampled_call(parent if arm=="parent" else child,history,obs,cfg,
                                    19073+7*t+ticket["seed"]+seat)) for arm in sequence]
        menus = len({c["menu"] for _,c in calls}) == 1
        work = lambda c: tuple(c["clock"][k] for k in ("K","H","rolls"))
        equal_work = len({work(c) for _,c in calls}) == 1
        stable = all(len({trace.base.digest(c["order"]) for name,c in calls if name==arm}) == 1 for arm in ("parent","child"))
        changed = next(c["order"] for a,c in calls if a=="parent") != next(c["order"] for a,c in calls if a=="child")
        result["roots"].append(dict(ordinal=i,seed=ticket["seed"],seat=seat,step=t,common_menu=menus,
                                    same_work=equal_work,within_arm_stable=stable,changed=changed,calls=calls))
        print(i,menus,equal_work,stable,changed,flush=True)
    result["completed"] = True
    trace.save(HERE/"checks.json.gz", result)
    print(result["contracts"],flush=True)


if __name__ == "__main__":
    run()
