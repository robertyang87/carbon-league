"""Build and verify a one-intervention-per-game deployment candidate from the frozen source parent."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from fit import HERE, ROOT, FastCarbon, load, save

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def build(out):
    base=HERE.parent/"planter_capture_full_dev/candidate.py"
    if sha(base)!="543a9d927ec9a2f6b6e2020ac2236eb9aecb594c0ee16a44c1a0ddee86f23797":raise ValueError("source parent changed")
    parms=load(HERE/"weights.json")["weights"]
    extra="\n_HARVEST_IMPL = {}\nexec("+repr((HERE/"policy.py").read_text())+", _HARVEST_IMPL)\n"
    extra+="_HARVEST_WEIGHTS = "+repr(parms)+"\n"
    extra+='''
_HARVEST_PREV = {}
_HARVEST_USED = set()
_harvest_parent_agent = agent

def agent(observation, configuration=None):
    orders = _harvest_parent_agent(observation, configuration)
    seat, tick = observation["player"], observation["step"]
    if tick == 0:
        _HARVEST_PREV.pop(seat, None)
        _HARVEST_USED.discard(seat)
    old = _HARVEST_PREV.get(seat)
    _HARVEST_PREV[seat] = {"players": observation["players"]}
    if seat in _HARVEST_USED:
        return orders
    cfg = configuration if configuration is not None else {"size": 15, "episodeSteps": 300, "treeLifespan": 50}
    choice = _HARVEST_IMPL["choose"](observation, old, cfg, orders, _HARVEST_WEIGHTS)
    if choice:
        _HARVEST_USED.add(seat)
        orders = dict(orders)
        orders.pop(choice[1])
    return orders
'''
    out.write_bytes(base.read_bytes()+extra.encode())
    if out.stat().st_size>10000000:raise ValueError("oversize candidate")

def verify(path):
    result=[]
    for q in load(HERE/"selected.json"):
        raw=load(ROOT/q["record"]["source"]);seed=raw["seed"];s=q["seat"]
        cfg=raw["game_configuration"];sim=FastCarbon(cfg).reset(seed=seed)
        spec=importlib.util.spec_from_file_location("harvest_standalone_test",path)
        deployed=importlib.util.module_from_spec(spec);spec.loader.exec_module(deployed)
        for t,taped in enumerate(raw["actions"][:q["root_t"]+1]):
            obs=sim.obs(s);got=deployed.agent(obs,cfg);expected=dict(taped[s])
            if t==q["root_t"]:del expected[q["root"]["worker"]]
            if got!=expected:raise ValueError(f"deployment action mismatch at map={seed}, t={t}")
            sim.step(*taped)
        result.append(dict(seed=seed,seat=s,prefix_actions_checked=q["root_t"]+1,changed_worker=q["root"]["worker"]))
    save(HERE/"package_parity.json",dict(parent_sha=sha(HERE.parent/"planter_capture_full_dev/candidate.py"),
         candidate_sha=sha(path),bytes=path.stat().st_size,release="first qualifying root once in each game",cases=result))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        out=args.out or Path(folder)/"candidate.py"
        build(out);verify(out)
