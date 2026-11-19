"""One separately registered reproduction of the first fixed root."""
import fcntl
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import responses as r


def run():
    plan=r.load(HERE/"protocol.json")
    case=r.load(r.HERE/"selected_roots.json")["cases"][0]
    if r.sha(r.PLAN)!=plan["original_registration"] or case["t"]!=265 or case["source_sha256"]!=plan["source"]["source_sha256"]:
        raise ValueError("changed source or root")
    original=[r.load(r.HERE/f"responses_{case['seed']}_{case['seat']}_{a}.json.gz") for a in ("parent","candidate")]
    if original[0]["default_warm_cost"]==original[1]["default_warm_cost"]:raise ValueError("recorded fault absent")
    with r.LOCK.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        warm=HERE/"shared_warm.json"
        if not warm.exists():
            with ProcessPoolExecutor(1) as pool:cost=pool.submit(r.native_warm).result()
            r.write(warm,dict(protocol=r.sha(HERE/"protocol.json"),cost=cost))
        receipt=r.load(warm)
        if receipt["protocol"]!=r.sha(HERE/"protocol.json"):raise ValueError("warmup binding")
        with ProcessPoolExecutor(2) as pool:
            old,new=pool.map(r.arm,[(case,"parent",0,receipt["cost"],HERE),(case,"candidate",1,receipt["cost"],HERE)])
    if not (old["initial_rng"]==new["initial_rng"]==original[0]["initial_rng"]==original[1]["initial_rng"]==plan["initial_seed"] and old["default_warm_cost"]==new["default_warm_cost"]==receipt["cost"]):
        raise ValueError("replication not matched")
    diff=[i+case["t"] for i,(x,y) in enumerate(zip(old["tape"],new["tape"])) if x[1-case["seat"]]!=y[1-case["seat"]]]
    r.write(HERE/"result.json",dict(protocol=r.sha(HERE/"protocol.json"),matched_initial_rng_and_cost=True,
            seed=case["seed"],seat=case["seat"],t=case["t"],signed_margins=[old["margin"],new["margin"]],
            delta=new["margin"]-old["margin"],response_changed_turns=diff,delta_tree=new["cash"]["tree_income_margin"]-old["cash"]["tree_income_margin"],
            delta_deposits=new["cash"]["delivery_margin"]-old["cash"]["delivery_margin"],
            original_invalid_margins=[x["margin"] for x in original]))


if __name__=="__main__":run()
