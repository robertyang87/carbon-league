"""Check how a learned opponent prior changes actual native search under its real clock."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
STEPS = (85, 215)
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
CANDIDATE = "245b5814d1855c5449435ee6bd4a53560017a8cc7a46d08729fa9dfdc13e9654"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def same(rows, key):
    return all(row[key] == rows[0][key] for row in rows[1:])


def root_call(mod, tracker, previous, obs, cfg, policy_seed, cost):
    import numpy as np

    me, t = obs["player"], obs["step"]
    mod._ST = dict(tr=copy.deepcopy(tracker), prev=previous, rng=np.random.RandomState(policy_seed))
    mod._S_COST[:] = cost
    mod._S_REC = []
    mod._S_STATS.update(searched=0, changed=0, aborted=0, single=0)
    mod._S_STATS.pop("last", None)
    enemy_samples, roots = [], []
    native_sample, native_search = mod._sample, mod._search

    def sample(players, player, lw, lb, rng, step, fixed=None):
        order = native_sample(players, player, lw, lb, rng, step, fixed)
        if players is obs["players"] and player != me and step == t:
            enemy_samples.append(order)
        return order

    def search(observation, configuration, est, act0, lw, lb, prev, t0, home=()):
        roots.append(act0)
        return native_search(observation, configuration, est, act0, lw, lb, prev, t0, home)

    mod._sample, mod._search = sample, search
    try:
        wall, cpu = time.perf_counter(), time.process_time()
        issued = mod.agent(obs, cfg)
        cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    finally:
        mod._sample, mod._search = native_sample, native_search
    stats = mod._S_STATS
    assert len(roots) == 1 and len(enemy_samples) <= 1
    rec = mod._S_REC
    last = stats.get("last")
    menu, ds, scores, best = [], [], [], None
    if rec:
        step, side, menu, ds, scores, best = rec[0]
        assert len(rec) == 1 and (step, side) == (t, me) and menu[0] == roots[0] and issued == menu[best]
    count, depth, rolls = (last[:3] if last else (1 if stats["single"] else 0, 0, 0))
    return dict(
        act0=roots[0], opponent_samples=enemy_samples, issued=issued, menu=menu, depth_by_arm=ds,
        sorted_depths=sorted(ds), leaf_scores=[float(x) for x in scores], best=best,
        candidates=count, depth=depth, rollouts=rolls, native_aborted=stats["aborted"],
        single=stats["single"], wall_s=wall, cpu_s=cpu, internal_wall_s=last[3] if last else None,
        ending_costs=[float(i) for i in mod._S_COST],
    )


def assess(rows):
    a, b, x, y = [rows[k] for k in ("original_1", "original_2", "candidate_1", "candidate_2")]
    traced = all(r["depth"] > 0 and len(r["opponent_samples"]) == 1 for r in (a, b, x, y))
    assert same([a,b,x,y], "act0")
    if traced:
        assert a["opponent_samples"] == b["opponent_samples"] and x["opponent_samples"] == y["opponent_samples"]
    control = a["issued"] == b["issued"]
    repeat = x["issued"] == y["issued"]
    same_work = traced and all(same([a,b,x,y], k) for k in
                               ("menu","sorted_depths","rollouts","depth","native_aborted"))
    opponent = traced and a["opponent_samples"] != x["opponent_samples"]
    changed = a["issued"] != x["issued"]
    return dict(four_arms_evaluated=traced, old_repeat_stable=control, new_repeat_stable=repeat,
                equal_menu_and_depth_work=same_work, opponent_changed=opponent, final_changed=changed,
                attributable_changed=bool(same_work and control and repeat and opponent and changed))


def main(core):
    if core not in os.sched_getaffinity(0):
        raise ValueError("Requested CPU is unavailable")
    os.sched_setaffinity(0, {core})
    for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ["CARBON_DEBUG"] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from original_policy import PROTOCOL, original, module, free_entry

    original_src = original()
    cand = (HERE.parent/"candidate.py").read_bytes()
    assert sha(original_src) == ORIGINAL and sha(cand) == CANDIDATE
    mods = {name: module(code) for name,code in (("original",original_src),("candidate",cand))}
    assert all(m._S_ABORT == .58 and m._TEMP == 1. and m._S_D2 == 8 and m._S_K == 32
               and m._SEED is None and m._net() is not None for m in mods.values())
    opening_cost = [sum(m._S_COST[i] for m in mods.values())/2 for i in range(4)]
    assert all(math.isfinite(v) and v>0 for v in opening_cost)
    held = [c for c in PROTOCOL["games"] if c["split"] == "val"]
    assert len(held)==8 and len({c["terrain"] for c in held})==4 and set(STEPS) <= set(PROTOCOL["steps"])
    assert all({c["candidate_seat"] for c in held if c["terrain"]==m}=={0,1} for m in {c["terrain"] for c in held})
    roots = []
    for c in held:
        seat = c["candidate_seat"]
        source = mods["original"]
        tracker = source.BeliefTracker(source._Bel,seat)
        prev = None
        for t,obs,cfg,archived in free_entry.public_history({**c,"expert_seat":seat}):
            if t in STEPS:
                assert "full_carbon" not in obs
                token = f"m3s2-opponent-probe-v1/{c['terrain']}/{seat}/{t}".encode()
                seed = int.from_bytes(hashlib.sha256(token).digest()[:4],"big")
                arms = (["original_1","candidate_1","candidate_2","original_2"] if len(roots)%2==0 else
                        ["candidate_1","original_1","original_2","candidate_2"])
                public_cfg = {k:v for k,v in cfg.items() if k!="randomSeed"}
                rows = {tag:root_call(mods[tag.split("_")[0]],tracker,prev,obs,public_cfg,seed,opening_cost)
                        for tag in arms}
                roots.append(dict(seed=c["seed"],terrain=c["terrain"],archive_sha256=c["sha256"],
                                  t=t,seat=seat,private_rng_seed=seed,invocation_order=arms,arms=rows,
                                  assessment=assess(rows)))
            tracker.step(obs["players"],t,obs["carbon"])
            tracker.set_actions(obs["players"],archived)
            prev = obs["players"]
        print(c["seed"],seat,flush=True)
    assert len(roots)==16
    counts={k:sum(r["assessment"][k] for r in roots) for k in roots[0]["assessment"]}
    result=dict(schema="native-master3s2-opponent-search-crossover/1",original_sha256=ORIGINAL,
                candidate_sha256=CANDIDATE,source_protocol_sha256=sha((HERE.parent/"protocol.json").read_bytes()),
                verifier_sha256=sha(Path(__file__).read_bytes()),steps=STEPS,cpu_affinity=[core],
                torch_threads=torch.get_num_threads(),python=platform.python_version(),
                opening_costs=opening_cost,scope="16 fixed historical roots. Full real 0.58-second search with paired private samples and a common measured starting cost; no new games or ladder qualification.",
                gate="At least one root repeats within each package, changes the shared opponent order and own issued command across packages, with identical candidate menu, sorted completed depths, rollout and native abort status.",
                counts=counts,roots=roots,dispatch_eligible=bool(counts["attributable_changed"]>0))
    path=HERE/"full_search.json.gz"
    raw=(json.dumps(result,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n").encode()
    tmp=path.with_suffix(".tmp"); tmp.write_bytes(gzip.compress(raw,mtime=0)); tmp.replace(path)
    print(counts,sha(path.read_bytes()),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu",type=int,required=True)
    main(ap.parse_args().cpu)
