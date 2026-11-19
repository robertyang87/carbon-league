"""Trace already-recorded certified pairs to public antecedents and actual terminal outcomes."""
from collections import Counter
import gzip
import json
import sys

from fit import HERE, NATIVE, checked, digest, read
from runtime import distance, step

sys.path.insert(0, str(HERE / "evidence"))
from account import settled


def compressed(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def state(obs, side, uid):
    own, enemy = obs["players"][side], obs["players"][1-side]
    state = own[2].get(uid)
    if not state:
        return None
    pos, cargo, role = state
    assert role == "COLLECTOR"
    near = [[other, w[0], round(w[1], 4), distance(pos,w[0],15)]
            for other, w in enemy[2].items() if w[2] == "COLLECTOR" and distance(pos,w[0],15) <= 2]
    return {"pos": pos, "cargo": cargo, "distance_to_own_base": min(distance(pos,b,15) for b in own[1].values()),
            "public_tile_carbon": obs["carbon"][pos], "visible_enemy_collectors_within_2": sorted(near)}


def main():
    rows = checked("train")
    sources = {(r["seed"],r["seat"]): r for r in read(HERE / "protocol.json")["source_groups"]}
    cases = []
    for r in rows:
        if r["status"] != "paired_full_original":
            continue
        seed,side,t,uid = r["seed"],r["seat"],r["t"],r["proposal"]["uid"]
        folder = NATIVE / f"train_{seed}_{side}"
        source = sources[seed,side]
        tape = compressed(HERE.parents[3] / source["raw"])
        sim = settled.FastCarbon(tape["game_configuration"]).reset(seed=seed)
        pre = []
        for j in tape["actions"][:t]:
            turn = sim.step_count
            if turn in [t-10,t-5,t-3,t-1]:
                pre.append({"t":turn,"public_state":state(sim.obs(side),side,uid),"executed":j[side].get(uid,"NONE")})
            sim.step(*j)
        snap = sim.get_state()
        case = {"seed":seed,"seat":side,"root_t":t,"uid":uid,"history":pre,"arms":{}}
        cases.append(case)
        for key in ("base","alt"):
            sim = settled.FastCarbon(tape["game_configuration"])
            sim.set_state(snap)
            arm = compressed(folder / f"{key}.json.gz")
            trace = []
            teams = Counter()
            for j in arm["orders"]:
                current = sim.obs(side)
                worker = state(current,side,uid)
                cmd = j[side].get(uid,"NONE") if worker else None
                t_now = sim.step_count
                occupants = []
                if worker:
                    target = step(worker["pos"],cmd,15)
                    for p in (0,1):
                        for wid,(pos,cargo,role) in current["players"][p][2].items():
                            order = j[p].get(wid,"NONE")
                            if step(pos,order,15)==target:
                                occupants.append([p,wid,round(cargo,4),role,order])
                settled.one_step(sim,j,side,verify=True)
                cash = [list(event) for event in sim.cash.events if event[0]=="own_delivery"]
                for event in cash:
                    teams[event[1]]+=event[2]
                trace.append({"t":t_now,"pre":worker,"own_action":cmd,"joint_arrivals_at_focal_destination":occupants,
                              "next":state(sim.obs(side),side,uid),"true_deposits":cash})
            cert = compressed(folder/f"{key}_account.json.gz")["account"]
            for p in (0,1):
                assert abs(teams[p] - cert["team"]["treasury_flows_physical"][p].get("own_delivery",0))<1e-5
            case["arms"][key]={"turns":trace,"principal_ledger":cert["per_worker"][uid],"final_cash":cert["terminal_cash_physical"],
                               "official_money_margin":cert["official_margin"]}
    raw=(json.dumps({"train_labels_sha256":digest(NATIVE/"train_labels.json"),"cases":cases},
                    ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n").encode()
    out=HERE/"turning_paths.json.gz"
    with out.open("wb") as f:
        with gzip.GzipFile(filename="",fileobj=f,mode="wb",mtime=0) as g:
            g.write(raw)
    print(len(cases),digest(out))


if __name__=="__main__":
    main()
