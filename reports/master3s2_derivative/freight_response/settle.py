"""Replay the suffix orders and reconcile both seats' official cash and ten-step receipts."""
from collections import Counter
import gzip
import json
from zipfile import ZipFile

import run as r


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def clock_stats(rows, s):
    clocks=[row["clocks"][s] for row in rows]
    walls=[c["wall"] for c in clocks]
    return dict(calls=len(rows),total_wall=sum(walls),total_cpu=sum(c["cpu"] for c in clocks),
                wall_p95=float(r.np.quantile(walls,.95)),wall_p99=float(r.np.quantile(walls,.99)),max_wall=max(walls),
                slower_than_one=sum(w>1 for w in walls),above_hard_limit=sum(w>3 for w in walls),
                overage=sum(max(0,w-1) for w in walls),deadline_triggers=sum(c["aborted"] for c in clocks),
                candidates=dict(sorted(Counter(c["K"] for c in clocks).items())),
                depth=dict(sorted(Counter(c["H"] for c in clocks).items())),
                rollouts=dict(sorted(Counter(c["rolls"] for c in clocks).items())))


def settle():
    games=json.loads((r.D/"dev1/hosted/settlement.json").read_bytes())["games"]
    results=[]
    for ticket in r.P["roots"]:
        seed,seat,t=ticket["seed"],ticket["seat"],ticket["t"]
        game=next(g for g in games if (g["seed"],g["seat"])==(seed,seat))
        stem=f"{seed}_{seat}_{t}";root_file=r.HERE/(stem+"_root.json.gz");root=read(root_file)
        default=root["default"][0];default_index=next(x["index"] for x in root["menu"] if x["command"]==default)
        result=dict(seed=seed,seat=seat,root_step=t,source_archive_sha256=root["source_archive"],
                    root_file_sha256=r.sha(root_file.read_bytes()),K=len(root["menu"]),
                    default_index=default_index,default_score=root["menu"][default_index]["score"],arms={})
        results.append(result)
        alt=root["alternative"]
        if alt is None:
            result["status"]="no_improving_native_alternative"
            continue
        result.update(status="paired",alternative_index=alt["index"],alternative_score=alt["score"],
                      changed={k:[default.get(k,"NONE"),alt["command"].get(k,"NONE")]
                               for k in sorted(default.keys()|alt["command"].keys()) if default.get(k)!=alt["command"].get(k)})
        zpath=r.D/"dev1/hosted"/game["archive_path"]
        with ZipFile(zpath) as z:
            turns=[json.loads(q) for q in z.read("opponent_policy/turns.jsonl").splitlines()]
        actual=[ [c["response"]["action"] for c in row["calls"]] for row in turns]
        shared=None
        for arm in ticket["arm_order"]:
            path=r.HERE/(stem+"_"+arm+".json.gz");evidence=read(path)
            cfg=r.econ.config();cfg.pop("randomSeed",None)
            r.np.random.seed(seed);env=r.make("carbon",configuration={**cfg,"randomSeed":seed})
            sim=r.econ.FastCarbon(cfg).reset(seed=seed);sim.cash=r.econ.CashEvents(sim.cash)
            for orders in actual[:t]:
                sim.step(*orders);env.step(orders,[])
            bank=turns[t-1]["overage_remaining_seconds"].copy()
            expected=default if arm=="default" else alt["command"]
            rows=evidence["records"]
            r.check(rows[0]["orders"][seat]==expected,"wrong requested intervention")
            signature=(rows[0]["orders"][1-seat],rows[0]["clocks"][1-seat],rows[0]["public_sha256"])
            if shared is None: shared=signature
            else:r.check(shared==signature,"opponent or information at root differs")
            for i,row in enumerate(rows,t):
                obs=[r.public(env,s,bank) for s in (0,1)]
                r.check(i==row["t"] and row["public_sha256"]==[r.digest(o) for o in obs],"public request not reproducible")
                sim.cash.events.clear();sim.step(*row["orders"]);env.step(row["orders"],[])
                r.check(row["end_cash"]==sim.cash==[p[0] for p in env.states[0].observation.players],"scoring divergence")
                deposits=[sum(ev[2] for ev in sim.cash.events if ev[0]=="own_delivery" and ev[1]==s) for s in range(2)]
                r.check(deposits==row["delivered"],"incorrect acquisition origin")
                bank=[v-max(0,c["wall"]-1) for v,c in zip(bank,row["clocks"])]
            r.check(env.done and [s.reward for s in env.states]==evidence["scores"],"invalid terminal score")
            outcome=dict(scores=evidence["scores"],margin=evidence["official_margin"],deposits10=evidence["deliveries10"],
                         real_suffix_orders_different_from_historical=sum(row["orders"][1-seat]!=actual[row["t"]][1-seat] for row in rows[1:]),
                         calls_by_seat=[clock_stats(rows,s) for s in range(2)],
                         first10_opponent_orders=[row["orders"][1-seat] for row in rows[:10]],
                         suffix_sha256=r.sha(path.read_bytes()))
            r.check(outcome["deposits10"]==[sum(row["delivered"][s] for row in rows[:10]) for s in range(2)],"ten-step deposits")
            result["arms"][arm]=outcome
        a,b=(result["arms"][arm] for arm in ("default","deviation"))
        result["deviation_minus_default"]=dict(official_margin=b["margin"]-a["margin"],deposits10_by_seat=[y-x for x,y in zip(a["deposits10"],b["deposits10"])])
    r.put("results.json",dict(scope="Two retrospectively selected dev1 roots. Paired exploratory labels, never independent wins.",roots=results))


if __name__=="__main__":
    settle()
