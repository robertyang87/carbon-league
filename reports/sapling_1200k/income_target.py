"""Reconstruct spendable receipts and the paired-map income difference from real action tapes."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

import numpy as np
from turning_point_ledger import CashEvents, FastCarbon, compact, sha

RECEIPTS = frozenset(("own_delivery", "opponent_worker_delivery_to_this_base", "tree_absorption",
                      "tree_collision_cash", "base_collision_cash"))
SPENDING = frozenset(("hire_collector", "hire_planter", "plant", "seize"))
METRICS = ("gross_receipts", "net_income", "official_margin")
TARGET = 500.
FORMAL_METRIC = "official_margin"


def reconstruct(row, raw):
    if sha(compact(raw["actions"])) != row["actions_sha256"] or len(raw["actions"])!=row["actual_transitions"]:
        raise ValueError("income ledger action mismatch")
    sim = FastCarbon(raw["game_configuration"]).reset(seed=row["seed"])
    initial = list(sim.cash)
    sim.cash = CashEvents(sim.cash)
    entries = [Counter(), Counter()]
    for actions in raw["actions"]:
        sim.cash.events.clear()
        sim.step(*actions)
        for kind, seat, value, *_ in sim.cash.events:
            if (not math.isfinite(value) or kind not in RECEIPTS | SPENDING | {"rounding"}
                    or (kind in RECEIPTS and value < 0) or (kind in SPENDING and value > 0)):
                raise ValueError("invalid economic transaction")
            entries[seat][kind] += value
    if raw["last_post_cash"] != sim.cash or sha(compact(sim.get_state()))!=row["final_state_sha256"]:
        raise ValueError("income ledger terminal mismatch")
    gross, net = [], []
    for s in (0, 1):
        if not math.isclose(initial[s]+sum(entries[s].values()),sim.cash[s],rel_tol=0,abs_tol=1e-6):
            raise ValueError("cash reconciliation failed")
        gross.append(math.fsum(entries[s][k] for k in sorted(RECEIPTS)))
        net.append(sim.cash[s]-initial[s])
    me = row["candidate_seat"]
    return dict(gross_receipts=gross[me]-gross[1-me], net_income=net[me]-net[1-me],
                official_margin=row["margin"],gross_by_seat=gross,net_by_seat=net,
                cash_entries_by_seat=[dict(sorted(e.items())) for e in entries])


def summarize(rows, metric):
    if metric not in METRICS:
        raise ValueError("unknown income definition")
    pairs = {}
    for r in rows:
        key=(r["seed"],r["candidate_seat"])
        if key in pairs or key[1] not in (0,1): raise ValueError("repeated or invalid map and seat")
        value=r["income_account"][metric]
        if not math.isfinite(value): raise ValueError("nonfinite income")
        if metric == FORMAL_METRIC:
            if "margin" in r and not math.isclose(r["margin"],value,rel_tol=0,abs_tol=1e-7):
                raise ValueError("income ledger differs from the official score margin")
            value=round(value*100)/100
        pairs[key]=value
    maps=sorted({k[0] for k in pairs})
    if not maps or len(pairs)!=2*len(maps):
        raise ValueError("both seats are required for every map")
    across=np.array([(pairs[m,0]+pairs[m,1])/2 for m in maps])
    mean=float(across.mean())
    met=mean>=TARGET
    if metric == FORMAL_METRIC:
        # Official terminal scores are in cents. A sample exactly on +500
        # must pass regardless of the order in which games were played.
        total_cents=sum(round(v*100) for v in pairs.values())
        mean=total_cents/(100*len(rows))
        met=total_cents>=round(100*TARGET)*len(rows)
    # A two-sided, percentile map bootstrap is descriptive for a single frozen
    # exam. With one or two maps report the observed pairs without a nominal CI.
    interval=None
    if len(maps)>=10:
        rng=np.random.default_rng(551218)
        means=np.empty(10000)
        for i in range(0,len(means),100):
            sample=rng.integers(0,len(maps),size=(100,len(maps)))
            means[i:i+100]=across[sample].mean(1)
        interval=np.quantile(means,[.025,.975]).tolist()
    return dict(metric=metric,game_count=len(rows),map_count=len(maps),mean_difference=mean,
                seat_mean_difference=[math.fsum(pairs[m,s] for m in maps)/len(maps) for s in (0,1)],
                map_cluster_bootstrap_95pct=interval,observed_at_least_500=met)


def development(folder):
    from tree_win_gate import validate_game, verify_raw
    source=json.loads((folder/"results.json").read_bytes())
    if source["status"]!="complete": raise ValueError("game run is incomplete")
    expected={(seed,s) for seed in source["header"]["seeds"] for s in (0,1)}
    if len(source["games"])!=len(expected) or {(r["seed"],r["candidate_seat"]) for r in source["games"]}!=expected:
        raise ValueError("map schedule differs")
    rows=[]
    for row in source["games"]:
        header=source["header"]
        verify_raw(folder,row,header["pack_sha256"],header["weight_sha256"])
        validate_game(row)
        raw=json.loads(gzip.decompress((folder/row["raw_file"]).read_bytes()))
        rows.append(dict(seed=row["seed"],candidate_seat=row["candidate_seat"],result=row["result"],
                         raw_sha256=row["raw_sha256"], income_account=reconstruct(row,raw)))
    return dict(games=rows,wins=sum(r["result"]=="win" for r in rows),
                summary={metric:summarize(rows,metric) for metric in METRICS})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",type=Path,nargs="+")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if len({p.name for p in args.runs})!=len(args.runs): parser.error("duplicate run")
    doc={p.name:development(p) for p in args.runs}
    args.out.write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n")
