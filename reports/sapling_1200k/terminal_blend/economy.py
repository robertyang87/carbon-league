"""Account for score divergence during the paired terminal-policy search games."""
from collections import Counter
import gzip
import json

import train
from turning_point_ledger import CashEvents


def replay(row):
    path = train.HERE / row["raw_file"]
    if train.prior.digest(path) != row["raw_sha256"]:
        raise ValueError("tape hash mismatch")
    raw = json.loads(gzip.decompress(path.read_bytes()))
    sim = train.exam.FastCarbon(raw["game_configuration"]).reset(seed=row["seed"])
    sim.cash = CashEvents(sim.cash)
    seat = row["candidate_seat"]
    running = [Counter(),Counter()]
    points = [{"t":0,"lead":sim.cash[seat]-sim.cash[1-seat],"sources":[{},{}]}]
    for joint in raw["actions"]:
        old = list(sim.cash)
        sim.cash.events=[]
        sim.step(*joint)
        counts = [Counter(), Counter()]
        for name, p, value, *_ in sim.cash.events:
            counts[p][name] += value
        for p in (0,1):
            if abs(sum(counts[p].values())-(sim.cash[p]-old[p]))>1e-6:
                raise ValueError("unexplained cash movement")
            running[p].update(counts[p])
        points.append({"t":sim.step_count,"lead":sim.cash[seat]-sim.cash[1-seat],
                       "sources":[dict(running[p]) for p in (seat,1-seat)]})
    if not any(raw["extinct_by_seat"]) and abs(points[-1]["lead"]-row["margin"]) > 1e-6:
        raise ValueError("score differs from tournament tape")
    return points


def main():
    results = json.loads((train.HERE/"results.json").read_bytes())
    summaries=[]
    for seed in train.HOLD:
        for seat in (0,1):
            games={g["arm"]:g for g in results["games"] if g["stage"]=="holdout" and g["seed"]==seed and g["candidate_seat"]==seat}
            lines=[replay(games[a]) for a in ("parent",results["chosen"])]
            n=min(map(len,lines))
            def gap(t):
                return lines[1][t]["lead"]-lines[0][t]["lead"]
            start=min(range(n-20),key=lambda t:round(gap(t+20)-gap(t),2))
            def decompose(t0,t1):
                effects=Counter()
                for i,side in enumerate(lines):
                    for p in range(2):
                        a,b=[side[t]["sources"][p] for t in (t0,t1)]
                        for k in a.keys()|b.keys():
                            effects[k]+=(b.get(k,0)-a.get(k,0))*(1 if i==1 else -1)*(1 if p==0 else -1)
                if abs(sum(effects.values())-(gap(t1)-gap(t0)))>1e-6:
                    raise ValueError("unexplained differential swing")
                return {k:round(v,3) for k,v in sorted(effects.items(),key=lambda item:-abs(item[1]))}
            times=sorted({t for t in (0,60,120,180,240,n-1) if t<n})
            summaries.append({"seed":seed,"seat":seat,"gap_over_parent_at":{str(t):round(gap(t),2) for t in times},
                              "all_turn_cash_components":decompose(0,n-1),
                              "worst_20":{"from":start,"to":start+20,"differential_swing":round(gap(start+20)-gap(start),2),
                                          "lead_gap_at_start":round(gap(start),2),"lead_gap_ten_steps_before":round(gap(max(0,start-10)),2),
                                          "components":decompose(start,start+20)}})
    train.prior.shared.atomic_json(train.HERE/"economy.json",{
        "scope":"Difference in (our cash minus opponent cash) between two independently played deterministic policies on the same map and seat. Mechanism is audited accounting; these trajectories do not identify the causal value of any individual move.",
        "rows":summaries})


if __name__=="__main__":
    main()
