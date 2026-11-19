"""Trace provisional cash leads and the largest signed advances in every fresh candidate loss."""
import gzip
import json
from pathlib import Path
from audit import HERE,ROOT,FastCarbon,CashEvents,read,digest,require,worker_state,terms_window


def sample(g):
    p=ROOT/g["transcript"];require(digest(p)==g["raw_sha256"],"transcript no longer signed")
    raw=read(p);seat=g["seat"];sim=FastCarbon(raw["game_configuration"]).reset(seed=g["seed"])
    sim.cash=CashEvents(sim.cash);leads=[sim.cash[seat]-sim.cash[1-seat]];rows=[]
    for t,orders in enumerate(raw["actions"]):
        before=worker_state(sim.full_observation());cash=list(sim.cash)
        sim.cash.events.clear();sim.step(*orders);after=worker_state(sim.full_observation());leads.append(sim.cash[seat]-sim.cash[1-seat])
        rows.append({"t":t,"cash_before":cash,"cash_after":list(sim.cash),"joint":orders,"cash_events":list(sim.cash.events),
            "workers":{k:{"pre":v,"command":orders[v["seat"]].get(k,"NONE"),"post":after.get(k)} for k,v in before.items()},
            "born":{k:v for k,v in after.items() if k not in before},"collisions":list(sim.last_collisions)})
    require(list(sim.cash)==g["final_cash"],"actual final cash mismatch")
    first=next((t for t,x in enumerate(leads) if x>500),None)
    windows=[terms_window(rows,leads,n,seat,first) for n in (3,5,10)]
    sets={"paid_lead_or_fallback":windows}
    if first is not None:
        # Explain the largest subsequent ten-turn reversal against the losing candidate from its first crossing.
        fall=[]
        for n in (3,5,10):
            if first+n>len(rows):continue
            start=min(range(first,len(rows)-n+1),key=lambda j:round(leads[j+n]-leads[j],2))
            fall.append(terms_window(rows,leads,n,seat,start+n))
        sets["largest_subsequent_fall"]=fall
    used=sorted({t for v in sets.values() for w in v for t in w["step_rows"]})
    return {"seed":g["seed"],"seat":seat,"result":g["result"],"official_margin":g["margin"],"raw_sha256":g["raw_sha256"],
            "first_cash_lead_above_500":first,"largest_cash_lead":round(max(leads),2),"first_peak_state":leads.index(max(leads)),
            "windows":sets,"turns":[rows[t] for t in used]}


def main():
    base=HERE/"replication.json.gz";trial=read(base)
    candidates=[g for g in trial["games"] if g["arm"]=="candidate"];require(len(candidates)==4,"missing candidate games")
    results=[sample(g) for g in candidates if g["result"]!="win"]
    out={"script_sha256":digest(Path(__file__)),"verified_cohort_sha256":digest(base),"participants":"All fresh-cohort candidate games without a win.",
         "precedence":"First strict focal cash lead >500. If absent, choose the largest positive or least negative gain separately for each horizon; ties go to the earliest interval. If present, also trace the largest subsequent decrease over each full horizon.",
         "games":results}
    p=HERE/"replication_comebacks.json.gz"
    data=gzip.compress(json.dumps(out,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode(),mtime=0)
    if p.exists():require(p.read_bytes()==data,"sealed previous cash record differs")
    else:
        temp=p.with_suffix(".tmp");temp.write_bytes(data);temp.replace(p)
    print(digest(p),[(v["seed"],v["seat"],v["first_cash_lead_above_500"],v["largest_cash_lead"]) for v in results])


if __name__=="__main__":main()
