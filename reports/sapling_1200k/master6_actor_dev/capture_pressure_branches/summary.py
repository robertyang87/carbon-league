"""Pair actual continuation ledgers and separate local cash flow from the whole team's outcome."""
from compare import HERE,sr

def compare(seat,records):
    p,a=[next(r for r in records if r["seat"]==seat and r["arm"]==arm) for arm in ("parent","approach")]
    out=HERE/f"seat{seat}"
    tapes=[sr.load(out/f"live_{arm}_actions.json.gz") for arm in ("parent","approach")]
    assert tapes[0][0][1-seat]==tapes[1][0][1-seat]
    for r in (p,a):
        assert r["root"]==p["root"] and r["root_context"]==p["root_context"]
    rivals=p["nearby_opponents"]
    def difference(offset,ids):
        def command(tape,uid):return tape[offset][1-seat].get(uid)
        return sorted(uid for uid in ids if command(tapes[0],uid)!=command(tapes[1],uid))
    first=next(((i,difference(i,rivals)) for i in range(1,len(tapes[0])) if difference(i,rivals)),None)
    windows={}
    for k,y in a["snapshots"].items():
        x=p["snapshots"][k];flows=[]
        for s in (seat,1-seat):
            keys=y["accounts"][s].keys()|x["accounts"][s].keys()
            flows.append({key:y["accounts"][s].get(key,0)-x["accounts"][s].get(key,0) for key in sorted(keys)})
        windows[k]=dict(delta_own_cash=y["cash"][seat]-x["cash"][seat],
                        delta_enemy_cash=y["cash"][1-seat]-x["cash"][1-seat],delta_own_flows=flows[0],
                        delta_enemy_flows=flows[1],
                        nearby_enemy_deposits=[sum(e["amount"] for e in r["their_deposits"] if e["offset"]<=int(k)) for r in (p,a)])
    return dict(seat=seat,first_local_enemy_order_divergence=first,
                root=p["root"],age=p["root_context"]["age"],margin=[p["margin"],a["margin"]],
                windows=windows)
if __name__=="__main__":
    rows=sr.load(HERE/"ledger.json")
    sr.retain(HERE/"summary.json",[compare(s,rows) for s in (0,1)])
