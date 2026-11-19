"""Separate the count of purchases from the realized return per acquisition."""
from statistics import mean
from verify import HERE,TEN,read,write,sha


def summarize(holdings):
    n=len(holdings)
    net=[h['gross'] + h['collision'] - h['price'] for h in holdings]
    return {"purchases":n, "direct_profit_positive":sum(x>0 for x in net),"direct_profit_sum":sum(net),
            "later_seized_by_opponent":sum(h["exit"]=="seized_by_opponent" for h in holdings),
            "mean_price":mean(h["price"] for h in holdings) if n else None,
            "mean_direct_profit":mean(net) if n else None,
            "mean_held_turns":mean(h["end"]-h["start"] for h in holdings) if n else None}


def main():
    raw=TEN/"tenures.json.gz"
    records=read(raw)
    result={}
    for name,lo,hi in (("51_250",51,250),("251_274",251,274),("275_284",275,284),("285_294",285,294),("295_299",295,299)):
        result[name]={}
        for origin in ("plant","seize"):
            pair={}
            for actor,flip in (("sapling",0),("opponent",1)):
                row=[h for g in records for h in g["holdings"] if h["owner"]==(g["seat"]^flip)
                     and lo<=h["start"]<=hi and h["origin"]==origin]
                pair[actor]=summarize(row)
            result[name][origin]=pair
    by_map={}
    for m in sorted({g["seed"] for g in records}):
        by_map[str(m)]={name:summarize([h for g in records if g["seed"]==m for h in g["holdings"]
                                       if h["owner"]==(g["seat"]^flip) and 50<h["start"]<=250 and h["origin"]=="plant"])
                        for name,flip in (("sapling",0),("opponent",1))}
    pair=result["51_250"]["plant"];a=pair["sapling"];b=pair["opponent"]
    level=(a["purchases"]-b["purchases"])*(a["mean_direct_profit"]+b["mean_direct_profit"])/2/len(records)
    rate=(a["mean_direct_profit"]-b["mean_direct_profit"])*(a["purchases"]+b["purchases"])/2/len(records)
    delta=(a["direct_profit_sum"]-b["direct_profit_sum"])/len(records)
    if abs(level+rate-delta)>1e-8:
        raise ValueError("count / rate identity")
    write(HERE/"cohort.json",{"source_sha256":sha(raw),"implementation_sha256":sha(HERE/"cohort.py"),
                             "counts_and_realized_returns":result,"per_map_mid_new_plant":by_map,"per_game_mid_new_tree_profit_difference":
                             {"total":delta,"count_component":level,"mean_realized_profit_component":rate}})


if __name__=="__main__":
    main()
