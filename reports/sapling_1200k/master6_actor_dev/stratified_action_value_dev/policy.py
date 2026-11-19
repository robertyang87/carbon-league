"""Pure public features to at most one eligible action in a state."""
TARGETS = ("margin","principal","own_delivery","enemy_delivery")


def vector(p, weights):
    return list(p["features"]) + [float(p["kind"] == k) for k in weights["kinds"]]


def predictions(p, weights):
    x=vector(p,weights)
    return {k:sum(v*c for v,c in zip(x,w,strict=True)) for k,w in weights["heads"].items()}


def choose(proposals, weights):
    if not weights or not weights.get("enabled"):
        return None
    best={}
    for order,p in enumerate(proposals):
        k=p["kind"]
        # Input order is deployment priority then worker identity and physical action order.
        if k not in best or p["priority"] > best[k][0]["priority"]:
            best[k]=p,order
    eligible=[]
    for p,order in best.values():
        y=predictions(p,weights)
        if y["margin"]>15 and y["principal"]>=-5 and y["own_delivery"]>=-5 and y["enemy_delivery"]<=10:
            eligible.append((y["margin"],-order,p))
    return max(eligible,key=lambda x:(x[0],x[1]))[2] if eligible else None
