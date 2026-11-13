"""ledger_books -- cash accounting per side from a replay: deposits, tree income, recruit/seize spend,
plant spend (residual = start cash + deposits + tree income - recruits*30 - seizes*20 - end cash; also absorbs
collision captures, so it is an estimate)."""
def books(st, p):
    dep = tree = 0.0; rec = plants = seizes = 0; prevw = None; prevtrees = set()
    cash0 = st[0][0]['observation']['players'][p][0]
    for t in range(len(st)):
        o = st[t][0]['observation']; pl = o['players'][p]
        mine = set(pl[3].keys()) if len(pl) > 3 else set()
        if t > 0:
            for k in mine:
                v = (o.get('trees') or {}).get(k)
                if v: tree += v[1] or 0
            new = mine - prevtrees
            plants += sum(1 for k in new if k.startswith(f'player-{p}-'))
            seizes += sum(1 for k in new if not k.startswith(f'player-{p}-'))
        if prevw is not None:
            bp = set(pl[1].values())
            for wid, w0 in prevw.items():
                w1 = pl[2].get(wid)
                if w1 and w0[2] == 'COLLECTOR' and w0[1] > 0 and w1[1] == 0 and w1[0] in bp: dep += w0[1]
            rec += sum(1 for w in pl[2] if w not in prevw)
        prevw = {k: list(v) for k, v in pl[2].items()}; prevtrees = mine
    cashT = st[-1][0]['observation']['players'][p][0]
    return dict(dep=dep, tree=tree, rec=rec, plants=plants, seizes=seizes,
                plant_spend=cash0 + dep + tree - rec * 30 - seizes * 20 - cashT, cash=cashT)
