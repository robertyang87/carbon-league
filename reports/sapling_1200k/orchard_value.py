"""Public features for predicting the receipts of a planter's next purchase."""
import math

def orchard_features(board, carbon, seat, cell, step, cfg):
    own, enemy = board["players"][seat], board["players"][1-seat]
    size = cfg.get("size", 15)
    row, col = divmod(cell, size)
    neighbors = [((row+dr) % size)*size+(col+dc) % size
                 for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))]
    def distance(pos):
        r, c = divmod(int(pos), size)
        dy, dx = abs(row-r), abs(col-c)
        return min(dy,size-dy)+min(dx,size-dx)
    # The planter sees all four adjacent cells before committing. Its current
    # cell is not generally visible, so none of these inputs uses its carbon.
    own_trees = {p for p, _ in own[3].values()}
    enemy_trees = {p:(tid, age) for tid,(p,age) in enemy[3].items()}
    tree = enemy_trees.get(cell)
    if cell in own_trees or cell in own[1].values() or cell in enemy[1].values():
        return None
    life = cfg.get("treeLifespan", 50)
    if tree and tree[1] >= life:
        return None  # expiry can turn a planned seizure into a new purchase
    trees_alive = sum(age < life for p in (own,enemy) for _,age in p[3].values())
    price = cfg.get("seizeCost",20) if tree else cfg.get("plantCost",20)
    if not tree and len(own[3]) > cfg.get("playerTreeProtectiveNumber",0):
        price += cfg.get("plantCostInflationRatio",5)*cfg.get("plantCostInflationBase",1.235)**trees_alive
    horizon = min(life-(tree[1] if tree else 0), cfg.get("episodeSteps",300)-1-step)
    geo = [(distance(pos),role,float(load)) for pos,load,role in enemy[2].values()]
    near = lambda role: min((d for d,r,_ in geo if r==role), default=14)
    previous = float(board.get("trees",{}).get(tree[0],[None,0])[1]) if tree else 0.
    ncarbon = sum(float(carbon[p]) for p in neighbors if p not in own_trees and p not in enemy_trees)
    features = [horizon/50, ncarbon/100, horizon*ncarbon/5000, previous/10,
                previous*horizon/500, near("PLANTER")/10, near("COLLECTOR")/10,
                sum(load for d,r,load in geo if r=="COLLECTOR" and d<=2)/100,
                min(map(distance,own[1].values()))/10, min(map(distance,enemy[1].values()))/10,
                sum(p in own_trees or p in enemy_trees for p in neighbors)/4,
                min(math.log1p(max(0,own[0])),10)/10,
                min((distance(p) for p,_,role in own[2].values() if role=="COLLECTOR"),default=14)/10]
    return ("seize" if tree else "plant"), price, features


def orchard_predict(model, features):
    return max(0., model["intercept"]+sum(x*w for x,w in zip(features,model["weights"])))
