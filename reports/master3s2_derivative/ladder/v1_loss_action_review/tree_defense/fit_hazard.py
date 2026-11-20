"""Learn the hazard of an existing tree changing owner using only pre-action public observations."""
import gzip
import hashlib
import io
import tarfile
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1]
sys.path.insert(0, str(LADDER / "tree_payback"))
from register import CACHE, blob
from runtime import distance

FEATURES = [
    "enemy_planter_d0", "enemy_planter_d1", "enemy_planter_d2", "enemy_planter_d3",
    "remaining_life", "posted_rent", "second_enemy_planter_distance", "enemies_within_3",
    "nearest_own_collector", "nearest_own_planter", "nearest_enemy_collector",
    "own_collectors_within_2", "own_empty_collectors_within_2", "opponent_can_pay",
    "nearest_enemy_closing_3", "nearest_enemy_known_3", "nearest_enemy_closing_5",
    "nearest_enemy_known_5", "enemy_stayed_last_action", "threat_has_adjacent_own_collector",
]
def digest(data):
    return hashlib.sha256(data).hexdigest()

def features(obs, history, config, owner, tree, distances):
    players = obs["players"]
    spot, age = players[owner][3][tree]
    enemy = players[1-owner]
    nearest = sorted((distances[spot, entity[0]], uid) for uid, entity in enemy[2].items()
                     if entity[2] == "PLANTER")
    if not nearest or nearest[0][0] > 4:
        return None
    d, uid = nearest[0]
    n = config["size"]
    enemy_spot = enemy[2][uid][0]
    neighbor = []
    for seat, role in ((owner, "COLLECTOR"), (owner, "PLANTER"), (1-owner, "COLLECTOR")):
        neighbor.append([(distances[spot,w[0]], w) for w in players[seat][2].values() if w[2] == role])
    friendly = neighbor[0]
    life = min(config["episodeSteps"]-1-obs["step"], config["treeLifespan"]-age+1)
    x = [float(d == i) for i in range(4)]
    x.extend([life, float(obs["trees"][tree][1]), nearest[1][0] if len(nearest)>1 else n,
              sum(pair[0]<=3 for pair in nearest)])
    x.extend(min((pair[0] for pair in group),default=n) for group in neighbor)
    x.extend([sum(dd<=2 for dd,_ in friendly), sum(dd<=2 and w[1]==0 for dd,w in friendly),
              int(enemy[0] >= config["seizeCost"])])
    for lag in (3,5):
        past = history.get(lag)
        former = past["players"][1-owner][2].get(uid) if past is not None else None
        x.extend([distances[spot, former[0]]-d if former else 0, int(former is not None)])
    previous = history.get(1)
    previous_worker = previous["players"][1-owner][2].get(uid) if previous else None
    x.extend([int(previous_worker is not None and previous_worker[0]==enemy_spot),
              int(any(distances[enemy_spot,w[0]]<=1 for _,w in friendly))])
    return int(d), int(life<=5), [float(v) for v in x]

def collect(game, name, terrain, seats):
    frames = game["steps"]
    cfg = game["configuration"]
    if len(frames)!=cfg["episodeSteps"] or game["statuses"] != ["DONE","DONE"]:
        return []
    if set(("players", "trees", "step")) - {k for k,v in game["specification"]["observation"].items() if v.get("shared")}:
        raise ValueError("Expected shared public observations")
    obs = [step[0]["observation"] for step in frames]
    n = cfg["size"]
    dist = np.array([[distance(i,j,n) for j in range(n*n)] for i in range(n*n)],dtype=int)
    rows = []
    for t in range(0,len(frames)-5,5):
        history = {k:obs[t-k] for k in (1,3,5) if t>=k}
        for seat in seats:
            for tree,(spot,_) in sorted(obs[t]["players"][seat][3].items()):
                inp=features(obs[t],history,cfg,seat,tree,dist)
                if inp is None:
                    continue
                first=next((i for i in range(1,6) if tree in obs[t+i]["players"][1-seat][3]),None)
                d,short,x=inp
                rows.append(dict(game=name, terrain=terrain, t=t, seat=seat, tree=tree,
                                 cell=spot, d=d, short=short, y=int(first is not None), lead=first, x=x))
    return rows

def load_archive(ticket):
    cached = CACHE / (ticket["id"] + ".json.gz")
    source = ticket["source"]
    if cached.is_file():
        raw = cached.read_bytes()
    elif "repo_path" in source:
        raw = blob(source["repo_path"])
    else:
        with tarfile.open(fileobj=io.BytesIO(blob(source["archive"])), mode="r:*") as archive:
            raw = archive.extractfile(source["member"]).read()
    if digest(raw) != ticket["compressed_sha256"]:
        raise ValueError("Training replay changed")
    game = json.loads(gzip.decompress(raw))
    terrain = digest(bytes(game["steps"][0][0]["observation"]["full_carbon"]))
    if terrain != ticket["terrain"]:
        raise ValueError("Training map changed")
    return game, terrain

def load_games():
    tickets = json.loads((LADDER/"tree_payback/maps.json").read_text())["tickets"]
    splits = {"train":[], "held":[], "placement":[]}
    terrains = {split:set() for split in splits}
    for ticket in tickets:
        game,terrain = load_archive(ticket)
        terrains[ticket["split"]].add(terrain)
        splits[ticket["split"]].extend(collect(game,ticket["id"],terrain,(0,1)))
    cards = json.loads((LADDER/"placement_refresh/summary.json").read_text())["all_settled_games"]
    for card in cards:
        if card["kind"]!="newrank-placement" or not card["rated"]:
            continue
        name = f'{card["match"]}-{card["game"]}'
        path = LADDER/"replays"/(name+".json.gz")
        if not path.exists():
            path = LADDER/"placement_refresh/replays"/(name+".json.gz")
        game=json.loads(gzip.decompress(path.read_bytes()))
        terrain=digest(bytes(game["steps"][0][0]["observation"]["full_carbon"]))
        if terrain in terrains["placement"]:
            raise ValueError("Duplicate placement terrain")
        terrains["placement"].add(terrain)
        splits["placement"].extend(collect(game,name,terrain,(card["seat"],)))
    if any(terrains[a]&terrains[b] for a,b in (("train","held"),("train","placement"),("held","placement"))):
        raise ValueError("Repeated terrain across splits")
    counts = {s:len(v) for s,v in terrains.items()}
    if counts != {"train":24,"held":8,"placement":33}:
        raise ValueError("Map set differs from the registered cohort")
    return splits, counts

def fit(rows):
    x=np.array([r["x"] for r in rows])
    y=np.array([r["y"] for r in rows])
    count=Counter(r["game"] for r in rows)
    m=np.array([1/(len(count)*count[r["game"]]) for r in rows])
    mean=m@x
    scale=np.maximum(0.05,np.sqrt(m@((x-mean)**2)))
    z=np.column_stack((np.ones(len(y)),(x-mean)/scale))
    def objective(w):
        logits=z@w
        l=m@(np.logaddexp(0,logits)-y*logits)+0.005*(w[1:]@w[1:])
        g=z.T@(m*(expit(logits)-y))
        g[1:]+=0.01*w[1:]
        return l,g
    result=minimize(objective,np.zeros(z.shape[1]),jac=True,method="L-BFGS-B",tol=1e-10)
    if not result.success:
        raise RuntimeError(result.message)
    base={}
    for d in range(5):
        for short in (0,1):
            in_bin=np.array([r["d"]==d and r["short"]==short for r in rows])
            # A small empirical-Bayes prior stabilizes rare end-of-life cells.
            base[f"{d}-{short}"]=float((m[in_bin]@y[in_bin]+0.002*(m@y))/(m[in_bin].sum()+0.002))
    return dict(features=FEATURES, mean=mean.tolist(),scale=scale.tolist(),coef=result.x.tolist(),base=base)

def evaluate(rows,model):
    group={}
    for r in rows:
        x=(np.array(r.pop("x"))-model["mean"])/model["scale"]
        r["p"]=float(expit(model["coef"][0]+x@model["coef"][1:]))
        r["baseline"]=model["base"][f'{r["d"]}-{r["short"]}']
        group.setdefault(r["game"],[]).append(r)
    by_game={}
    for name,game in group.items():
        y=np.array([r["y"] for r in game])
        rec={"n":len(y),"captures":int(y.sum())}
        for key in ("baseline","p"):
            p=np.clip([r[key] for r in game],1e-9,1-1e-9)
            rank=np.argsort(-p,kind="stable")[:(len(y)+3)//4]
            rec[key]={"brier":float(np.mean((p-y)**2)),
                      "log_loss":float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
                      "top_quarter_hits":int(y[rank].sum()),"top_quarter_n":len(rank)}
        by_game[name]=rec
    mean={k:{metric:float(np.mean([r[k][metric] for r in by_game.values()]))
             for metric in ("brier","log_loss")} for k in ("baseline","p")}
    return dict(maps=len(group),n=len(rows),captures=sum(r["y"] for r in rows),
                mean_over_maps=mean,by_game=by_game)

def main():
    splits,terrains=load_games()
    model=fit(splits["train"])
    scores={s:evaluate(rows,model) for s,rows in splits.items()}
    (HERE/"weights.json").write_text(json.dumps(model,indent=2,allow_nan=False)+"\n")
    (HERE/"results.json").write_text(json.dumps({"terrain_counts":terrains,"scores":scores},indent=2,allow_nan=False)+"\n")
    tested={k:v for k,v in splits.items() if k!="train"}
    (HERE/"predictions.json.gz").write_bytes(gzip.compress(json.dumps(tested,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    print([(k,v["maps"],v["n"],v["captures"],v["mean_over_maps"]) for k,v in scores.items()])

if __name__=="__main__":
    main()
